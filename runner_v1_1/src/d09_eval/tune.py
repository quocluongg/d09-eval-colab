"""Hyperparameter tuning with equal budget per model (FR-RSK-02)."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from d09_eval.runner import (
    DEFAULT_CONFIG_PATH,
    _assert_not_forbidden_path,
    _prepare_matrices,
    _train_and_score_single,
    load_and_split_data,
    verify_config_sha256,
    verify_manifest_and_get_data_sha,
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def run_tuning(
    data_dir_or_file: Path,
    ranges_path: Path,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    max_configs: int | None = None,
) -> int:
    _assert_not_forbidden_path(data_dir_or_file)
    _assert_not_forbidden_path(ranges_path)
    _assert_not_forbidden_path(out_dir)

    cfg_file = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    _assert_not_forbidden_path(cfg_file)
    base_cfg, _ = verify_config_sha256(cfg_file)

    if not ranges_path.exists():
        raise FileNotFoundError(f"Tune ranges file not found: {ranges_path}")
    ranges_sha = compute_sha256(ranges_path)
    ranges_raw: dict[str, list[dict[str, Any]]] = json.loads(ranges_path.read_text(encoding="utf-8"))

    if data_dir_or_file.is_file():
        features_path = data_dir_or_file
        data_dir = data_dir_or_file.parent
    else:
        features_path, _ = verify_manifest_and_get_data_sha(data_dir_or_file)
        data_dir = data_dir_or_file

    train_df, val_df = load_and_split_data(features_path, base_cfg, target_split="validation", data_dir=data_dir)
    target_col = next((c for c in train_df.columns if c.lower() == "label"), None)
    if target_col is None:
        raise ValueError("Target column missing in training split")
    y_train = train_df[target_col].values.astype(int)
    y_val = val_df[target_col].values.astype(int)

    feature_names: list[str] = list(base_cfg["features"])
    missing_cfg: dict[str, Any] = base_cfg.get("missing_values", {})

    seeds_5 = [42, 43, 44, 45, 46]
    model_order = ["logistic_regression", "random_forest", "xgboost", "mlp", "isolation_forest", "autoencoder"]

    tuning_scores_table: dict[str, list[dict[str, Any]]] = {}
    chosen_settings: dict[str, dict[str, Any]] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting tuning process (max_configs={max_configs}) across {len(model_order)} base models...")

    for m_name in model_order:
        configs = ranges_raw.get(m_name, [])
        if max_configs is not None:
            configs = configs[:max_configs]

        if not configs:
            continue

        model_start_time = time.perf_counter()
        scores_for_model: list[dict[str, Any]] = []
        best_cfg: dict[str, Any] | None = None
        best_pr_auc = -1.0

        base_model_cfg = base_cfg.get("models", {}).get(m_name, {})

        for cfg_idx, cfg_item in enumerate(configs):
            merged_m_cfg = copy.deepcopy(base_model_cfg)
            merged_m_cfg.update(cfg_item)

            prep = merged_m_cfg.get("preprocessing", {})
            log1p_cols = list(prep.get("log1p_features", []))
            apply_imputer = prep.get("imputation", "median_with_indicator") != "none_native_nan"
            apply_scaler = prep.get("scaler") == "StandardScaler"
            strategy = str(missing_cfg.get("strategy", "median"))
            add_indicator = bool(missing_cfg.get("add_indicator", True))

            x_train, x_val = _prepare_matrices(
                train_df,
                val_df,
                feature_names,
                apply_log1p=log1p_cols,
                apply_imputer=apply_imputer,
                apply_scaler=apply_scaler,
                imputer_strategy=strategy,
                imputer_indicator=add_indicator,
            )
            precomputed = (x_train, x_val)

            if m_name == "logistic_regression":
                eval_seeds: list[int | None] = [None]
            else:
                eval_seeds = list(seeds_5)

            pr_aucs: list[float] = []
            for s in eval_seeds:
                scores, _, _, _ = _train_and_score_single(
                    m_name,
                    merged_m_cfg,
                    missing_cfg,
                    feature_names,
                    train_df,
                    val_df,
                    y_train,
                    s,
                    precomputed_matrices=precomputed,
                )
                pr_auc = float(average_precision_score(y_val, scores))
                pr_aucs.append(pr_auc)

            del x_train, x_val, precomputed
            gc.collect()

            mean_pr_auc = float(np.mean(pr_aucs))
            entry = {
                "config_index": cfg_idx,
                "config": cfg_item,
                "mean_val_pr_auc": round(mean_pr_auc, 6),
                "seed_val_pr_aucs": [round(x, 6) for x in pr_aucs],
            }
            scores_for_model.append(entry)

            if mean_pr_auc > best_pr_auc:
                best_pr_auc = mean_pr_auc
                best_cfg = copy.deepcopy(merged_m_cfg)

        model_duration = time.perf_counter() - model_start_time
        tuning_scores_table[m_name] = scores_for_model
        chosen_settings[m_name] = best_cfg if best_cfg is not None else base_model_cfg
        print(f"tune model={m_name} configs={len(configs)} best_pr_auc={best_pr_auc:.6f} duration={model_duration:.2f}s")
        gc.collect()

    # Combinations reuse chosen settings from XGBoost and Isolation Forest
    chosen_xgb_cfg = chosen_settings.get("xgboost", base_cfg.get("models", {}).get("xgboost", {}))
    chosen_if_cfg = chosen_settings.get("isolation_forest", base_cfg.get("models", {}).get("isolation_forest", {}))

    for combo_name in ("xgboost_if_feature", "combo_rank_mean", "combo_or"):
        c_cfg = copy.deepcopy(chosen_xgb_cfg)
        c_cfg["if_n_estimators"] = chosen_if_cfg.get("n_estimators", 300)
        c_cfg["contamination"] = chosen_if_cfg.get("contamination", "auto")
        chosen_settings[combo_name] = c_cfg

    run_cfg: dict[str, Any] = copy.deepcopy(base_cfg)
    run_cfg["models"] = chosen_settings

    run_cfg_dir = out_dir / "run_config"
    run_cfg_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_path = run_cfg_dir / "experiment.json"
    run_cfg_bytes = (json.dumps(run_cfg, indent=2, sort_keys=True) + "\n").encode("utf-8")
    run_cfg_path.write_bytes(run_cfg_bytes)
    run_sha = compute_sha256(run_cfg_path)
    (run_cfg_dir / "experiment.sha256").write_text(f"{run_sha}  experiment.json\n", encoding="utf-8")

    tuned_exp: dict[str, Any] = copy.deepcopy(run_cfg)
    tuned_exp["tune_ranges_sha256"] = ranges_sha
    tuned_exp["tuning_table"] = tuning_scores_table

    out_file = out_dir / "tuned_experiment.json"
    out_file.write_text(json.dumps(tuned_exp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Tuning complete. Saved configuration to {out_file} and runnable config to {run_cfg_path}")
    return 0
