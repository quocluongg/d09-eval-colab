"""Core implementation of the d09_eval runner."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

KEY_COLUMNS = ("sellerTaxCode", "series", "number", "invoiceId")
EVAL_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = EVAL_PKG_ROOT / "experiment.json"
DEFAULT_REFERENCE_PATH = EVAL_PKG_ROOT / "reference" / "lr_v1_reference.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _assert_not_forbidden_path(path: Path, *, allow_truth: bool = False) -> None:
    resolved = path.resolve()
    parts_lower = [p.lower() for p in resolved.parts]
    name_lower = resolved.name.lower()
    if not allow_truth:
        if "hidden" in parts_lower or "ground_truth" in name_lower:
            raise ValueError(f"Refusing to access restricted path: {path}")


def verify_config_sha256(config_path: Path) -> tuple[dict[str, Any], str]:
    _assert_not_forbidden_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    sha_file = config_path.with_suffix(".sha256")
    if not sha_file.exists():
        alt_sha = config_path.parent / "experiment.sha256"
        if alt_sha.exists() and config_path.name == "experiment.json":
            sha_file = alt_sha
        else:
            raise FileNotFoundError(f"Configuration sha256 file not found: {sha_file}")

    expected_sha = sha_file.read_text(encoding="utf-8").strip().split()[0].lower()
    actual_sha = compute_sha256(config_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Configuration sha256 mismatch for {config_path}: expected {expected_sha}, got {actual_sha}"
        )

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    return cfg, actual_sha


def verify_manifest_and_get_data_sha(data_dir: Path) -> tuple[Path, str]:
    _assert_not_forbidden_path(data_dir)
    manifest_path = data_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MANIFEST.json not found in {data_dir}")

    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_map = manifest_raw.get("files", manifest_raw) if isinstance(manifest_raw, dict) else {}
    if not isinstance(files_map, dict) or not files_map:
        raise ValueError(f"Invalid MANIFEST.json structure in {manifest_path}")

    for rel_name, expected_hash in sorted(files_map.items()):
        rel_lower = str(rel_name).lower()
        if "hidden" in rel_lower or "ground_truth" in rel_lower:
            continue
        file_path = data_dir / rel_name
        _assert_not_forbidden_path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file listed in MANIFEST.json is missing: {file_path}")
        actual_hash = compute_sha256(file_path)
        if actual_hash != str(expected_hash).strip().lower():
            raise ValueError(
                f"Input sha256 mismatch for {rel_name}: expected {expected_hash}, got {actual_hash}"
            )

    features_path = data_dir / "features.csv"
    _assert_not_forbidden_path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"features.csv not found in {data_dir}")

    features_sha = compute_sha256(features_path)
    return features_path, features_sha


def load_and_split_data(
    features_path: Path,
    cfg: dict[str, Any],
    *,
    target_split: str = "validation",
    allow_test: bool = False,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _assert_not_forbidden_path(features_path)
    if target_split == "test":
        base_dir = data_dir if data_dir is not None else features_path.parent
        unlock_path = base_dir / "TEST_UNLOCK.txt"
        if not allow_test or not unlock_path.exists():
            raise PermissionError(
                "Loading the test split is refused unless both --allow-test is passed and TEST_UNLOCK.txt exists."
            )
        testlog_path = base_dir / "TESTLOG.txt"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with testlog_path.open("a", encoding="utf-8") as logf:
            logf.write(f"{ts}\ttest_split_accessed\t{features_path.name}\n")
    elif target_split != "validation":
        raise ValueError(f"Unsupported target split: {target_split}")

    dtype_spec = {col: str for col in KEY_COLUMNS}
    df = pd.read_csv(features_path, dtype=dtype_spec)

    feature_names: list[str] = list(cfg["features"])
    for col in feature_names:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    split_cfg = cfg["split"]
    train_end = str(split_cfg["trainEnd"])
    val_end = str(split_cfg["validationEnd"])
    date_col = str(split_cfg.get("dateColumn", "decidedAt"))
    fallback_col = str(split_cfg.get("fallbackDateColumn", "issueDate"))

    if date_col in df.columns:
        raw_dates = df[date_col].fillna("").astype(str).str.strip()
        if fallback_col in df.columns:
            fb_dates = df[fallback_col].fillna("").astype(str).str.strip()
            raw_dates = raw_dates.where(raw_dates != "", fb_dates)
    elif fallback_col in df.columns:
        raw_dates = df[fallback_col].fillna("").astype(str).str.strip()
    else:
        raise ValueError(f"Neither {date_col} nor {fallback_col} found in {features_path}")

    date_str = raw_dates.str[:10]

    train_mask = date_str < train_end
    val_mask = (date_str >= train_end) & (date_str < val_end)
    test_mask = date_str >= val_end

    # Drop test rows immediately when not explicitly unlocked for test
    if target_split == "validation":
        df_non_test = df[~test_mask].copy()
        train_df = df_non_test[train_mask[~test_mask]].copy()
        eval_df = df_non_test[val_mask[~test_mask]].copy()
    else:
        train_df = df[train_mask].copy()
        eval_df = df[test_mask].copy()

    return train_df, eval_df


def _prepare_matrices(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_names: list[str],
    *,
    apply_log1p: list[str] | None = None,
    apply_imputer: bool = True,
    apply_scaler: bool = False,
    imputer_strategy: str = "median",
    imputer_indicator: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    tr_feat = train_df[feature_names].copy()
    ev_feat = eval_df[feature_names].copy()

    if apply_log1p:
        for col in apply_log1p:
            if col in tr_feat.columns:
                tr_feat[col] = np.log1p(np.maximum(0.0, tr_feat[col].values.astype(float)))
                ev_feat[col] = np.log1p(np.maximum(0.0, ev_feat[col].values.astype(float)))

    x_train = tr_feat.values.astype(float)
    x_eval = ev_feat.values.astype(float)

    if apply_imputer:
        imputer = SimpleImputer(strategy=imputer_strategy, add_indicator=imputer_indicator)
        x_train = imputer.fit_transform(x_train)
        x_eval = imputer.transform(x_eval)

    if apply_scaler:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_eval = scaler.transform(x_eval)

    return x_train, x_eval


def _train_and_score_single(
    model_name: str,
    model_cfg: dict[str, Any],
    missing_cfg: dict[str, Any],
    feature_names: list[str],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_train: np.ndarray,
    seed: int | None,
    *,
    use_log1p_override: bool | None = None,
) -> tuple[np.ndarray, str, float, float]:
    prep = model_cfg.get("preprocessing", {})
    log1p_cols: list[str] = list(prep.get("log1p_features", []))
    if use_log1p_override is False:
        log1p_cols = []

    imp_mode = prep.get("imputation", "median_with_indicator")
    apply_imputer = imp_mode != "none_native_nan"
    apply_scaler = prep.get("scaler") == "StandardScaler"

    strategy = str(missing_cfg.get("strategy", "median"))
    add_indicator = bool(missing_cfg.get("add_indicator", True))

    start_fit = time.perf_counter()
    x_train, x_eval = _prepare_matrices(
        train_df,
        eval_df,
        feature_names,
        apply_log1p=log1p_cols,
        apply_imputer=apply_imputer,
        apply_scaler=apply_scaler,
        imputer_strategy=strategy,
        imputer_indicator=add_indicator,
    )

    if model_name == "logistic_regression":
        rs = 42 if seed is None else int(seed)
        clf = LogisticRegression(
            C=float(model_cfg.get("C", 10.0)),
            class_weight=model_cfg.get("class_weight", "balanced"),
            solver=str(model_cfg.get("solver", "lbfgs")),
            max_iter=int(model_cfg.get("max_iter", 2000)),
            random_state=rs,
        )
        clf.fit(x_train, y_train)
        fit_sec = time.perf_counter() - start_fit
        start_score = time.perf_counter()
        scores = clf.predict_proba(x_eval)[:, 1].astype(float)
        score_sec = time.perf_counter() - start_score
        return scores, type(clf).__name__, fit_sec, score_sec

    if model_name == "random_forest":
        rs = 42 if seed is None else int(seed)
        max_depth_val = model_cfg.get("max_depth", None)
        clf = RandomForestClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            class_weight=model_cfg.get("class_weight", "balanced_subsample"),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            max_depth=(int(max_depth_val) if max_depth_val is not None else None),
            random_state=rs,
            n_jobs=int(model_cfg.get("n_jobs", 1)),
        )
        clf.fit(x_train, y_train)
        fit_sec = time.perf_counter() - start_fit
        start_score = time.perf_counter()
        scores = clf.predict_proba(x_eval)[:, 1].astype(float)
        score_sec = time.perf_counter() - start_score
        return scores, type(clf).__name__, fit_sec, score_sec

    if model_name == "xgboost":
        import xgboost as xgb

        if xgb is None:
            raise ImportError("xgboost is required and must never fall back")
        rs = 42 if seed is None else int(seed)
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = float(n_neg / n_pos) if n_pos > 0 else 1.0
        clf = xgb.XGBClassifier(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            learning_rate=float(model_cfg.get("learning_rate", 0.1)),
            max_depth=int(model_cfg.get("max_depth", 3)),
            scale_pos_weight=spw,
            random_state=rs,
            n_jobs=int(model_cfg.get("n_jobs", 1)),
            eval_metric=str(model_cfg.get("eval_metric", "logloss")),
        )
        clf.fit(x_train, y_train)
        fit_sec = time.perf_counter() - start_fit
        start_score = time.perf_counter()
        scores = clf.predict_proba(x_eval)[:, 1].astype(float)
        score_sec = time.perf_counter() - start_score
        return scores, type(clf).__name__, fit_sec, score_sec

    if model_name == "isolation_forest":
        rs = 42 if seed is None else int(seed)
        clf = IsolationForest(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            contamination=model_cfg.get("contamination", "auto"),
            random_state=rs,
            n_jobs=int(model_cfg.get("n_jobs", 1)),
        )
        clf.fit(x_train)
        tr_neg_samples = -clf.score_samples(x_train)
        tr_min = float(np.min(tr_neg_samples))
        tr_max = float(np.max(tr_neg_samples))
        fit_sec = time.perf_counter() - start_fit
        start_score = time.perf_counter()
        ev_neg_samples = -clf.score_samples(x_eval)
        if tr_max > tr_min:
            scores = ((ev_neg_samples - tr_min) / (tr_max - tr_min)).astype(float)
        else:
            scores = np.zeros_like(ev_neg_samples, dtype=float)
        scores = np.clip(scores, 0.0, 1.0)
        score_sec = time.perf_counter() - start_score
        return scores, type(clf).__name__, fit_sec, score_sec

    if model_name == "mlp":
        rs = 42 if seed is None else int(seed)
        hidden = tuple(int(x) for x in model_cfg.get("hidden_layer_sizes", [64, 32]))
        clf = MLPClassifier(
            hidden_layer_sizes=hidden,
            early_stopping=bool(model_cfg.get("early_stopping", True)),
            max_iter=int(model_cfg.get("max_iter", 200)),
            random_state=rs,
        )
        clf.fit(x_train, y_train)
        fit_sec = time.perf_counter() - start_fit
        start_score = time.perf_counter()
        scores = clf.predict_proba(x_eval)[:, 1].astype(float)
        score_sec = time.perf_counter() - start_score
        return scores, type(clf).__name__, fit_sec, score_sec

    raise ValueError(f"Unknown model name: {model_name}")


def select_threshold_for_recall(y_true: np.ndarray, y_score: np.ndarray, target_recall: float = 0.90) -> tuple[float, str]:
    p, r, th = precision_recall_curve(y_true, y_score)
    cand: list[tuple[float, float]] = []
    for i in range(len(th)):
        if r[i] >= target_recall:
            cand.append((float(p[i]), float(th[i])))
    if cand:
        _, best_th = max(cand, key=lambda x: (x[0], x[1]))
        return float(best_th), f"recall>={target_recall:.2f}"
    best_f1, best_th = -1.0, 0.5
    for i in range(len(th)):
        prec = float(p[i])
        rec = float(r[i])
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1 or (f1 == best_f1 and float(th[i]) > best_th):
            best_f1 = f1
            best_th = float(th[i])
    return float(best_th), "best_f1"


def build_reference_from_dataset(data_dir: Path, config_path: Path, out_ref_path: Path) -> None:
    cfg, _ = verify_config_sha256(config_path)
    features_path, _ = verify_manifest_and_get_data_sha(data_dir)
    train_df, val_df = load_and_split_data(features_path, cfg, target_split="validation", data_dir=data_dir)

    target_col = next((c for c in train_df.columns if c.lower() == "label"), None)
    if target_col is None:
        raise ValueError("Target column missing in training data")

    y_train = train_df[target_col].values.astype(int)
    y_val = val_df[target_col].values.astype(int)

    scores, _, _, _ = _train_and_score_single(
        "logistic_regression",
        cfg["models"]["logistic_regression"],
        cfg.get("missing_values", {}),
        list(cfg["features"]),
        train_df,
        val_df,
        y_train,
        None,
        use_log1p_override=False,
    )
    th, _ = select_threshold_for_recall(y_val, scores, 0.90)
    cm = confusion_matrix(y_val, (scores >= th).astype(int)).tolist()

    rows = []
    for i, (_, r) in enumerate(val_df.iterrows()):
        rows.append(
            {
                "sellerTaxCode": str(r["sellerTaxCode"]),
                "series": str(r["series"]),
                "number": str(r["number"]),
                "invoiceId": str(r["invoiceId"]),
                "score": float(scores[i]),
            }
        )

    ref_obj = {
        "source_predictions_sha256": compute_sha256(features_path),
        "source_choices_sha256": compute_sha256(config_path),
        "threshold": float(th),
        "confusion_matrix": cm,
        "row_count": len(rows),
        "rows": rows,
    }
    out_ref_path.parent.mkdir(parents=True, exist_ok=True)
    out_ref_path.write_text(json.dumps(ref_obj, indent=2, sort_keys=True), encoding="utf-8")


def run_selfcheck(
    data_dir: Path,
    config_path: Path | None = None,
    reference_path: Path | None = None,
) -> int:
    cfg_file = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    ref_file = reference_path if reference_path is not None else DEFAULT_REFERENCE_PATH
    _assert_not_forbidden_path(data_dir)
    _assert_not_forbidden_path(cfg_file)
    _assert_not_forbidden_path(ref_file)

    cfg, _ = verify_config_sha256(cfg_file)
    features_path, _ = verify_manifest_and_get_data_sha(data_dir)
    if not ref_file.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    ref = json.loads(ref_file.read_text(encoding="utf-8"))
    ref_rows = ref["rows"]
    ref_th = float(ref["threshold"])
    ref_cm = ref["confusion_matrix"]

    train_df, val_df = load_and_split_data(features_path, cfg, target_split="validation", data_dir=data_dir)
    target_col = next((c for c in train_df.columns if c.lower() == "label"), None)
    if target_col is None:
        raise ValueError("Target column missing in dataset")

    y_train = train_df[target_col].values.astype(int)
    y_val = val_df[target_col].values.astype(int)

    # Retrain LR on train and score validation using the reference baseline settings
    scores, _, _, _ = _train_and_score_single(
        "logistic_regression",
        cfg["models"]["logistic_regression"],
        cfg.get("missing_values", {}),
        list(cfg["features"]),
        train_df,
        val_df,
        y_train,
        None,
        use_log1p_override=False,
    )

    if len(scores) != len(ref_rows):
        print(f"SELFCheck FAILED: row count mismatch (actual {len(scores)} vs reference {len(ref_rows)})")
        return 1

    ref_scores = np.array([float(r["score"]) for r in ref_rows], dtype=float)
    max_abs_diff = float(np.max(np.abs(scores - ref_scores))) if len(scores) > 0 else 0.0

    actual_side = (scores >= ref_th).astype(int)
    ref_side = (ref_scores >= ref_th).astype(int)
    side_flips = int(np.sum(actual_side != ref_side))

    actual_cm = confusion_matrix(y_val, actual_side).tolist()
    ref_cm_from_ref_scores = confusion_matrix(y_val, ref_side).tolist()

    print(f"selfcheck max_abs_diff={max_abs_diff:.12e} threshold={ref_th:.12f}")
    print(f"selfcheck actual_cm={actual_cm} reference_cm={ref_cm}")

    if max_abs_diff > 1e-6:
        print(f"SELFCheck FAILED: max_abs_diff {max_abs_diff:.12e} > 1e-6")
        return 1

    if side_flips > 0:
        print(f"WARNING: {side_flips} row(s) changed side of threshold {ref_th} within tolerance (max_abs_diff={max_abs_diff:.12e})")
        return 0

    if actual_cm != ref_cm and ref_cm_from_ref_scores == ref_cm:
        print(f"SELFCheck FAILED: confusion matrix mismatch {actual_cm} != {ref_cm}")
        return 1

    print("SELFCheck OK")
    return 0


def _collect_library_versions(require_xgboost: bool = True) -> dict[str, str]:
    versions: dict[str, str] = {
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "scikit-learn": importlib_metadata.version("scikit-learn"),
    }
    if require_xgboost:
        import xgboost as xgb

        if xgb is None:
            raise ImportError("xgboost is required")
        versions["xgboost"] = str(xgb.__version__)
    return versions


def run_models(
    data_dir: Path,
    config_path: Path,
    out_dir: Path,
    *,
    target_split: str = "validation",
    allow_test: bool = False,
    only_models: list[str] | None = None,
    only_seeds: list[int] | None = None,
) -> int:
    _assert_not_forbidden_path(data_dir)
    _assert_not_forbidden_path(config_path)
    _assert_not_forbidden_path(out_dir)

    cfg, cfg_sha = verify_config_sha256(config_path)
    features_path, data_sha = verify_manifest_and_get_data_sha(data_dir)

    models_cfg: dict[str, Any] = cfg["models"]
    seeds_cfg: dict[str, list[int]] = cfg.get("seeds", {})
    missing_cfg: dict[str, Any] = cfg.get("missing_values", {})
    feature_names: list[str] = list(cfg["features"])

    model_order = ["logistic_regression", "random_forest", "xgboost", "isolation_forest", "mlp"]
    if only_models:
        model_order = [m for m in model_order if m in only_models]

    if "xgboost" in model_order:
        import xgboost as xgb

        if xgb is None:
            raise ImportError("xgboost is required and must never fall back")

    lib_versions = _collect_library_versions(require_xgboost=("xgboost" in model_order))

    train_df, eval_df = load_and_split_data(
        features_path,
        cfg,
        target_split=target_split,
        allow_test=allow_test,
        data_dir=data_dir,
    )

    target_col = next((c for c in train_df.columns if c.lower() == "label"), None)
    if target_col is None:
        raise ValueError("Target column missing in training split")
    y_train = train_df[target_col].values.astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    key_df = pd.DataFrame({col: eval_df[col].astype(str).values for col in KEY_COLUMNS})

    for m_name in model_order:
        if m_name not in models_cfg:
            continue
        m_cfg = models_cfg[m_name]
        m_seeds = list(seeds_cfg.get(m_name, []))
        if only_seeds is not None and m_seeds:
            m_seeds = [s for s in m_seeds if s in only_seeds]

        runs_to_execute: list[tuple[int | None, Path]] = []
        if not m_seeds:
            runs_to_execute.append((None, out_dir / m_name))
        else:
            for s in m_seeds:
                runs_to_execute.append((int(s), out_dir / m_name / f"seed_{int(s)}"))

        for seed_val, run_subdir in runs_to_execute:
            run_subdir.mkdir(parents=True, exist_ok=True)
            scores, class_name, fit_sec, score_sec = _train_and_score_single(
                m_name,
                m_cfg,
                missing_cfg,
                feature_names,
                train_df,
                eval_df,
                y_train,
                seed_val,
            )

            pred_df = key_df.copy()
            pred_df["score"] = scores
            pred_path = run_subdir / "predictions.csv"
            pred_df.to_csv(pred_path, index=False, lineterminator="\n")

            meta: dict[str, Any] = {
                "config_sha256": cfg_sha,
                "data_sha256": data_sha,
                "library_versions": lib_versions,
                "model_class": class_name,
                "model_name": m_name,
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "row_count": int(len(pred_df)),
                "seed": seed_val,
                "split": target_split,
                "timings_seconds": {
                    "fit": round(float(fit_sec), 6),
                    "score": round(float(score_sec), 6),
                    "total": round(float(fit_sec + score_sec), 6),
                },
            }
            meta_path = run_subdir / "run_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            seed_tag = f"seed={seed_val}" if seed_val is not None else "no-seed"
            print(
                f"run model={m_name} {seed_tag} class={class_name} rows={len(pred_df)} "
                f"fit={fit_sec:.3f}s score={score_sec:.3f}s"
            )

    return 0


def pack_outputs(out_dir: Path, zip_path: Path | None = None) -> Path:
    _assert_not_forbidden_path(out_dir)
    if not out_dir.exists() or not out_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {out_dir}")

    files_hash: dict[str, str] = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json" and not p.name.endswith(".zip"):
            _assert_not_forbidden_path(p)
            rel = p.relative_to(out_dir).as_posix()
            files_hash[rel] = compute_sha256(p)

    manifest_obj = {
        "file_count": len(files_hash),
        "files": files_hash,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_zip = zip_path if zip_path is not None else out_dir.parent / f"{out_dir.name}.zip"
    _assert_not_forbidden_path(target_zip)
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and not p.name.endswith(".zip"):
                rel = p.relative_to(out_dir).as_posix()
                zf.write(p, arcname=rel)

    zip_sha = compute_sha256(target_zip)
    print(f"pack zip={target_zip} sha256={zip_sha} files={len(files_hash) + 1}")
    return target_zip


def _grouped_bootstrap_prauc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    unique_groups, group_inv = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    group_indices: list[np.ndarray] = [np.where(group_inv == i)[0] for i in range(n_groups)]

    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []
    for _ in range(n_resamples):
        sampled_g = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_indices[g] for g in sampled_g])
        yt = y_true[idx]
        ys = y_score[idx]
        if 0 < int(yt.sum()) < len(yt):
            boot_vals.append(float(average_precision_score(yt, ys)))

    if not boot_vals:
        base = float(average_precision_score(y_true, y_score))
        return base, base
    low, high = np.percentile(boot_vals, [2.5, 97.5])
    return float(low), float(high)


def score_predictions(
    pred_dir: Path,
    truth_path: Path,
    *,
    extra_truth_path: Path | None = None,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not truth_path.exists():
        raise FileNotFoundError(f"Truth file not found: {truth_path}")

    dtype_spec = {"sellerTaxCode": str, "series": str, "number": str}
    truth_df = pd.read_csv(truth_path, dtype=dtype_spec)
    for col in ("sellerTaxCode", "series", "number"):
        if col in truth_df.columns:
            truth_df[col] = truth_df[col].astype(str).str.strip()
    truth_df["key_number"] = [int(x) for x in truth_df["number"]]

    if "label" not in truth_df.columns and extra_truth_path is not None and extra_truth_path.exists():
        extra_df = pd.read_csv(extra_truth_path, dtype=dtype_spec)
        for col in ("sellerTaxCode", "series", "number"):
            if col in extra_df.columns:
                extra_df[col] = extra_df[col].astype(str).str.strip()
        extra_df["key_number"] = [int(x) for x in extra_df["number"]]
        if "label" in extra_df.columns:
            truth_df = pd.merge(
                truth_df,
                extra_df[["sellerTaxCode", "series", "key_number", "label"]],
                on=["sellerTaxCode", "series", "key_number"],
                how="inner",
                validate="one_to_one",
            )
        elif "decision" in extra_df.columns:
            extra_df["label"] = (~extra_df["decision"].astype(str).str.strip().isin(["APPROVED", "APPROVED_FALSE_ALARM"])).astype(int)
            truth_df = pd.merge(
                truth_df,
                extra_df[["sellerTaxCode", "series", "key_number", "label"]],
                on=["sellerTaxCode", "series", "key_number"],
                how="inner",
                validate="one_to_one",
            )

    if "label" not in truth_df.columns:
        if "flagged" in truth_df.columns:
            truth_df["label"] = truth_df["flagged"].astype(int)
        elif "decision" in truth_df.columns:
            truth_df["label"] = (~truth_df["decision"].astype(str).str.strip().isin(["APPROVED", "APPROVED_FALSE_ALARM"])).astype(int)
        else:
            raise ValueError(f"Truth file {truth_path} has no 'label', 'decision', or 'flagged' column")

    pred_files = sorted(pred_dir.rglob("predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions.csv files found under {pred_dir}")

    results_by_run: dict[str, dict[str, Any]] = {}
    by_model_runs: dict[str, list[dict[str, Any]]] = {}

    for pf in pred_files:
        rel_dir = pf.parent.relative_to(pred_dir).as_posix()
        model_name = pf.parent.parent.name if pf.parent.name.startswith("seed_") else pf.parent.name
        p_df = pd.read_csv(pf, dtype=dtype_spec)
        for col in ("sellerTaxCode", "series", "number"):
            p_df[col] = p_df[col].astype(str).str.strip()
        p_df["key_number"] = [int(x) for x in p_df["number"]]

        merged = pd.merge(
            p_df,
            truth_df,
            on=["sellerTaxCode", "series", "key_number"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_truth"),
        )
        if len(merged) != len(p_df):
            raise ValueError(f"Truth join matched {len(merged)} of {len(p_df)} rows for {pf}")

        y_true = merged["label"].values.astype(int)
        y_score = merged["score"].values.astype(float)

        pr_auc = float(average_precision_score(y_true, y_score))
        roc_auc = float(roc_auc_score(y_true, y_score))
        th, rule_used = select_threshold_for_recall(y_true, y_score, 0.90)
        y_pred = (y_score >= th).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()

        if "poNumber" in merged.columns:
            po_vals = merged["poNumber"].fillna("").astype(str).str.strip().values
            supp_vals = merged["sellerTaxCode"].astype(str).values
            groups = np.where(po_vals != "", po_vals, supp_vals)
            group_col_used = "poNumber"
        elif "po_number" in merged.columns:
            po_vals = merged["po_number"].fillna("").astype(str).str.strip().values
            supp_vals = merged["sellerTaxCode"].astype(str).values
            groups = np.where(po_vals != "", po_vals, supp_vals)
            group_col_used = "po_number"
        else:
            groups = merged["sellerTaxCode"].astype(str).values
            group_col_used = "sellerTaxCode"

        ci_low, ci_high = _grouped_bootstrap_prauc(
            y_true, y_score, groups, n_resamples=n_bootstrap, seed=42
        )

        entry = {
            "model": model_name,
            "run": rel_dir,
            "n": int(len(merged)),
            "positives": int(y_true.sum()),
            "pr_auc": pr_auc,
            "pr_auc_ci95": [ci_low, ci_high],
            "bootstrap_group": group_col_used,
            "roc_auc": roc_auc,
            "threshold": th,
            "threshold_rule": rule_used,
            "precision_at_r90": prec,
            "recall_at_r90": rec,
            "confusion_matrix": cm,
        }
        results_by_run[rel_dir] = entry
        by_model_runs.setdefault(model_name, []).append(entry)

    summary_by_model: dict[str, Any] = {}
    for m_name, runs in by_model_runs.items():
        first = runs[0]
        if len(runs) == 1:
            summary_by_model[m_name] = first
        else:
            pr_list = [r["pr_auc"] for r in runs]
            roc_list = [r["roc_auc"] for r in runs]
            prec_list = [r["precision_at_r90"] for r in runs]
            rec_list = [r["recall_at_r90"] for r in runs]
            summary_by_model[m_name] = {
                "model": m_name,
                "seeds_count": len(runs),
                "n": first["n"],
                "positives": first["positives"],
                "pr_auc_mean": float(np.mean(pr_list)),
                "pr_auc_std": float(np.std(pr_list, ddof=0)),
                "roc_auc_mean": float(np.mean(roc_list)),
                "roc_auc_std": float(np.std(roc_list, ddof=0)),
                "precision_at_r90_mean": float(np.mean(prec_list)),
                "precision_at_r90_std": float(np.std(prec_list, ddof=0)),
                "recall_at_r90_mean": float(np.mean(rec_list)),
                "seed_42": first,
                "runs": runs,
            }

    print("=== SCORE PER RUN ===")
    for rel_dir, r in results_by_run.items():
        ci = r["pr_auc_ci95"]
        print(
            f"{rel_dir}: PR-AUC={r['pr_auc']:.6f} (95% CI [{ci[0]:.6f}, {ci[1]:.6f}] by {r['bootstrap_group']}) "
            f"ROC-AUC={r['roc_auc']:.6f} th={r['threshold']:.6f} "
            f"P@R90={r['precision_at_r90']:.6f} R@R90={r['recall_at_r90']:.6f} CM={r['confusion_matrix']}"
        )

    print("\n=== SCORE SUMMARY BY MODEL ===")
    for m_name, s in summary_by_model.items():
        if "pr_auc_mean" in s:
            s42 = s["seed_42"]
            ci = s42["pr_auc_ci95"]
            print(
                f"{m_name} ({s['seeds_count']} seeds): mean PR-AUC={s['pr_auc_mean']:.6f} (std={s['pr_auc_std']:.6f}), "
                f"mean ROC-AUC={s['roc_auc_mean']:.6f}, mean P@R90={s['precision_at_r90_mean']:.6f} | "
                f"seed_42 PR-AUC={s42['pr_auc']:.6f} CI95=[{ci[0]:.6f}, {ci[1]:.6f}] "
                f"P@R90={s42['precision_at_r90']:.6f} R@R90={s42['recall_at_r90']:.6f} CM={s42['confusion_matrix']}"
            )
        else:
            ci = s["pr_auc_ci95"]
            print(
                f"{m_name} (single): PR-AUC={s['pr_auc']:.6f} CI95=[{ci[0]:.6f}, {ci[1]:.6f}] "
                f"ROC-AUC={s['roc_auc']:.6f} P@R90={s['precision_at_r90']:.6f} "
                f"R@R90={s['recall_at_r90']:.6f} CM={s['confusion_matrix']}"
            )

    return {"by_run": results_by_run, "by_model": summary_by_model}
