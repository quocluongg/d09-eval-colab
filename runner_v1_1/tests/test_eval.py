"""Unit tests for the d09_eval runner (FR-RSK-02, NFR-AI-02)."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from d09_eval.__main__ import main

EVAL_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_JSON = EVAL_ROOT / "experiment.json"
EXPERIMENT_SHA256 = EVAL_ROOT / "experiment.sha256"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().lower()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _make_synthetic_dataset(
    data_dir: Path,
    *,
    n_train: int = 80,
    n_val: int = 40,
    n_test: int = 20,
    seed: int = 123,
) -> Path:
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(EXPERIMENT_JSON.read_text(encoding="utf-8"))
    features: list[str] = cfg["features"]

    rows: list[dict[str, Any]] = []
    idx = 0
    for split_name, count, date_prefix in [
        ("train", n_train, "2026-03-"),
        ("validation", n_val, "2026-08-"),
        ("test", n_test, "2026-11-"),
    ]:
        for i in range(count):
            idx += 1
            day = (i % 25) + 1
            issue_date = f"{date_prefix}{day:02d}"
            decided_at = f"{issue_date}T10:00:00Z"
            label = 1 if (i % 5 == 0) else 0
            row: dict[str, Any] = {}
            for f_idx, feat in enumerate(features):
                if feat in ("duplicate_flag", "missing_po", "cash_split_flag"):
                    row[feat] = bool((i + f_idx) % 7 == 0)
                elif feat in ("invoice_amount", "average_unit_price"):
                    base = 5_000_000.0 if label == 0 else 25_000_000.0
                    row[feat] = float(base + rng.uniform(1000.0, 500000.0))
                elif feat in ("price_diff_ratio", "qty_diff_ratio", "amount_diff_ratio"):
                    row[feat] = float(0.08 * label + rng.uniform(0.0, 0.02))
                elif feat == "previous_error_rate" and i % 11 == 0:
                    row[feat] = np.nan
                else:
                    row[feat] = float(label * 0.5 + rng.normal(0.0, 1.0))
            row["sellerTaxCode"] = f"0100000{(i % 9) + 1:03d}"
            row["series"] = "C26TAA"
            row["number"] = str(idx)
            row["invoiceId"] = f"inv-{idx:05d}"
            row["issueDate"] = issue_date
            row["decidedAt"] = decided_at
            row["label"] = label
            row["split"] = split_name
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = data_dir / "features.csv"
    df.to_csv(csv_path, index=False)

    manifest = {
        "files": {
            "features.csv": _sha256_file(csv_path),
        }
    }
    (data_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return data_dir


def _make_fast_config(tmp_path: Path, seeds: list[int] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(EXPERIMENT_JSON.read_text(encoding="utf-8"))
    seed_list = [42] if seeds is None else seeds
    cfg["seeds"] = {
        "autoencoder": list(seed_list),
        "combo_or": list(seed_list),
        "combo_rank_mean": list(seed_list),
        "isolation_forest": list(seed_list),
        "logistic_regression": [],
        "mlp": list(seed_list),
        "random_forest": list(seed_list),
        "xgboost": list(seed_list),
        "xgboost_if_feature": list(seed_list),
    }
    cfg["models"]["random_forest"]["n_estimators"] = 15
    cfg["models"]["xgboost"]["n_estimators"] = 15
    cfg["models"]["isolation_forest"]["n_estimators"] = 15
    cfg["models"]["mlp"]["max_iter"] = 30
    cfg["models"]["autoencoder"]["max_iter"] = 30
    cfg["models"]["xgboost_if_feature"]["n_estimators"] = 15
    cfg["models"]["combo_rank_mean"]["n_estimators"] = 15
    cfg["models"]["combo_or"]["n_estimators"] = 15
    cfg_path = tmp_path / "experiment.json"
    cfg_bytes = json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")
    cfg_path.write_bytes(cfg_bytes)
    (tmp_path / "experiment.sha256").write_text(
        _sha256_bytes(cfg_bytes) + "\n",
        encoding="utf-8",
    )
    return cfg_path


def test_fr_rsk_02_feature_names_match_srs_when_importable() -> None:
    d09_risk_vec = pytest.importorskip("d09_risk.features.vector")
    cfg = json.loads(EXPERIMENT_JSON.read_text(encoding="utf-8"))
    assert cfg["features"] == list(d09_risk_vec.SRS_FEATURE_NAMES)


def test_fr_rsk_02_config_sha256_mismatch_refuses_to_run(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")
    (tmp_path / "cfg" / "experiment.sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
    assert rc != 0


def test_fr_rsk_02_input_manifest_sha256_mismatch_refuses_to_run(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")
    manifest_path = data_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"files": {"features.csv": "f" * 64}}, indent=2),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    rc = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
    assert rc != 0


def test_fr_rsk_02_split_uses_parameters_changing_dates_changes_row_counts(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=40, n_test=20)
    cfg_dir_a = tmp_path / "cfg_a"
    cfg_dir_a.mkdir()
    cfg_a = _make_fast_config(cfg_dir_a)

    out_a = tmp_path / "out_a"
    rc_a = main(["run", "--data", str(data_dir), "--config", str(cfg_a), "--out", str(out_a)])
    assert rc_a == 0

    cfg_dir_b = tmp_path / "cfg_b"
    cfg_dir_b.mkdir()
    cfg_b = _make_fast_config(cfg_dir_b)
    cfg_obj = json.loads(cfg_b.read_text(encoding="utf-8"))
    cfg_obj["split"]["validationEnd"] = "2026-08-15"
    cfg_bytes = json.dumps(cfg_obj, indent=2, sort_keys=True).encode("utf-8")
    cfg_b.write_bytes(cfg_bytes)
    (cfg_dir_b / "experiment.sha256").write_text(_sha256_bytes(cfg_bytes) + "\n", encoding="utf-8")

    out_b = tmp_path / "out_b"
    rc_b = main(["run", "--data", str(data_dir), "--config", str(cfg_b), "--out", str(out_b)])
    assert rc_b == 0

    pred_a = pd.read_csv(out_a / "logistic_regression" / "predictions.csv")
    pred_b = pd.read_csv(out_b / "logistic_regression" / "predictions.csv")
    assert len(pred_a) != len(pred_b)
    assert len(pred_b) < len(pred_a)


def test_fr_rsk_02_test_split_refused_without_allow_test_and_unlock_file(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")

    # 1. Requesting test split without --allow-test fails
    rc1 = main(
        [
            "run",
            "--data",
            str(data_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "out1"),
            "--split",
            "test",
        ]
    )
    assert rc1 != 0

    # 2. Requesting --allow-test without TEST_UNLOCK.txt fails
    rc2 = main(
        [
            "run",
            "--data",
            str(data_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "out2"),
            "--split",
            "test",
            "--allow-test",
        ]
    )
    assert rc2 != 0

    # 3. With TEST_UNLOCK.txt but without --allow-test fails
    unlock_file = data_dir / "TEST_UNLOCK.txt"
    unlock_file.write_text("unlock test evaluation\n", encoding="utf-8")
    rc3 = main(
        [
            "run",
            "--data",
            str(data_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(tmp_path / "out3"),
            "--split",
            "test",
        ]
    )
    assert rc3 != 0

    # 4. With both --allow-test and TEST_UNLOCK.txt succeeds and appends one line to TESTLOG.txt
    out4 = tmp_path / "out4"
    rc4 = main(
        [
            "run",
            "--data",
            str(data_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(out4),
            "--split",
            "test",
            "--allow-test",
        ]
    )
    assert rc4 == 0
    testlog = data_dir / "TESTLOG.txt"
    assert testlog.exists()
    lines = [ln for ln in testlog.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1

    # Running a second time appends a second line
    out5 = tmp_path / "out5"
    rc5 = main(
        [
            "run",
            "--data",
            str(data_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(out5),
            "--split",
            "test",
            "--allow-test",
        ]
    )
    assert rc5 == 0
    lines2 = [ln for ln in testlog.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines2) == 2


def test_fr_rsk_02_decoy_ground_truth_and_hidden_never_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")

    # Create decoy files in data_dir
    decoy_gt = data_dir / "ground_truth.csv"
    decoy_gt.write_text("sellerTaxCode,series,number,cause\n0100,C26,1,spike\n", encoding="utf-8")
    decoy_hidden_dir = data_dir / "hidden"
    decoy_hidden_dir.mkdir()
    decoy_hidden_file = decoy_hidden_dir / "ground_truth.csv"
    decoy_hidden_file.write_text("sellerTaxCode,series,number,cause\n0100,C26,1,spike\n", encoding="utf-8")

    # Build a synthetic reference file for selfcheck on this dataset
    from d09_eval.runner import build_reference_from_dataset

    ref_path = tmp_path / "ref.json"
    build_reference_from_dataset(data_dir, cfg_path, ref_path)

    accessed_paths: list[str] = []
    orig_open = builtins.open
    orig_path_open = Path.open
    orig_read_csv = pd.read_csv

    def record_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)):
            accessed_paths.append(str(Path(file).resolve()))
        return orig_open(file, *args, **kwargs)

    def record_path_open(self_path: Path, *args: Any, **kwargs: Any) -> Any:
        accessed_paths.append(str(self_path.resolve()))
        return orig_path_open(self_path, *args, **kwargs)

    def record_read_csv(filepath_or_buffer: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(filepath_or_buffer, (str, Path)):
            accessed_paths.append(str(Path(filepath_or_buffer).resolve()))
        return orig_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", record_open)
    monkeypatch.setattr(Path, "open", record_path_open)
    monkeypatch.setattr(pd, "read_csv", record_read_csv)

    out_dir = tmp_path / "out"
    rc_self = main(["selfcheck", "--data", str(data_dir), "--config", str(cfg_path), "--reference", str(ref_path)])
    assert rc_self == 0

    rc_run = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
    assert rc_run == 0

    rc_pack = main(["pack", "--out", str(out_dir)])
    assert rc_pack == 0

    forbidden = [
        p
        for p in accessed_paths
        if "ground_truth" in p.lower() or "hidden" in Path(p).parts
    ]
    assert forbidden == []


def test_fr_rsk_02_run_meta_has_no_label_cause_or_hidden_profile_key(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")
    out_dir = tmp_path / "out"

    rc = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
    assert rc == 0

    meta_files = list(out_dir.rglob("run_meta.json"))
    assert len(meta_files) >= 5

    forbidden_substrings = ("label", "cause", "profile", "hidden")
    for mf in meta_files:
        raw_text = mf.read_text(encoding="utf-8")
        meta = json.loads(raw_text)

        def check_keys(obj: Any) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_low = str(k).lower()
                    for bad in forbidden_substrings:
                        assert bad not in k_low, f"Forbidden key '{k}' in {mf}"
                    check_keys(v)
            elif isinstance(obj, list):
                for item in obj:
                    check_keys(item)

        check_keys(meta)
        lower_text = raw_text.lower()
        for bad in forbidden_substrings:
            assert bad not in lower_text, f"Forbidden word '{bad}' in {mf} text"


def test_fr_rsk_02_selfcheck_tolerance_and_side_flip_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg")

    from d09_eval.runner import build_reference_from_dataset

    ref_path = tmp_path / "ref.json"
    build_reference_from_dataset(data_dir, cfg_path, ref_path)
    base_ref = json.loads(ref_path.read_text(encoding="utf-8"))

    # Case 1: 2e-6 shift in score -> exit non-zero
    ref_shift = json.loads(json.dumps(base_ref))
    ref_shift["rows"][0]["score"] = float(ref_shift["rows"][0]["score"]) + 2e-6
    ref_shift_path = tmp_path / "ref_shift.json"
    ref_shift_path.write_text(json.dumps(ref_shift, indent=2), encoding="utf-8")

    rc_fail = main(
        ["selfcheck", "--data", str(data_dir), "--config", str(cfg_path), "--reference", str(ref_shift_path)]
    )
    assert rc_fail != 0

    # Case 2: tiny shift (<= 1e-6, e.g. 5e-7) that flips a row across the threshold -> WARNING, exit 0
    ref_flip = json.loads(json.dumps(base_ref))
    target_score = float(ref_flip["rows"][0]["score"])
    # Set threshold just above the actual model score by 2e-7, and set reference score to threshold + 2e-7
    # so |actual - ref_score| = 4e-7 <= 1e-6, but actual < threshold while ref_score >= threshold
    new_th = target_score + 2e-7
    ref_flip["threshold"] = new_th
    ref_flip["rows"][0]["score"] = target_score + 4e-7
    ref_flip_path = tmp_path / "ref_flip.json"
    ref_flip_path.write_text(json.dumps(ref_flip, indent=2), encoding="utf-8")

    capsys.readouterr()
    rc_warn = main(
        ["selfcheck", "--data", str(data_dir), "--config", str(cfg_path), "--reference", str(ref_flip_path)]
    )
    captured = capsys.readouterr()
    combined_out = (captured.out + captured.err).upper()
    assert rc_warn == 0
    assert "WARNING" in combined_out


def test_fr_rsk_02_same_seed_gives_identical_predictions_on_two_runs(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    assert main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out1)]) == 0
    assert main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out2)]) == 0

    pred_files_1 = sorted(p.relative_to(out1) for p in out1.rglob("predictions.csv"))
    pred_files_2 = sorted(p.relative_to(out2) for p in out2.rglob("predictions.csv"))
    assert pred_files_1 == pred_files_2
    assert len(pred_files_1) == 9

    for rel in pred_files_1:
        assert (out1 / rel).read_bytes() == (out2 / rel).read_bytes()


def test_fr_rsk_02_xgboost_missing_leads_to_error_never_silent_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data")
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])
    out_dir = tmp_path / "out"

    import sys

    monkeypatch.setitem(sys.modules, "xgboost", None)

    orig_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "xgboost" or name.startswith("xgboost."):
            raise ImportError("xgboost is hidden for test")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="xgboost"):
        rc = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
        if rc != 0:
            raise ImportError("xgboost missing")


def test_fr_rsk_02_score_computes_metrics_and_grouped_bootstrap(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])
    out_dir = tmp_path / "out"

    rc_run = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir)])
    assert rc_run == 0

    df_feat = pd.read_csv(data_dir / "features.csv", dtype={"sellerTaxCode": str, "series": str, "number": str})
    val_df = df_feat[df_feat["split"] == "validation"].copy()
    truth_df = pd.DataFrame(
        {
            "sellerTaxCode": val_df["sellerTaxCode"],
            "series": val_df["series"],
            "number": [f"{int(x):08d}" for x in val_df["number"]],
            "poNumber": [f"PO-2026-{i // 3:05d}" for i in range(len(val_df))],
            "label": val_df["label"].astype(int),
        }
    )
    truth_path = tmp_path / "truth.csv"
    truth_df.to_csv(truth_path, index=False)

    rc_score = main(["score", "--pred", str(out_dir), "--truth", str(truth_path), "--bootstrap", "20"])
    assert rc_score == 0


def test_fr_rsk_02_autoencoder_model(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["seeds"]["autoencoder"] = [42]
    cfg["models"]["autoencoder"] = {
        "alpha": 0.0001,
        "hidden_layer_sizes": [16, 8, 16],
        "max_iter": 30,
        "preprocessing": {
            "imputation": "median_with_indicator",
            "scaler": "StandardScaler",
        },
    }
    cfg_bytes = json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")
    cfg_path.write_bytes(cfg_bytes)
    (tmp_path / "cfg" / "experiment.sha256").write_text(_sha256_bytes(cfg_bytes) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = main(["run", "--data", str(data_dir), "--config", str(cfg_path), "--out", str(out_dir), "--models", "autoencoder"])
    assert rc == 0
    pred_csv = out_dir / "autoencoder" / "seed_42" / "predictions.csv"
    assert pred_csv.exists()
    df_pred = pd.read_csv(pred_csv)
    assert len(df_pred) == 30
    assert (df_pred["score"] >= 0.0).all()


def test_fr_rsk_02_leak_free_combinations(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for m in ["xgboost_if_feature", "combo_rank_mean", "combo_or"]:
        cfg["seeds"][m] = [42]
        cfg["models"][m] = {
            "eval_metric": "logloss",
            "learning_rate": 0.1,
            "max_depth": 3,
            "n_estimators": 15,
            "n_jobs": 1,
            "preprocessing": {"imputation": "none_native_nan"},
            "scale_pos_weight": "train_neg_over_pos",
        }
    cfg_bytes = json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")
    cfg_path.write_bytes(cfg_bytes)
    (tmp_path / "cfg" / "experiment.sha256").write_text(_sha256_bytes(cfg_bytes) + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    rc = main([
        "run",
        "--data", str(data_dir),
        "--config", str(cfg_path),
        "--out", str(out_dir),
        "--models", "xgboost_if_feature", "combo_rank_mean", "combo_or",
    ])
    assert rc == 0
    for m in ["xgboost_if_feature", "combo_rank_mean", "combo_or"]:
        pred_csv = out_dir / m / "seed_42" / "predictions.csv"
        assert pred_csv.exists()
        df_pred = pd.read_csv(pred_csv)
        assert len(df_pred) == 30
        assert (df_pred["score"] >= 0.0).all()
        assert (df_pred["score"] <= 1.0).all()


def test_fr_rsk_02_validation_scores_do_not_depend_on_other_validation_rows(tmp_path: Path) -> None:
    """Validation scores for any row must not depend on other validation rows."""
    from d09_eval.runner import _train_and_score_single, load_and_split_data

    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    cfg_path = _make_fast_config(tmp_path / "cfg", seeds=[42])
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    features_path = data_dir / "features.csv"

    train_df, val_df = load_and_split_data(features_path, cfg, target_split="validation", data_dir=data_dir)
    target_col = next((c for c in train_df.columns if c.lower() == "label"), "label")
    y_train = train_df[target_col].values.astype(int)
    feature_names = list(cfg["features"])
    missing_cfg = cfg.get("missing_values", {})

    modified_val_df = val_df.copy()
    col_to_change = feature_names[0]
    modified_val_df.iloc[0, modified_val_df.columns.get_loc(col_to_change)] = 999999999.0

    for model_name in ("xgboost_if_feature", "combo_rank_mean", "combo_or"):
        model_cfg = cfg["models"][model_name]
        scores_orig, *_ = _train_and_score_single(
            model_name,
            model_cfg,
            missing_cfg,
            feature_names,
            train_df,
            val_df,
            y_train,
            seed=42,
        )
        scores_mod, *_ = _train_and_score_single(
            model_name,
            model_cfg,
            missing_cfg,
            feature_names,
            train_df,
            modified_val_df,
            y_train,
            seed=42,
        )
        np.testing.assert_array_equal(
            scores_orig[1:],
            scores_mod[1:],
            err_msg=f"Validation scores for other rows changed for {model_name}",
        )

        y_train_mod = y_train.copy()
        y_train_mod[:10] = 1 - y_train_mod[:10]
        scores_train_mod, *_ = _train_and_score_single(
            model_name,
            model_cfg,
            missing_cfg,
            feature_names,
            train_df,
            val_df,
            y_train_mod,
            seed=42,
        )
        assert not np.array_equal(scores_orig, scores_train_mod), (
            f"control: scores must react to a train change ({model_name})"
        )


def test_fr_rsk_02_tune_command(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    ranges_path = EVAL_ROOT / "config" / "tune_ranges.json"
    out_dir = tmp_path / "tune_out"

    rc = main([
        "tune",
        "--data", str(data_dir),
        "--ranges", str(ranges_path),
        "--out", str(out_dir),
        "--max-configs", "1",
    ])
    assert rc == 0
    tuned_json = out_dir / "tuned_experiment.json"
    assert tuned_json.exists()
    t_cfg = json.loads(tuned_json.read_text(encoding="utf-8"))
    assert "tune_ranges_sha256" in t_cfg
    assert "tuning_table" in t_cfg
    assert len(t_cfg["tuning_table"]) == 6
    for m in ["logistic_regression", "random_forest", "xgboost", "mlp", "isolation_forest", "autoencoder"]:
        assert len(t_cfg["tuning_table"][m]) == 1
    for combo in ["xgboost_if_feature", "combo_rank_mean", "combo_or"]:
        assert combo in t_cfg["models"]


def test_fr_rsk_02_tune_writes_runnable_config(tmp_path: Path) -> None:
    data_dir = _make_synthetic_dataset(tmp_path / "data", n_train=60, n_val=30, n_test=10)
    ranges_path = EVAL_ROOT / "config" / "tune_ranges.json"
    tune_out = tmp_path / "tune_out"

    rc_tune = main([
        "tune",
        "--data", str(data_dir),
        "--ranges", str(ranges_path),
        "--out", str(tune_out),
        "--max-configs", "1",
    ])
    assert rc_tune == 0

    run_cfg_path = tune_out / "run_config" / "experiment.json"
    run_sha_path = tune_out / "run_config" / "experiment.sha256"
    assert run_cfg_path.exists()
    assert run_sha_path.exists()

    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8"))
    assert "tuning_table" not in run_cfg
    assert "tune_ranges_sha256" not in run_cfg

    run_out = tmp_path / "run_out"
    rc_run = main([
        "run",
        "--data", str(data_dir),
        "--config", str(run_cfg_path),
        "--out", str(run_out),
        "--models", "logistic_regression", "xgboost",
    ])
    assert rc_run == 0
    assert (run_out / "logistic_regression" / "predictions.csv").exists()
    assert (run_out / "xgboost" / "seed_42" / "predictions.csv").exists()



