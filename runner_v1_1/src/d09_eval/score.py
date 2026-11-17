"""Scoring and grouped bootstrap evaluation for d09_eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from d09_eval.runner import select_threshold_for_recall


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
            extra_df["label"] = (
                ~extra_df["decision"].astype(str).str.strip().isin(["APPROVED", "APPROVED_FALSE_ALARM"])
            ).astype(int)
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
            truth_df["label"] = (
                ~truth_df["decision"].astype(str).str.strip().isin(["APPROVED", "APPROVED_FALSE_ALARM"])
            ).astype(int)
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
