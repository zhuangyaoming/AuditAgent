"""Macro/Micro metric aggregation and ablation comparison.

Computes dataset-level aggregates from per-case LLM Judge metrics,
and compares results across ablation experiment variants.
"""

import json
from pathlib import Path

import pandas as pd


def compute_macro_micro(df: pd.DataFrame) -> dict:
    """Compute macro and micro averages from per-case Eval_Metrics DataFrame.

    Args:
        df: DataFrame with columns from compute_case_metrics() output
            (R_I, P_I, F1_I, R_E, n_gt(dedup)_R_I_denom, etc.)

    Returns:
        Dict with Macro_* and Micro_* aggregates.
    """
    df = df[df["CaseId"].astype(str) != "SUMMARY"]

    if len(df) == 0:
        return {
            "N": 0, "Macro_R_I": 0.0, "Macro_P_I": 0.0, "Macro_F1_I": 0.0,
            "Macro_R_E": 0.0, "Micro_R_I": 0.0, "Micro_P_I": 0.0,
            "Micro_F1_I": 0.0, "Micro_R_E": 0.0,
        }

    n_cases = len(df)

    # Macro: simple average of per-case rates
    macro_r_i = pd.to_numeric(df["R_I"], errors="coerce").fillna(0).mean()
    macro_p_i = pd.to_numeric(df["P_I"], errors="coerce").fillna(0).mean()
    macro_f1_i = pd.to_numeric(df["F1_I"], errors="coerce").fillna(0).mean()
    macro_r_e = pd.to_numeric(df["R_E"], errors="coerce").fillna(0).mean()

    # Micro: global sum of numerators / global sum of denominators
    total_gt = pd.to_numeric(df["n_gt(dedup)_R_I_denom"], errors="coerce").fillna(0).sum()
    total_rf = pd.to_numeric(df["n_risk_facts(P_I_denom)"], errors="coerce").fillna(0).sum()
    total_rt1_ri = pd.to_numeric(df["n_risk_title_1(R_I_num)"], errors="coerce").fillna(0).sum()
    total_rt1_pi = pd.to_numeric(df["n_risk_title_1(P_I_num)"], errors="coerce").fillna(0).sum()
    total_gt_ev = pd.to_numeric(df["n_gt_evidence(R_E_denom)"], errors="coerce").fillna(0).sum()
    total_ev1 = pd.to_numeric(df["n_ev_chain_1(R_E_num)"], errors="coerce").fillna(0).sum()

    micro_r_i = min(total_rt1_ri, total_gt) / total_gt if total_gt > 0 else 0.0
    micro_p_i = total_rt1_pi / total_rf if total_rf > 0 else 0.0
    micro_f1_i = (2 * micro_p_i * micro_r_i / (micro_p_i + micro_r_i)
                  if (micro_p_i + micro_r_i) > 0 else 0.0)
    micro_r_e = min(total_ev1, total_gt_ev) / total_gt_ev if total_gt_ev > 0 else 0.0

    return {
        "N": n_cases,
        "Macro_R_I": round(macro_r_i, 6),
        "Macro_P_I": round(macro_p_i, 6),
        "Macro_F1_I": round(macro_f1_i, 6),
        "Macro_R_E": round(macro_r_e, 6),
        "Micro_R_I": round(micro_r_i, 6),
        "Micro_P_I": round(micro_p_i, 6),
        "Micro_F1_I": round(micro_f1_i, 6),
        "Micro_R_E": round(micro_r_e, 6),
    }


def compare_ablations(
    baseline_path: Path,
    variant_paths: dict[str, Path],
) -> pd.DataFrame:
    """Compare multiple ablation experiment results against baseline.

    Args:
        baseline_path: Path to baseline Eval_Metrics Excel file.
        variant_paths: {variant_name: path} dict of ablation variants.

    Returns:
        DataFrame with comparison metrics (deltas and relative changes).
    """
    df_base = pd.read_excel(baseline_path, sheet_name="Eval_Metrics")
    df_base = df_base[df_base["CaseId"].astype(str) != "SUMMARY"]
    base_agg = compute_macro_micro(df_base)

    rows = []
    # Baseline row
    rows.append({"Variant": "Baseline", **base_agg, "Delta_F1_I": 0.0, "RelChange_F1_I": 0.0})

    for var_name, var_path in variant_paths.items():
        if not Path(var_path).exists():
            rows.append({
                "Variant": var_name, "N": 0,
                "Macro_R_I": 0, "Macro_P_I": 0, "Macro_F1_I": 0, "Macro_R_E": 0,
                "Micro_R_I": 0, "Micro_P_I": 0, "Micro_F1_I": 0, "Micro_R_E": 0,
                "Delta_F1_I": 0, "RelChange_F1_I": 0,
            })
            continue

        df_var = pd.read_excel(var_path, sheet_name="Eval_Metrics")
        df_var = df_var[df_var["CaseId"].astype(str) != "SUMMARY"]
        var_agg = compute_macro_micro(df_var)

        delta_f1 = var_agg["Macro_F1_I"] - base_agg["Macro_F1_I"]
        rel_f1 = delta_f1 / base_agg["Macro_F1_I"] if base_agg["Macro_F1_I"] > 0 else 0.0

        rows.append({
            "Variant": var_name,
            **var_agg,
            "Delta_F1_I": round(delta_f1, 6),
            "RelChange_F1_I": round(rel_f1, 6),
        })

    return pd.DataFrame(rows)


def per_case_diff(
    baseline_path: Path,
    variant_path: Path,
) -> pd.DataFrame:
    """Compute per-case metric differences between baseline and one variant.

    Args:
        baseline_path: Baseline Excel file.
        variant_path: Variant Excel file.

    Returns:
        DataFrame with per-case R_I, F1_I, R_E deltas.
    """
    df_base = pd.read_excel(baseline_path, sheet_name="Eval_Metrics")
    df_var = pd.read_excel(variant_path, sheet_name="Eval_Metrics")

    df_base = df_base[df_base["CaseId"].astype(str) != "SUMMARY"]
    df_var = df_var[df_var["CaseId"].astype(str) != "SUMMARY"]

    merged = df_base.merge(df_var, on="CaseId", suffixes=("_base", "_var"), how="inner")

    result = pd.DataFrame()
    result["CaseId"] = merged["CaseId"]
    for metric in ["R_I", "P_I", "F1_I", "R_E"]:
        base_col = f"{metric}_base"
        var_col = f"{metric}_var"
        result[f"{metric}_delta"] = (
            pd.to_numeric(merged[var_col], errors="coerce").fillna(0)
            - pd.to_numeric(merged[base_col], errors="coerce").fillna(0)
        ).round(6)

    result["Direction"] = result["F1_I_delta"].apply(
        lambda x: "Improved" if x > 0.001 else ("Degraded" if x < -0.001 else "Same")
    )
    return result.sort_values("F1_I_delta")


def print_summary_table(agg: dict) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 60)
    print(f"EVALUATION SUMMARY (N={agg['N']})")
    print("=" * 60)
    print(f"  {'Macro':<12} R_I={agg['Macro_R_I']:.4f}  P_I={agg['Macro_P_I']:.4f}  "
          f"F1_I={agg['Macro_F1_I']:.4f}  R_E={agg['Macro_R_E']:.4f}")
    print(f"  {'Micro':<12} R_I={agg['Micro_R_I']:.4f}  P_I={agg['Micro_P_I']:.4f}  "
          f"F1_I={agg['Micro_F1_I']:.4f}  R_E={agg['Micro_R_E']:.4f}")
    print("=" * 60)
