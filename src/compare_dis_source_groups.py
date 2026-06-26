#! /usr/bin/python

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dis_fit_data_sources import SOURCE_GROUPS, build_3he_g1f1_group_bundle, load_source_manifest
from dis_fit_models import evaluate_dis_fit, get_dis_fit_model_keys
from get_dis_fit import fit_dis_model_suite
from utility import project_display_path, project_path

DEFAULT_GROUPS = [
    "plots_baseline",
    "plots_baseline_plus_hermes",
    "current_global_2025",
    "current_global_2025_no_kramer",
    "legacy_mingyu",
]
DEFAULT_MODELS = ["fullx", "quad_alpha", "cubic_alpha", "quad2", "quad", "cubic", "all"]
LOW_X_VALUES = np.array([0.017, 0.024, 0.033, 0.047, 0.065, 0.087, 0.119, 0.168], dtype=np.double)
LOW_Q2_VALUES = np.array([1.5, 2.0, 3.0, 5.0], dtype=np.double)
PARAMETER_COLUMNS = [
    "alpha",
    "a",
    "b",
    "c",
    "beta",
    "d",
    "x0",
    "sigma",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare source-aware 3He DIS fits across source groups.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Source groups to compare. Defaults to the nominal comparison groups.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="DIS model keys to compare. Defaults to canonical model keys plus 'all'.",
    )
    parser.add_argument(
        "--output-dir",
        default=project_path("fit_data", "dis_source_group_comparison"),
        help="Directory where the comparison outputs will be written.",
    )
    return parser.parse_args()


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def _parameter_payload(result):
    payload = {f"param_{name}": np.nan for name in PARAMETER_COLUMNS}
    payload.update({f"param_{name}_err": np.nan for name in PARAMETER_COLUMNS})
    for name, value, error in zip(
        result["parameter_names"],
        result["par_quad"],
        result["par_err_quad"],
    ):
        payload[f"param_{name}"] = _safe_float(value)
        payload[f"param_{name}_err"] = _safe_float(error)
    return payload


def _fit_group_model(group_name, dis_df, model_key):
    indep_data = [dis_df["X"], dis_df["Q2"]]
    x_dense = np.linspace(float(dis_df["X"].min()), float(dis_df["X"].max()), 1000)
    q2_dense = np.full(x_dense.size, 5.0)
    suite = fit_dis_model_suite(
        indep_data,
        dis_df,
        x_dense,
        q2_dense,
        dis_fit_model=model_key,
        source_group=group_name,
    )
    return suite


def _suite_row(group_name, model_key, suite):
    selected_result = suite["selected_result"]
    selected_timing_row = next(
        (
            row for row in suite["timing_rows"]
            if row["model_key"] == selected_result["model_key"] and row["fit_status"] == "success"
        ),
        None,
    )
    total_runtime = float(np.nansum([row["runtime_sec"] for row in suite["timing_rows"]]))
    row = {
        "source_group": group_name,
        "model_key": model_key,
        "selected_model_key": selected_result["model_key"],
        "N": int(selected_timing_row["n_points"]) if selected_timing_row else int(len(selected_result["residuals"])),
        "fit_status": "success",
        "chi2": selected_timing_row["chi2"] if selected_timing_row else np.nan,
        "ndf": selected_timing_row["ndf"] if selected_timing_row else np.nan,
        "chi2_ndf": selected_result["chi2_quad"],
        "runtime_sec": total_runtime if model_key == "all" else (selected_timing_row["runtime_sec"] if selected_timing_row else np.nan),
        "selected_by_all": bool(model_key == "all"),
    }
    row.update(_parameter_payload(selected_result))
    return row


def _lowx_rows(group_name, suite, model_key):
    selected_result = suite["selected_result"]
    rows = []
    for q2_value in LOW_Q2_VALUES:
        q2_array = np.full(LOW_X_VALUES.size, q2_value, dtype=np.double)
        predictions = evaluate_dis_fit(selected_result, LOW_X_VALUES, q2_array)
        for x_value, prediction in zip(LOW_X_VALUES, predictions):
            rows.append(
                {
                    "source_group": group_name,
                    "model_key": model_key,
                    "selected_model_key": selected_result["model_key"],
                    "x": float(x_value),
                    "Q2": float(q2_value),
                    "G1F1_fit": float(prediction),
                    "G1F1_err": np.nan,
                }
            )
    return rows


def _plot_group_data_overlays(group_payloads, output_dir):
    n_groups = len(group_payloads)
    fig, axes = plt.subplots(n_groups, 1, figsize=(14, max(4, 4 * n_groups)), squeeze=False)
    for axis, payload in zip(axes[:, 0], group_payloads):
        dis_df = payload["dis_df"]
        for label in dis_df["Label"].unique():
            label_df = dis_df[dis_df["Label"] == label]
            axis.errorbar(
                label_df["X"],
                label_df["G1F1"],
                yerr=label_df["G1F1.err"],
                fmt="o",
                markersize=5,
                capsize=2,
                linewidth=1.0,
                label=label,
            )
        axis.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        axis.set_title(payload["group_name"])
        axis.set_xlabel("x")
        axis.set_ylabel(r"$g_1^{3He}/F_1^{3He}$")
        axis.set_ylim(-0.04, 0.04)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "source_group_data_overlay.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_model_family_overlays(group_payloads, output_dir):
    canonical_model_keys = get_dis_fit_model_keys()
    fig, axes = plt.subplots(len(group_payloads), 1, figsize=(14, max(4, 4 * len(group_payloads))), squeeze=False)
    x_grid = np.linspace(0.001, 0.85, 1000, dtype=np.double)
    q2_grid = np.full(x_grid.size, 5.0, dtype=np.double)
    for axis, payload in zip(axes[:, 0], group_payloads):
        dis_df = payload["dis_df"]
        axis.errorbar(
            dis_df["X"],
            dis_df["G1F1"],
            yerr=dis_df["G1F1.err"],
            fmt="o",
            color="black",
            markersize=4,
            alpha=0.6,
            label="Data",
        )
        for model_key in canonical_model_keys:
            suite = payload["suite_by_model"].get(model_key)
            if not suite:
                continue
            fit_curve = evaluate_dis_fit(suite["selected_result"], x_grid, q2_grid)
            axis.plot(x_grid, fit_curve, linewidth=1.5, label=f"{model_key}: chi2red={suite['selected_result']['chi2_quad']:.2f}")
        axis.axhline(0.0, color="black", linestyle="--", alpha=0.5)
        axis.set_title(payload["group_name"])
        axis.set_xlabel("x")
        axis.set_ylabel(r"$g_1^{3He}/F_1^{3He}$")
        axis.set_ylim(-0.04, 0.04)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path = os.path.join(output_dir, "model_family_overlay.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _impact_summary_rows(comparison_df, lowx_df):
    lines = []
    baseline = comparison_df[comparison_df["source_group"] == "plots_baseline"]
    baseline_plus_hermes = comparison_df[comparison_df["source_group"] == "plots_baseline_plus_hermes"]
    current_2025 = comparison_df[comparison_df["source_group"] == "current_global_2025"]

    for model_key in ("fullx", "cubic_alpha"):
        base_row = baseline[baseline["model_key"] == model_key]
        hermes_row = baseline_plus_hermes[baseline_plus_hermes["model_key"] == model_key]
        current_row = current_2025[current_2025["model_key"] == model_key]
        if base_row.empty or hermes_row.empty:
            lines.append(f"{model_key}: insufficient successful fits to compare HERMES impact.")
            continue

        base_chi2 = float(base_row.iloc[0]["chi2_ndf"])
        hermes_chi2 = float(hermes_row.iloc[0]["chi2_ndf"])
        lines.append(
            f"{model_key}: HERMES changed chi2/ndf from {base_chi2:.3f} to {hermes_chi2:.3f} "
            f"(delta={hermes_chi2 - base_chi2:+.3f})."
        )

        if not current_row.empty:
            current_chi2 = float(current_row.iloc[0]["chi2_ndf"])
            lines.append(
                f"{model_key}: adding current 2025 DIS points moved chi2/ndf to {current_chi2:.3f} "
                f"(delta vs baseline+HERMES={current_chi2 - hermes_chi2:+.3f})."
            )

        base_lowx = lowx_df[(lowx_df["source_group"] == "plots_baseline") & (lowx_df["model_key"] == model_key)]
        hermes_lowx = lowx_df[(lowx_df["source_group"] == "plots_baseline_plus_hermes") & (lowx_df["model_key"] == model_key)]
        if not base_lowx.empty and not hermes_lowx.empty:
            merged = base_lowx.merge(
                hermes_lowx,
                on=["x", "Q2", "model_key"],
                suffixes=("_baseline", "_hermes"),
            )
            max_shift = np.max(np.abs(merged["G1F1_fit_hermes"] - merged["G1F1_fit_baseline"]))
            lines.append(
                f"{model_key}: max low-x prediction shift from HERMES over the diagnostic grid = {max_shift:.4e}."
            )

    return lines


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_source_manifest()
    group_payloads = []
    comparison_rows = []
    lowx_rows = []
    audit_frames = []

    for group_name in args.groups:
        bundle = build_3he_g1f1_group_bundle(
            group_name,
            manifest,
            source_groups=SOURCE_GROUPS,
            q2_min=1.0,
            dis_w_min=2.0,
        )
        dis_df = bundle["dis_df"]
        suite_by_model = {}

        audit_df = pd.DataFrame(bundle["metadata"]["audit_rows"]).copy()
        audit_df.insert(0, "source_group", group_name)
        audit_frames.append(audit_df)

        for model_key in args.models:
            suite = _fit_group_model(group_name, dis_df, model_key)
            suite_by_model[model_key] = suite
            comparison_rows.append(_suite_row(group_name, model_key, suite))
            lowx_rows.extend(_lowx_rows(group_name, suite, model_key))

        group_payloads.append(
            {
                "group_name": group_name,
                "dis_df": dis_df,
                "suite_by_model": suite_by_model,
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    lowx_df = pd.DataFrame(lowx_rows)
    audit_df = pd.concat(audit_frames, ignore_index=True)

    comparison_path = os.path.join(output_dir, "dis_fit_model_comparison.csv")
    lowx_path = os.path.join(output_dir, "dis_fit_lowx_predictions.csv")
    audit_path = os.path.join(output_dir, "dis_fit_source_audit_summary.csv")
    comparison_df.to_csv(comparison_path, index=False)
    lowx_df.to_csv(lowx_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    plot_files = [
        _plot_group_data_overlays(group_payloads, output_dir),
        _plot_model_family_overlays(group_payloads, output_dir),
    ]

    summary_lines = [
        "3He DIS source-group impact summary",
        f"Output directory: {project_display_path(output_dir)}",
        f"Source groups: {', '.join(args.groups)}",
        f"Models: {', '.join(args.models)}",
        "",
    ]
    summary_lines.extend(_impact_summary_rows(comparison_df, lowx_df))
    summary_lines.extend(
        [
            "",
            "Generated files:",
            f"  {project_display_path(comparison_path)}",
            f"  {project_display_path(lowx_path)}",
            f"  {project_display_path(audit_path)}",
        ]
    )
    for plot_file in plot_files:
        summary_lines.append(f"  {project_display_path(plot_file)}")

    summary_path = os.path.join(output_dir, "dis_fit_impact_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
