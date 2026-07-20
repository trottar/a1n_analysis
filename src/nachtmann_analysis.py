#! /usr/bin/python

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from complete_fit_helpers import evaluate_complete_fit_from_x
from functions import nachtmann_x, quad_nucl_curve_k, x_to_W
from utility import prefix_generated_output_name, project_path

A1N_ALL_SOURCE_KEY = "a1n_all"
SPIN_DUALITY_SOURCE_KEY = "psolv_e01012_g1g2"
NACHTMANN_MASS_GEV = 0.93870319

_SPIN_BIN_COLORS = ["#6a3d9a", "#1f78b4", "#33a02c"]
_SPIN_BIN_MARKERS = ["^", "s", "D"]


def _analysis_output_dir(analysis_tag):
    if analysis_tag == "legacy":
        return project_path("fit_data")
    return project_path("fit_data", analysis_tag)


def _build_artifact_path(analysis_tag, filename):
    output_dir = _analysis_output_dir(analysis_tag)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, prefix_generated_output_name(filename))


def _source_key_mask(df, source_key):
    if "source_key" in df.columns:
        return df["source_key"].fillna("").astype(str).eq(source_key)
    return pd.Series(False, index=df.index)


def _label_mask(df, label_fragment):
    if "Label" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Label"].fillna("").astype(str).str.contains(label_fragment, case=False, regex=False)


def write_nachtmann_g1f1_output(g1f1_df, analysis_tag):
    output_path = _build_artifact_path(analysis_tag, "g1f1_canonical_with_nachtmann.csv")
    preferred_columns = [
        "source_key",
        "source_group",
        "Label",
        "Q2",
        "Q2_labels",
        "X",
        "Nachtmann_x",
        "W",
        "G1F1",
        "G1F1.err",
        "reference",
        "table",
        "notes",
    ]
    available_columns = [column for column in preferred_columns if column in g1f1_df.columns]
    remaining_columns = [column for column in g1f1_df.columns if column not in available_columns]
    export_df = g1f1_df.loc[:, available_columns + remaining_columns].copy()
    export_df.to_csv(output_path, index=False)
    return output_path


def _fallback_spin_q2_labels(spin_df):
    fallback_df = spin_df.copy()
    fallback_df["Q2"] = pd.to_numeric(fallback_df["Q2"], errors="coerce")
    valid_q2 = fallback_df["Q2"].dropna().sort_values().unique()
    if len(valid_q2) == 0:
        fallback_df["Q2_labels"] = "Unbinned spin-duality data"
        return fallback_df

    q2_diffs = np.diff(valid_q2)
    if len(q2_diffs) == 0:
        threshold = 0.05
    else:
        positive_diffs = q2_diffs[q2_diffs > 0]
        median_gap = float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
        threshold = max(0.05, 2.0 * median_gap)

    cluster_edges = [valid_q2[0]]
    for q2_value, gap in zip(valid_q2[1:], q2_diffs):
        if gap > threshold:
            cluster_edges.append(q2_value)

    cluster_map = {}
    cluster_id = 0
    last_q2 = None
    for q2_value in valid_q2:
        if last_q2 is not None and (q2_value - last_q2) > threshold:
            cluster_id += 1
        cluster_map[q2_value] = cluster_id
        last_q2 = q2_value

    fallback_df["_spin_q2_cluster"] = fallback_df["Q2"].map(cluster_map)
    label_map = {}
    for cluster_id, cluster_frame in fallback_df.groupby("_spin_q2_cluster", dropna=True):
        q2_min = float(cluster_frame["Q2"].min())
        q2_max = float(cluster_frame["Q2"].max())
        q2_mean = float(cluster_frame["Q2"].mean())
        n_points = int(len(cluster_frame))
        label_map[cluster_id] = (
            f"Spin bin {int(cluster_id) + 1}: "
            f"{q2_mean:.3f} GeV^2 [{q2_min:.3f}, {q2_max:.3f}] (n={n_points})"
        )
    fallback_df["Q2_labels"] = fallback_df["_spin_q2_cluster"].map(label_map)
    return fallback_df.drop(columns=["_spin_q2_cluster"])


def _spin_q2_bin_stats(spin_df):
    if "Q2_labels" in spin_df.columns:
        candidate_labels = spin_df["Q2_labels"].dropna().astype(str)
        if candidate_labels.nunique() > 0:
            usable_labels = candidate_labels.nunique() > 1 or spin_df["Q2"].nunique() <= 1
            if usable_labels:
                grouped_df = spin_df.copy()
            else:
                grouped_df = _fallback_spin_q2_labels(spin_df)
        else:
            grouped_df = _fallback_spin_q2_labels(spin_df)
    else:
        grouped_df = _fallback_spin_q2_labels(spin_df)

    stats = []
    for label, group in grouped_df.groupby("Q2_labels", dropna=True):
        q2_values = pd.to_numeric(group["Q2"], errors="coerce").dropna()
        if q2_values.empty:
            continue
        stats.append(
            {
                "label": str(label),
                "mean_q2": float(q2_values.mean()),
                "min_q2": float(q2_values.min()),
                "max_q2": float(q2_values.max()),
                "n_points": int(len(group)),
            }
        )
    return grouped_df, sorted(stats, key=lambda item: item["mean_q2"])


def select_nachtmann_display_subset(
    g1f1_df,
    requested_high_q2_bins=3,
    *,
    a1n_source_key=A1N_ALL_SOURCE_KEY,
    spin_source_key=SPIN_DUALITY_SOURCE_KEY,
):
    if requested_high_q2_bins not in {2, 3}:
        raise ValueError(
            "NACHTMANN_SPIN_DUALITY_HIGH_Q2_BINS must be 2 or 3."
        )

    working_df = g1f1_df.copy()
    missing_warnings = []

    a1n_mask = _source_key_mask(working_df, a1n_source_key)
    if not bool(a1n_mask.any()):
        a1n_mask = _label_mask(working_df, "A1n all")
    a1n_frame = working_df.loc[a1n_mask].copy()
    if a1n_frame.empty:
        missing_warnings.append(
            f"A1n ALL source '{a1n_source_key}' is not present in the active normalized g1/F1 DataFrame."
        )

    spin_mask = _source_key_mask(working_df, spin_source_key)
    if not bool(spin_mask.any()):
        spin_mask = _label_mask(working_df, "Solvg. E01-012")
    spin_frame = working_df.loc[spin_mask].copy()
    if spin_frame.empty:
        missing_warnings.append(
            f"Spin-duality source '{spin_source_key}' is not present in the active normalized g1/F1 DataFrame."
        )

    selected_spin_frame = spin_frame.iloc[0:0].copy()
    selected_bin_stats = []
    all_bin_stats = []
    selected_bin_labels = []
    if not spin_frame.empty:
        binned_spin_frame, all_bin_stats = _spin_q2_bin_stats(spin_frame)
        ranked_stats = sorted(all_bin_stats, key=lambda item: item["mean_q2"])
        selected_bin_stats = ranked_stats[-requested_high_q2_bins:]
        selected_bin_labels = [item["label"] for item in sorted(selected_bin_stats, key=lambda item: item["mean_q2"], reverse=True)]
        if selected_bin_stats:
            selected_label_set = {item["label"] for item in selected_bin_stats}
            selected_spin_frame = binned_spin_frame[
                binned_spin_frame["Q2_labels"].astype(str).isin(selected_label_set)
            ].copy()

    selected_df = pd.concat([a1n_frame, selected_spin_frame], ignore_index=True)
    selected_df = selected_df.replace([np.inf, -np.inf], np.nan)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "a1n_all_source_key": a1n_source_key,
        "a1n_all_points": int(len(a1n_frame)),
        "spin_duality_source_key": spin_source_key,
        "requested_high_q2_bins": int(requested_high_q2_bins),
        "selected_bin_labels": selected_bin_labels,
        "selected_spin_duality_bin_stats": selected_bin_stats,
        "all_spin_duality_bin_stats": all_bin_stats,
        "selected_spin_duality_points": int(len(selected_spin_frame)),
        "mass_used_gev": NACHTMANN_MASS_GEV,
        "missing_source_warnings": missing_warnings,
    }
    return selected_df.reset_index(drop=True), metadata


def _resolve_data_ylim(y_values):
    finite_values = np.asarray(y_values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return -0.05, 0.05

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_min == y_max:
        padding = max(0.005, 0.15 * abs(y_min) if y_min != 0.0 else 0.01)
    else:
        padding = max(0.002, 0.12 * (y_max - y_min))
    return y_min - padding, y_max + padding


def create_nachtmann_data_only_outputs(
    g1f1_df,
    analysis_tag,
    pdf,
    mode_label,
    *,
    requested_high_q2_bins=3,
):
    selected_df, metadata = select_nachtmann_display_subset(
        g1f1_df,
        requested_high_q2_bins=requested_high_q2_bins,
    )

    csv_path = _build_artifact_path(analysis_tag, "nachtmann_data_points.csv")
    json_path = _build_artifact_path(analysis_tag, "nachtmann_data_selection.json")
    pdf_path = _build_artifact_path(analysis_tag, "nachtmann_data_only.pdf")
    png_path = _build_artifact_path(analysis_tag, "nachtmann_data_only.png")

    export_columns = [
        column
        for column in [
            "source_key",
            "Label",
            "Q2",
            "Q2_labels",
            "X",
            "Nachtmann_x",
            "W",
            "G1F1",
            "G1F1.err",
            "reference",
            "table",
        ]
        if column in selected_df.columns
    ]
    selected_df.loc[:, export_columns].to_csv(csv_path, index=False)

    print(f"[{mode_label}] Stage: Nachtmann data-only comparison")
    print(f"[{mode_label}] A1n ALL points selected: {metadata['a1n_all_points']}")
    print(f"[{mode_label}] Spin-duality source: {metadata['spin_duality_source_key']}")
    print(f"[{mode_label}] Selected high-Q2 spin-duality bins: {metadata['selected_bin_labels'] or 'none'}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, alpha=0.8)

    if selected_df.empty:
        ax.text(
            0.5,
            0.5,
            "No A1n ALL or selected spin-duality points were available\nfor the Nachtmann data-only comparison.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.05, 0.05)
    else:
        a1n_mask = _source_key_mask(selected_df, A1N_ALL_SOURCE_KEY)
        if not bool(a1n_mask.any()):
            a1n_mask = _label_mask(selected_df, "A1n all")
        a1n_frame = selected_df.loc[a1n_mask].copy()
        if not a1n_frame.empty:
            ax.errorbar(
                a1n_frame["Nachtmann_x"],
                a1n_frame["G1F1"],
                yerr=np.abs(a1n_frame["G1F1.err"]),
                fmt="o",
                linestyle="none",
                color="#17becf",
                ecolor="#17becf",
                capsize=2,
                linewidth=1.0,
                markersize=5,
                label="A1n ALL",
            )

        if "Q2_labels" in selected_df.columns:
            for idx, bin_info in enumerate(metadata["selected_spin_duality_bin_stats"]):
                bin_frame = selected_df[selected_df["Q2_labels"].astype(str) == bin_info["label"]].copy()
                if bin_frame.empty:
                    continue
                ax.errorbar(
                    bin_frame["Nachtmann_x"],
                    bin_frame["G1F1"],
                    yerr=np.abs(bin_frame["G1F1.err"]),
                    fmt=_SPIN_BIN_MARKERS[idx % len(_SPIN_BIN_MARKERS)],
                    linestyle="none",
                    color=_SPIN_BIN_COLORS[idx % len(_SPIN_BIN_COLORS)],
                    ecolor=_SPIN_BIN_COLORS[idx % len(_SPIN_BIN_COLORS)],
                    capsize=2,
                    linewidth=1.0,
                    markersize=6,
                    label=(
                        f"Spin duality: "
                        f"$\\langle Q^2 \\rangle$={bin_info['mean_q2']:.3f} GeV$^2$"
                    ),
                )

        ax.set_xlim(
            max(0.0, float(np.nanmin(selected_df["Nachtmann_x"])) - 0.02),
            min(1.0, float(np.nanmax(selected_df["Nachtmann_x"])) + 0.02),
        )
        ax.set_ylim(*_resolve_data_ylim(selected_df["G1F1"]))

    ax.set_xlabel(r"Nachtmann $\xi$")
    ax.set_ylabel(r"$g_1^{3\mathrm{He}}/F_1^{3\mathrm{He}}$")
    ax.set_title("Nachtmann data-only comparison: A1n ALL + highest spin-duality bins")
    ax.grid(True, linestyle="--", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    metadata["generated_filenames"] = {
        "plot_pdf": os.path.basename(pdf_path),
        "plot_png": os.path.basename(png_path),
        "data_csv": os.path.basename(csv_path),
        "selection_json": os.path.basename(json_path),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return {
        "selected_df": selected_df,
        "metadata": metadata,
        "files": {
            "plot_pdf": pdf_path,
            "plot_png": png_path,
            "data_csv": csv_path,
            "selection_json": json_path,
        },
    }


def _deduplicate_q2_values(values, tolerance=0.05):
    unique_values = []
    for value in sorted(float(candidate) for candidate in values if np.isfinite(candidate) and candidate > 0.0):
        if not unique_values or all(abs(value - existing) > tolerance for existing in unique_values):
            unique_values.append(value)
    return unique_values


def resolve_nachtmann_complete_fit_q2_values(selection_result, q2_override=None):
    if q2_override is not None:
        if not isinstance(q2_override, (list, tuple, np.ndarray)):
            raise ValueError("NACHTMANN_COMPLETE_FIT_Q2_VALUES must be None or a list-like of positive Q2 values.")
        override_values = _deduplicate_q2_values(q2_override, tolerance=1.0e-9)
        if not override_values:
            raise ValueError("NACHTMANN_COMPLETE_FIT_Q2_VALUES did not contain any finite positive Q2 values.")
        return override_values

    metadata = selection_result["metadata"]
    selected_df = selection_result["selected_df"]
    spin_values = [
        item["mean_q2"]
        for item in metadata.get("selected_spin_duality_bin_stats", [])
        if np.isfinite(item.get("mean_q2", np.nan))
    ]
    q2_values = _deduplicate_q2_values(spin_values)

    if "source_key" in selected_df.columns:
        a1n_frame = selected_df[selected_df["source_key"].astype(str) == A1N_ALL_SOURCE_KEY].copy()
    else:
        a1n_frame = selected_df[_label_mask(selected_df, "A1n all")].copy()

    if len(q2_values) < 3 and not a1n_frame.empty:
        a1n_q2 = pd.to_numeric(a1n_frame["Q2"], errors="coerce").dropna().to_numpy(dtype=float)
        if a1n_q2.size:
            candidate_values = [
                float(np.median(a1n_q2)),
                float(np.quantile(a1n_q2, 0.25)),
                float(np.quantile(a1n_q2, 0.75)),
                float(np.mean(a1n_q2)),
            ]
            for candidate in candidate_values:
                if len(q2_values) >= 3:
                    break
                merged = _deduplicate_q2_values(q2_values + [candidate])
                if len(merged) > len(q2_values):
                    q2_values = merged

    if len(q2_values) < 3:
        fallback_spin_values = [
            item["mean_q2"]
            for item in metadata.get("all_spin_duality_bin_stats", [])
            if np.isfinite(item.get("mean_q2", np.nan))
        ]
        for candidate in fallback_spin_values:
            if len(q2_values) >= 3:
                break
            merged = _deduplicate_q2_values(q2_values + [candidate])
            if len(merged) > len(q2_values):
                q2_values = merged

    return q2_values[:4]


def build_nachtmann_complete_fit_curve_frame(
    q2_values,
    complete_curve_evaluator,
    *,
    x_grid=None,
    w_min=1.1,
    w_max=None,
):
    if x_grid is None:
        x_grid = np.linspace(1.0e-4, 1.0, 4000, dtype=np.double)
    x_grid = np.asarray(x_grid, dtype=np.double)

    rows = []
    for q2 in q2_values:
        q2_value = float(q2)
        q2_array = np.full_like(x_grid, q2_value, dtype=np.double)
        w_values = x_to_W(x_grid, q2_array)
        valid_mask = np.isfinite(x_grid) & np.isfinite(w_values) & (w_values >= float(w_min))
        if w_max is not None:
            valid_mask &= w_values <= float(w_max)
        x_valid = x_grid[valid_mask]
        w_valid = w_values[valid_mask]
        if x_valid.size == 0:
            continue

        curve_payload = complete_curve_evaluator(x_valid, q2_value, w_valid)
        y_complete = np.asarray(curve_payload["y_complete"], dtype=np.double)
        nachtmann_values = nachtmann_x(x_valid, np.full_like(x_valid, q2_value), mass=NACHTMANN_MASS_GEV)

        curve_df = pd.DataFrame(
            {
                "Q2": np.full_like(x_valid, q2_value, dtype=np.double),
                "X": x_valid,
                "Nachtmann_x": nachtmann_values,
                "W": w_valid,
                "y_complete": y_complete,
            }
        )
        curve_df = curve_df.replace([np.inf, -np.inf], np.nan)
        curve_df = curve_df.dropna(subset=["Q2", "X", "Nachtmann_x", "W", "y_complete"])
        curve_df = curve_df.sort_values("Nachtmann_x")
        rows.append(curve_df)

    if not rows:
        return pd.DataFrame(columns=["Q2", "X", "Nachtmann_x", "W", "y_complete"])
    return pd.concat(rows, ignore_index=True)


def create_nachtmann_complete_fit_outputs(
    selection_result,
    analysis_tag,
    pdf,
    mode_label,
    dis_fit_params,
    dis_transition_fit,
    k_nucl_par,
    k_nucl_err,
    gamma_nucl_par,
    gamma_nucl_err,
    mass_nucl_par,
    mass_nucl_err,
    k_P_vals,
    gamma_P_vals,
    mass_P_vals,
    *,
    full_w_max,
    q2_override=None,
    w_min=1.1,
    quad_nucl_curve_k_func=quad_nucl_curve_k,
):
    q2_values = resolve_nachtmann_complete_fit_q2_values(selection_result, q2_override=q2_override)
    if not q2_values:
        raise RuntimeError("Could not determine any Q2 values for the Nachtmann complete-fit comparison.")

    print(f"[{mode_label}] Stage: Nachtmann complete-fit comparison")
    print(f"[{mode_label}] Complete-fit Q2 values: {q2_values}")

    def complete_curve_evaluator(x_values, q2_value, w_values):
        return evaluate_complete_fit_from_x(
            q2_value,
            x_values,
            dis_fit_params,
            dis_transition_fit,
            k_nucl_par,
            k_nucl_err,
            gamma_nucl_par,
            gamma_nucl_err,
            mass_nucl_par,
            mass_nucl_err,
            k_P_vals,
            gamma_P_vals,
            mass_P_vals,
            w_values=w_values,
            quad_nucl_curve_k_func=quad_nucl_curve_k_func,
        )

    curve_df = build_nachtmann_complete_fit_curve_frame(
        q2_values,
        complete_curve_evaluator,
        w_min=w_min,
        w_max=full_w_max,
    )

    csv_path = _build_artifact_path(analysis_tag, "nachtmann_complete_fit_curves.csv")
    json_path = _build_artifact_path(analysis_tag, "nachtmann_complete_fit_metadata.json")
    pdf_path = _build_artifact_path(analysis_tag, "nachtmann_complete_fit_only.pdf")
    png_path = _build_artifact_path(analysis_tag, "nachtmann_complete_fit_only.png")
    curve_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, alpha=0.8)

    color_cycle = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for idx, q2_value in enumerate(q2_values):
        q2_frame = curve_df[np.isclose(curve_df["Q2"], q2_value)].copy()
        if q2_frame.empty:
            continue
        ax.plot(
            q2_frame["Nachtmann_x"],
            q2_frame["y_complete"],
            color=color_cycle[idx % len(color_cycle)],
            linewidth=2.0,
            label=fr"$Q^2={q2_value:.3f}$ GeV$^2$",
        )

    ax.set_xlabel(r"Nachtmann $\xi$")
    ax.set_ylabel(r"$g_1^{3\mathrm{He}}/F_1^{3\mathrm{He}}$")
    ax.set_title("Complete-fit curves in Nachtmann ξ")
    ax.grid(True, linestyle="--", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True)
    else:
        ax.text(
            0.5,
            0.5,
            "No finite complete-fit curves were available in the requested W domain.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
    ax.text(
        0.02,
        0.02,
        "Model evaluated in Bjorken x and displayed versus Nachtmann ξ.",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "Q2_values": q2_values,
        "dis_model_key": dis_fit_params["model_key"],
        "requested_dis_model_key": dis_fit_params.get("requested_model_key", dis_fit_params["model_key"]),
        "evaluation_coordinate": "Bjorken x",
        "display_coordinate": "Nachtmann xi",
        "w_domain": {"min": float(w_min), "max": float(full_w_max)},
        "mass_used_gev": NACHTMANN_MASS_GEV,
        "generated_filenames": {
            "plot_pdf": os.path.basename(pdf_path),
            "plot_png": os.path.basename(png_path),
            "curve_csv": os.path.basename(csv_path),
            "metadata_json": os.path.basename(json_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return {
        "curve_df": curve_df,
        "metadata": metadata,
        "files": {
            "plot_pdf": pdf_path,
            "plot_png": png_path,
            "curve_csv": csv_path,
            "metadata_json": json_path,
        },
    }
