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
_Q2_DUPLICATE_TOLERANCE = 1.0e-9
_SPIN_BIN_MARKERS = (
    "^", "s", "D", "P", "X", "v", "<", ">", "h", "p", "*", "8", "H", "d", "o",
)


def _normalize_requested_q2_values(values, setting_name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{setting_name} must be a nonempty list-like sequence of finite positive Q2 values.")
    try:
        normalized_values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{setting_name} must be a nonempty list-like sequence of finite positive Q2 values."
        ) from exc

    if not normalized_values:
        raise ValueError(f"{setting_name} must not be empty.")
    for value in normalized_values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{setting_name} values must be finite and strictly positive; received {value!r}.")
    for value_index, value in enumerate(normalized_values):
        for prior_value in normalized_values[:value_index]:
            if abs(value - prior_value) <= _Q2_DUPLICATE_TOLERANCE:
                raise ValueError(
                    f"{setting_name} contains duplicate or effectively identical values "
                    f"({prior_value:.12g} and {value:.12g})."
                )
    return normalized_values


def validate_nachtmann_q2_configuration(
    requested_data_q2_values,
    q2_match_tolerance,
    complete_fit_q2_values=None,
):
    """Validate and normalize the user-controlled Nachtmann Q2 settings."""
    normalized_requested_values = _normalize_requested_q2_values(
        requested_data_q2_values,
        "NACHTMANN_Q2_VALUES",
    )
    try:
        normalized_match_tolerance = float(q2_match_tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("NACHTMANN_Q2_MATCH_TOLERANCE must be a finite nonnegative number.") from exc
    if not np.isfinite(normalized_match_tolerance) or normalized_match_tolerance < 0.0:
        raise ValueError("NACHTMANN_Q2_MATCH_TOLERANCE must be a finite nonnegative number.")

    normalized_complete_values = None
    if complete_fit_q2_values is not None:
        normalized_complete_values = _normalize_requested_q2_values(
            complete_fit_q2_values,
            "NACHTMANN_COMPLETE_FIT_Q2_VALUES",
        )
    return (
        normalized_requested_values,
        normalized_match_tolerance,
        normalized_complete_values,
    )


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


def _spin_source_mask(df, source_key=SPIN_DUALITY_SOURCE_KEY):
    # Normalized data carry manifest provenance.  Once that column exists,
    # never recover a missing E01-012 source from a similar human label.
    if "source_key" in df.columns:
        return _source_key_mask(df, source_key)
    # Retain the legacy-label fallback only for older tables with no source
    # provenance column at all.
    return _label_mask(df, "Solvg. E01-012")


def _available_spin_bin_summary(all_bin_stats):
    if not all_bin_stats:
        return "none"
    return "; ".join(
        f"{item['label']} (mean={item['mean_q2']:.6g} GeV^2)"
        for item in all_bin_stats
    )


def _resolve_requested_q2_bins(all_bin_stats, requested_q2_values, q2_match_tolerance):
    """Resolve requested Q2 values against the normalized analysis-bin structure."""
    requested_values = _normalize_requested_q2_values(requested_q2_values, "NACHTMANN_Q2_VALUES")
    if not all_bin_stats:
        raise ValueError("No normalized Q2-label bins are available for Nachtmann matching.")

    available_summary = _available_spin_bin_summary(all_bin_stats)
    matches = []
    unmatched_values = []
    matched_labels = set()
    duplicate_match_values = []
    for requested_value in requested_values:
        nearest_bin = min(
            all_bin_stats,
            key=lambda item: abs(requested_value - item["mean_q2"]),
        )
        absolute_difference = abs(requested_value - nearest_bin["mean_q2"])
        if absolute_difference > q2_match_tolerance:
            unmatched_values.append(requested_value)
            continue
        if nearest_bin["label"] in matched_labels:
            duplicate_match_values.append(
                (requested_value, nearest_bin["label"], nearest_bin["mean_q2"])
            )
            continue
        matched_labels.add(nearest_bin["label"])
        matches.append(
            {
                "requested_q2": float(requested_value),
                "resolved_label": str(nearest_bin["label"]),
                "resolved_mean_q2": float(nearest_bin["mean_q2"]),
                "min_q2": float(nearest_bin["min_q2"]),
                "max_q2": float(nearest_bin["max_q2"]),
                "n_points": int(nearest_bin["n_points"]),
                "absolute_difference": float(absolute_difference),
            }
        )

    if unmatched_values:
        requested_text = ", ".join(f"{value:.12g}" for value in unmatched_values)
        raise ValueError(
            "Unmatched requested normalized Q2 value(s): "
            f"{requested_text}. Available normalized bin means and labels: {available_summary}. "
            f"Configured NACHTMANN_Q2_MATCH_TOLERANCE={q2_match_tolerance:.12g} GeV^2."
        )
    if duplicate_match_values:
        collision_text = "; ".join(
            f"{requested_value:.12g} -> {label} (mean={mean_q2:.12g})"
            for requested_value, label, mean_q2 in duplicate_match_values
        )
        raise ValueError(
            "NACHTMANN_Q2_VALUES cannot be matched one-to-one to normalized Q2 bins: "
            f"{collision_text}. Available normalized bin means and labels: {available_summary}. "
            f"Configured NACHTMANN_Q2_MATCH_TOLERANCE={q2_match_tolerance:.12g} GeV^2."
        )
    return matches


def _select_resolved_spin_bin_rows(plot_df, resolved_bin_label, spin_source_key=SPIN_DUALITY_SOURCE_KEY):
    """Recover one plotted E01-012 bin without admitting another source sharing its label."""
    if "Q2_labels" not in plot_df.columns:
        return plot_df.iloc[0:0].copy()
    spin_source_mask = _spin_source_mask(plot_df, spin_source_key)
    return plot_df.loc[
        spin_source_mask & plot_df["Q2_labels"].astype(str).eq(str(resolved_bin_label))
    ].copy()


def select_nachtmann_display_subset(
    g1f1_df,
    requested_q2_values,
    q2_match_tolerance,
    *,
    a1n_source_key=A1N_ALL_SOURCE_KEY,
    spin_source_key=SPIN_DUALITY_SOURCE_KEY,
):
    requested_values, match_tolerance, _unused_complete_values = validate_nachtmann_q2_configuration(
        requested_q2_values,
        q2_match_tolerance,
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

    spin_mask = _spin_source_mask(working_df, spin_source_key)
    spin_frame = working_df.loc[spin_mask].copy()
    if spin_frame.empty:
        raise ValueError(
            f"Spin-duality source '{spin_source_key}' is not present in the active normalized g1/F1 DataFrame; "
            "cannot resolve NACHTMANN_Q2_VALUES."
        )

    # Q2_labels are created for the full normalized analysis frame.  Resolve
    # requested display bins against that same structure (including the high
    # Q2 A1n bin near 7.5 GeV^2), then retain only E01-012 rows when plotting
    # the spin-duality overlay.
    binned_spin_frame, spin_bin_stats = _spin_q2_bin_stats(spin_frame)
    _binned_display_frame, all_bin_stats = _spin_q2_bin_stats(working_df)
    resolved_bin_matches = _resolve_requested_q2_bins(
        all_bin_stats,
        requested_values,
        match_tolerance,
    )
    selected_spin_frames = [
        _select_resolved_spin_bin_rows(
            binned_spin_frame,
            match["resolved_label"],
            spin_source_key=spin_source_key,
        )
        for match in resolved_bin_matches
    ]
    selected_spin_frame = pd.concat(selected_spin_frames, ignore_index=True)
    spin_stats_by_label = {item["label"]: item for item in spin_bin_stats}
    for match, selected_spin_bin_frame in zip(resolved_bin_matches, selected_spin_frames):
        spin_bin_stats_item = spin_stats_by_label.get(match["resolved_label"])
        match["spin_duality_n_points"] = int(len(selected_spin_bin_frame))
        if spin_bin_stats_item is not None:
            match["spin_duality_mean_q2"] = float(spin_bin_stats_item["mean_q2"])
        else:
            match["spin_duality_mean_q2"] = None
            missing_warnings.append(
                "Normalized Q2 bin "
                f"'{match['resolved_label']}' has no {spin_source_key} rows; "
                "A1n ALL points remain included without a spin-duality overlay for that bin."
            )
    selected_df = pd.concat([a1n_frame, selected_spin_frame], ignore_index=True)
    selected_df = selected_df.replace([np.inf, -np.inf], np.nan)

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "a1n_all_source_key": a1n_source_key,
        "a1n_all_points": int(len(a1n_frame)),
        "spin_duality_source_key": spin_source_key,
        "requested_data_q2_values": requested_values,
        "q2_match_tolerance": match_tolerance,
        "resolved_data_bin_matches": resolved_bin_matches,
        "selected_bin_labels": [match["resolved_label"] for match in resolved_bin_matches],
        "selected_spin_duality_bin_stats": resolved_bin_matches,
        "all_normalized_q2_bin_stats": all_bin_stats,
        "all_spin_duality_bin_stats": spin_bin_stats,
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
    requested_q2_values,
    q2_match_tolerance,
):
    selected_df, metadata = select_nachtmann_display_subset(
        g1f1_df,
        requested_q2_values=requested_q2_values,
        q2_match_tolerance=q2_match_tolerance,
    )
    plot_columns = ["Nachtmann_x", "G1F1", "G1F1.err"]
    plot_df = selected_df.copy()
    if all(column in plot_df.columns for column in plot_columns):
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=plot_columns)
    else:
        plot_df = plot_df.iloc[0:0].copy()
    metadata["plotted_points"] = int(len(plot_df))
    metadata["dropped_nonfinite_plot_points"] = int(len(selected_df) - len(plot_df))
    if metadata["dropped_nonfinite_plot_points"]:
        metadata.setdefault("plot_warnings", []).append(
            "Rows with nonfinite Nachtmann_x, G1F1, or G1F1.err were retained in the CSV "
            "but omitted from the plotted points."
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
    print(f"[{mode_label}] Requested normalized Q2 values: {metadata['requested_data_q2_values']}")
    print(f"[{mode_label}] Resolved normalized Q2 bins: {metadata['selected_bin_labels'] or 'none'}")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, alpha=0.8)

    if plot_df.empty:
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
        a1n_mask = _source_key_mask(plot_df, A1N_ALL_SOURCE_KEY)
        if not bool(a1n_mask.any()):
            a1n_mask = _label_mask(plot_df, "A1n all")
        a1n_frame = plot_df.loc[a1n_mask].copy()
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
                label="A1n ALL — individual $Q^2$ values",
            )

        if "Q2_labels" in plot_df.columns:
            resolved_bin_matches = metadata["resolved_data_bin_matches"]
            color_map = plt.get_cmap("turbo", max(len(resolved_bin_matches), 1))
            for idx, bin_match in enumerate(resolved_bin_matches):
                bin_frame = _select_resolved_spin_bin_rows(
                    plot_df,
                    bin_match["resolved_label"],
                    spin_source_key=metadata["spin_duality_source_key"],
                )
                if bin_frame.empty:
                    continue
                requested_value = bin_match["requested_q2"]
                normalized_bin_mean = bin_match["resolved_mean_q2"]
                spin_mean = bin_match.get("spin_duality_mean_q2")
                if spin_mean is None:
                    continue
                if f"{requested_value:.3f}" == f"{normalized_bin_mean:.3f}":
                    legend_label = (
                        "E01-012 spin duality: "
                        f"$\\langle Q^2 \\rangle$={spin_mean:.3f} GeV$^2$"
                    )
                else:
                    legend_label = (
                        "E01-012 spin duality: "
                        f"requested $Q^2$={requested_value:.3f}, "
                        f"$\\langle Q^2 \\rangle_{{\\mathrm{{E01-012}}}}$={spin_mean:.3f} GeV$^2$ "
                        f"(normalized bin={normalized_bin_mean:.3f})"
                    )
                color = color_map(idx)
                ax.errorbar(
                    bin_frame["Nachtmann_x"],
                    bin_frame["G1F1"],
                    yerr=np.abs(bin_frame["G1F1.err"]),
                    fmt=_SPIN_BIN_MARKERS[idx % len(_SPIN_BIN_MARKERS)],
                    linestyle="none",
                    color=color,
                    ecolor=color,
                    capsize=2,
                    linewidth=1.0,
                    markersize=6,
                    label=legend_label,
                )

        ax.set_xlim(
            max(0.0, float(np.min(plot_df["Nachtmann_x"])) - 0.02),
            min(1.0, float(np.max(plot_df["Nachtmann_x"])) + 0.02),
        )
        y_error = np.abs(plot_df["G1F1.err"].to_numpy(dtype=float))
        y_values = np.concatenate(
            [
                plot_df["G1F1"].to_numpy(dtype=float),
                plot_df["G1F1"].to_numpy(dtype=float) - y_error,
                plot_df["G1F1"].to_numpy(dtype=float) + y_error,
            ]
        )
        ax.set_ylim(*_resolve_data_ylim(y_values))

    ax.set_xlabel(r"Nachtmann $\xi$")
    ax.set_ylabel(r"$g_1^{3\mathrm{He}}/F_1^{3\mathrm{He}}$")
    ax.set_title("A1n ALL versus E01-012 spin-duality comparison in Nachtmann $\\xi$")
    ax.text(
        0.02,
        0.02,
        "A1n ALL points use their individual measured $Q^2$ values.",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
    )
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


def resolve_nachtmann_complete_fit_q2_values(selection_result, q2_override=None):
    if q2_override is not None:
        return _normalize_requested_q2_values(
            q2_override,
            "NACHTMANN_COMPLETE_FIT_Q2_VALUES",
        )

    resolved_matches = selection_result["metadata"].get("resolved_data_bin_matches", [])
    if not resolved_matches:
        raise ValueError(
            "No resolved E01-012 bins are available for the complete-fit Nachtmann comparison."
        )
    return [float(match["resolved_mean_q2"]) for match in resolved_matches]


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
    row_counts = []
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
            row_counts.append(
                {
                    "Q2": q2_value,
                    "candidate_rows": 0,
                    "dropped_nonfinite_rows": 0,
                    "exported_rows": 0,
                }
            )
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
        candidate_rows = int(len(curve_df))
        curve_df = curve_df.replace([np.inf, -np.inf], np.nan)
        curve_df = curve_df.dropna(subset=["Q2", "X", "Nachtmann_x", "W", "y_complete"])
        curve_df = curve_df.sort_values("Nachtmann_x")
        row_counts.append(
            {
                "Q2": q2_value,
                "candidate_rows": candidate_rows,
                "dropped_nonfinite_rows": int(candidate_rows - len(curve_df)),
                "exported_rows": int(len(curve_df)),
            }
        )
        rows.append(curve_df)

    if not rows:
        result_df = pd.DataFrame(columns=["Q2", "X", "Nachtmann_x", "W", "y_complete"])
    else:
        result_df = pd.concat(rows, ignore_index=True)
    result_df.attrs["curve_row_counts"] = row_counts
    return result_df


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
            fill_invalid_with_zero=False,
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

    color_map = plt.get_cmap("turbo", max(len(q2_values), 1))
    for idx, q2_value in enumerate(q2_values):
        q2_frame = curve_df[np.isclose(curve_df["Q2"], q2_value)].copy()
        if q2_frame.empty:
            continue
        ax.plot(
            q2_frame["Nachtmann_x"],
            q2_frame["y_complete"],
            color=color_map(idx),
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
        "requested_data_q2_values": selection_result["metadata"]["requested_data_q2_values"],
        "q2_match_tolerance": selection_result["metadata"]["q2_match_tolerance"],
        "resolved_data_bin_matches": selection_result["metadata"]["resolved_data_bin_matches"],
        "complete_fit_q2_source": "explicit_override" if q2_override is not None else "resolved_e01012_bins",
        "complete_fit_q2_values": q2_values,
        "nonfinite_curve_rows_by_q2": curve_df.attrs.get("curve_row_counts", []),
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
