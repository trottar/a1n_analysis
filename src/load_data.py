#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-22 10:53:24 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import os
import re

import numpy as np
import pandas as pd

from dis_fit_data_sources import (
    DEFAULT_SOURCE_GROUP,
    SOURCE_GROUPS,
    build_3he_g1f1_group_bundle,
    load_source_manifest,
    resolve_source_group_name,
    source_group_breakdown_lines,
)
from functions import x_to_W
from utility import project_display_path, project_path

##################################################################################################################################################

LEGACY_EXCLUDED_LABELS = {"Flay E06-014 (2014)", "Kramer E97-103 (2003)"}
LEGACY_BASE_LABELS = {"E94-010", "E97-110"}
TWENTY25_COLUMNS = [
    "Ep", "xbj", "Q2",
    "Apar", "Apar_stat", "Apar_syst",
    "Aperp", "Aperp_stat", "Aperp_syst",
    "vpar", "vperp",
    "g1F1_He3", "g1F1_stat", "g1F1_syst",
    "vpar2", "vperp2",
    "g2F1_He3", "g2F1_stat", "g2F1_syst",
]
EMPTY_FRAME_COLUMNS = {
    "g2f1": ["Q2", "W", "X", "G2F1", "G2F1.err", "Label"],
    "a1": ["Q2", "W", "X", "A1", "A1.err", "Label"],
    "a2": ["Q2", "W", "X", "A2", "A2.err", "Label"],
}
LEGACY_G1F1_PATH = project_path("data", "g1f1_comb.csv")
LEGACY_G2F1_PATH = project_path("data", "g2f1_comb.csv")
LEGACY_A1_PATH = project_path("data", "a1_comb.csv")
LEGACY_A2_PATH = project_path("data", "a2_comb.csv")
MINGYU_DIS_PATH = project_path("data", "mingyu_g1f1_g2f1_dis.csv")
VALID_DIS_DATA_MODES = {"legacy_combined_csv", "source_group"}


def _convert_q2(q2):
    try:
        return float(q2)
    except Exception:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(q2))
        if match:
            return float(match.group())
        return np.nan


def _assign_q2_category(q2):
    if pd.isna(q2):
        return None
    if q2 < 0.1:
        return "Low Q2"
    if 0.1 <= q2 < 1.0:
        return "Mid Q2"
    if q2 >= 1.0:
        return "High Q2"
    return None


def _create_bins_for_category_maximize(df, category, min_count=5, gap_factor=2.0):
    """
    For the given category, this function:
      1. Sorts the Q² values.
      2. Computes the differences between consecutive Q² values.
      3. Flags potential splits when a gap exceeds (gap_factor * median_gap).
      4. Splits the data at those indices.
      5. Merges any bins that have fewer than min_count data points.
    The final label for each bin shows the central value and the bin size.
    """
    subset = df[df["Q2_category"] == category].copy()
    if subset.empty:
        return pd.Series(dtype=object)

    subset.sort_values("Q2", inplace=True)
    q2_vals = subset["Q2"].values
    n = len(q2_vals)
    indices = subset.index.tolist()

    if n < min_count:
        center = (q2_vals[0] + q2_vals[-1]) / 2
        bin_size = q2_vals[-1] - q2_vals[0]
        label = f"{category} bin 1: {center:.3f} ± {bin_size:.3f} (n={n})"
        return pd.Series([label] * n, index=indices)

    diffs = np.diff(q2_vals)
    median_diff = np.median(diffs) if len(diffs) > 0 else 0
    threshold = gap_factor * median_diff

    potential_splits = [i for i, diff in enumerate(diffs) if diff > threshold]

    bins = []
    start = 0
    for split_idx in potential_splits:
        end = split_idx + 1
        bins.append((start, end))
        start = end
    bins.append((start, n))

    merged_bins = []
    idx = 0
    while idx < len(bins):
        start, end = bins[idx]
        count = end - start
        if count < min_count:
            if merged_bins:
                prev_start, _ = merged_bins[-1]
                merged_bins[-1] = (prev_start, end)
            elif idx + 1 < len(bins):
                _, next_end = bins[idx + 1]
                merged_bins.append((start, next_end))
                idx += 1
            else:
                merged_bins.append((start, end))
        else:
            merged_bins.append((start, end))
        idx += 1

    changed = True
    while changed and len(merged_bins) > 1:
        changed = False
        new_bins = []
        idx = 0
        while idx < len(merged_bins):
            start, end = merged_bins[idx]
            if (end - start) < min_count and idx > 0:
                prev_start, _ = new_bins[-1]
                new_bins[-1] = (prev_start, end)
                changed = True
            else:
                new_bins.append((start, end))
            idx += 1
        merged_bins = new_bins

    bin_labels = [None] * n
    for bin_idx, (start, end) in enumerate(merged_bins):
        count = end - start
        lower = q2_vals[start]
        upper = q2_vals[end - 1]
        center = (lower + upper) / 2
        bin_size = upper - lower
        label = f"{category} bin {bin_idx + 1}: {center:.3f} ± {bin_size:.3f} (n={count})"
        for label_idx in range(start, end):
            bin_labels[label_idx] = label

    return pd.Series(bin_labels, index=indices)


def _prepare_g1f1_df(g1f1_df, excluded_labels=None):
    g1f1_df = g1f1_df.copy()

    print("Columns:", g1f1_df.columns.tolist())

    g1f1_df["Q2"] = g1f1_df["Q2"].apply(_convert_q2)

    unique_q2 = sorted(g1f1_df["Q2"].dropna().unique())
    print("\nUnique Q2 values after cleaning:", unique_q2)

    if excluded_labels and "Label" in g1f1_df.columns:
        g1f1_df = g1f1_df[~g1f1_df["Label"].isin(excluded_labels)].copy()

    g1f1_df["Q2_category"] = g1f1_df["Q2"].apply(_assign_q2_category)

    print("\nDistribution of Q² categories:")
    print(g1f1_df["Q2_category"].value_counts(dropna=False))

    low_bins = _create_bins_for_category_maximize(g1f1_df, "Low Q2", min_count=5, gap_factor=2.0)
    mid_bins = _create_bins_for_category_maximize(g1f1_df, "Mid Q2", min_count=5, gap_factor=2.0)
    high_bins = _create_bins_for_category_maximize(g1f1_df, "High Q2", min_count=5, gap_factor=2.0)

    all_bins = pd.concat([low_bins, mid_bins, high_bins])
    g1f1_df["Q2_labels"] = all_bins

    return g1f1_df.reset_index(drop=True)


def _exclude_labels(df, labels):
    if not labels or "Label" not in df.columns:
        return df.copy()
    return df[~df["Label"].isin(labels)].copy().reset_index(drop=True)


def _load_2025_g1f1_frame(path, label):
    raw_df = pd.read_csv(
        path,
        sep=r"\s+",
        names=TWENTY25_COLUMNS,
        skiprows=1,
        engine="python",
    )

    for column in TWENTY25_COLUMNS:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    g1f1_err = np.sqrt(raw_df["g1F1_stat"] ** 2 + raw_df["g1F1_syst"] ** 2)
    w_values = x_to_W(
        raw_df["xbj"].to_numpy(dtype=np.double),
        raw_df["Q2"].to_numpy(dtype=np.double),
    )

    normalized_df = pd.DataFrame(
        {
            "Q2": raw_df["Q2"],
            "W": w_values,
            "X": raw_df["xbj"],
            "G1F1": raw_df["g1F1_He3"],
            "G1F1.err": g1f1_err,
            "Label": label,
        }
    )

    normalized_df = normalized_df.dropna(subset=["Q2", "W", "X", "G1F1", "G1F1.err"])
    return normalized_df.reset_index(drop=True)


def _empty_frame(columns):
    return pd.DataFrame(columns=columns)


def _build_dis_cut_df(g1f1_df, label=None):
    dis_df = g1f1_df.copy()
    dis_df["Q2"] = dis_df["Q2"].apply(_convert_q2)
    dis_df["W"] = pd.to_numeric(dis_df["W"], errors="coerce")
    dis_df = dis_df[(dis_df["Q2"] > 1.0) & (dis_df["W"] > 2.0)].copy()
    if label is not None:
        dis_df["Label"] = label
    return dis_df.reset_index(drop=True)


def _build_mingyu_dis_df():
    mingyu_df = pd.read_csv(MINGYU_DIS_PATH)
    return pd.DataFrame(
        {
            "Q2": mingyu_df["Q2"],
            "W": mingyu_df["W.cal"],
            "X": mingyu_df["x"],
            "G1F1": mingyu_df["g1F1_3He"],
            "G1F1.err": mingyu_df["g1f1.err"],
            "Label": ["Mingyu DIS" for _ in range(len(mingyu_df["Q2"]))],
        }
    )


def _load_legacy_fit_support(analysis_scope):
    legacy_g1f1_df = pd.read_csv(LEGACY_G1F1_PATH)
    g2f1_df = pd.read_csv(LEGACY_G2F1_PATH)
    a1_df = pd.read_csv(LEGACY_A1_PATH)
    a2_df = pd.read_csv(LEGACY_A2_PATH)

    legacy_dis_df = _build_dis_cut_df(legacy_g1f1_df)

    mingyu_dis_df = _build_mingyu_dis_df()
    if analysis_scope == "dis_only":
        legacy_dis_df = pd.concat([legacy_dis_df, mingyu_dis_df], ignore_index=True)
    else:
        legacy_dis_df = pd.concat([mingyu_dis_df, legacy_dis_df], ignore_index=True)

    return legacy_g1f1_df, g2f1_df, a1_df, a2_df, legacy_dis_df


def _load_6gev_support(analysis_scope):
    g1f1_df, g2f1_df, a1_df, a2_df, _legacy_dis_df = _load_legacy_fit_support(analysis_scope)
    g1f1_df = _exclude_labels(g1f1_df, LEGACY_BASE_LABELS)
    g2f1_df = _exclude_labels(g2f1_df, LEGACY_BASE_LABELS)
    a1_df = _exclude_labels(a1_df, LEGACY_BASE_LABELS)
    a2_df = _exclude_labels(a2_df, LEGACY_BASE_LABELS)
    dis_df = _build_dis_cut_df(g1f1_df)
    return g1f1_df, g2f1_df, a1_df, a2_df, dis_df


def _format_label_counts(df):
    if "Label" not in df.columns or df.empty:
        return "none"
    counts = df["Label"].value_counts().sort_index()
    return ", ".join(f"{label}: {count}" for label, count in counts.items())


def _collect_frame_lines(name, df, sources, cuts):
    return [
        f"[load_data] {name}",
        f"  rows: {len(df)}",
        f"  sources: {', '.join(project_display_path(source) for source in sources)}",
        f"  cuts: {', '.join(cuts) if cuts else 'none'}",
        f"  labels: {_format_label_counts(df)}",
    ]


def _build_splash_output_path(dataset_mode, analysis_scope, g1f1_2025_path=None, dis_2025_path=None):
    if dataset_mode == "2025":
        if analysis_scope == "dis_only":
            tag = os.path.splitext(os.path.basename(dis_2025_path or "2025_dis"))[0]
        else:
            tag = os.path.splitext(os.path.basename(g1f1_2025_path or "2025_all"))[0]
        filename = f"load_data_splash_{tag}_{analysis_scope}.txt"
    else:
        filename = f"load_data_splash_{dataset_mode}_{analysis_scope}.txt"
    return project_path("fit_data", filename)


def _write_splash_report(lines, dataset_mode, analysis_scope, g1f1_2025_path=None, dis_2025_path=None):
    output_path = _build_splash_output_path(
        dataset_mode,
        analysis_scope,
        g1f1_2025_path=g1f1_2025_path,
        dis_2025_path=dis_2025_path,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as splash_file:
        splash_file.write("\n".join(lines) + "\n")
    print(f"[load_data] splash report saved to {project_display_path(output_path)}")


def _print_dataset_splash(dataset_mode, analysis_scope, g1f1_df, g2f1_df, a1_df, a2_df, dis_df,
                         g1f1_2025_path=None, dis_2025_path=None):
    lines = ["=" * 100, f"[load_data] dataset_mode={dataset_mode} analysis_scope={analysis_scope}"]

    if dataset_mode == "2025":
        if analysis_scope == "dis_only":
            g1f1_sources = [LEGACY_G1F1_PATH]
            dis_sources = [LEGACY_G1F1_PATH, dis_2025_path]
            dis_cuts = [
                "legacy DIS-cut baseline uses Q2 > 1.0 and W > 2.0",
                "2025 DIS file loaded directly",
                "2025 all excluded in 2025/dis_only",
                "Mingyu DIS excluded in 2025 mode",
            ]
        else:
            g1f1_sources = [LEGACY_G1F1_PATH, g1f1_2025_path]
            dis_sources = [LEGACY_G1F1_PATH, g1f1_2025_path]
            dis_cuts = [
                "legacy DIS-cut baseline uses Q2 > 1.0 and W > 2.0",
                "2025 all DIS-cut uses Q2 > 1.0 and W > 2.0",
                "2025 DIS excluded in 2025/full",
                "Mingyu DIS excluded in 2025 mode",
            ]
        excluded_labels = sorted(LEGACY_EXCLUDED_LABELS)
    elif dataset_mode == "6gev":
        g1f1_sources = [LEGACY_G1F1_PATH]
        dis_sources = [LEGACY_G1F1_PATH]
        dis_cuts = [
            "6gev mode excludes legacy base labels E94-010 and E97-110",
            "DIS-cut uses Q2 > 1.0 and W > 2.0",
            "Mingyu DIS excluded",
            "2025 datasets excluded",
        ]
        excluded_labels = sorted(LEGACY_BASE_LABELS)
    else:
        g1f1_sources = [LEGACY_G1F1_PATH]
        dis_sources = [LEGACY_G1F1_PATH, MINGYU_DIS_PATH]
        dis_cuts = [
            "legacy DIS-cut uses Q2 > 1.0 and W > 2.0",
            "Mingyu DIS file appended",
        ]
        excluded_labels = sorted(LEGACY_EXCLUDED_LABELS)

    lines.extend(
        _collect_frame_lines(
            "g1f1_df",
            g1f1_df,
            g1f1_sources,
            [f"excluded labels from g1f1_df: {', '.join(excluded_labels)}"],
        )
    )
    lines.extend(_collect_frame_lines("dis_df", dis_df, dis_sources, dis_cuts))
    lines.extend(_collect_frame_lines("g2f1_df", g2f1_df, [LEGACY_G2F1_PATH], []))
    lines.extend(_collect_frame_lines("a1_df", a1_df, [LEGACY_A1_PATH], []))
    lines.extend(_collect_frame_lines("a2_df", a2_df, [LEGACY_A2_PATH], []))
    lines.append("=" * 100)

    print()
    print("\n".join(lines))
    print()
    _write_splash_report(
        lines,
        dataset_mode,
        analysis_scope,
        g1f1_2025_path=g1f1_2025_path,
        dis_2025_path=dis_2025_path,
    )


def _print_source_group_splash(dataset_mode, analysis_scope, dis_data_mode, dis_source_group,
                               g1f1_df, g2f1_df, a1_df, a2_df, dis_df):
    metadata = dict(g1f1_df.attrs.get("source_group_metadata", {}))
    lines = [
        "=" * 100,
        (
            f"[load_data] dataset_mode={dataset_mode} analysis_scope={analysis_scope} "
            f"dis_data_mode={dis_data_mode} dis_source_group={dis_source_group}"
        ),
    ]
    lines.extend(source_group_breakdown_lines(metadata)[1:-1])
    lines.extend(
        _collect_frame_lines(
            "g1f1_df",
            g1f1_df,
            [metadata["source_file_map"][source_key] for source_key in metadata.get("source_keys", [])],
            [
                f"assembled from source group {dis_source_group}",
                "no source-level W cuts applied",
            ],
        )
    )
    lines.extend(
        _collect_frame_lines(
            "dis_df",
            dis_df,
            [metadata["source_file_map"][source_key] for source_key in metadata.get("source_keys", [])],
            [
                f"assembled from source group {dis_source_group}",
                f"Q2 > {metadata.get('q2_min', 'none')}",
                (
                    f"W > {metadata['dis_w_min']}"
                    if metadata.get("dis_w_min") is not None
                    else "no DIS W cut"
                ),
                (
                    f"uncut DIS sources: {', '.join(metadata['dis_uncut_source_keys'])}"
                    if metadata.get("dis_uncut_source_keys")
                    else "no source-specific DIS-cut overrides"
                ),
            ],
        )
    )
    lines.extend(_collect_frame_lines("g2f1_df", g2f1_df, [LEGACY_G2F1_PATH], ["legacy combined support"]))
    lines.extend(_collect_frame_lines("a1_df", a1_df, [LEGACY_A1_PATH], ["legacy combined support"]))
    lines.extend(_collect_frame_lines("a2_df", a2_df, [LEGACY_A2_PATH], ["legacy combined support"]))
    lines.append("=" * 100)

    print()
    print("\n".join(lines))
    print()


def load_data(dataset_mode="legacy", g1f1_2025_path=None, dis_2025_path=None, analysis_scope="full",
              dis_data_mode="legacy_combined_csv", dis_source_group=None, dis_w_min=None,
              dis_uncut_source_keys=None):

    dataset_mode = dataset_mode.lower()
    if dataset_mode not in {"legacy", "2025", "6gev"}:
        raise ValueError(f"Unsupported dataset_mode '{dataset_mode}'. Expected 'legacy', '2025', or '6gev'.")

    analysis_scope = analysis_scope.lower()
    if analysis_scope == "dis":
        analysis_scope = "dis_only"
    if analysis_scope not in {"full", "dis_only"}:
        raise ValueError(f"Unsupported analysis_scope '{analysis_scope}'. Expected 'full' or 'dis_only'.")

    dis_data_mode = str(dis_data_mode).strip().lower()
    if dis_data_mode not in VALID_DIS_DATA_MODES:
        supported = ", ".join(sorted(VALID_DIS_DATA_MODES))
        raise ValueError(f"Unsupported dis_data_mode '{dis_data_mode}'. Expected one of: {supported}.")

    if dis_data_mode == "source_group":
        manifest = load_source_manifest()
        source_group = resolve_source_group_name(dataset_mode, analysis_scope, dis_source_group or DEFAULT_SOURCE_GROUP)
        bundle = build_3he_g1f1_group_bundle(
            source_group,
            manifest,
            source_groups=SOURCE_GROUPS,
            q2_min=1.0,
            dis_w_min=dis_w_min,
            dis_uncut_source_keys=dis_uncut_source_keys,
        )
        raw_g1f1_df = bundle["g1f1_df"]
        dis_df = bundle["dis_df"]
        source_metadata = bundle["metadata"]

        g1f1_df = _prepare_g1f1_df(raw_g1f1_df)
        g1f1_df.attrs["source_group_metadata"] = source_metadata
        dis_df.attrs["source_group_metadata"] = source_metadata

        _legacy_g1f1_df, g2f1_df, a1_df, a2_df, _legacy_dis_df = _load_legacy_fit_support(analysis_scope)
        _print_source_group_splash(
            dataset_mode,
            analysis_scope,
            dis_data_mode,
            source_group,
            g1f1_df,
            g2f1_df,
            a1_df,
            a2_df,
            dis_df,
        )
        return g1f1_df, g2f1_df, a1_df, a2_df, dis_df

    if dataset_mode == "2025":
        if not g1f1_2025_path or not dis_2025_path:
            raise ValueError("2025 mode requires both g1f1_2025_path and dis_2025_path.")

        legacy_g1f1_df, g2f1_df, a1_df, a2_df, _legacy_dis_df = _load_legacy_fit_support(analysis_scope)
        legacy_dis_cut_df = _build_dis_cut_df(legacy_g1f1_df)

        if analysis_scope == "dis_only":
            g1f1_df = _prepare_g1f1_df(legacy_g1f1_df, excluded_labels=LEGACY_EXCLUDED_LABELS)
            dis_2025_df = _load_2025_g1f1_frame(dis_2025_path, "2025 DIS")
            dis_df = pd.concat([legacy_dis_cut_df, dis_2025_df], ignore_index=True)
        else:
            g1f1_2025_df = _load_2025_g1f1_frame(g1f1_2025_path, "2025 all")
            g1f1_df = pd.concat([legacy_g1f1_df, g1f1_2025_df], ignore_index=True)
            g1f1_df = _prepare_g1f1_df(g1f1_df, excluded_labels=LEGACY_EXCLUDED_LABELS)
            dis_2025_all_df = _build_dis_cut_df(g1f1_2025_df, label="2025 all DIS-cut")
            dis_df = pd.concat([legacy_dis_cut_df, dis_2025_all_df], ignore_index=True)

        _print_dataset_splash(
            dataset_mode,
            analysis_scope,
            g1f1_df,
            g2f1_df,
            a1_df,
            a2_df,
            dis_df,
            g1f1_2025_path=g1f1_2025_path,
            dis_2025_path=dis_2025_path,
        )

        return g1f1_df, g2f1_df, a1_df, a2_df, dis_df

    if dataset_mode == "6gev":
        g1f1_df, g2f1_df, a1_df, a2_df, dis_df = _load_6gev_support(analysis_scope)
        g1f1_df = _prepare_g1f1_df(g1f1_df, excluded_labels=LEGACY_BASE_LABELS)

        _print_dataset_splash(
            dataset_mode,
            analysis_scope,
            g1f1_df,
            g2f1_df,
            a1_df,
            a2_df,
            dis_df,
        )

        return g1f1_df, g2f1_df, a1_df, a2_df, dis_df

    g1f1_df, g2f1_df, a1_df, a2_df, dis_df = _load_legacy_fit_support(analysis_scope)
    g1f1_df = _prepare_g1f1_df(g1f1_df, excluded_labels=LEGACY_EXCLUDED_LABELS)

    _print_dataset_splash(
        dataset_mode,
        analysis_scope,
        g1f1_df,
        g2f1_df,
        a1_df,
        a2_df,
        dis_df,
    )

    return g1f1_df, g2f1_df, a1_df, a2_df, dis_df
