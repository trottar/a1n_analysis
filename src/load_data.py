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
import re

import numpy as np
import pandas as pd

from functions import x_to_W
from utility import project_path

##################################################################################################################################################

LEGACY_EXCLUDED_LABELS = {"Flay E06-014 (2014)", "Kramer E97-103 (2003)"}
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


def _prepare_g1f1_df(g1f1_df, remove_unwanted_labels=False):
    g1f1_df = g1f1_df.copy()

    print("Columns:", g1f1_df.columns.tolist())

    g1f1_df["Q2"] = g1f1_df["Q2"].apply(_convert_q2)

    unique_q2 = sorted(g1f1_df["Q2"].dropna().unique())
    print("\nUnique Q2 values after cleaning:", unique_q2)

    if remove_unwanted_labels and "Label" in g1f1_df.columns:
        g1f1_df = g1f1_df[~g1f1_df["Label"].isin(LEGACY_EXCLUDED_LABELS)].copy()

    g1f1_df["Q2_category"] = g1f1_df["Q2"].apply(_assign_q2_category)

    print("\nDistribution of Q² categories:")
    print(g1f1_df["Q2_category"].value_counts(dropna=False))

    low_bins = _create_bins_for_category_maximize(g1f1_df, "Low Q2", min_count=5, gap_factor=2.0)
    mid_bins = _create_bins_for_category_maximize(g1f1_df, "Mid Q2", min_count=5, gap_factor=2.0)
    high_bins = _create_bins_for_category_maximize(g1f1_df, "High Q2", min_count=5, gap_factor=2.0)

    all_bins = pd.concat([low_bins, mid_bins, high_bins])
    g1f1_df["Q2_labels"] = all_bins

    return g1f1_df.reset_index(drop=True)


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
    dis_df = dis_df[dis_df["Q2"] > 1.0].copy()
    if label is not None:
        dis_df["Label"] = label
    return dis_df.reset_index(drop=True)


def _build_mingyu_dis_df():
    mingyu_df = pd.read_csv(project_path("data", "mingyu_g1f1_g2f1_dis.csv"))
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
    legacy_g1f1_df = pd.read_csv(project_path("data", "g1f1_comb.csv"))
    g2f1_df = pd.read_csv(project_path("data", "g2f1_comb.csv"))
    a1_df = pd.read_csv(project_path("data", "a1_comb.csv"))
    a2_df = pd.read_csv(project_path("data", "a2_comb.csv"))

    legacy_dis_df = _build_dis_cut_df(legacy_g1f1_df)

    mingyu_dis_df = _build_mingyu_dis_df()
    if analysis_scope == "dis_only":
        legacy_dis_df = pd.concat([legacy_dis_df, mingyu_dis_df], ignore_index=True)
    else:
        legacy_dis_df = pd.concat([mingyu_dis_df, legacy_dis_df], ignore_index=True)

    return legacy_g1f1_df, g2f1_df, a1_df, a2_df, legacy_dis_df


def load_data(dataset_mode="legacy", g1f1_2025_path=None, dis_2025_path=None, analysis_scope="full"):

    dataset_mode = dataset_mode.lower()
    if dataset_mode not in {"legacy", "2025"}:
        raise ValueError(f"Unsupported dataset_mode '{dataset_mode}'. Expected 'legacy' or '2025'.")

    analysis_scope = analysis_scope.lower()
    if analysis_scope == "dis":
        analysis_scope = "dis_only"
    if analysis_scope not in {"full", "dis_only"}:
        raise ValueError(f"Unsupported analysis_scope '{analysis_scope}'. Expected 'full' or 'dis_only'.")

    if dataset_mode == "2025":
        if not g1f1_2025_path or not dis_2025_path:
            raise ValueError("2025 mode requires both g1f1_2025_path and dis_2025_path.")

        legacy_g1f1_df, g2f1_df, a1_df, a2_df, legacy_dis_df = _load_legacy_fit_support(analysis_scope)

        g1f1_2025_df = _load_2025_g1f1_frame(g1f1_2025_path, "2025 all")
        g1f1_df = pd.concat([legacy_g1f1_df, g1f1_2025_df], ignore_index=True)
        g1f1_df = _prepare_g1f1_df(g1f1_df, remove_unwanted_labels=True)

        dis_2025_df = _load_2025_g1f1_frame(dis_2025_path, "2025 DIS")
        if analysis_scope == "dis_only":
            dis_2025_all_df = _build_dis_cut_df(g1f1_2025_df, label="2025 all DIS-cut")
            dis_df = pd.concat([dis_2025_all_df, dis_2025_df], ignore_index=True)
        else:
            dis_df = pd.concat([legacy_dis_df, dis_2025_df], ignore_index=True)

        return g1f1_df, g2f1_df, a1_df, a2_df, dis_df

    g1f1_df, g2f1_df, a1_df, a2_df, dis_df = _load_legacy_fit_support(analysis_scope)
    g1f1_df = _prepare_g1f1_df(g1f1_df, remove_unwanted_labels=True)

    return g1f1_df, g2f1_df, a1_df, a2_df, dis_df
