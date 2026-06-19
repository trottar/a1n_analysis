#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2026-06-17 17:45:25 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import os
import traceback

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata, interp1d

##################################################################################################################################################
# Importing utility functions

from dis_fit_models import derive_dis_fit_tag, evaluate_dis_fit, normalize_dis_fit_model
from utility import project_path, show_pdf_with_evince

##################################################################################################################################################

# Redefine W-range
w_min = 1.1
w_max = 3.0

# Initial resonance region range (optimized later on)
w_res_min = 1.1
w_res_max = 1.45

# Dataset mode variants:
# DATASET_MODE = "legacy"
# DATASET_MODE = "2025"
# DATASET_MODE = "6gev"
DATASET_MODE = "2025"

# DIS fit model variants:
# DIS_FIT_MODEL = "fullx"
# DIS_FIT_MODEL = "quad_new"
# DIS_FIT_MODEL = "quad2"
# DIS_FIT_MODEL = "quad"
# DIS_FIT_MODEL = "cubic"
# DIS_FIT_MODEL = "all"
DIS_FIT_MODEL = "fullx"

# Analysis scope variants:
# ANALYSIS_SCOPE = "full"
# ANALYSIS_SCOPE = "dis_only"
# ANALYSIS_SCOPE = "dis"
ANALYSIS_SCOPE = "full"

# Full-scope fallback variants:
# FALLBACK_TO_DIS_ON_FULL_FAILURE = True
# FALLBACK_TO_DIS_ON_FULL_FAILURE = False
FALLBACK_TO_DIS_ON_FULL_FAILURE = True

# Full-failure debug variants:
# DEBUG_FULL_FAILURE_TRACEBACK = True
# DEBUG_FULL_FAILURE_TRACEBACK = False
DEBUG_FULL_FAILURE_TRACEBACK = True

# Sparse-2025 full-mode variants:
# ALLOW_SPARSE_2025_FULL = True
# ALLOW_SPARSE_2025_FULL = False
ALLOW_SPARSE_2025_FULL = False

# Hybrid-2025 support variants:
# USE_LEGACY_FIT_SUPPORT_FOR_2025 = True
# USE_LEGACY_FIT_SUPPORT_FOR_2025 = False
USE_LEGACY_FIT_SUPPORT_FOR_2025 = True

DATASET_2025_ALL_PATH = project_path("data", "g1F1he3_2025_all.csv")
DATASET_2025_DIS_PATH = project_path("data", "g1F1he3_2025_dis.csv")

DATASET_MODE = DATASET_MODE.lower()
DIS_FIT_MODEL = normalize_dis_fit_model(DIS_FIT_MODEL)
ANALYSIS_SCOPE = ANALYSIS_SCOPE.lower()
if ANALYSIS_SCOPE == "dis":
    ANALYSIS_SCOPE = "dis_only"

LEGACY_G1F1_INPUT = os.path.join("data", "g1f1_comb.csv")
MINGYU_DIS_INPUT = os.path.join("data", "mingyu_g1f1_g2f1_dis.csv")


def sanitize_dataset_tag(tag):
    safe_tag = tag.replace("(", "_").replace(")", "")
    safe_tag = safe_tag.replace(" ", "_")
    while "__" in safe_tag:
        safe_tag = safe_tag.replace("__", "_")
    return safe_tag.strip("_")


def derive_dataset_tag(dataset_mode, g1f1_path=None, dis_path=None):
    if dataset_mode == "legacy":
        return "legacy"

    g1f1_stem = os.path.splitext(os.path.basename(g1f1_path or dataset_mode))[0]
    dis_stem = os.path.splitext(os.path.basename(dis_path or ""))[0]

    if dis_stem and dis_stem != g1f1_stem:
        common_prefix = os.path.commonprefix([g1f1_stem, dis_stem])
        g1f1_suffix = g1f1_stem[len(common_prefix):]
        dis_suffix = dis_stem[len(common_prefix):]
        if common_prefix and g1f1_suffix and dis_suffix:
            return sanitize_dataset_tag(f"{common_prefix}{g1f1_suffix}_{dis_suffix}")
        return sanitize_dataset_tag(f"{g1f1_stem}__{dis_stem}")

    return sanitize_dataset_tag(g1f1_stem)


if DATASET_MODE == "2025":
    DATASET_TAG = derive_dataset_tag(DATASET_MODE, DATASET_2025_ALL_PATH, DATASET_2025_DIS_PATH)
else:
    DATASET_TAG = derive_dataset_tag(DATASET_MODE)
if DATASET_MODE == "2025" and ALLOW_SPARSE_2025_FULL:
    DATASET_TAG = f"{DATASET_TAG}_sparse_full"
ANALYSIS_TAG = derive_dis_fit_tag(DATASET_TAG, DIS_FIT_MODEL)


def build_output_path(base_path, dataset_tag, analysis_scope):
    if dataset_tag == "legacy" and analysis_scope == "full":
        return base_path

    output_dir, filename = os.path.split(base_path)
    tag_parts = []
    if dataset_tag != "legacy":
        tag_parts.append(dataset_tag)
    if analysis_scope != "full":
        tag_parts.append(analysis_scope)
    tagged_dir = os.path.join(output_dir, *tag_parts)
    os.makedirs(tagged_dir, exist_ok=True)
    return os.path.join(tagged_dir, filename)


def describe_fit_inputs(dataset_mode, analysis_scope):
    if dataset_mode == "2025":
        if analysis_scope == "dis_only":
            return (
                f"legacy DIS baseline from '{LEGACY_G1F1_INPUT}' plus "
                f"2025 DIS input '{os.path.join('data', os.path.basename(DATASET_2025_DIS_PATH))}'"
            )
        return (
            f"legacy DIS baseline from '{LEGACY_G1F1_INPUT}' plus "
            f"2025 all input '{os.path.join('data', os.path.basename(DATASET_2025_ALL_PATH))}'"
        )

    if dataset_mode == "6gev":
        return (
            f"'{LEGACY_G1F1_INPUT}' with E94-010 and E97-110 removed; "
            "Mingyu DIS excluded; 2025 datasets excluded"
        )

    if analysis_scope == "dis_only":
        return f"legacy DIS baseline from '{LEGACY_G1F1_INPUT}' plus Mingyu DIS '{MINGYU_DIS_INPUT}'"
    return f"legacy full input '{LEGACY_G1F1_INPUT}'"


def validate_bw_global_fit_support(dataset_mode, analysis_scope, res_df, delta_par_df, input_description, strict=True):
    min_bw_fit_points = 11
    min_unique_q2_bins = 4

    if dataset_mode == "6gev":
        min_bw_fit_points = 5

    resonance_points = len(res_df)
    resonance_bins = len(res_df["Q2_labels"].dropna().unique())
    bw_fit_points = len(delta_par_df)
    unique_q2_bins = delta_par_df["Q2"].nunique()

    if bw_fit_points >= min_bw_fit_points and unique_q2_bins >= min_unique_q2_bins:
        return True

    mode_label = f"{dataset_mode}/{analysis_scope}"
    message = (
        f"{mode_label} cannot continue to BW/global-fit stages. "
        f"Required at least {min_bw_fit_points} resonance fit points and {min_unique_q2_bins} unique Q2 bins, "
        f"but observed {bw_fit_points} fit points, {unique_q2_bins} unique Q2 bins, "
        f"{resonance_points} resonance-region rows, and {resonance_bins} resonance labels. "
        f"Inputs: {input_description}."
    )
    if strict:
        raise RuntimeError(message)

    print(f"[{mode_label}] Continuing despite validation failure: {message}")
    return False


def validate_resonance_fit_support(dataset_mode, analysis_scope, res_df, w_lims, input_description, strict=True):
    label_counts = []
    q2_labels = list(res_df["Q2_labels"].dropna().unique())

    for idx, label in enumerate(q2_labels):
        if idx >= len(w_lims):
            break

        w_min_fit, w_max_fit = w_lims[idx]
        fit_count = len(
            res_df[
                (res_df["Q2_labels"] == label)
                & (res_df["W"] > w_min_fit)
                & (res_df["W"] < w_max_fit)
            ]
        )
        label_counts.append((label, fit_count, w_min_fit, w_max_fit))

    insufficient = [entry for entry in label_counts if entry[1] < 3]
    if not insufficient:
        return True

    details = "; ".join(
        f"{label}: {count} points in [{w_min_fit:.3f}, {w_max_fit:.3f}]"
        for label, count, w_min_fit, w_max_fit in insufficient
    )
    mode_label = f"{dataset_mode}/{analysis_scope}"
    message = (
        f"{mode_label} cannot start resonance Breit-Wigner fits. "
        "Each resonance Q2 label needs at least 3 points inside its configured W fit window. "
        f"Observed: {details}. Inputs: {input_description}."
    )
    if strict:
        raise RuntimeError(message)

    print(f"[{mode_label}] Continuing despite validation failure: {message}")
    return False


def prepare_resonance_fit_inputs(dataset_mode, analysis_scope, res_df, w_lims):
    q2_labels = list(res_df["Q2_labels"].dropna().unique())
    fit_w_lims = list(w_lims[:len(q2_labels)])

    if dataset_mode != "6gev" or analysis_scope != "full":
        return res_df.copy(), fit_w_lims

    default_w_min = 1.1
    default_w_max = 1.4
    target_window_points = 5
    min_window_points = 3
    adjusted_labels = []
    dropped_labels = []
    kept_labels = []
    adjusted_w_lims = []

    for label in q2_labels:
        label_w = np.sort(
            res_df.loc[res_df["Q2_labels"] == label, "W"].to_numpy(dtype=np.double)
        )
        eligible_w = label_w[label_w > default_w_min]
        w_max_fit = default_w_max

        if eligible_w.size >= target_window_points:
            w_max_fit = max(
                w_max_fit,
                float(np.nextafter(eligible_w[target_window_points - 1], np.inf)),
            )
        elif eligible_w.size >= min_window_points:
            w_max_fit = max(
                w_max_fit,
                float(np.nextafter(eligible_w[min_window_points - 1], np.inf)),
            )

        fit_count = int(
            np.count_nonzero((label_w > default_w_min) & (label_w < w_max_fit))
        )

        if fit_count >= min_window_points:
            kept_labels.append(label)
            adjusted_w_lims.append((default_w_min, w_max_fit))
            if abs(w_max_fit - default_w_max) > 1.0e-9:
                adjusted_labels.append((label, default_w_min, default_w_max, w_max_fit, fit_count))
        else:
            dropped_labels.append((label, fit_count, default_w_min, w_max_fit))

    for label, w_min_fit, old_w_max, new_w_max, fit_count in adjusted_labels:
        print(
            f"[6gev/full] Adjusted BW window for {label}: "
            f"[{w_min_fit:.3f}, {old_w_max:.3f}] -> [{w_min_fit:.3f}, {new_w_max:.3f}] "
            f"to capture {fit_count} fit points."
        )

    for label, fit_count, w_min_fit, w_max_fit in dropped_labels:
        print(
            f"[6gev/full] Dropping BW fit bin {label}: only {fit_count} fit points "
            f"remain in [{w_min_fit:.3f}, {w_max_fit:.3f}]."
        )

    if not kept_labels:
        raise RuntimeError(
            "6gev/full could not construct any resonance BW-fit bins with at least "
            f"{min_window_points} usable points above W>{default_w_min:.3f}."
        )

    fit_res_df = res_df[res_df["Q2_labels"].isin(kept_labels)].copy().reset_index(drop=True)
    return fit_res_df, adjusted_w_lims


def build_sparse_2025_bw_fit_input(delta_par_df):
    def finite_value(value, fallback):
        return value if np.isfinite(value) else fallback

    def positive_error(value, fallback):
        if np.isfinite(value) and abs(value) > 0:
            return abs(value)
        return fallback

    sparse_rows = []
    for _, row in delta_par_df.iterrows():
        q2_value = finite_value(row.get("Q2", np.nan), np.nan)
        if not np.isfinite(q2_value):
            continue

        use_variable_fit = all(
            np.isfinite(row.get(key, np.nan)) and abs(row.get(key, np.nan)) > 0
            for key in ("k.err", "gamma.err", "M.err")
        )

        if use_variable_fit:
            k_value = finite_value(row.get("k", np.nan), 0.0)
            gamma_value = abs(finite_value(row.get("gamma", np.nan), 0.25))
            mass_value = finite_value(row.get("M", np.nan), 1.232)
            k_err = positive_error(row.get("k.err", np.nan), 0.02)
            gamma_err = positive_error(row.get("gamma.err", np.nan), 0.05)
            mass_err = positive_error(row.get("M.err", np.nan), 0.02)
        else:
            k_value = finite_value(row.get("k_constM", np.nan), 0.0)
            gamma_value = abs(finite_value(row.get("gamma_constM", np.nan), 0.25))
            mass_value = 1.232
            k_err = positive_error(row.get("k_constM.err", np.nan), 0.02)
            gamma_err = positive_error(row.get("gamma_constM.err", np.nan), 0.05)
            mass_err = 0.02

        sparse_rows.append(
            {
                "Q2": q2_value,
                "k": k_value,
                "k.err": k_err,
                "gamma": gamma_value,
                "gamma.err": gamma_err,
                "M": mass_value,
                "M.err": mass_err,
                "Experiment": row.get("Experiment", "2025 data"),
                "Label": row.get("Label", "2025 sparse full"),
            }
        )

    if not sparse_rows:
        raise RuntimeError("2025 sparse-full override could not construct BW fit inputs from delta_par_df.")

    reference_row = sparse_rows[-1]
    for q2_anchor in range(4, 11):
        sparse_rows.append(
            {
                "Q2": float(q2_anchor),
                "k": 0.0,
                "k.err": reference_row["k.err"],
                "gamma": 0.25,
                "gamma.err": reference_row["gamma.err"],
                "M": 1.232,
                "M.err": reference_row["M.err"],
                "Experiment": "2025 sparse anchor",
                "Label": "2025 sparse anchor",
            }
        )

    bw_input_df = pd.DataFrame(sparse_rows)
    print(
        f"[2025/full] Sparse-full override: augmenting BW fits with "
        f"{len(bw_input_df) - len(delta_par_df)} synthetic anchor rows."
    )
    return bw_input_df


def sanitize_bw_delta_par_df(delta_par_df):
    if delta_par_df.empty:
        raise RuntimeError("No resonance Breit-Wigner fit rows were produced.")

    def finite_positive(value):
        return np.isfinite(value) and value > 0

    def reasonable_mass(value):
        return np.isfinite(value) and 1.0 <= value <= 1.6

    def reasonable_k(value):
        return np.isfinite(value) and abs(value) <= 1.0

    def reasonable_gamma(value):
        return np.isfinite(value) and 1.0e-6 <= abs(value) <= 1.0

    cleaned_rows = []
    recovered_labels = []
    dropped_labels = []

    for _, row in delta_par_df.iterrows():
        cleaned_row = row.copy()
        label = row.get("Label", row.get("Experiment", "<unknown>"))

        if pd.isna(cleaned_row.get("Experiment", np.nan)):
            cleaned_row["Experiment"] = label

        variable_fit_valid = all(
            [
                np.isfinite(cleaned_row.get("Q2", np.nan)),
                reasonable_mass(cleaned_row.get("M", np.nan)),
                reasonable_k(cleaned_row.get("k", np.nan)),
                reasonable_gamma(cleaned_row.get("gamma", np.nan)),
                finite_positive(abs(cleaned_row.get("M.err", np.nan))),
                finite_positive(abs(cleaned_row.get("k.err", np.nan))),
                finite_positive(abs(cleaned_row.get("gamma.err", np.nan))),
            ]
        )

        if not variable_fit_valid:
            constm_fit_valid = all(
                [
                    np.isfinite(cleaned_row.get("Q2", np.nan)),
                    reasonable_k(cleaned_row.get("k_constM", np.nan)),
                    reasonable_gamma(cleaned_row.get("gamma_constM", np.nan)),
                    finite_positive(abs(cleaned_row.get("k_constM.err", np.nan))),
                    finite_positive(abs(cleaned_row.get("gamma_constM.err", np.nan))),
                ]
            )

            if constm_fit_valid:
                cleaned_row["M"] = 1.232
                cleaned_row["M.err"] = 0.02
                cleaned_row["k"] = cleaned_row["k_constM"]
                cleaned_row["k.err"] = abs(cleaned_row["k_constM.err"])
                cleaned_row["gamma"] = abs(cleaned_row["gamma_constM"])
                cleaned_row["gamma.err"] = abs(cleaned_row["gamma_constM.err"])
                recovered_labels.append(label)
            else:
                dropped_labels.append(label)
                continue
        else:
            cleaned_row["M.err"] = abs(cleaned_row["M.err"])
            cleaned_row["k.err"] = abs(cleaned_row["k.err"])
            cleaned_row["gamma"] = abs(cleaned_row["gamma"])
            cleaned_row["gamma.err"] = abs(cleaned_row["gamma.err"])

        cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        raise RuntimeError(
            "No usable resonance Breit-Wigner rows remained after sanitizing the fit results."
        )

    sanitized_df = pd.DataFrame(cleaned_rows).reset_index(drop=True)

    if recovered_labels:
        print(
            "[BW sanitize] Recovered variable-mass failures with constant-mass fits for: "
            + ", ".join(str(label) for label in recovered_labels)
        )

    if dropped_labels:
        print(
            "[BW sanitize] Dropped unusable resonance rows before global BW fits: "
            + ", ".join(str(label) for label in dropped_labels)
        )

    print(
        f"[BW sanitize] Using {len(sanitized_df)} of {len(delta_par_df)} resonance fit rows for global BW fits."
    )
    return sanitized_df

##################################################################################################################################################

from load_data import load_data
from get_dis_fit import get_dis_fit
from plot_dis_x import plot_dis_x
from plot_3he_data_W import plot_3he_data_W
from get_res_fit import get_res_fit
from plot_BW_params import plot_BW_params
from fit_BW_params import fit_BW_params
from fit_dis_transition import fit_dis_transition
from get_g1f1_W_fits import get_g1f1_W_fits, get_g1f1_W_fits_q2_bin

from g1f1_grid import create_g1f1_grid
from functions import fit_error, weighted_avg

##################################################################################################################################################

if DATASET_MODE not in {"legacy", "2025", "6gev"}:
    raise ValueError(f"Unsupported DATASET_MODE '{DATASET_MODE}'. Expected 'legacy', '2025', or '6gev'.")

if ANALYSIS_SCOPE not in {"full", "dis_only"}:
    raise ValueError(f"Unsupported ANALYSIS_SCOPE '{ANALYSIS_SCOPE}'. Expected 'full' or 'dis_only'.")

def load_analysis_data(analysis_scope):
    load_data_kwargs = {
        "dataset_mode": DATASET_MODE,
        "analysis_scope": analysis_scope,
    }
    if DATASET_MODE == "2025":
        load_data_kwargs.update(
            {
                "g1f1_2025_path": DATASET_2025_ALL_PATH,
                "dis_2025_path": DATASET_2025_DIS_PATH,
            }
        )
    return load_data(**load_data_kwargs)

def run_analysis(analysis_scope):
    g1f1_df, g2f1_df, a1_df, a2_df, dis_df = load_analysis_data(analysis_scope)
    uses_hybrid_2025_support = DATASET_MODE == "2025" and USE_LEGACY_FIT_SUPPORT_FOR_2025
    force_sparse_2025_full = (
        DATASET_MODE == "2025"
        and analysis_scope == "full"
        and ALLOW_SPARSE_2025_FULL
        and not uses_hybrid_2025_support
    )
    input_description = describe_fit_inputs(DATASET_MODE, analysis_scope)

    # independent variable data to feed to curve fit, X and Q2
    indep_data = [dis_df['X'], dis_df['Q2']]

    outputpdf = build_output_path(project_path("plots", "g1f1_fits.pdf"), ANALYSIS_TAG, analysis_scope)
    print(f"[{DATASET_MODE}/{analysis_scope}] Writing PDF to {outputpdf}")
    print(f"[{DATASET_MODE}/{analysis_scope}] Requested DIS fit model: {DIS_FIT_MODEL}")

    # Create a PdfPages object to manage the PDF file
    with PdfPages(outputpdf) as pdf:

        # DIS fit
        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: DIS fit")
        q2_interp = interp1d(dis_df['X'].values, dis_df['Q2'].values, kind='linear')
        x_dense = np.linspace(dis_df['X'].min(), dis_df['X'].max(), 10000)
        q2_dense = np.full(x_dense.size, 5.0) # array of q2 = 5.0 GeV^2

        dis_fit_params = get_dis_fit(
            indep_data,
            dis_df,
            q2_interp,
            x_dense,
            q2_dense,
            pdf,
            dataset_tag=ANALYSIS_TAG,
            dis_fit_model=DIS_FIT_MODEL,
        )

        # Generate fitted curve using the fitted parameters for constant q2
        x = np.linspace(1e-6, 1.0, 1000, dtype=np.double)
        q2 = np.full(x.size, 5.0) # array of q2 = 5.0 GeV^2

        quad_new_fit_curve = evaluate_dis_fit(dis_fit_params, x, q2)
        quad_fit_err = fit_error(x, q2, dis_fit_params["par_quad"], dis_fit_params["par_err_quad"], dis_fit_params["corr_quad"], dis_fit_params["partials"])

        # Plot dis fit vs x
        plot_dis_x(x, quad_new_fit_curve, quad_fit_err, dis_fit_params, dis_df, pdf)

        if analysis_scope == "dis_only":
            print("DIS-only scope selected. Skipping resonance, BW, transition, and grid stages.")
            return outputpdf

        # make dataframe of Resonance values (1<W<2)
        res_df = g1f1_df[g1f1_df['W']<2.0]
        res_df = res_df[res_df['W']>1.0]

        n_bins = len(res_df['Q2_labels'])

        # Plot g1/f1 vs W
        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Resonance data overview")
        plot_3he_data_W(res_df, pdf)

        # initial guesses for k and M
        k_init = [-.025, -.06, -.01, .02,
                  -.1, -.08, -.08, -.07,
                  -.06, -.06, -.05, -.2,
                  -.2, -.15, -.14, -.13,
                  -.13, -0.1, .01]

        mass_init = [1.3, 1.35, 1.2, 1.25,
                  1.23, 1.23, 1.23, 1.23,
                  1.2, 1.22, 1.22, 1.2,
                  1.22, 1.25, 1.3, 1.3,
                  1.3, 1.3, 1.5]

        gamma_init = [0.1, 0.3, 0.1, 0.1,
                  0.1, 0.1, 0.1, 0.1,
                  0.1, 0.1, 0.1, 0.1,
                  0.1, 0.2, 0.2, 0.2,
                  0.2, 0.1, 0.1]

        # RLT (10/16/2024)
        w_lims = [(1.125, 1.4), (1.125, 1.4), (1.100, 1.4), (1.100, 1.4),
                  (1.100, 1.4), (1.100, 1.4), (1.100, 1.4), (1.100, 1.35),
                  (1.085, 1.4), (1.085, 1.4), (1.085, 1.5), (1.100, 1.5),
                  (1.100, 1.45), (1.100, 1.5), (1.100, 1.5), (1.100, 1.5),
                  (1.100, 1.5), (1.100, 1.65), (1.100, 1.8)]
        bw_res_df, bw_w_lims = prepare_resonance_fit_inputs(
            DATASET_MODE, analysis_scope, res_df, w_lims
        )

        if DATASET_MODE in {"2025", "6gev"}:
            validate_resonance_fit_support(
                DATASET_MODE,
                analysis_scope,
                bw_res_df,
                bw_w_lims,
                input_description,
                strict=(
                    not (force_sparse_2025_full or uses_hybrid_2025_support)
                    if DATASET_MODE == "2025"
                    else True
                ),
            )

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Resonance Breit-Wigner fits")
        delta_par_df = get_res_fit(k_init, gamma_init, mass_init, bw_w_lims, bw_res_df, pdf)
        delta_par_df = sanitize_bw_delta_par_df(delta_par_df)

        # Plot k, gamma, M
        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: BW parameter summary")
        plot_BW_params(delta_par_df, pdf)
        bw_delta_par_df = delta_par_df
        if force_sparse_2025_full:
            bw_delta_par_df = build_sparse_2025_bw_fit_input(delta_par_df)

        # Generate fitted curves using the fitted parameters
        q2 = np.linspace(0.0, bw_delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double)
        #q2 = np.linspace(0.1, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Ignore small q2 region for fits
        #q2 = np.linspace(1.0, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Q2>1.0

        if DATASET_MODE in {"2025", "6gev"}:
            validate_bw_global_fit_support(
                DATASET_MODE,
                analysis_scope,
                bw_res_df,
                delta_par_df,
                input_description,
                strict=(
                    not (force_sparse_2025_full or uses_hybrid_2025_support)
                    if DATASET_MODE == "2025"
                    else True
                ),
            )

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: BW parameter global fits")
        bw_fit_params = fit_BW_params(q2, bw_delta_par_df, pdf, dataset_tag=ANALYSIS_TAG)

        # Redefine the upper W coverage for the full combined fit stages.
        full_w_max = g1f1_df['W'].max()

        # Redefine dataframe for complete fit
        res_df = g1f1_df
        res_df = res_df[res_df['W']<2.0]
        res_df = res_df[res_df['W']>1.0]

        w = np.linspace(w_res_min, w_res_max, 1000, dtype=np.double)

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: DIS-transition fits")
        dis_transition_fit = fit_dis_transition(w_min, full_w_max, res_df, dis_fit_params,
                                                bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                                                bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                                                bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],
                                                bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                                                w_lims,
                                                pdf,
                                                dataset_tag=ANALYSIS_TAG,
        )

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Combined W-fit pages")
        get_g1f1_W_fits(w, w_min, full_w_max, w_res_min, w_res_max, quad_fit_err,
                        res_df, dis_fit_params, dis_transition_fit,
                        bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                        bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                        bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],
                        bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                        dis_fit_params["beta_val"],
                        w_lims,
                        pdf,
                        g1f1_df
        )

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Combined W-fit pages by Q2 bin")
        get_g1f1_W_fits_q2_bin(w, w_min, full_w_max, w_res_min, w_res_max, quad_fit_err,
                        res_df, dis_fit_params, dis_transition_fit,
                        bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                        bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                        bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],
                        bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                        dis_fit_params["beta_val"],
                        w_lims,
                        pdf,
                        g1f1_df
        )

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Fit grid export")
        create_g1f1_grid(w, w_min, full_w_max, w_res_min, w_res_max, quad_fit_err,
                         res_df, dis_fit_params, dis_transition_fit,
                         bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                         bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                         bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],
                         bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                         dis_fit_params["beta_val"],
                         w_lims,
                         pdf,
                         dataset_tag=ANALYSIS_TAG,
            )

    return outputpdf


disable_sparse_2025_dis_fallback = DATASET_MODE == "2025" and ANALYSIS_SCOPE == "full" and ALLOW_SPARSE_2025_FULL

if ANALYSIS_SCOPE == "full" and FALLBACK_TO_DIS_ON_FULL_FAILURE and not disable_sparse_2025_dis_fallback:
    try:
        outputpdf = run_analysis("full")
    except Exception as exc:
        if DEBUG_FULL_FAILURE_TRACEBACK:
            print("Full-scope traceback:")
            print(traceback.format_exc())
        print(f"Full scope failed ({exc.__class__.__name__}: {exc}). Falling back to DIS-only scope.")
        outputpdf = run_analysis("dis_only")
else:
    outputpdf = run_analysis(ANALYSIS_SCOPE)

show_pdf_with_evince(outputpdf)
