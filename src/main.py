#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-22 01:16:03 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import os
import traceback

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata, interp1d

##################################################################################################################################################
# Importing utility functions

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
DATASET_MODE = "legacy"

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

DATASET_2025_ALL_PATH = project_path("data", "g1F1he3_2025_all.csv")
DATASET_2025_DIS_PATH = project_path("data", "g1F1he3_2025_dis.csv")

DATASET_MODE = DATASET_MODE.lower()
ANALYSIS_SCOPE = ANALYSIS_SCOPE.lower()
if ANALYSIS_SCOPE == "dis":
    ANALYSIS_SCOPE = "dis_only"


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


DATASET_TAG = derive_dataset_tag(DATASET_MODE, DATASET_2025_ALL_PATH, DATASET_2025_DIS_PATH)


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


def validate_2025_only_support(res_df, delta_par_df, g1f1_path, dis_path):
    min_bw_fit_points = 11
    min_unique_q2_bins = 4

    resonance_points = len(res_df)
    resonance_bins = len(res_df["Q2_labels"].dropna().unique())
    bw_fit_points = len(delta_par_df)
    unique_q2_bins = delta_par_df["Q2"].nunique()

    if bw_fit_points >= min_bw_fit_points and unique_q2_bins >= min_unique_q2_bins:
        return

    raise RuntimeError(
        "2025-only mode cannot continue to BW/global-fit stages. "
        f"Required at least {min_bw_fit_points} resonance fit points and {min_unique_q2_bins} unique Q2 bins, "
        f"but observed {bw_fit_points} fit points, {unique_q2_bins} unique Q2 bins, "
        f"{resonance_points} resonance-region rows, and {resonance_bins} resonance labels. "
        f"Inputs: all='{g1f1_path}', dis='{dis_path}'."
    )


def validate_2025_resonance_fit_support(res_df, w_lims, g1f1_path, dis_path):
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
        return

    details = "; ".join(
        f"{label}: {count} points in [{w_min_fit:.3f}, {w_max_fit:.3f}]"
        for label, count, w_min_fit, w_max_fit in insufficient
    )
    raise RuntimeError(
        "2025-only mode cannot start resonance Breit-Wigner fits. "
        "Each resonance Q2 label needs at least 3 points inside its configured W fit window. "
        f"Observed: {details}. Inputs: all='{g1f1_path}', dis='{dis_path}'."
    )

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
from functions import g1f1_quad_fullx_DIS, \
    partial_alpha_fullx, partial_a_fullx, partial_b_fullx, partial_c_fullx, partial_beta_fullx, partial_d_fullx, partial_x0_fullx, partial_sigma_fullx,\
    fit_error, weighted_avg

##################################################################################################################################################

if DATASET_MODE not in {"legacy", "2025"}:
    raise ValueError(f"Unsupported DATASET_MODE '{DATASET_MODE}'. Expected 'legacy' or '2025'.")

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

    # independent variable data to feed to curve fit, X and Q2
    indep_data = [dis_df['X'], dis_df['Q2']]

    outputpdf = build_output_path(project_path("plots", "g1f1_fits.pdf"), DATASET_TAG, analysis_scope)
    print(f"[{DATASET_MODE}/{analysis_scope}] Writing PDF to {outputpdf}")

    # Create a PdfPages object to manage the PDF file
    with PdfPages(outputpdf) as pdf:

        # DIS fit
        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: DIS fit")
        q2_interp = interp1d(dis_df['X'].values, dis_df['Q2'].values, kind='linear')
        x_dense = np.linspace(dis_df['X'].min(), dis_df['X'].max(), 10000)
        q2_dense = np.full(x_dense.size, 5.0) # array of q2 = 5.0 GeV^2

        dis_fit_params = get_dis_fit(indep_data, dis_df, q2_interp, x_dense, q2_dense, pdf)

        # Generate fitted curve using the fitted parameters for constant q2
        x = np.linspace(0,1.0,1000, dtype=np.double)
        q2 = np.full(x.size, 5.0) # array of q2 = 5.0 GeV^2

        args_new = [[x, q2]] + [p for p in dis_fit_params["par_quad"]]

        quad_new_fit_curve = g1f1_quad_fullx_DIS(*args_new)

        # Table F.1 from XZ's thesis
        dis_fit_params["partials"] = [partial_alpha_fullx, partial_a_fullx, partial_b_fullx, partial_c_fullx, partial_beta_fullx, partial_d_fullx, partial_x0_fullx, partial_sigma_fullx]

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

        if DATASET_MODE == "2025":
            validate_2025_resonance_fit_support(res_df, w_lims, DATASET_2025_ALL_PATH, DATASET_2025_DIS_PATH)

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: Resonance Breit-Wigner fits")
        delta_par_df = get_res_fit(k_init, gamma_init, mass_init, w_lims, res_df, pdf)

        # Plot k, gamma, M
        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: BW parameter summary")
        plot_BW_params(delta_par_df, pdf)

        x = list(delta_par_df["Q2"].unique())
        k_unique = []
        k_err_unique = []
        gamma_unique = []
        gamma_err_unique = []
        mass_unique = []
        mass_err_unique = []

        # Go through each unique Q2 value and do weighted average of the points
        for Q2 in x:
            # average k's and their errors
            k_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["k"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])
            k_unique.append(k_avg)
            k_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"]**2))
            k_err_unique.append(k_avg_err)

            # average gammas and their errors
            gamma_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["gamma"], w=(1/delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]))
            gamma_unique.append(gamma_avg)
            gamma_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]**2))
            gamma_err_unique.append(gamma_avg_err)

            mass_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["M"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])
            mass_unique.append(mass_avg)
            mass_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"]**2))
            mass_err_unique.append(mass_avg_err)

        x0 = 4.0
        for i in range(7):
          x.append(x0+i)
          k_unique.append(0.0)
          k_err_unique.append(k_avg_err)
          gamma_unique.append(0.25)
          gamma_err_unique.append(gamma_avg_err)
          mass_unique.append(1.232)
          mass_err_unique.append(mass_avg_err)

        # Generate fitted curves using the fitted parameters
        q2 = np.linspace(0.0, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double)
        #q2 = np.linspace(0.1, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Ignore small q2 region for fits
        #q2 = np.linspace(1.0, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Q2>1.0

        if DATASET_MODE == "2025":
            validate_2025_only_support(res_df, delta_par_df, DATASET_2025_ALL_PATH, DATASET_2025_DIS_PATH)

        print(f"[{DATASET_MODE}/{analysis_scope}] Stage: BW parameter global fits")
        bw_fit_params = fit_BW_params(q2, delta_par_df, pdf, dataset_tag=DATASET_TAG)

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
                                                dataset_tag=DATASET_TAG,
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
                         dataset_tag=DATASET_TAG,
            )

    return outputpdf


if ANALYSIS_SCOPE == "full" and FALLBACK_TO_DIS_ON_FULL_FAILURE:
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
