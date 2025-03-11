#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-03-10 14:43:00 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import os
import json

##################################################################################################################################################

from functions import (
    k_gamma_mass_loop,
    k_new_new, k_new_new_err, 
    x_to_W, W_to_x, red_chi_sqr,
    breit_wigner_bump_wrapper, breit_wigner_bump,
    breit_wigner_wrapper, breit_wigner_res, propagate_bw_error,
    quad_nucl_curve_k, quad_nucl_curve_gamma, quad_nucl_curve_mass, 
    g1f1_quad_new_DIS, calculate_param_error, 
    damping_function, damping_function_err, 
    propagate_transition_error, propagate_complete_error, calculate_fit_residuals,
    propagate_dis_error
)

##################################################################################################################################################

def get_g1f1_W_fits(
        w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
        res_df, dis_fit_params, dis_transition_fit,
        k_nucl_par, k_nucl_err,
        gamma_nucl_par, gamma_nucl_err,
        mass_nucl_par, mass_nucl_err,
        k_P_vals, gamma_P_vals, mass_P_vals,
        beta_val, w_lims, pdf, g1f1_df
):
    """
    Refactored version of the original code. 
    The only significant difference is that we introduce a single helper 
    function `k_gamma_mass_loop(...)` to eliminate repeated triple-nested loops.
    All final plots, lines, error-band fills, and printed statements remain
    exactly as in the original code to guarantee identical output.
    """

    # drop Flay data
    g1f1_df = g1f1_df.drop(g1f1_df[g1f1_df.Label == "Flay E06-014 (2014)"].index)

    # drop Kramer data
    g1f1_df = g1f1_df.drop(g1f1_df[g1f1_df.Label == 'Kramer E97-103 (2003)'].index)
    
    # --- Process g1f1_df: Compute Q² labels ---
    q2_labels_g1f1 = []
    # Loop through each experiment in g1f1_df
    for name in g1f1_df['Label'].unique():
        data = g1f1_df[g1f1_df['Label'] == name]
        if name == "Flay E06-014 (2014)":
            # Skip this experiment's data
            continue
        # For each unique Q² in the experiment, create Q² label entries
        for q2 in data['Q2'].unique():
            count = len(data[data['Q2'] == q2])
            q2_labels_g1f1 += [f"{name} $Q^2={q2}\\ GeV^2$" for _ in range(count)]
            print(name, q2, count)

    # Assign the computed Q² labels to the g1f1_df DataFrame
    g1f1_df['Q2_labels'] = q2_labels_g1f1

    # --- Process res_df: Extrapolate Q² values ---
    extrapolated_q2 = [5, 10, 25]
    # Identify numeric columns that need extrapolation (excluding Q2 since it's set manually)
    numeric_cols = res_df.select_dtypes(include=[np.number]).columns.tolist()

    extrapolated_entries = []
    for q2 in extrapolated_q2:
        new_entry = {'Label': "Extrapolated Q²", 'Q2': q2, 'Q2_labels': f"Extrapolated $Q^2={q2}\\ GeV^2$"}
        for col in numeric_cols:
            if col == 'Q2':
                continue  # Q² is explicitly set
            # Use available Q² data points for interpolation/extrapolation
            available_q2 = res_df['Q2'].unique()
            available_values = res_df[col].dropna().values

            if len(available_q2) == len(available_values) and len(available_q2) > 1:
                if q2 > max(available_q2):
                    # Extrapolate using a quadratic fit
                    poly_fit = np.polyfit(available_q2, available_values, deg=2)
                    new_entry[col] = np.polyval(poly_fit, q2)
                else:
                    # Interpolate if q2 is within the known range
                    new_entry[col] = np.interp(q2, available_q2, available_values)
            else:
                new_entry[col] = np.nan  # Insufficient data
        extrapolated_entries.append(new_entry)

    # Convert the extrapolated entries into a DataFrame and append to res_df
    extrapolated_df = pd.DataFrame(extrapolated_entries)
    res_df = pd.concat([res_df, extrapolated_df], ignore_index=True)

    # Print confirmation with unique Q² values in the final DataFrame
    print(f"Added extrapolated Q² values and merged non-overlapping g1f1_df data. Unique Q²: {res_df['Q2'].unique()}")
    
    ##################################################################################################################################################
    # Original function code below (unchanged in logic), with triple loops replaced by the above helper yields
    ##################################################################################################################################################

    best_fit_results, q2_bin_params, q2_bin_errors = dis_transition_fit

    # Evaluate w_dis_transition, damping_dis_width at given Q^2
    def w_dis_transition_wrapper(q2):
        try:
            return best_fit_results['w_dis_transition']['eval_func'](q2)
        except:
            return np.nan, np.nan  # Prevent crashes on extrapolated Q²

    def damping_dis_width_wrapper(q2):
        try:
            return best_fit_results['damping_dis_width']['eval_func'](q2)
        except:
            return np.nan, np.nan  # Prevent crashes on extrapolated Q²
        
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 6.5, n_rows * 6))

    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = i // n_col
        col = i % n_col

        q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]

        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]
        fit_names = ["New"]
        chosen_fits = [(0, 0, 0)]

        for (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) in k_gamma_mass_loop(
            q2, w, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ):
            y = breit_wigner_res(w, mass, k, gamma)

            W = res_df['W']
            y_cal = breit_wigner_res(W, mass, k, gamma)
            y_act = res_df['G1F1']
            y_act_err = res_df['G1F1.err']
            nu = abs(len(y_act) - 3)
            chi2 = red_chi_sqr(y_cal, y_act, y_act_err, nu)

            axs[row, col].plot(
                w, y,
                linestyle="dashed",
                label=f"$\\chi_v^2$={chi2:.2f}",
                linewidth=config["error_bar"]["line_width"]
            )

        quad_new_dis_par = dis_fit_params["par_quad"]
        w_dis = np.linspace(2.0, 3.0, 1000)
        q2_array = np.ones(w_dis.size) * q2
        x_dis = W_to_x(w_dis, q2_array)
        y_dis = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)

        axs[row, col].plot(
            w_dis, y_dis,
            color=config["colors"]["error_band"],
            linestyle="--",
            label=f"Quad DIS Fit, $\\beta$ = {beta_val:.4f}",
            linewidth=config["error_bar"]["line_width"]
        )

        axs[row, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].axhline(0, color="black", linestyle="--")
        axs[row, col].set_ylim(-.15, 0.1)
        axs[row, col].set_xlim(0.9, 2.5)
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")

    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 6.5, n_rows * 6))

    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = i // n_col
        col = i % n_col

        q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]

        '''
        w_dis_transition = q2_bin_params[l]['w_dis_transition']
        damping_dis_width = q2_bin_params[l]['damping_dis_width']
        w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
        damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
        '''
        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]
        fit_names = ["New"]
        chosen_fits = [(0, 0, 0)]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

        for (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) in k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ):
            damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)

            axs[row, col].plot(
                w_res,
                damping_dis,
                color=config["colors"]["error_band"],
                linestyle="-.",
                linewidth=config["error_bar"]["line_width"],
                label=f"damping_dis",
            )

        axs[row, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].axhline(0, color="black", linestyle="--")
        axs[row, col].set_xlim(0.9, 2.5)
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")

    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1
    fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 6.5, n_rows * 6))

    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = i // n_col
        col = i % n_col

        q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]

        '''
        w_dis_transition = q2_bin_params[l]['w_dis_transition']
        damping_dis_width = q2_bin_params[l]['damping_dis_width']
        w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
        damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
        '''
        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]
        chosen_fits = [(0, 0, 0)]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

        for (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) in k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ):
            y_bw = breit_wigner_res(w_res, mass, k, gamma)
            x_dis = W_to_x(w_res, np.full_like(w_res, q2))
            quad_new_dis_par = dis_fit_params["par_quad"]
            y_dis = g1f1_quad_new_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

            k_new = k_new_new(q2)
            y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)
            y_transition = y_bw_bump + (y_bw - y_dis)
            damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)

            axs[row, col].plot(
                w_res, y_transition,
                color=config["colors"]["scatter"], linestyle="-.",
                linewidth=config["error_bar"]["line_width"],
                label=f"y_transition"
            )
            axs[row, col].plot(
                w_res, (1 - damping_dis) * y_dis,
                color=config["colors"]["error_bar"], linestyle="-.",
                linewidth=config["error_bar"]["line_width"],
                label=f"(1-damping_dis)*y_dis"
            )
            axs[row, col].plot(
                w_res, y_transition * damping_dis,
                color=config["colors"]["error_band"], linestyle="-.",
                linewidth=config["error_bar"]["line_width"],
                label=f"y_transition * damping_dis"
            )

            y_complete = y_transition * damping_dis + y_dis
            y_complete = np.nan_to_num(y_complete, nan=0.0)            
            axs[row, col].plot(
                w_res, y_complete,
                color=config["colors"]["grid"], linestyle="solid",
                linewidth=config["error_bar"]["line_width"],
                label=f"y_complete"
            )
            axs[row, col].plot(
                w_res, y_dis,
                color="purple", linestyle=":",
                linewidth=config["error_bar"]["line_width"],
                label=f"y_dis"
            )

        axs[row, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].axhline(0, color="black", linestyle="--")
        axs[row, col].set_ylim(-.15, 0.1)
        axs[row, col].set_xlim(0.9, 2.5)
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")

    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1

    # Define custom height ratios to force residuals to be attached but keep gaps between Q² bins
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([3, 1, 1.5])  # g₁/F₁ and residuals connected, then gap

    fig = plt.figure(figsize=(n_col * 6.5, n_rows * 6))
    gs = gridspec.GridSpec(n_rows * 3, n_col, height_ratios=height_ratios, hspace=0.0)  # hspace=0 for connection

    axs = np.empty((n_rows * 3, n_col), dtype=object)
    
    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = (i // n_col) * 3  # Ensure each Q2 bin gets one set of subplots (fit + residuals) with a gap
        col = i % n_col

        q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]

        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

        (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) = next(k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ))

        y_bw = breit_wigner_res(w_res, mass, k, gamma)
        x_dis = W_to_x(w_res, np.full_like(w_res, q2))
        quad_new_dis_par = dis_fit_params["par_quad"]
        y_dis = g1f1_quad_new_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

        k_new = k_new_new(q2)
        k_new_err = k_new_new_err(q2, 0.01)  # 1% Q2 error
        y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)
        y_transition = y_bw_bump + (y_bw - y_dis)

        damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
        y_complete = y_transition * damping_dis + y_dis
        y_complete = np.nan_to_num(y_complete, nan=0.0)        

        interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_complete_interpolated = interp_func(res_df['W'][res_df['Q2_labels'] == l])

        nu = abs(len(y_complete_interpolated) - len([w_dis_transition, damping_dis_width]))
        chi2 = red_chi_sqr(y_complete_interpolated, res_df['G1F1'][res_df['Q2_labels'] == l], res_df['G1F1.err'][res_df['Q2_labels'] == l], nu)

        # Compute Residuals
        residuals, normalized_residuals = calculate_fit_residuals(
            y_complete_interpolated,
            res_df['G1F1'][res_df['Q2_labels'] == l],  # Experimental data
            res_df['G1F1.err'][res_df['Q2_labels'] == l]  # Measurement uncertainties
        )

        ### **TOP PLOT: Fit vs. Data**
        axs[row, col] = fig.add_subplot(gs[row, col])
        axs[row, col].plot(
            w_res, y_complete,
            color=config["colors"]["scatter"],
            linestyle="solid",
            linewidth=config["error_bar"]["line_width"],
            label=f"$\\chi_v^2$={chi2:.2f}",
        )

        axs[row, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].set_ylabel("$g_1^{3He}/F_1^{3He}$", fontsize=config["font_sizes"]["labels"])
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        axs[row, col].set_xlim(0.9, 2.5)
        axs[row, col].set_ylim(-0.15, 0.1)

        # Get current y-tick labels
        yticks = axs[row, col].get_yticks()

        # Remove only the lowest bound tick by replacing it with an empty label
        yticklabels = ["" if y == min(yticks) else f"{y:.2f}" for y in yticks]

        # Apply modified labels to keep all but the lowest one
        axs[row, col].set_yticklabels(yticklabels)        
        
        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
        ### **BOTTOM PLOT: Residuals**
        axs[row + 1, col] = fig.add_subplot(gs[row + 1, col])
        axs[row + 1, col].scatter(
            res_df['W'][res_df['Q2_labels'] == l],
            residuals,
            color=config["colors"]["scatter"],
            marker=config["marker"]["type"],
            s=config["marker"]["size"] * 1.5
        )

        axs[row + 1, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            residuals,
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt="none",
            ecolor=config["colors"]["error_bar"],
            capsize=2,
            capthick=1,
            linewidth=1,
        )

        axs[row + 1, col].set_xlabel("W (GeV)", fontsize=config["font_sizes"]["x_axis"])
        axs[row + 1, col].set_ylabel("Residuals", fontsize=config["font_sizes"]["labels"])

        axs[row + 1, col].set_xlim(0.9, 2.5)
        axs[row + 1, col].set_ylim(-0.05, 0.05)

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row + 1, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
    # Force gaps between Q² bins, but keep residuals attached to g₁/F₁
    fig.subplots_adjust(hspace=0.0)  

    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")

    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1

    # Define custom height ratios to force residuals to be attached but keep gaps between Q² bins
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([3, 0, 0])  # g₁/F₁ and residuals connected, then gap

    fig = plt.figure(figsize=(n_col * 6.5, n_rows * 6))
    gs = gridspec.GridSpec(n_rows * 3, n_col, height_ratios=height_ratios, hspace=0.0)  # hspace=0 for connection

    axs = np.empty((n_rows * 3, n_col), dtype=object)
    
    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = (i // n_col) * 3  # Ensure each Q2 bin gets one set of subplots (fit + residuals) with a gap
        col = i % n_col

        q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]        

        xbj = W_to_x(res_df['W'][res_df['Q2_labels'] == l], np.full_like(res_df['W'][res_df['Q2_labels'] == l], q2))
        
        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)        

        (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) = next(k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ))

        y_bw = breit_wigner_res(w_res, mass, k, gamma)
        x_dis = W_to_x(w_res, np.full_like(w_res, q2))
        quad_new_dis_par = dis_fit_params["par_quad"]
        y_dis = g1f1_quad_new_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

        k_new = k_new_new(q2)
        k_new_err = k_new_new_err(q2, 0.01)  # 1% Q2 error
        y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)
        y_transition = y_bw_bump + (y_bw - y_dis)

        damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
        y_complete = y_transition * damping_dis + y_dis
        y_complete = np.nan_to_num(y_complete, nan=0.0)        

        interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_complete_interpolated = interp_func(res_df['W'][res_df['Q2_labels'] == l])

        nu = abs(len(y_complete_interpolated) - len([w_dis_transition, damping_dis_width]))
        chi2 = red_chi_sqr(y_complete_interpolated, res_df['G1F1'][res_df['Q2_labels'] == l], res_df['G1F1.err'][res_df['Q2_labels'] == l], nu)

        # Compute Residuals
        residuals, normalized_residuals = calculate_fit_residuals(
            y_complete_interpolated,
            res_df['G1F1'][res_df['Q2_labels'] == l],  # Experimental data
            res_df['G1F1.err'][res_df['Q2_labels'] == l]  # Measurement uncertainties
        )

        ### **TOP PLOT: Fit vs. Data**
        axs[row, col] = fig.add_subplot(gs[row, col])
        axs[row, col].plot(
            x_dis, y_complete,
            color=config["colors"]["scatter"],
            linestyle="solid",
            linewidth=config["error_bar"]["line_width"],
            label=f"$\\chi_v^2$={chi2:.2f}",
        )

        axs[row, col].errorbar(
            xbj,
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=abs(res_df['G1F1.err'][res_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].set_ylabel("$g_1^{3He}/F_1^{3He}$", fontsize=config["font_sizes"]["labels"])
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        axs[row, col].set_xlim(0.0, 1.0)
        axs[row, col].set_ylim(-0.15, 0.1)

        # Get current y-tick labels
        yticks = axs[row, col].get_yticks()

        # Remove only the lowest bound tick by replacing it with an empty label
        yticklabels = ["" if y == min(yticks) else f"{y:.2f}" for y in yticks]

        # Apply modified labels to keep all but the lowest one
        axs[row, col].set_yticklabels(yticklabels)        
        
        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
    # Force gaps between Q² bins, but keep residuals attached to g₁/F₁
    fig.subplots_adjust(hspace=0.0)  

    fig.text(0.5, 0.001, "x", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
    
    print("All figures generated and saved to PDF.")


def get_g1f1_W_fits_q2_bin(
        w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
        res_df, dis_fit_params, dis_transition_fit,
        k_nucl_par, k_nucl_err,
        gamma_nucl_par, gamma_nucl_err,
        mass_nucl_par, mass_nucl_err,
        k_P_vals, gamma_P_vals, mass_P_vals,
        beta_val, w_lims, pdf, g1f1_df
):
    """
    Refactored version of the original code. 
    The only significant difference is that we introduce a single helper 
    function `k_gamma_mass_loop(...)` to eliminate repeated triple-nested loops.
    All final plots, lines, error-band fills, and printed statements remain
    exactly as in the original code to guarantee identical output.
    """
    
    # --- Debug: Print column names and a sample of data to check the format ---
    print("Columns:", g1f1_df.columns.tolist())

    # Apply W minimum cut
    g1f1_df = g1f1_df[g1f1_df['W'] > w_min]

    # --- Function to convert Q² values to float safely, removing any extra characters ---
    def convert_q2(q2):
        try:
            return float(q2)
        except Exception:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(q2))
            if match:
                return float(match.group())
            return np.nan

    # Clean the Q2 column
    g1f1_df['Q2'] = g1f1_df['Q2'].apply(convert_q2)

    # Debug: Print unique Q2 values after cleaning
    unique_q2 = sorted(g1f1_df['Q2'].dropna().unique())
    print("\nUnique Q2 values after cleaning:", unique_q2)

    # --- Remove unwanted experiments ---
    g1f1_df = g1f1_df[~g1f1_df['Label'].isin(["Flay E06-014 (2014)", "Kramer E97-103 (2003)"])]

    # --- Assign Q² category based on the numeric Q² value ---
    def assign_q2_category(q2):
        if pd.isna(q2):
            return None
        if q2 < 0.1:
            return "Low Q2"
        elif 0.1 <= q2 < 1.0:
            return "Mid Q2"
        elif q2 >= 1.0:
            return "High Q2"
        else:
            return None

    g1f1_df['Q2_category'] = g1f1_df['Q2'].apply(assign_q2_category)

    # Debug: Print distribution of Q² categories
    print("\nDistribution of Q² categories:")
    print(g1f1_df['Q2_category'].value_counts(dropna=False))

    # --- New function to create bins with as many points as possible per bin,
    #     but ensuring every bin has at least min_count data points.
    def create_bins_for_category_maximize(df, category, min_count=5, gap_factor=2.0):
        """
        For the given category, this function:
          1. Sorts the Q² values.
          2. Computes the differences between consecutive Q² values.
          3. Flags potential splits when a gap exceeds (gap_factor * median_gap).
          4. Splits the data at those indices.
          5. Merges any bins that have fewer than min_count data points.
        The final label for each bin shows the central value and the bin size.
        """
        subset = df[df['Q2_category'] == category].copy()
        if subset.empty:
            return pd.Series(dtype=object)

        subset.sort_values('Q2', inplace=True)
        q2_vals = subset['Q2'].values
        n = len(q2_vals)
        indices = subset.index.tolist()

        if n < min_count:
            # Not enough points even for one bin; return one bin.
            center = (q2_vals[0] + q2_vals[-1]) / 2
            bin_size = q2_vals[-1] - q2_vals[0]
            label = f"{category} bin 1: {center:.3f} ± {bin_size:.3f} (n={n})"
            return pd.Series([label] * n, index=indices)

        # Compute gaps between consecutive Q² values.
        diffs = np.diff(q2_vals)
        median_diff = np.median(diffs) if len(diffs) > 0 else 0
        threshold = gap_factor * median_diff

        # Identify indices where the gap is "large"
        potential_splits = [i for i, d in enumerate(diffs) if d > threshold]

        # Create initial bins using these split indices.
        bins = []
        start = 0
        for split_idx in potential_splits:
            end = split_idx + 1
            bins.append((start, end))
            start = end
        bins.append((start, n))

        # Merge any bins that have fewer than min_count points.
        merged_bins = []
        i = 0
        while i < len(bins):
            s, e = bins[i]
            count = e - s
            if count < min_count:
                if merged_bins:
                    prev_s, prev_e = merged_bins[-1]
                    merged_bins[-1] = (prev_s, e)
                else:
                    if i + 1 < len(bins):
                        next_s, next_e = bins[i+1]
                        merged_bins.append((s, next_e))
                        i += 1  # Skip the next bin
                    else:
                        merged_bins.append((s, e))
            else:
                merged_bins.append((s, e))
            i += 1

        # It may be possible that after merging, an internal bin is still too small.
        changed = True
        while changed and len(merged_bins) > 1:
            changed = False
            new_bins = []
            i = 0
            while i < len(merged_bins):
                s, e = merged_bins[i]
                if (e - s) < min_count and i > 0:
                    prev_s, prev_e = new_bins[-1]
                    new_bins[-1] = (prev_s, e)
                    changed = True
                else:
                    new_bins.append((s, e))
                i += 1
            merged_bins = new_bins

        # Assign bin labels using the merged bins, computing central value and bin size.
        bin_labels = [None] * n
        for bin_idx, (s, e) in enumerate(merged_bins):
            count = e - s
            lower = q2_vals[s]
            upper = q2_vals[e-1]
            center = (lower + upper) / 2
            bin_size = upper - lower
            label = f"{category} bin {bin_idx+1}: {center:.3f} ± {bin_size:.3f} (n={count})"
            for j in range(s, e):
                bin_labels[j] = label

        return pd.Series(bin_labels, index=indices)

    # --- Create bins for each Q² category using the new algorithm ---
    low_bins = create_bins_for_category_maximize(g1f1_df, "Low Q2", min_count=5, gap_factor=2.0)
    mid_bins = create_bins_for_category_maximize(g1f1_df, "Mid Q2", min_count=5, gap_factor=2.0)
    high_bins = create_bins_for_category_maximize(g1f1_df, "High Q2", min_count=5, gap_factor=2.0)

    # Combine the bin labels and assign them as the new Q² labels column
    all_bins = pd.concat([low_bins, mid_bins, high_bins])
    g1f1_df['Q2_labels'] = all_bins

    # Redefine w_max (if needed)
    w_max = g1f1_df['W'].max()
    
    ##################################################################################################################################################
    # Original function code below (unchanged in logic), with triple loops replaced by the above helper yields
    ##################################################################################################################################################

    best_fit_results, q2_bin_params, q2_bin_errors = dis_transition_fit

    # Evaluate w_dis_transition, damping_dis_width at given Q^2
    def w_dis_transition_wrapper(q2):
        try:
            return best_fit_results['w_dis_transition']['eval_func'](q2)
        except:
            return np.nan, np.nan  # Prevent crashes on extrapolated Q²

    def damping_dis_width_wrapper(q2):
        try:
            return best_fit_results['damping_dis_width']['eval_func'](q2)
        except:
            return np.nan, np.nan  # Prevent crashes on extrapolated Q²
        
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    n_col = 5
    num_plots = len(g1f1_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1

    # Define custom height ratios to force residuals to be attached but keep gaps between Q² bins
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([3, 1, 1.5])  # g₁/F₁ and residuals connected, then gap

    fig = plt.figure(figsize=(n_col * 6.5, n_rows * 6))
    gs = gridspec.GridSpec(n_rows * 3, n_col, height_ratios=height_ratios, hspace=0.0)  # hspace=0 for connection

    axs = np.empty((n_rows * 3, n_col), dtype=object)
    
    for i, l in enumerate(g1f1_df['Q2_labels'].unique()):
        row = (i // n_col) * 3  # Ensure each Q2 bin gets one set of subplots (fit + residuals) with a gap
        col = i % n_col

        q2 = g1f1_df['Q2'][g1f1_df['Q2_labels'] == l].unique()[0]

        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

        (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) = next(k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ))

        y_bw = breit_wigner_res(w_res, mass, k, gamma)
        x_dis = W_to_x(w_res, np.full_like(w_res, q2))
        quad_new_dis_par = dis_fit_params["par_quad"]
        y_dis = g1f1_quad_new_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

        k_new = k_new_new(q2)
        k_new_err = k_new_new_err(q2, 0.01)  # 1% Q2 error
        y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)
        y_transition = y_bw_bump + (y_bw - y_dis)

        damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
        y_complete = y_transition * damping_dis + y_dis
        y_complete = np.nan_to_num(y_complete, nan=0.0)        

        interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_complete_interpolated = interp_func(g1f1_df['W'][g1f1_df['Q2_labels'] == l])

        nu = abs(len(y_complete_interpolated) - len([w_dis_transition, damping_dis_width]))
        chi2 = red_chi_sqr(y_complete_interpolated, g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l], g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l], nu)

        # Compute Residuals
        residuals, normalized_residuals = calculate_fit_residuals(
            y_complete_interpolated,
            g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l],  # Experimental data
            g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l]  # Measurement uncertainties
        )

        ### **TOP PLOT: Fit vs. Data**
        axs[row, col] = fig.add_subplot(gs[row, col])
        axs[row, col].plot(
            w_res, y_complete,
            color=config["colors"]["scatter"],
            linestyle="solid",
            linewidth=config["error_bar"]["line_width"],
            label=f"$\\chi_v^2$={chi2:.2f}",
        )

        axs[row, col].errorbar(
            g1f1_df['W'][g1f1_df['Q2_labels'] == l],
            g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l],
            yerr=abs(g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].set_ylabel("$g_1^{3He}/F_1^{3He}$", fontsize=config["font_sizes"]["labels"])
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        w_min_data = g1f1_df['W'][g1f1_df['Q2_labels'] == l].min() - 0.1 * g1f1_df['W'][g1f1_df['Q2_labels'] == l].min()
        w_max_data = g1f1_df['W'][g1f1_df['Q2_labels'] == l].max() + 0.1 * g1f1_df['W'][g1f1_df['Q2_labels'] == l].max()
        axs[row, col].set_xlim(w_min_data, w_max_data)
        
        #axs[row, col].set_xlim(0.9, 2.5)
        #axs[row, col].set_ylim(-0.15, 0.1)

        # Get current y-tick labels
        yticks = axs[row, col].get_yticks()

        # Remove only the lowest bound tick by replacing it with an empty label
        yticklabels = ["" if y == min(yticks) else f"{y:.2f}" for y in yticks]

        # Apply modified labels to keep all but the lowest one
        axs[row, col].set_yticklabels(yticklabels)        
        
        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
        ### **BOTTOM PLOT: Residuals**
        axs[row + 1, col] = fig.add_subplot(gs[row + 1, col])
        axs[row + 1, col].scatter(
            g1f1_df['W'][g1f1_df['Q2_labels'] == l],
            residuals,
            color=config["colors"]["scatter"],
            marker=config["marker"]["type"],
            s=config["marker"]["size"] * 1.5
        )

        axs[row + 1, col].errorbar(
            g1f1_df['W'][g1f1_df['Q2_labels'] == l],
            residuals,
            yerr=abs(g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l]),
            fmt="none",
            ecolor=config["colors"]["error_bar"],
            capsize=2,
            capthick=1,
            linewidth=1,
        )

        axs[row + 1, col].set_xlabel("W (GeV)", fontsize=config["font_sizes"]["x_axis"])
        axs[row + 1, col].set_ylabel("Residuals", fontsize=config["font_sizes"]["labels"])

        axs[row + 1, col].set_xlim(w_min_data, w_max_data)
        
        #axs[row + 1, col].set_xlim(0.9, 2.5)
        axs[row + 1, col].set_ylim(-0.05, 0.05)

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row + 1, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
    # Force gaps between Q² bins, but keep residuals attached to g₁/F₁
    fig.subplots_adjust(hspace=0.0)  

    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")

    n_col = 5
    num_plots = len(g1f1_df['Q2_labels'].unique())
    n_rows = num_plots // n_col + 1

    # Define custom height ratios to force residuals to be attached but keep gaps between Q² bins
    height_ratios = []
    for _ in range(n_rows):
        height_ratios.extend([3, 0, 0])  # g₁/F₁ and residuals connected, then gap

    fig = plt.figure(figsize=(n_col * 6.5, n_rows * 6))
    gs = gridspec.GridSpec(n_rows * 3, n_col, height_ratios=height_ratios, hspace=0.0)  # hspace=0 for connection

    axs = np.empty((n_rows * 3, n_col), dtype=object)
    
    for i, l in enumerate(g1f1_df['Q2_labels'].unique()):
        row = (i // n_col) * 3  # Ensure each Q2 bin gets one set of subplots (fit + residuals) with a gap
        col = i % n_col

        q2 = g1f1_df['Q2'][g1f1_df['Q2_labels'] == l].unique()[0]        

        xbj = W_to_x(g1f1_df['W'][g1f1_df['Q2_labels'] == l], np.full_like(g1f1_df['W'][g1f1_df['Q2_labels'] == l], q2))
        
        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]

        w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)        

        (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) = next(k_gamma_mass_loop(
            q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
            fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
            k_P_vals, gamma_P_vals, mass_P_vals,
            k_nucl_err, gamma_nucl_err, mass_nucl_err
        ))

        y_bw = breit_wigner_res(w_res, mass, k, gamma)
        x_dis = W_to_x(w_res, np.full_like(w_res, q2))
        quad_new_dis_par = dis_fit_params["par_quad"]
        y_dis = g1f1_quad_new_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

        k_new = k_new_new(q2)
        k_new_err = k_new_new_err(q2, 0.01)  # 1% Q2 error
        y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)
        y_transition = y_bw_bump + (y_bw - y_dis)

        damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
        y_complete = y_transition * damping_dis + y_dis
        y_complete = np.nan_to_num(y_complete, nan=0.0)        

        interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
        y_complete_interpolated = interp_func(g1f1_df['W'][g1f1_df['Q2_labels'] == l])

        nu = abs(len(y_complete_interpolated) - len([w_dis_transition, damping_dis_width]))
        chi2 = red_chi_sqr(y_complete_interpolated, g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l], g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l], nu)

        # Compute Residuals
        residuals, normalized_residuals = calculate_fit_residuals(
            y_complete_interpolated,
            g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l],  # Experimental data
            g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l]  # Measurement uncertainties
        )

        ### **TOP PLOT: Fit vs. Data**
        axs[row, col] = fig.add_subplot(gs[row, col])
        axs[row, col].plot(
            x_dis, y_complete,
            color=config["colors"]["scatter"],
            linestyle="solid",
            linewidth=config["error_bar"]["line_width"],
            label=f"$\\chi_v^2$={chi2:.2f}",
        )

        axs[row, col].errorbar(
            xbj,
            g1f1_df['G1F1'][g1f1_df['Q2_labels'] == l],
            yerr=abs(g1f1_df['G1F1.err'][g1f1_df['Q2_labels'] == l]),
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=5,
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"]
        )

        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])
        axs[row, col].set_ylabel("$g_1^{3He}/F_1^{3He}$", fontsize=config["font_sizes"]["labels"])
        axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

        axs[row, col].set_xlim(0.0, 1.0)
        #axs[row, col].set_ylim(-0.15, 0.1)

        # Get current y-tick labels
        yticks = axs[row, col].get_yticks()

        # Remove only the lowest bound tick by replacing it with an empty label
        yticklabels = ["" if y == min(yticks) else f"{y:.2f}" for y in yticks]

        # Apply modified labels to keep all but the lowest one
        axs[row, col].set_yticklabels(yticklabels)        
        
        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )
        
    # Force gaps between Q² bins, but keep residuals attached to g₁/F₁
    fig.subplots_adjust(hspace=0.0)  

    fig.text(0.5, 0.001, "x", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
    
    print("All figures generated and saved to PDF.")
    
