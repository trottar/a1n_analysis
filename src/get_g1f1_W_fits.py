#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-02-25 11:11:46 trottar"
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
        beta_val, w_lims, pdf
):
    """
    Refactored version of the original code. 
    The only significant difference is that we introduce a single helper 
    function `k_gamma_mass_loop(...)` to eliminate repeated triple-nested loops.
    All final plots, lines, error-band fills, and printed statements remain
    exactly as in the original code to guarantee identical output.
    """

    ##################################################################################################################################################
    # Original function code below (unchanged in logic), with triple loops replaced by the above helper yields
    ##################################################################################################################################################

    best_fit_results, q2_bin_params, q2_bin_errors = dis_transition_fit

    # Evaluate w_dis_transition, damping_dis_width at given Q^2
    def w_dis_transition_wrapper(q2):
        return best_fit_results['w_dis_transition']['eval_func'](q2)
    def damping_dis_width_wrapper(q2):
        return best_fit_results['damping_dis_width']['eval_func'](q2)
        
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
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
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
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
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
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
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
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
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
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
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

    print("All figures generated and saved to PDF.")
