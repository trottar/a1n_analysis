#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-03-04 13:23:09 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json

from functions import (
    k_gamma_mass_loop,
    k_new_new, k_new_new_err, 
    x_to_W, W_to_x, red_chi_sqr,
    breit_wigner_bump_wrapper, breit_wigner_bump,
    breit_wigner_wrapper, breit_wigner_res,
    quad_nucl_curve_k, quad_nucl_curve_gamma, quad_nucl_curve_mass, 
    g1f1_quad_new_DIS, damping_function,
)

def create_g1f1_grid(
        w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
        res_df, dis_fit_params, dis_transition_fit,
        k_nucl_par, k_nucl_err,
        gamma_nucl_par, gamma_nucl_err,
        mass_nucl_par, mass_nucl_err,
        k_P_vals, gamma_P_vals, mass_P_vals,
        beta_val, w_lims,
        pdf
):

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)    
    
    ##########################################
    # 1) Build the Q² grid with three segments:
    #    - 0.001 to 0.1 (100 points)
    #    - 0.1 to 1.0   (90 points)
    #    - 1.0 to 20.0  (190 points)
    ##########################################
    q2_part1 = np.linspace(0.001, 0.1, 100)
    q2_part2 = np.linspace(0.1, 1.0, 90)
    q2_part3 = np.linspace(1.0, 20.0, 190)
    q2_grid = np.unique(np.concatenate([q2_part1, q2_part2, q2_part3]))

    ##########################################
    # 2) Define your x range (0.0 to 1.0) and compute corresponding W values
    ##########################################
    n_x_points = 1000
    x_grid = np.linspace(0.0, 1.0, n_x_points, dtype=np.double)

    best_fit_results, q2_bin_params, q2_bin_errors = dis_transition_fit

    # Wrappers to safely evaluate transition functions for a given Q²
    def w_dis_transition_wrapper(q2):
        try:
            return best_fit_results['w_dis_transition']['eval_func'](q2)
        except Exception:
            return np.nan, np.nan

    def damping_dis_width_wrapper(q2):
        try:
            return best_fit_results['damping_dis_width']['eval_func'](q2)
        except Exception:
            return np.nan, np.nan

    ##########################################
    # 3) Loop over Q² values, compute y_dis and y_complete, plus uncertainties
    ##########################################
    data_rows = []

    for q2_val in q2_grid:
        # Compute W values from x for the current Q² value
        w_vals = x_to_W(x_grid, np.full_like(x_grid, q2_val))
        
        # -- Compute transition parameters --
        w_dis_transition, w_dis_transition_err = w_dis_transition_wrapper(q2_val)
        damping_dis_width, damping_dis_width_err = damping_dis_width_wrapper(q2_val)

        k_fit_params = [k_nucl_par]
        gamma_fit_params = [gamma_nucl_par]
        mass_fit_params = [mass_nucl_par]
        fit_funcs_k = [quad_nucl_curve_k]
        fit_funcs_gamma = [quad_nucl_curve_gamma]
        fit_funcs_mass = [quad_nucl_curve_mass]

        # Extract (k, gamma, mass) using your loop generator
        (ii, jj, ijj, k, k_err,
         gamma, gamma_err,
         mass, mass_err) = next(
            k_gamma_mass_loop(
                q2_val, w_vals,
                k_fit_params, gamma_fit_params, mass_fit_params,
                fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
                k_P_vals, gamma_P_vals, mass_P_vals,
                k_nucl_err, gamma_nucl_err, mass_nucl_err
            )
        )

        # -- Compute the resonance shape --
        y_bw = breit_wigner_res(w_vals, mass, k, gamma)

        # In this formulation, x_bj is simply the x_grid
        xbj = x_grid

        # -- Compute DIS part --
        quad_new_dis_par = dis_fit_params["par_quad"]
        y_dis = g1f1_quad_new_DIS([xbj, np.full_like(xbj, q2_val)], *quad_new_dis_par)

        # -- Compute the transition/damping part and complete function --
        k_new_val = k_new_new(q2_val)
        k_new_err_val = k_new_new_err(q2_val, 0.01)
        y_bw_bump = breit_wigner_bump(w_vals, 1.55, k_new_val, 0.25)
        y_transition = y_bw_bump + (y_bw - y_dis)
        damping_dis = damping_function(w_vals, w_dis_transition, damping_dis_width)
        y_complete = y_transition * damping_dis + y_dis
        y_complete = np.nan_to_num(y_complete, nan=0.0)

        # -- Quantify extrapolation uncertainties (3% relative error placeholder) --
        y_dis_err = np.abs(0.03 * y_dis)
        y_complete_err = np.abs(0.03 * y_complete)

        # -- Store each x value (with corresponding W and computed values) as a row --
        for i in range(len(x_grid)):
            data_rows.append({
                "Q2":               q2_val,
                "W":                w_vals[i],
                "xbj":              xbj[i],
                "y_dis":            y_dis[i],
                "y_dis_err":        y_dis_err[i],
                "y_complete":       y_complete[i],
                "y_complete_err":   y_complete_err[i],
            })

    ##########################################
    # 4) Create DataFrame & Save to CSV
    ##########################################
    # Create DataFrame & Save to CSV
    grid_df = pd.DataFrame(data_rows)

    # Compute range strings for Q² and x
    q2_range = f"{q2_grid[0]:.3f}-{q2_grid[-1]:.3f}"
    x_range = f"{x_grid[0]:.1f}-{x_grid[-1]:.1f}"

    # Update CSV filename to include Q² and x ranges
    csv_filename = f"../fit_data/3He_fit_grid_Q{q2_range}_x{x_range}.csv"
    grid_df.to_csv(csv_filename, index=False)
    print("Done! Saved new grid with extrapolated values and uncertainties to '{}'.".format(csv_filename))

    ##########################################
    # 5) Plot the CSV output to check y_complete and y_dis vs Q2, x (xbj) and W
    ##########################################
    df = pd.read_csv(csv_filename)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    # Plot y vs Q2
    axes[0].plot(df["Q2"], df["y_complete"], label="y_complete")
    axes[0].plot(df["Q2"], df["y_dis"], color=config["colors"]["fit"], label="y_dis")
    axes[0].set_xlabel("Q2")
    axes[0].set_ylabel("y values")
    axes[0].set_title("y vs Q2")
    axes[0].legend()

    axes[0].set_xscale('log')
    
    # Plot y vs x (xbj)
    axes[1].plot(df["xbj"], df["y_complete"], label="y_complete")
    axes[1].plot(df["xbj"], df["y_dis"], color=config["colors"]["fit"], label="y_dis")
    axes[1].set_xlabel("xbj")
    axes[1].set_ylabel("y values")
    axes[1].set_title("y vs x (xbj)")
    axes[1].legend()

    axes[1].set_xscale('log')
    
    # Plot y vs W
    axes[2].plot(df["W"], df["y_complete"], label="y_complete")
    axes[2].plot(df["W"], df["y_dis"], color=config["colors"]["fit"], label="y_dis")
    axes[2].set_xlabel("W")
    axes[2].set_ylabel("y values")
    axes[2].set_title("y vs W")
    axes[2].legend()

    axes[2].set_xscale('log')
    
    plt.tight_layout()

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
