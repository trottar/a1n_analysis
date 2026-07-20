#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-21 18:17:29 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json

from complete_fit_helpers import evaluate_complete_fit_from_x
from functions import (
    nachtmann_x,
    quad_nucl_curve_k,
    x_to_W,
)
from utility import prefix_generated_output_name, project_path, src_path

def _build_artifact_path(filename, dataset_tag):
    dated_filename = prefix_generated_output_name(filename)
    if dataset_tag == "legacy":
        return project_path("fit_data", dated_filename)

    tagged_dir = project_path("fit_data", dataset_tag)
    os.makedirs(tagged_dir, exist_ok=True)
    return os.path.join(tagged_dir, dated_filename)


def create_g1f1_grid(
        w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
        res_df, dis_fit_params, dis_transition_fit,
        k_nucl_par, k_nucl_err,
        gamma_nucl_par, gamma_nucl_err,
        mass_nucl_par, mass_nucl_err,
        k_P_vals, gamma_P_vals, mass_P_vals,
        beta_val, w_lims,
        pdf,
        dataset_tag="legacy",
        quad_nucl_curve_k_func=quad_nucl_curve_k,
):

    # Load configuration
    with open(src_path("config.json"), "r") as f:
        config = json.load(f)    
    
    ##########################################
    # 1) Build the Q² grid with three segments:
    #    - 0.001 to 0.1 (0.001 steps)
    #    - 0.1 to 1.0   (0.1 steps)
    #    - 1.0 to 20.0  (1.0 steps)
    ##########################################
    q2_part1 = np.arange(0.001, 0.100, 0.001, dtype=np.double)
    q2_part2 = np.arange(0.1, 1.0, 0.01, dtype=np.double)
    q2_part3 = np.arange(1.0, 20.1, 0.1, dtype=np.double)
    q2_grid = np.unique(np.concatenate([q2_part1, q2_part2, q2_part3]))

    ##########################################
    # 2) Define your x range (0.0 to 1.0) and compute corresponding W values
    ##########################################
    n_x_points = 1000
    x_grid = np.arange(0.001, 1.001, 0.001, dtype=np.double)

    ##########################################
    # 3) Loop over Q² values, compute y_dis and y_complete, plus uncertainties
    ##########################################
    data_rows = []

    for q2_val in q2_grid:
        # Compute W values from x for the current Q² value
        w_vals = x_to_W(x_grid, np.full_like(x_grid, q2_val))

        curve_payload = evaluate_complete_fit_from_x(
            q2_val,
            x_grid,
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
            w_values=w_vals,
            quad_nucl_curve_k_func=quad_nucl_curve_k_func,
        )
        xbj = curve_payload["X"]
        y_dis = curve_payload["y_dis"]
        y_complete = curve_payload["y_complete"]
        nachtmann_vals = nachtmann_x(xbj, np.full_like(xbj, q2_val))

        # -- Quantify extrapolation uncertainties (3% relative error placeholder) --
        y_dis_err = np.abs(0.03 * y_dis)
        y_complete_err = np.abs(0.03 * y_complete)

        # -- Store each x value (with corresponding W and computed values) as a row --
        for i in range(len(x_grid)):
            data_rows.append({
                    "Q2":               q2_val,
                    "W":                w_vals[i],
                    "xbj":              xbj[i],
                    "Nachtmann_x":      nachtmann_vals[i],
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
    csv_filename = _build_artifact_path(f"3He_fit_grid_Q{q2_range}_x{x_range}.csv", dataset_tag)
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
    plt.close(fig)
