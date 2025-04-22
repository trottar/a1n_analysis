#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-22 10:53:00 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt
import numpy as np
import json

##################################################################################################################################################

def plot_dis_x(x, quad_new_fit_curve, quad_fit_err, dis_fit_params, dis_df, pdf):

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Get unique labels and assign unique colors & markers dynamically
    unique_labels = dis_df['Label'].unique()
    color_map = plt.cm.get_cmap("tab10", len(unique_labels))  # Generates distinct colors

    # Define distinct marker types
    marker_types = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'x', '+', '*']
    marker_cycle = {label: marker_types[i % len(marker_types)] for i, label in enumerate(unique_labels)}

    # Make figure
    fig, ax1 = plt.subplots(figsize=(18, 10))  # Proper aspect ratio

    # Plot data with unique colors and marker types per dataset
    for i, label in enumerate(unique_labels):
        ax1.errorbar(
            dis_df['X'][dis_df['Label'] == label],
            dis_df['G1F1'][dis_df['Label'] == label],
            yerr=dis_df['G1F1.err'][dis_df['Label'] == label],
            fmt=marker_cycle[label],  # Assign unique marker
            color=color_map(i),  # Assign unique color to each dataset
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=color_map(i),  # Ensure error bars match dataset colors
            label=label  # Add to legend
        )

    # Plot fit and fit error
    ax1.plot(
        x, quad_new_fit_curve,
        label="Quadratic Fit, $Q^2=5\ {GeV}^2$" + f" $\chi_v^2={dis_fit_params['chi2_quad']:.2f}$",
        color=config["colors"]["fit"],
        linewidth=config["error_bar"]["line_width"]
    )

    ax1.fill_between(
        x, quad_new_fit_curve - quad_fit_err, quad_new_fit_curve + quad_fit_err,
        alpha=0.5, color=config["colors"]["error_band"]
    )

    ax1.axhline(y=0, color="black", linestyle="dashed", alpha=0.7, linewidth=1.2)  # Improved visibility

    # Set labels and tick sizes
    ax1.set_xlabel("X", fontsize=config["font_sizes"]["x_axis"])
    ax1.set_ylabel('$g_1^{^{3}He}/F_1^{^{3}He}$', fontsize=config["font_sizes"]["y_axis"])
    ax1.tick_params(axis='both', which='major', labelsize=config["font_sizes"]["ticks"])

    # Grid settings
    if config["grid"]["enabled"]:
        ax1.grid(
            True, linestyle=config["grid"]["line_style"],
            linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
            color=config["colors"]["grid"]
        )

    # Legend settings
    ax1.legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"], loc=config["legend"]["location"])

    #ax1.set_ylim(-0.05, 0.05)
    
    # Adjust layout
    plt.tight_layout()

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
