#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-03-12 16:31:07 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt
import json

##################################################################################################################################################

def plot_3he_data_W(res_df, pdf):

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Determine number of subplots
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots // 4 + 1
    fig, axs = plt.subplots(n_rows, 4, figsize=(20, n_rows * 5))

    # Plot resonance w/ labels
    for i, l in enumerate(res_df['Q2_labels'].unique()):
        row = i // 4
        col = i % 4

        axs[row, col].errorbar(
            res_df['W'][res_df['Q2_labels'] == l],
            res_df['G1F1'][res_df['Q2_labels'] == l],
            yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],  # Config-based scatter color
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],  # Error bar color from config
            label=l
        )

        w_min_data = res_df['W'][res_df['Q2_labels'] == l].min() - 0.1 * res_df['W'][res_df['Q2_labels'] == l].min()
        w_max_data = res_df['W'][res_df['Q2_labels'] == l].max() + 0.1 * res_df['W'][res_df['Q2_labels'] == l].max()
        axs[row, col].set_xlim(w_min_data, w_max_data)
        
        # Apply axis limits
        #axs[row, col].set_ylim(-0.15, 0.1)
        #axs[row, col].set_xlim(0.9, 2.1)

        # Apply grid settings if enabled
        if config["grid"]["enabled"]:
            axs[row, col].grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

        # Legend settings
        axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

    # Apply overall figure layout
    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
