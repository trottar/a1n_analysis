#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-03-13 12:17:10 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt
import json

##################################################################################################################################################

def plot_BW_params(delta_par_df, pdf):
    
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 10))

    # Maintain distinct colors between plots using index from color map
    color_index = 0

    # Plot all the parameters vs Q²
    for i, exp_name in enumerate(delta_par_df["Label"].unique()):
        axs[0].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["k"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["k.err"],
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],  # Config-based scatter color
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],
        )

        axs[1].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["gamma"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["gamma.err"],
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],
        )

        axs[2].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["M"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["M.err"],
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],
        )

    # Set y-axis labels with configurable font size
    axs[0].set_ylabel("k", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_ylabel("$\Gamma$", fontsize=config["font_sizes"]["y_axis"])
    axs[2].set_ylabel("M", fontsize=config["font_sizes"]["y_axis"])

    # Add reference horizontal lines
    axs[0].axhline(y=0, color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0, color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)

    # Apply grid settings if enabled
    for ax in axs:
        if config["grid"]["enabled"]:
            ax.grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

    # Adjust layout and add global x-axis label
    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
