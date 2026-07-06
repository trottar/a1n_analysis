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

from utility import src_path

##################################################################################################################################################

_EXPERIMENT_STYLE_OVERRIDES = {
    "Flay E06-014 (2014)": ("#1f77b4", "o"),
    "Kramer E97-103 (2003)": ("#ff7f0e", "s"),
    "E94-010": ("#2ca02c", "^"),
    "E97-110": ("#d62728", "D"),
    "Solvg. E01-012 (2006)": ("#9467bd", "v"),
    "SLAC E142 (1996)": ("#8c564b", "P"),
    "SLAC E154 (1997)": ("#e377c2", "X"),
    "Zheng E99-117 (2002)": ("#7f7f7f", "<"),
    "HERMES (2000)": ("#bcbd22", ">"),
    "2025 all": ("#17becf", "h"),
    "2025 DIS": ("#17becf", "s"),
}

_FALLBACK_EXPERIMENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_FALLBACK_EXPERIMENT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]


def _canonical_experiment_style_key(label):
    label_str = str(label).strip()
    label_lower = label_str.lower()

    if "2025" in label_lower and "dis" in label_lower:
        return "2025 DIS"
    if "2025" in label_lower and "all" in label_lower:
        return "2025 all"
    if "flay" in label_lower:
        return "Flay E06-014 (2014)"
    if "kramer" in label_lower:
        return "Kramer E97-103 (2003)"
    if "e94-010" in label_lower:
        return "E94-010"
    if "e97-110" in label_lower:
        return "E97-110"
    if "solvg." in label_lower or "e01-012" in label_lower:
        return "Solvg. E01-012 (2006)"
    if "e142" in label_lower:
        return "SLAC E142 (1996)"
    if "e154" in label_lower:
        return "SLAC E154 (1997)"
    if "zheng" in label_lower or "e99-117" in label_lower:
        return "Zheng E99-117 (2002)"
    if "hermes" in label_lower:
        return "HERMES (2000)"

    return label_str


def _build_experiment_styles(labels):
    experiment_styles = {}
    fallback_index = 0

    for label in labels:
        style_key = _canonical_experiment_style_key(label)
        if style_key in _EXPERIMENT_STYLE_OVERRIDES:
            color, marker = _EXPERIMENT_STYLE_OVERRIDES[style_key]
        else:
            color = _FALLBACK_EXPERIMENT_COLORS[fallback_index % len(_FALLBACK_EXPERIMENT_COLORS)]
            marker = _FALLBACK_EXPERIMENT_MARKERS[fallback_index % len(_FALLBACK_EXPERIMENT_MARKERS)]
            fallback_index += 1
        experiment_styles[label] = {"color": color, "marker": marker}

    return experiment_styles

def plot_BW_params(delta_par_df, pdf):
    
    # Load configuration
    with open(src_path("config.json"), "r") as f:
        config = json.load(f)
    experiment_styles = _build_experiment_styles(delta_par_df["Label"].dropna().unique())

    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 10))

    # Plot all the parameters vs Q²
    for i, exp_name in enumerate(delta_par_df["Label"].unique()):
        style = experiment_styles[exp_name]
        axs[0].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["k"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["k.err"],
            fmt=style["marker"],
            linestyle="none",
            color=style["color"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            elinewidth=config["error_bar"]["line_width"],
            ecolor=style["color"],
            markeredgecolor=config["marker"]["edge_color"],
            markeredgewidth=max(0.5, config["marker"]["edge_width"] / 2.0),
            label=exp_name,
        )

        axs[1].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["gamma"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["gamma.err"],
            fmt=style["marker"],
            linestyle="none",
            color=style["color"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            elinewidth=config["error_bar"]["line_width"],
            ecolor=style["color"],
            markeredgecolor=config["marker"]["edge_color"],
            markeredgewidth=max(0.5, config["marker"]["edge_width"] / 2.0),
            label=exp_name,
        )

        axs[2].errorbar(
            delta_par_df[delta_par_df["Label"] == exp_name]["Q2"],
            delta_par_df[delta_par_df["Label"] == exp_name]["M"],
            yerr=delta_par_df[delta_par_df["Label"] == exp_name]["M.err"],
            fmt=style["marker"],
            linestyle="none",
            color=style["color"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            elinewidth=config["error_bar"]["line_width"],
            ecolor=style["color"],
            markeredgecolor=config["marker"]["edge_color"],
            markeredgewidth=max(0.5, config["marker"]["edge_width"] / 2.0),
            label=exp_name,
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
        ax.legend(fontsize=config["font_sizes"]["legend"])

    # Adjust layout and add global x-axis label
    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figure
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
