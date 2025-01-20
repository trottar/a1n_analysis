#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-01-15 13:44:33 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt

##################################################################################################################################################

def plot_BW_params(delta_par_df, pdf):

    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    
    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")
    
    # plot M, k, gamma vs Q2 from variable M fit
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # maintain distinct colors between plots by keeping track of the index in the color map
    color_index = 0

    # plot all the parameters vs Q2
    for i, exp_name in enumerate(delta_par_df["Experiment"].unique()):
        axs[0].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==exp_name]["k"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["k.err"], fmt=m_type,
                        color=colors[i], markersize=m_size, capsize=cap_size,
                        label=exp_name, capthick=cap_thick)

        axs[1].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma.err"], fmt=m_type,
                        color=colors[i], markersize=m_size, capsize=cap_size,
                        label=exp_name, capthick=cap_thick)

        axs[2].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==exp_name]["M"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["M.err"], fmt=m_type,
                        color=colors[i], markersize=m_size, capsize=cap_size,
                        label=exp_name, capthick=cap_thick)

    axs[0].set_ylabel("k")
    axs[1].set_ylabel("$\Gamma$")
    axs[2].set_ylabel("M")

    axs[0].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")
