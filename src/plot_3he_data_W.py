#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-01-15 13:21:31 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt

##################################################################################################################################################

def plot_3he_data_W(res_df, pdf):

    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    
    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

    # make figure
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//4 + 1
    fig, axs = plt.subplots(num_plots//4 + 1, 4, figsize=(20,n_rows*5))

    # plot resonance w/ labels
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//4
      col = i%4
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                  res_df['G1F1'][res_df['Q2_labels']==l],
                  yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                  fmt=m_type, color=colors[i], markersize=m_size, capsize=cap_size,
                  label=l, capthick=cap_thick)

      axs[row,col].legend()
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.1)

      fig.tight_layout()
      fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center')
      fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")
