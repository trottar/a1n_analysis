#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-01-15 13:24:00 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import matplotlib.pyplot as plt

##################################################################################################################################################

def plot_dis_x(x, quad_new_fit_curve, quad_fit_err, dis_fit_params, dis_df, pdf):

    # make figure
    fig, (ax1) = plt.subplots(1, 1, figsize=(18,10))

    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    colors = ("red", "darkorange", "limegreen",
              "darkslategray", "darkblue", "rebeccapurple",
              "darkmagenta")

    # plot w/ labels
    for i,l in enumerate(dis_df['Label'].unique()):
      ax1.errorbar(dis_df['X'][dis_df['Label']==l],
                    dis_df['G1F1'][dis_df['Label']==l],
                    yerr=dis_df['G1F1.err'][dis_df['Label']==l],
                    fmt=m_type, color=colors[i], markersize=m_size, capsize=cap_size,
                    label=l, capthick=cap_thick)


    # plot fit and fit error
    ax1.plot(x, quad_new_fit_curve, label="Quadratic Fit, $Q^2=5\ {GeV}^2$" + f" $\chi_v^2={dis_fit_params['chi2_quad']:.2f}$", color="darkred")
    ax1.fill_between(x, quad_new_fit_curve-quad_fit_err, quad_new_fit_curve+quad_fit_err, alpha=0.5, color="darkred")
    ax1.axhline(y=0, color="black", linestyle="dashed")

    ax1.legend()
    fig.tight_layout()
    fig.text(0.53, 0.001, "X", ha='center', va='center')
    fig.text(0.001, 0.56, '$g_1^{^{3}He}/F_1^{^{3}He}$', ha='center', va='center', rotation='vertical')

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")
