#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-22 01:16:03 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata, interp1d

##################################################################################################################################################
# Importing utility functions

from utility import show_pdf_with_evince

##################################################################################################################################################

# Redefine W-range
w_min = 1.1
w_max = 3.0

# Initial resonance region range (optimized later on)
w_res_min = 1.1
w_res_max = 1.45

##################################################################################################################################################

from load_data import load_data
from get_dis_fit import get_dis_fit
from plot_dis_x import plot_dis_x
from plot_3he_data_W import plot_3he_data_W
from get_res_fit import get_res_fit
from plot_BW_params import plot_BW_params
from fit_BW_params import fit_BW_params
from fit_dis_transition import fit_dis_transition
from get_g1f1_W_fits import get_g1f1_W_fits, get_g1f1_W_fits_q2_bin

from g1f1_grid import create_g1f1_grid
from functions import g1f1_quad_fullx_DIS, \
    partial_alpha_fullx, partial_a_fullx, partial_b_fullx, partial_c_fullx, partial_beta_fullx, partial_d_fullx, partial_x0_fullx, partial_sigma_fullx,\
    fit_error, weighted_avg

##################################################################################################################################################

g1f1_df, g2f1_df, a1_df, a2_df, dis_df = load_data()

# independent variable data to feed to curve fit, X and Q2
indep_data = [dis_df['X'], dis_df['Q2']]

outputpdf = "../plots/g1f1_fits.pdf"

# Create a PdfPages object to manage the PDF file
with PdfPages(outputpdf) as pdf:

    # DIS fit    
    q2_interp = interp1d(dis_df['X'].values, dis_df['Q2'].values, kind='linear')
    x_dense = np.linspace(dis_df['X'].min(), dis_df['X'].max(), 10000)
    q2_dense = np.full(x_dense.size, 5.0) # array of q2 = 5.0 GeV^2
    
    dis_fit_params = get_dis_fit(indep_data, dis_df, q2_interp, x_dense, q2_dense, pdf)    

    # Generate fitted curve using the fitted parameters for constant q2
    x = np.linspace(0,1.0,1000, dtype=np.double)
    q2 = np.full(x.size, 5.0) # array of q2 = 5.0 GeV^2

    args_new = [[x, q2]] + [p for p in dis_fit_params["par_quad"]]

    quad_new_fit_curve = g1f1_quad_fullx_DIS(*args_new)

    # Table F.1 from XZ's thesis
    dis_fit_params["partials"] = [partial_alpha_fullx, partial_a_fullx, partial_b_fullx, partial_c_fullx, partial_beta_fullx, partial_d_fullx, partial_x0_fullx, partial_sigma_fullx]
    
    quad_fit_err = fit_error(x, q2, dis_fit_params["par_quad"], dis_fit_params["par_err_quad"], dis_fit_params["corr_quad"], dis_fit_params["partials"])

    # Plot dis fit vs x
    plot_dis_x(x, quad_new_fit_curve, quad_fit_err, dis_fit_params, dis_df, pdf)

    # make dataframe of Resonance values (1<W<2)
    res_df = g1f1_df[g1f1_df['W']<2.0]    
    res_df = res_df[res_df['W']>1.0]    
    
    n_bins = len(res_df['Q2_labels'])

    # Plot g1/f1 vs W
    plot_3he_data_W(res_df, pdf)
        
    # initial guesses for k and M
    k_init = [-.025, -.06, -.01, .02,
              -.1, -.08, -.08, -.07,
              -.06, -.06, -.05, -.2,
              -.2, -.15, -.14, -.13,
              -.13, -0.1, .01]

    mass_init = [1.3, 1.35, 1.2, 1.25,
              1.23, 1.23, 1.23, 1.23,
              1.2, 1.22, 1.22, 1.2,
              1.22, 1.25, 1.3, 1.3,
              1.3, 1.3, 1.5]

    gamma_init = [0.1, 0.3, 0.1, 0.1,
              0.1, 0.1, 0.1, 0.1,
              0.1, 0.1, 0.1, 0.1,
              0.1, 0.2, 0.2, 0.2,
              0.2, 0.1, 0.1]
    
    # RLT (10/16/2024)
    w_lims = [(1.125, 1.4), (1.125, 1.4), (1.100, 1.4), (1.100, 1.4),
              (1.100, 1.4), (1.100, 1.4), (1.100, 1.4), (1.100, 1.35),
              (1.085, 1.4), (1.085, 1.4), (1.085, 1.5), (1.100, 1.5),
              (1.100, 1.45), (1.100, 1.5), (1.100, 1.5), (1.100, 1.5),
              (1.100, 1.5), (1.100, 1.65), (1.100, 1.8)]
    
    delta_par_df = get_res_fit(k_init, gamma_init, mass_init, w_lims, res_df, pdf)

    # Plot k, gamma, M
    plot_BW_params(delta_par_df, pdf)

    x = list(delta_par_df["Q2"].unique())
    k_unique = []
    k_err_unique = []
    gamma_unique = []
    gamma_err_unique = []
    mass_unique = []
    mass_err_unique = []    
    
    # Go through each unique Q2 value and do weighted average of the points
    for Q2 in x:
        # average k's and their errors
        k_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["k"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])
        k_unique.append(k_avg)
        k_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"]**2))
        k_err_unique.append(k_avg_err)

        # average gammas and their errors
        gamma_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["gamma"], w=(1/delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]))
        gamma_unique.append(gamma_avg)
        gamma_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]**2))
        gamma_err_unique.append(gamma_avg_err)

        mass_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["M"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])
        mass_unique.append(mass_avg)
        mass_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"]**2))
        mass_err_unique.append(mass_avg_err)
            
    x0 = 4.0
    for i in range(7):
      x.append(x0+i)
      k_unique.append(0.0)
      k_err_unique.append(k_avg_err)
      gamma_unique.append(0.25)
      gamma_err_unique.append(gamma_avg_err)
      mass_unique.append(1.232)
      mass_err_unique.append(mass_avg_err)
      
    # Generate fitted curves using the fitted parameters
    q2 = np.linspace(0.0, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double)
    #q2 = np.linspace(0.1, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Ignore small q2 region for fits
    #q2 = np.linspace(1.0, delta_par_df["Q2"].max()+w_max, 1000, dtype=np.double) # Q2>1.0

    bw_fit_params = fit_BW_params(q2, delta_par_df, pdf)    

    # Redefine w_max (if needed)
    w_max = g1f1_df['W'].max()

    # Redefine dataframe for complete fit
    res_df = g1f1_df
    res_df = res_df[res_df['W']<2.0]
    res_df = res_df[res_df['W']>1.0]
    
    w = np.linspace(w_res_min, w_res_max, 1000, dtype=np.double)

    dis_transition_fit = fit_dis_transition(w_min, w_max, res_df, dis_fit_params, 
                                            bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                                            bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                                            bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],                    
                                            bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                                            w_lims,
                                            pdf                                            
    )
    
    get_g1f1_W_fits(w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
                    res_df, dis_fit_params, dis_transition_fit,
                    bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                    bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                    bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],                    
                    bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                    dis_fit_params["beta_val"],
                    w_lims,                    
                    pdf,
                    g1f1_df
    )

    get_g1f1_W_fits_q2_bin(w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
                    res_df, dis_fit_params, dis_transition_fit,
                    bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                    bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                    bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],                    
                    bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                    dis_fit_params["beta_val"],
                    w_lims,                    
                    pdf,
                    g1f1_df
    )

    create_g1f1_grid(w, w_min, w_max, w_res_min, w_res_max, quad_fit_err,
                     res_df, dis_fit_params, dis_transition_fit,
                     bw_fit_params["k params"]["nucl_par"], bw_fit_params["k params"]["nucl_curve_err"],
                     bw_fit_params["gamma params"]["nucl_par"], bw_fit_params["gamma params"]["nucl_curve_err"],
                     bw_fit_params["mass params"]["nucl_par"], bw_fit_params["mass params"]["nucl_curve_err"],                    
                     bw_fit_params["k params"]["P_vals"], bw_fit_params["gamma params"]["P_vals"], bw_fit_params["mass params"]["P_vals"],
                     dis_fit_params["beta_val"],
                     w_lims,
                     pdf
        )
    
show_pdf_with_evince(outputpdf)
