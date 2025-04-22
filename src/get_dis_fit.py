#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-21 17:24:10 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate
import matplotlib.pyplot as plt
import json

##################################################################################################################################################

from functions import g1f1_quad_fullx_DIS

##################################################################################################################################################

def get_dis_fit(indep_data, dis_df, q2_interp, x_dense, q2_dense, pdf):

    # fit the g1f1 DIS data with constrained quadratic form

    # Initial parameterization guesses (irrelavent as algorthm is suffiecent with most initial params)
    quad_new_init = [0.66084205, -0.23606144,  -1.25499178, 2.65987975,  0.09666789, 0.0, 0.2, 0.35]
    quad_new_constr = ([-np.inf, -np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.1, 0.2],
                    [np.inf, np.inf, 0.0, np.inf, np.inf, np.inf, 0.3, 0.5]) # min and max bounds on alpha, a, b, c, beta, d, x0 and sigma

    quad_param_names = ["alpha", "a", "b", "c", "beta", "d", "x0", "sigma"]
    
    def optimize_init_params(func, x, y, y_err, params_init, bounds, n_tries=1000):
        """
        Optimize initial parameters by trying multiple starting points
        """
        best_chi2 = np.inf
        best_params = params_init

        # Generate random starting points within bounds
        for _ in range(n_tries):
            try:
                random_init = np.array([
                    np.random.uniform(low, high) 
                    for low, high in zip(bounds[0], bounds[1])
                ])
                params, _ = curve_fit(func, x, y, p0=random_init, sigma=y_err, 
                                    bounds=bounds, maxfev=50000)

                # Calculate chi2 for this fit
                y_fit = func(x, *params)
                chi2 = np.sum(((y - y_fit) / y_err) ** 2) / (len(y) - len(params))

                if abs(chi2 - 1) < abs(best_chi2 - 1):
                    best_chi2 = chi2
                    best_params = params
            except:
                continue

        return best_params

    def fit_new(func, x, y, y_err, params_init, param_names, constr=None, silent=False, optimize=True):
        """
        Enhanced fitting function with parameter optimization
        """
        if constr is None:
            constr = ([-np.inf for _ in param_names],
                     [np.inf for _ in param_names])

        # Optimize initial parameters if requested
        if optimize:
            params_init = optimize_init_params(func, x, y, y_err, params_init, constr)

        # Perform final fit with optimized initial parameters
        params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, 
                                     bounds=constr, maxfev=50000)

        param_sigmas = [np.sqrt(covariance[i][i]) for i in range(len(params))]
        table = [[f"{params[i]:.5f} ± {param_sigmas[i]:.5f}" 
                  for i in range(len(params))]]

        # Calculate reduced chi squared
        nu = len(y) - len(param_names)
        y_fit = func(x, *params)
        chi_2 = np.sum(((y - y_fit) / y_err) ** 2) / nu

        if not silent:
            print(tabulate(table, param_names, tablefmt="fancy_grid"))
            print(f"$\chi_v^2$ = {chi_2:.2f}")

        return params, covariance, param_sigmas, chi_2

    par_quad, cov_quad, par_err_quad, chi2_quad = fit_new(g1f1_quad_fullx_DIS, indep_data,
                                                          dis_df['G1F1'],
                                                          dis_df['G1F1.err'],
                                                          quad_new_init,
                                                          quad_param_names,
                                                          constr=quad_new_constr)
    def covariance_to_correlation(cov_matrix):
        # Calculate the standard deviations (square root of variances)
        std_devs = np.sqrt(np.diag(cov_matrix))

        # Create the correlation matrix
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

        return correlation_matrix

    corr_quad = covariance_to_correlation(cov_quad)
                                                      
    print("\n\n", "-"*25)
    print("Best set of new DIS parameters with uncertainties:")
    for names, param, error in zip(quad_param_names, par_quad, par_err_quad):
        print(f"{names}:  {param:.4e} ± {error:.4e}")
        if names == "beta":
            beta_val = param
    # Print covariance matrix
    print("Covariance matrix:")
    for row in cov_quad:
        print(" ".join(f"{val:6.2e}" for val in row))
    # Print correlation matrix
    print("\nCorrelation matrix:")
    for row in corr_quad:
        print(" ".join(f"{val:6.2e}" for val in row))
    print("-"*25, "\n\n")

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # g1/f1 fit vs x, residuals vs x
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})

    fit_vals = g1f1_quad_fullx_DIS([x_dense, q2_dense], *par_quad)

    # Top plot: Data with Fit
    axs[0].errorbar(
        dis_df['X'], dis_df['G1F1'], yerr=dis_df['G1F1.err'], 
        fmt=config["marker"]["type"], color=config["colors"]["scatter"], 
        label='Data', markersize=config["marker"]["size"], 
        capsize=config["error_bar"]["cap_size"], capthick=config["error_bar"]["cap_thick"],
        linewidth=config["error_bar"]["line_width"], ecolor=config["colors"]["error_bar"]
    )
    axs[0].plot(
        x_dense, fit_vals, 'r-', 
        label=f'Fit ($\chi^2_{{red}}$ = {chi2_quad:.2f})', 
        linewidth=config["error_bar"]["line_width"]
    )

    # Labels, Titles, Legends
    axs[0].set_xlabel('x', fontsize=config["font_sizes"]["x_axis"])
    axs[0].set_ylabel('$g_1F_1$', fontsize=config["font_sizes"]["y_axis"])
    axs[0].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

    # Grid settings
    if config["grid"]["enabled"]:
        axs[0].grid(True, linestyle=config["grid"]["line_style"], 
                    linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"], 
                    color=config["colors"]["grid"])

    # Bottom plot: Residuals
    residuals = (dis_df['G1F1'] - g1f1_quad_fullx_DIS([dis_df['X'], dis_df['Q2']], *par_quad)) / dis_df['G1F1.err']
    axs[1].scatter(
        dis_df['X'], residuals, color=config["colors"]["scatter"], 
        s=config["marker"]["size"] * 2  # Scale scatter point size
    )
    axs[1].axhline(y=0, color=config["colors"]["error_band"], linestyle='-', alpha=0.5)

    axs[1].set_xlabel('x', fontsize=config["font_sizes"]["x_axis"])
    axs[1].set_ylabel('Residuals ($\\sigma$)', fontsize=config["font_sizes"]["y_axis"])

    # Grid settings for residuals plot
    if config["grid"]["enabled"]:
        axs[1].grid(True, linestyle=config["grid"]["line_style"], 
                    linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"], 
                    color=config["colors"]["grid"])

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")

    return {"par_quad" : par_quad, "cov_quad" : cov_quad, "corr_quad" : corr_quad, "par_err_quad" : par_err_quad, "chi2_quad" : chi2_quad, "beta_val" : beta_val, "residuals" : residuals}
