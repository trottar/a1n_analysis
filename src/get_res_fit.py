#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-15 22:14:56 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import probplot
import pandas as pd
import json

##################################################################################################################################################

from functions import fit, breit_wigner_wrapper, breit_wigner_res

##################################################################################################################################################

def get_res_fit(k_init, gamma_init, mass_init, w_bounds, res_df, pdf):

    param_names = ["M", "k", "gamma"]
    n_params = len(param_names)
    
    def plot_res_fits(w_bounds, M, region_name, param_df):

        # Load configuration
        with open("config.json", "r") as f:
            config = json.load(f)

        # Make figure
        n_col = 5
        num_plots = len(res_df['Q2_labels'].unique())
        n_rows = num_plots // n_col + 1
        fig, axs = plt.subplots(n_rows, n_col, figsize=(n_col * 6, n_rows * 6))

        # Make fit curves and plot with data
        for i, l in enumerate(res_df['Q2_labels'].unique()):
            row = i // n_col
            col = i % n_col

            # Parameters for fit with 3 parameters (M, k, gamma) - Variable Mass
            params = [param_df[param_df['Label'] == l][f"{param_names[j]}"].unique()[0] for j in range(n_params)]

            # Parameters for fit with 2 parameters (k, gamma) - Fixed Mass
            params_constm = [param_df[param_df['Label'] == l][f"{param_names[j + 1]}_constM"].unique()[0] for j in range(n_params - 1)]

            # Generate fitted curve using the fitted parameters
            w = np.linspace(w_bounds[i][0], w_bounds[i][1], 1000, dtype=np.double)

            # Make fitted curve for all three parameter fit
            if 0 not in params:
                fit_curve = breit_wigner_res(w, params[0], params[1], params[2])
                axs[row, col].plot(
                    w, fit_curve,
                    color=config["colors"]["scatter"],
                    linewidth=config["error_bar"]["line_width"],
                    label="Fit = "
                )

            # Make fitted curve for two parameter fit (k, gamma)
            if 0 not in params_constm:
                fit_constm = breit_wigner_res(w, M, params_constm[0], params_constm[1])
                axs[row, col].plot(
                    w, fit_constm,
                    color=config["colors"]["error_bar"],
                    linestyle='dashed',
                    linewidth=config["error_bar"]["line_width"],
                    label=f"Fit M={M}"
                )

            # Plot the data
            axs[row, col].errorbar(
                res_df['W'][res_df['Q2_labels'] == l],
                res_df['G1F1'][res_df['Q2_labels'] == l],
                yerr=res_df['G1F1.err'][res_df['Q2_labels'] == l],
                fmt=config["marker"]["type"],
                color=config["colors"]["scatter"],
                markersize=config["marker"]["size"],
                capsize=config["error_bar"]["cap_size"],
                capthick=config["error_bar"]["cap_thick"],
                linewidth=config["error_bar"]["line_width"],
                ecolor=config["colors"]["error_bar"]
            )

            axs[row, col].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

            # Set axes limits
            axs[row, col].axhline(0, color="black", linestyle="--")

            w_min_data = res_df['W'][res_df['Q2_labels'] == l].min() - 0.1 * res_df['W'][res_df['Q2_labels'] == l].min()
            w_max_data = res_df['W'][res_df['Q2_labels'] == l].max() + 0.1 * res_df['W'][res_df['Q2_labels'] == l].max()
            axs[row, col].set_xlim(w_min_data, w_max_data)
            axs[row, col].set_title(l, fontsize=config["font_sizes"]["labels"])

            # Apply grid settings if enabled
            if config["grid"]["enabled"]:
                axs[row, col].grid(
                    True, linestyle=config["grid"]["line_style"],
                    linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                    color=config["colors"]["grid"]
                )

        fig.tight_layout()
        fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
        fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

        plt.tight_layout()

        # Save figure
        pdf.savefig(fig, bbox_inches="tight")
      
    ## Fitting Function
    def fit_breit_wigner(res_df, pdf, w_bounds, M, region_name):
        """
        w_bounds: list of bounds of W for fitting for each bin (ex: [(w min, w max),...])
        M: actual mass of resonance
        init: initial guesses for M, k, gamma
        region_name: unique name for resonance region that is being fitted

        returns a new dataframe with the parameters
        """
        # lists to make dataframe
        q2_list = []
        label_list = []
        exp_list = []
        # lists to hold k, gamma and their errors for constant M fits
        constM_par_lists = [[],[]]
        constM_par_err_lists = [[],[]]
        # lists to hold M, k, gamma and their errors
        par_lists = [[],[],[]]
        par_err_lists = [[],[],[]]

        # go through each experiment for constant Q2 and do a fit
        for i, name in enumerate(res_df['Q2_labels'].unique()):

            # get Q2 for this bin
            q2 = res_df['Q2'][res_df['Q2_labels']==name].unique()[0]
            
            n_points = len(res_df['W'][res_df['Q2_labels']==name])
            init = (mass_init[i], k_init[i], gamma_init[i])

            label_list.append(name) # add name to list of labels
            q2_list.append(q2)

            # add experiment name to list
            exp_list.append(res_df['Label'][res_df['Q2_labels']==name].unique()[0])

            # chop off data outside w_min and w_max
            w = res_df['W'][res_df['Q2_labels']==name][(res_df['W'][res_df['Q2_labels']==name] < w_bounds[i][1]) & (res_df['W'][res_df['Q2_labels']==name] > w_bounds[i][0])]
            g1f1 = res_df['G1F1'][res_df['Q2_labels']==name][(res_df['W'][res_df['Q2_labels']==name] < w_bounds[i][1]) & (res_df['W'][res_df['Q2_labels']==name] > w_bounds[i][0])]
            g1f1_err = res_df['G1F1.err'][res_df['Q2_labels']==name][(res_df['W'][res_df['Q2_labels']==name] < w_bounds[i][1]) & (res_df['W'][res_df['Q2_labels']==name] > w_bounds[i][0])]

            # default bounds for parameters M, k, gamma
            par_bounds = ([-np.inf, -np.inf, -np.inf],
                            [np.inf, np.inf, np.inf])
            
            try:
                # fit for (M, k, gamma) and get parameters and covariance matrix
                params, pcov, perr, chi2 = fit(breit_wigner_res, w, g1f1, g1f1_err, init, param_names, par_bounds, silent=True)
            except Exception as e:
              print(f"{name} Fit Failed\n   {e}")
              params = np.zeros(n_params)
              pcov = np.zeros((n_params, n_params))

            # add parameters and their errors to lists for M, k, gamma
            for j in range(len(params)):
              par_lists[j].append(params[j])
              par_err_lists[j].append(perr[j])

            try:
              # try fitting with constant M
              params_constm, pcov_constm, perr_constm, chi2_constm = fit(breit_wigner_wrapper(M), w, g1f1, g1f1_err, init[1:], ["k", "gamma"], [par_bounds[0][1:],par_bounds[1][1:]], silent=True)
            except Exception as e:
              print(f"{name} Constant M Fit Failed\n   {e}")
              params_constm = np.zeros(n_params-1)
              pcov_constm = np.zeros((n_params-1, n_params-1))

            # add parameters and their errors to lists for k & gamma for constant M fits
            for j in range(len(params_constm)):
              constM_par_lists[j].append(params_constm[j])
              constM_par_err_lists[j].append(perr_constm[j])

            table = [
                [f"{params[j]:.5f} ± {perr[j]:.5f}" for j in range(n_params)] + [f"{chi2:.2f}"],
                [f"{M}"] + [f"{params_constm[j]:.5f} ± {perr_constm[j]:.5f}" for j in range(n_params - 1)] + [f"{chi2_constm:.2f}"]
                ]

            # Print the table
            print(f"{name} Fit Parameters")
            print(tabulate(table, param_names + ["$\chi_v^2$"], tablefmt="fancy_grid"))

            # Load configuration
            with open("config.json", "r") as f:
                config = json.load(f)

            # Add diagnostic plots
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))

            # Apply global title with config font size
            fig.suptitle(f"Diagnostic Plots for {name}", fontsize=config["font_sizes"]["title"])

            # Plot 1: Data and fitted curve
            axs[0, 0].errorbar(
                w, g1f1, yerr=g1f1_err,
                fmt=config["marker"]["type"],
                markersize=config["marker"]["size"],
                color=config["colors"]["scatter"],
                capsize=config["error_bar"]["cap_size"],
                capthick=config["error_bar"]["cap_thick"],
                linewidth=config["error_bar"]["line_width"],
                ecolor=config["colors"]["error_bar"],
            )
            w_fine = np.linspace(w.min(), w.max(), 1000)
            axs[0, 0].plot(
                w_fine, breit_wigner_res(w_fine, *params),
                color=config["colors"]["error_band"],
                linewidth=config["error_bar"]["line_width"],
            )
            axs[0, 0].set_xlabel('W', fontsize=config["font_sizes"]["x_axis"])
            axs[0, 0].set_ylabel('G1F1', fontsize=config["font_sizes"]["y_axis"])
            axs[0, 0].set_title('Data and Fitted Curve', fontsize=config["font_sizes"]["labels"])
            axs[0, 0].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

            # Plot 2: Normalized Residuals
            residuals = g1f1 - breit_wigner_res(w, *params)  # Raw residuals
            normalized_residuals = residuals / g1f1_err  # Normalized by error

            axs[0, 1].scatter(
                w, normalized_residuals,
                color=config["colors"]["scatter"],
                s=config["marker"]["size"] * 2
            )
            axs[0, 1].axhline(y=0, color=config["colors"]["error_band"], linestyle='--', label='Zero Residual')
            axs[0, 1].set_xlabel('W', fontsize=config["font_sizes"]["x_axis"])
            axs[0, 1].set_ylabel('Normalized Residual', fontsize=config["font_sizes"]["y_axis"])
            axs[0, 1].set_title('Normalized Residuals vs W', fontsize=config["font_sizes"]["labels"])
            axs[0, 1].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

            if config["grid"]["enabled"]:
                axs[0, 1].grid(True, linestyle=config["grid"]["line_style"], linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"], color=config["colors"]["grid"])

            # Plot 3: Q-Q plot of normalized residuals
            (osm, osr), _ = probplot(normalized_residuals)
            axs[1, 0].plot(osm, osr, config["marker"]["type"], color=config["colors"]["scatter"], markersize=config["marker"]["size"])
            axs[1, 0].plot(osm, osm, config["grid"]["line_style"], color=config["colors"]["error_band"])
            axs[1, 0].set_xlabel('Theoretical Quantiles', fontsize=config["font_sizes"]["x_axis"])
            axs[1, 0].set_ylabel('Sample Quantiles', fontsize=config["font_sizes"]["y_axis"])
            axs[1, 0].set_title('Q-Q Plot of Normalized Residuals', fontsize=config["font_sizes"]["labels"])

            if config["grid"]["enabled"]:
                axs[1, 0].grid(True, linestyle=config["grid"]["line_style"], linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"], color=config["colors"]["grid"])

            # Compute overall reduced chi² and add annotation to Plot 1
            dof = len(w) - n_params  # Degrees of freedom
            overall_reduced_chi2 = np.sum(normalized_residuals ** 2) / dof
            axs[0, 0].text(0.05, 0.95, f"Reduced χ² = {overall_reduced_chi2:.2f}", transform=axs[0, 0].transAxes, fontsize=config["font_sizes"]["legend"], verticalalignment='top')

            # Remove the unused subplot (Plot 4) to clean up the figure
            fig.delaxes(axs[1, 1])

            plt.tight_layout()

            # Save figure
            pdf.savefig(fig, bbox_inches="tight")

        # make lists into dataframe
        params_df = pd.DataFrame({"Q2": q2_list,
                                  "Experiment": exp_list,
                                  "Label": label_list,
                                  "k_constM": constM_par_lists[0],
                                  "k_constM.err": constM_par_err_lists[0],
                                  "gamma_constM": constM_par_lists[1],
                                  "gamma_constM.err": constM_par_err_lists[1],
                                  "M": par_lists[0],
                                  "M.err": par_err_lists[0],
                                  "k": par_lists[1],
                                  "k.err": par_err_lists[1],
                                  "gamma": [abs(par) for par in par_lists[2]],
                                  "gamma.err": [abs(err) for err in par_err_lists[2]]})
        # plot
        plot_res_fits(w_bounds=w_bounds, M=M, region_name=region_name, param_df=params_df)
        
        return params_df        

    delta_par_df = fit_breit_wigner(res_df, pdf, w_bounds=w_bounds, M=1.232, region_name="1232MeV")

    return delta_par_df
