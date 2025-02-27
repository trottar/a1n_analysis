#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-02-23 11:46:45 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, differential_evolution
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re, ast, os
import json

##################################################################################################################################################

from functions import fit_with_dynamic_params, quad_nucl_curve_k, quad_nucl_curve_gamma, quad_nucl_curve_mass

##################################################################################################################################################

def fit_BW_params(q2, delta_par_df, pdf):

    fit_results_csv = "../fit_data/fit_results.csv"

    k_lb = [-1e10, -1e10, -1e10, -1e-10]
    k_ub = [1e10, 1e10, 1e10, 1e-10]
    k_bounds = Bounds(lb=k_lb, ub=k_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    k_p_vals_initial = [P0, P1, P2, Y1]

    gamma_lb = [-1e10, -1e10, -1e10, 0.0]
    gamma_ub = [1e10, 1e10, 1e10, 1e10]
    gamma_bounds = Bounds(lb=gamma_lb, ub=gamma_ub)        
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    gamma_p_vals_initial = [P0, P1, P2, Y1]

    mass_lb = [0.0, -1e10, 0.0, 0.0]
    mass_ub = [1e10, 1e10, 1e10, 2.0]
    mass_bounds = Bounds(lb=mass_lb, ub=mass_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    mass_p_vals_initial = [P0, P1, P2, Y1]                

    def quad_nucl_curve_gamma_wrapper(x, a, b, c, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_gamma(x, a, b, c, y0, P0, P1, P2, Y1)
    def quad_nucl_curve_k_wrapper(x, a, b, c, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_k(x, a, b, c, y0, P0, P1, P2, Y1)
    def quad_nucl_curve_mass_wrapper(x, a, b, c, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_mass(x, a, b, c, y0, P0, P1, P2, Y1)
    
    if not os.path.exists(fit_results_csv):
        print(f"\n\nFile '{fit_results_csv}' does not exist. Finding best fits!")
    
        # Initialize an empty list to store results
        fit_results = []

        # Perform fits for k, gamma, and mass
        print("-"*35)
        print("K Quad-Nucl Potential Fit Params")
        print("-"*35)
        k_best_params, k_best_p_vals, k_best_chi2, k_param_uncertainties, k_p_val_uncertainties = fit_with_dynamic_params(
            "k",
            x_data=delta_par_df["Q2"],
            y_data=delta_par_df["k"],
            y_err=delta_par_df["k.err"],
            param_bounds=k_bounds,
            p_vals_initial=k_p_vals_initial,
            fit_function=quad_nucl_curve_k_wrapper,
            N=10,
        )

        # Store results
        fit_results.append({
            "Parameter": "k",
            "Best Fit Parameters": k_best_params,
            "Best P Values": k_best_p_vals,
            "Chi-Squared": k_best_chi2,
            "Parameter Uncertainties" : k_param_uncertainties,
            "P Value Uncertainties": k_p_val_uncertainties,
        })

        # Repeat for gamma
        print("-"*35)
        print("Gamma Quad-Nucl Potential Fit Params")
        print("-"*35)
        gamma_best_params, gamma_best_p_vals, gamma_best_chi2, gamma_param_uncertainties, gamma_p_val_uncertainties = fit_with_dynamic_params(
            "gamma",
            x_data=delta_par_df["Q2"],
            y_data=delta_par_df["gamma"],
            y_err=delta_par_df["gamma.err"],
            param_bounds=gamma_bounds,
            p_vals_initial=gamma_p_vals_initial,
            fit_function=quad_nucl_curve_gamma_wrapper,
            N=10,
        )

        # Store results
        fit_results.append({
            "Parameter": "gamma",
            "Best Fit Parameters": gamma_best_params,
            "Best P Values": gamma_best_p_vals,
            "Chi-Squared": gamma_best_chi2,
            "Parameter Uncertainties" : gamma_param_uncertainties,            
            "P Value Uncertainties": gamma_p_val_uncertainties,
        })

        # Repeat for mass
        print("-"*35)
        print("Mass Quad-Nucl Potential Fit Params")
        print("-"*35)
        mass_best_params, mass_best_p_vals, mass_best_chi2, mass_param_uncertainties, mass_p_val_uncertainties = fit_with_dynamic_params(
            "mass",
            x_data=delta_par_df["Q2"],
            y_data=delta_par_df["M"],
            y_err=delta_par_df["M.err"],
            param_bounds=mass_bounds,
            p_vals_initial=mass_p_vals_initial,
            fit_function=quad_nucl_curve_mass_wrapper,
            N=10,
        )

        # Store results
        fit_results.append({
            "Parameter": "mass",
            "Best Fit Parameters": mass_best_params,
            "Best P Values": mass_best_p_vals,
            "Chi-Squared": mass_best_chi2,
            "Parameter Uncertainties" : mass_param_uncertainties,
            "P Value Uncertainties": mass_p_val_uncertainties,
        })

        # Save results to a CSV
        df_results = pd.DataFrame(fit_results)
        df_results.to_csv(fit_results_csv, index=False)

        print("Results saved to fit_results.csv")

    else:
        print(f"\n\nFile '{fit_results_csv}' exists. Loading variables from CSV.")

        # Load the CSV file
        fit_results_df = pd.read_csv(fit_results_csv)

        # Define a function to parse list-like strings with irregular formatting
        def parse_list(value):
            if pd.isna(value):  # Handle NaN values
                return []
            # Remove extra spaces and split by spaces or commas
            value = re.sub(r'\s+', ',', value.strip())  # Replace spaces with commas
            value = value.replace('[,', '[').replace(',]', ']')  # Clean up misplaced commas
            try:
                return ast.literal_eval(value)
            except Exception:
                print(f"Error parsing value: {value}")
                return []

        # Apply parsing to the relevant columns
        columns_to_parse = ["Best Fit Parameters", "P Value Uncertainties", "Parameter Uncertainties"]
        for col in columns_to_parse:
            fit_results_df[col] = fit_results_df[col].apply(parse_list)

        # Extract variables from the parsed dataframe
        for _, row in fit_results_df.iterrows():
            if row["Parameter"] == "k":
                k_best_params = row["Best Fit Parameters"]
                k_best_p_vals = ast.literal_eval(row["Best P Values"])  # Parse normally formatted column
                k_best_chi2 = row["Chi-Squared"]
                k_param_uncertainties = row["Parameter Uncertainties"]
                k_p_val_uncertainties = row["P Value Uncertainties"]

            elif row["Parameter"] == "gamma":
                gamma_best_params = row["Best Fit Parameters"]
                gamma_best_p_vals = ast.literal_eval(row["Best P Values"])
                gamma_best_chi2 = row["Chi-Squared"]
                gamma_param_uncertainties = row["Parameter Uncertainties"]
                gamma_p_val_uncertainties = row["P Value Uncertainties"]

            elif row["Parameter"] == "mass":
                mass_best_params = row["Best Fit Parameters"]
                mass_best_p_vals = ast.literal_eval(row["Best P Values"])
                mass_best_chi2 = row["Chi-Squared"]
                mass_param_uncertainties = row["Parameter Uncertainties"]                
                mass_p_val_uncertainties = row["P Value Uncertainties"]

        print("Variables successfully loaded from the CSV.")


    # Unpack the results
    print("k Parameters")
    print("-"*50)
    k_nucl_par = k_best_params
    k_P_vals = k_best_p_vals
    k_nucl_chi2 = k_best_chi2
    print("Best fit parameters (a, b, c, y0):", k_nucl_par)
    print("Best P values (P0, P1, P2, Y1):", k_P_vals)
    print("Best chi-squared:", k_nucl_chi2)
    print("Parameters uncertainties:", k_param_uncertainties)
    print("P value uncertainties:", k_p_val_uncertainties)
    print("\n")

    # Unpack the results
    print("Gamma Parameters")
    print("-"*50)
    gamma_nucl_par = gamma_best_params
    gamma_P_vals = gamma_best_p_vals
    gamma_nucl_chi2 = gamma_best_chi2
    print("Best fit parameters (a, b, c, y0):", gamma_nucl_par)
    print("Best P values (P0, P1, P2, Y1):", gamma_P_vals)
    print("Best chi-squared:", gamma_nucl_chi2)
    print("Parameters uncertainties:", gamma_param_uncertainties)
    print("P value uncertainties:", gamma_p_val_uncertainties)
    print("\n")    

    # Unpack the results
    print("Mass Parameters")
    print("-"*50)    
    mass_nucl_par = mass_best_params
    mass_P_vals = mass_best_p_vals
    mass_nucl_chi2 = mass_best_chi2
    print("Best fit parameters (a, b, c, y0):", mass_nucl_par)
    print("Best P values (P0, P1, P2, Y1):", mass_P_vals)
    print("Best chi-squared:", mass_nucl_chi2)
    print("Parameters uncertainties:", mass_param_uncertainties)    
    print("P value uncertainties:", mass_p_val_uncertainties)
    print("\n")

    # k
    k_nucl_args = [q2] + [p for p in k_nucl_par] + [P for P in k_P_vals]
    k_nucl = quad_nucl_curve_k(*k_nucl_args)
    k_nucl_err = [p for p in k_param_uncertainties] + [p for p in k_p_val_uncertainties]
    # gamma
    gamma_nucl_args = [q2] + [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
    gamma_nucl = quad_nucl_curve_gamma(*gamma_nucl_args)
    gamma_nucl_err = [p for p in gamma_param_uncertainties] + [p for p in gamma_p_val_uncertainties]
    # mass
    mass_nucl_args = [q2] + [p for p in mass_nucl_par] + [P for P in mass_P_vals]
    mass_nucl = quad_nucl_curve_mass(*mass_nucl_args)
    mass_nucl_err = [p for p in mass_param_uncertainties] + [p for p in mass_p_val_uncertainties]

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # plot M, k, gamma vs Q2 from variable M fit
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # maintain distinct colors between plots by keeping track of the index in the color map
    color_index = 0
    
    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))
    
    def find_param_errors(i, var_name):
        x_data = delta_par_df["Q2"]
        if var_name == "k":
            true_params = [p for p in k_nucl_par] + [P for P in k_P_vals]
            y_data = delta_par_df["k"]
            y_err = delta_par_df["k.err"]
            y_nucl = k_nucl
            bounds = (k_lb + [P-(1e-6) for P in k_P_vals], k_ub + [P+(1e-6) for P in k_P_vals])
            model = quad_nucl_curve_k
        elif var_name == "mass":
            true_params = [p for p in mass_nucl_par] + [P for P in mass_P_vals]
            y_data = delta_par_df["M"]
            y_err = delta_par_df["M.err"]
            y_nucl = mass_nucl            
            bounds = (mass_lb + [P-(1e-6) for P in mass_P_vals], mass_ub + [P+(1e-6) for P in mass_P_vals])
            model = quad_nucl_curve_mass
        elif var_name == "gamma":
            true_params = [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
            y_data = delta_par_df["gamma"]
            y_err = delta_par_df["gamma.err"]
            y_nucl = gamma_nucl
            bounds = (gamma_lb + [P-(1e-6) for P in gamma_P_vals], gamma_ub + [P+(1e-6) for P in gamma_P_vals])
            model = quad_nucl_curve_gamma
            
        else:
            print("ERROR: Invalid variable name!")
            return

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        y_err = np.array(y_err)

        # Perform the initial fit
        popt, pcov = curve_fit(
            model, x_data, y_data, p0=true_params, sigma=y_err, bounds=bounds, absolute_sigma=True
        )

        # Define file names for saving/loading bootstrap results
        params_filename = f"../fit_data/bootstrap_{var_name}_params.npy"
        fits_data_filename = f"../fit_data/bootstrap_{var_name}_fits_data.npy"
        fits_q2_filename = f"../fit_data/bootstrap_{var_name}_fits_q2.npy"
        
        if os.path.exists(params_filename) and os.path.exists(fits_q2_filename):
            print(f"Loading existing bootstrap results for {var_name}...")
            bootstrap_params = np.load(params_filename)
            bootstrap_fits_data = np.load(fits_data_filename)            
            bootstrap_fits_q2 = np.load(fits_q2_filename)
        else:
        
            # Bootstrap parameters
            n_bootstrap = 10000
            n_points = len(x_data)
            bootstrap_params = np.zeros((n_bootstrap, len(true_params)))
            bootstrap_fits_data = np.zeros((n_bootstrap, len(x_data)))
            bootstrap_fits_q2 = np.zeros((n_bootstrap, len(q2)))

            # Perform bootstrap iterations
            for b in range(n_bootstrap):
                # Generate bootstrap sample
                indices = np.random.randint(0, n_points, size=n_points)
                x_bootstrap = x_data[indices]
                y_bootstrap = y_data[indices]

                try:

                    # Fit the bootstrap sample
                    if var_name == "k":
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in k_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in k_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    elif var_name == "gamma":
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in gamma_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in gamma_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    elif var_name == "mass":
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in mass_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in mass_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    else:
                        bounds=bounds
                    boot_popt, _ = curve_fit(
                        model, 
                        x_bootstrap, 
                        y_bootstrap, 
                        p0=true_params,
                        bounds=bounds,
                        sigma=y_err,
                        absolute_sigma=True
                    )                    
                    bootstrap_params[b] = boot_popt
                    bootstrap_fits_data[b] = model(x_data, *boot_popt)
                    bootstrap_fits_q2[b] = model(q2, *boot_popt)
                except RuntimeError:
                    bootstrap_params[b] = popt
                    bootstrap_fits_data[b] = model(x_data, *popt)
                    bootstrap_fits_q2[b] = model(q2, *popt)

            # Save bootstrap results
            np.save(params_filename, bootstrap_params)
            np.save(fits_data_filename, bootstrap_fits_data)
            np.save(fits_q2_filename, bootstrap_fits_q2)
            print(f"Saved bootstrap results for {var_name}")
                    
        # Calculate bootstrap uncertainties
        param_stds = np.std(bootstrap_params, axis=0)

        # Print results
        print("\n\n", "-"*25)
        print("Best-fit parameters with uncertainties:")
        print("Original fit uncertainties:")
        perr = np.sqrt(np.diag(pcov))
        for param, error, boot_err in zip(popt, perr, param_stds):
            print(f"{param:.4e} ± {error:.4e} (fit) ± {boot_err:.4e} (bootstrap)")
        print("Covariance matrix:")
        for row in pcov:
            print(" ".join(f"{val:6.2e}" for val in row))
        print("-"*25)

        # Compute the Jacobian matrix (keep original error calculation)
        def jacobian(x, params):
            epsilon = np.sqrt(np.finfo(float).eps)
            return np.array([
                (model(x, *(params + epsilon * np.eye(len(params))[i])) - 
                 model(x, *(params - epsilon * np.eye(len(params))[i]))) / 
                (2 * epsilon) for i in range(len(params))
            ]).T

        # Compute fit and error bars
        fit = model(x_data, *popt)
        J = jacobian(x_data, popt)
        fit_var = np.sum(J @ pcov * J, axis=1)
        fit_err = np.sqrt(fit_var)

        def moving_average(data, window_size):
            window_size = min(window_size, len(data))
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 3:
                window_size = 3
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')

        # Calculate uncertainties for q2 points
        q2_fit = model(q2, *popt)
        q2_err = np.std(bootstrap_fits_q2, axis=0)
        window_size = min(len(q2) // 3, 15)
        smoothed_q2_err = moving_average(q2_err, window_size)
                        
        # Plot
        axs[i].errorbar(x_data, y_data, yerr=y_err, 
                        fmt=config["marker"]["type"], label='Data', 
                        color=config["colors"]["scatter"], markersize=config["marker"]["size"], 
                        capsize=config["error_bar"]["cap_size"], capthick=config["error_bar"]["cap_thick"])

        axs[i].plot(x_data, fit, label='Curve_fit', color=config["colors"]["fit"])
        axs[i].plot(q2, q2_fit, label='Extrapolation', color=config["colors"]["extrapolation"], linestyle='--')
        axs[i].plot(q2, y_nucl, label='Diff. Ev.', color=config["colors"]["diff_ev"])
        axs[i].fill_between(q2, 
                            q2_fit - smoothed_q2_err,
                            q2_fit + smoothed_q2_err,
                            alpha=0.5, color=config["colors"]["error_band"])

    for i, var_name in enumerate(["k", "gamma", "mass"]):
        find_param_errors(i, var_name)

    axs[0].set_ylabel("k", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_ylabel("$\Gamma$", fontsize=config["font_sizes"]["y_axis"])
    axs[2].set_ylabel("M", fontsize=config["font_sizes"]["y_axis"])    

    axs[0].legend(fontsize=config["font_sizes"]["legend"])
    axs[1].legend(fontsize=config["font_sizes"]["legend"])
    axs[2].legend(fontsize=config["font_sizes"]["legend"])

    axs[0].set_ylim(-.12, 0.05)
    axs[1].set_ylim(-.5, 0.5)
    axs[2].set_ylim(1.1, 1.6)
    axs[0].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[1].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[2].axhline(y=1.232, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figures
    pdf.savefig(fig, bbox_inches="tight")

    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # plot all the parameters vs Q2
    for i, label in enumerate(delta_par_df["Experiment"].unique()):
        axs[0].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==label]["k"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==label]["k.err"], 
                        fmt=config["marker"]["type"], color=config["colors"]["scatter"], 
                        markersize=config["marker"]["size"], capsize=config["error_bar"]["cap_size"], 
                        label=label, capthick=config["error_bar"]["cap_thick"])

        axs[1].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==label]["gamma"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==label]["gamma.err"], 
                        fmt=config["marker"]["type"], color=config["colors"]["scatter"], 
                        markersize=config["marker"]["size"], capsize=config["error_bar"]["cap_size"], 
                        label=label, capthick=config["error_bar"]["cap_thick"])

        axs[2].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==label]["M"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==label]["M.err"], 
                        fmt=config["marker"]["type"], color=config["colors"]["scatter"], 
                        markersize=config["marker"]["size"], capsize=config["error_bar"]["cap_size"], 
                        label=label, capthick=config["error_bar"]["cap_thick"])
        
    axs[0].plot(q2, k_nucl, label="New Fit $\chi_v^2$=" + f"{k_nucl_chi2:.2f}", color=config["colors"]["fit"])
    axs[1].plot(q2, gamma_nucl, label="New Fit $\chi_v^2$=" + f"{gamma_nucl_chi2:.2f}", color=config["colors"]["fit"])
    axs[2].plot(q2, mass_nucl, label="New Fit $\chi_v^2$=" + f"{mass_nucl_chi2:.2f}", color=config["colors"]["fit"])
    
    fig.tight_layout()

    axs[0].set_ylabel("k", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_ylabel("$\Gamma$", fontsize=config["font_sizes"]["y_axis"])
    axs[2].set_ylabel("M", fontsize=config["font_sizes"]["y_axis"])

    axs[0].legend(fontsize=config["font_sizes"]["legend"])
    axs[1].legend(fontsize=config["font_sizes"]["legend"])
    axs[2].legend(fontsize=config["font_sizes"]["legend"])

    axs[0].set_ylim(-.12, 0.05)
    axs[1].set_ylim(-.5, 0.5)
    axs[2].set_ylim(1.1, 1.6)
    axs[0].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[1].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[2].axhline(y=1.232, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    # Save figures
    pdf.savefig(fig, bbox_inches="tight")
    
    return {
        "k params" : {
            "nucl_par" : k_nucl_par,
            "par_err" : k_param_uncertainties,           
            "P_vals" : k_P_vals,
            "p_val_err" : k_p_val_uncertainties,
            "nucl_chi2" : k_nucl_chi2,
            "nucl_args" : k_nucl_args,
            "nucl_curve" : k_nucl,
            "nucl_curve_err" : k_nucl_err
        },
        "gamma params" : {
            "nucl_par" : gamma_nucl_par,
            "par_err" : gamma_param_uncertainties,           
            "P_vals" : gamma_P_vals,
            "p_val_err" : gamma_p_val_uncertainties,
            "nucl_chi2" : gamma_nucl_chi2,
            "nucl_args" : gamma_nucl_args,
            "nucl_curve" : gamma_nucl,
            "nucl_curve_err" : gamma_nucl_err
        },
        "mass params" : {
            "nucl_par" : mass_nucl_par,
            "par_err" : mass_param_uncertainties,           
            "P_vals" : mass_P_vals,
            "p_val_err" : mass_p_val_uncertainties,
            "nucl_chi2" : mass_nucl_chi2,
            "nucl_args" : mass_nucl_args,
            "nucl_curve" : mass_nucl,
            "nucl_curve_err" : mass_nucl_err
        },
    }

