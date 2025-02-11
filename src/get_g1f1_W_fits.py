#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-02-11 01:04:24 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.interpolate import interp1d
import os

##################################################################################################################################################

from functions import x_to_W, W_to_x, red_chi_sqr,\
    breit_wigner_wrapper, breit_wigner_res, propagate_bw_error, \
    quad_nucl_curve_k, quad_nucl_curve_gamma, quad_nucl_curve_mass, g1f1_quad_new_DIS, \
    calculate_param_error, \
    damping_function, damping_function_err, \
    propagate_transition_error, propagate_complete_error, \
    propagate_dis_error

##################################################################################################################################################

def get_g1f1_W_fits(
        w, w_min, w_max, w_res_min, w_res_max, quad_fit_err, \
        res_df, dis_fit_params, \
        k_nucl_par, k_nucl_err, gamma_nucl_par, gamma_nucl_err, mass_nucl_par, mass_nucl_err, k_P_vals, gamma_P_vals, mass_P_vals, beta_val, \
        w_lims, pdf
):

    # Use these functions for k and gamma to plot the delta peaks of the data
    def optimize_parameters_for_q2_bin(res_df, l, k_nucl_par, gamma_nucl_par, mass_nucl_par, k_P_vals, gamma_P_vals, mass_P_vals, w_lims):
        def objective_function(params):

            w_dis_transition, damping_dis_width = params

            q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

            # List to store chi-squared values for each Q2 label
            chi_squared_values = []

            k_fit_params = [k_nucl_par] #, k_spline]
            gamma_fit_params = [gamma_nucl_par] #, gamma_spline]
            mass_fit_params = [mass_nucl_par] #, mass_spline]      
            fit_funcs_k = [quad_nucl_curve_k] #, BSpline]
            fit_funcs_gamma = [quad_nucl_curve_gamma] #, BSpline]
            fit_funcs_mass = [quad_nucl_curve_mass] #, BSpline]
            fit_names = ["New"] # , f"Spline, s={sk}"]

            chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

            for i in range(len(k_fit_params)):
                k_params = k_fit_params[i]
                args = [q2]

                for j in range(len(gamma_fit_params)):

                  gamma_params = gamma_fit_params[j]

                  for ij in range(len(mass_fit_params)):

                      if (i, j, ij) not in chosen_fits:
                          # skip combinations that aren't desired
                          continue

                      mass_params = mass_fit_params[ij]

                      if i==1:
                        # spline k
                        k = k_fit_params[i](q2)
                      else:
                        k_args = args + [p for p in k_params]
                        if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                        k = fit_funcs_k[i](*k_args)

                      if j==1:
                        # spline gamma
                        gamma = gamma_fit_params[j](q2)
                      else:
                        gamma_args = args + [p for p in gamma_params]
                        if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                        gamma = fit_funcs_gamma[j](*gamma_args)

                      if ij==1:
                        # spline mass
                        mass = mass_fit_params[ij](q2)
                      else:
                        mass_args = args + [p for p in mass_params]
                        if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                        mass = fit_funcs_mass[ij](*mass_args)

                      # calculate fitted curve
                      y = breit_wigner_res(w, mass, k, gamma)

                      # try getting a chi squared for this curve for w_lims[i][0]<W<w_lims[i][1]
                      W = res_df['W']
                      y_cal = breit_wigner_res(W, mass, k, gamma)
                      y_act = res_df['G1F1']
                      y_act_err = res_df['G1F1.err']
              
            w_dis = np.linspace(2.0,3.0,1000)
            q2_array = np.ones(w_dis.size)*q2
            x_dis = W_to_x(w_dis, q2_array)
            
            # Table F.1 from XZ's thesis
            quad_new_dis_par = dis_fit_params["par_quad"]
            y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)

            for i in range(len(k_fit_params)):

              k_params = k_fit_params[i]
              args = [q2]

              for j in range(len(gamma_fit_params)):

                gamma_params = gamma_fit_params[j]

                for ij in range(len(mass_fit_params)):

                    if (i, j, ij) not in chosen_fits:
                        # skip combinations that aren't desired
                        continue

                    mass_params = mass_fit_params[ij]

                    if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                    else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                    if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                    else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                    if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                    else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                    w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                    # Calculate Breit-Wigner fit
                    k_new_new = ((0.25 * q2) / (1.0 + 1.55 * q2)) * np.exp(-q2 / (2 * 0.25))
                    y_bw = breit_wigner_res(w_res, mass, k, gamma)

                    # Calculate DIS fit
                    y_dis = g1f1_quad_new_DIS([W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)], *quad_new_dis_par)
                    
                    # Calculate the complete fit
                    y_bw_bump =  breit_wigner_res(w_res, 1.55, k_new_new, 0.25)
                    y_transition = y_bw_bump + (y_bw - y_dis)

                    # Ensure smooth transition to DIS
                    damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
                    y_complete = y_transition * damping_dis + y_dis
                    
                    interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
                    y_complete_interpolated = interp_func(res_df['W'][res_df['Q2_labels']==l])
                    
                    nu = abs(len(y_complete_interpolated)-len(params))
                    chi2 = red_chi_sqr(y_complete_interpolated, res_df['G1F1'][res_df['Q2_labels']==l], res_df['G1F1.err'][res_df['Q2_labels']==l], nu)
                    
                    chi_squared_values.append(chi2)
                  
            # Return mean chi-squared across all Q2 labels
            return np.mean(chi_squared_values)
    
        # Define parameter bounds
        bounds = [
            (1.5, 1.9),   # w_dis_transition
            (0.1, 1.0)   # damping_dis_width            
            
        ]

        # Run differential evolution for this Q2 bin
        result = differential_evolution(
            objective_function, 
            bounds, 
            strategy='best1bin', 
            popsize=15,
            maxiter=50000
        )
        refined_result = minimize(
            objective_function,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-8}
        )
        
        def compute_uncertainties(result, objective_func, epsilon=1e-8):
            """Compute parameter uncertainties using numerical Hessian"""
            params = result.x
            n_params = len(params)
            hessian = np.zeros((n_params, n_params))
            
            # Compute numerical Hessian using central differences
            for i in range(n_params):
                for j in range(i + 1):
                    # Create perturbation vectors
                    h_i = np.zeros(n_params)
                    h_j = np.zeros(n_params)
                    h_i[i] = epsilon
                    h_j[j] = epsilon
                    
                    # Compute mixed partial derivatives
                    f_ij = objective_func(params + h_i + h_j)
                    f_i = objective_func(params + h_i)
                    f_j = objective_func(params + h_j)
                    f_0 = objective_func(params)
                    
                    hessian[i, j] = (f_ij - f_i - f_j + f_0) / (epsilon * epsilon)
                    hessian[j, i] = hessian[i, j]
            
            try:
                # Compute covariance matrix
                covariance = np.linalg.inv(hessian)
                # Extract parameter uncertainties (standard errors)
                uncertainties = np.sqrt(np.diag(covariance))
                
                # Validate uncertainties
                if np.all(np.isreal(uncertainties)) and np.all(np.isfinite(uncertainties)):
                    return uncertainties, True
            except np.linalg.LinAlgError:
                pass
            
            # Fallback: use parameter scaling
            param_ranges = np.array([high - low for low, high in bounds])
            return param_ranges * 0.01, False  # 1% of parameter range as uncertainty
            
        # Compute uncertainties using the refined result
        uncertainties, success = compute_uncertainties(refined_result, objective_function)
        
        return refined_result.x, uncertainties        
        
    full_results_csv = "../fit_data/full_results.csv"

    # Assuming optimize_parameters_for_q2_bin now returns both best_params and uncertainties
    param_names = ['w_dis_transition', 'damping_dis_width']

    if not os.path.exists(full_results_csv):
        print(f"\n\nFile '{full_results_csv}' does not exist. Finding best parameters!")

        # Initialize a dictionary to store results for all Q2 bins
        full_results = {}

        # Iterate through unique Q2 labels
        for l in res_df['Q2_labels'].unique():
            # Optimize parameters for this specific Q2 bin
            best_params, param_uncertainties = optimize_parameters_for_q2_bin(
                res_df, l, k_nucl_par, gamma_nucl_par, mass_nucl_par, 
                k_P_vals, gamma_P_vals, mass_P_vals, w_lims
            )

            # Store both parameters and uncertainties for this Q2 label
            full_results[l] = {
                'params': best_params,
                'errors': param_uncertainties
            }

        # Create separate DataFrames for parameters and uncertainties
        params_df = pd.DataFrame({label: data['params'] for label, data in full_results.items()}, index=param_names)
        errors_df = pd.DataFrame({label: data['errors'] for label, data in full_results.items()}, index=param_names)

        # Save results to CSV files
        params_df.to_csv(full_results_csv, index=True)
        errors_df.to_csv(full_results_csv.replace('.csv', '_errors.csv'), index=True)

        print("Results and uncertainties saved to CSV files")

    else:
        print(f"\n\nFile '{full_results_csv}' exists. Loading variables from CSV.")

    # Initialize dictionaries to store parameters and errors for each Q2 bin
    q2_bin_params = {}
    q2_bin_errors = {}

    # Load both CSV files
    full_results_df = pd.read_csv(full_results_csv, index_col=0)
    full_errors_df = pd.read_csv(full_results_csv.replace('.csv', '_errors.csv'), index_col=0)

    # Display the columns and index for debugging purposes
    print("CSV Columns:", full_results_df.columns)
    print("CSV Index:", full_results_df.index)

    # Iterate through each unique Q2 label
    for l in res_df['Q2_labels'].unique():
        if l in full_results_df.columns:
            # Initialize dictionaries for this Q2 bin
            q2_bin_params[l] = {}
            q2_bin_errors[l] = {}

            # Extract parameters and errors for this Q2 bin
            for name in param_names:
                if name in full_results_df.index:
                    q2_bin_params[l][name] = full_results_df.at[name, l]
                    q2_bin_errors[l][name] = full_errors_df.at[name, l]
                else:
                    print(f"Warning: Parameter '{name}' not found in the CSV index.")

            # Print parameters and errors for this Q2 bin
            print(f"\nBest Parameters for Q2 bin {l}:")
            for p_name in param_names:
                print(f"{p_name}: {q2_bin_params[l][p_name]} ± {q2_bin_errors[l][p_name]}")
        else:
            print(f"Warning: Q2 label '{l}' not found in the CSV columns.")

    print("\nVariables successfully loaded from the CSV.")

    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    
    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

    for name in param_names:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))

        q2 = []
        param_lst = []
        error_lst = []

        # Collect data for plotting
        for i, l in enumerate(res_df['Q2_labels'].unique()):
            q2.append(res_df['Q2'][res_df['Q2_labels']==l].unique()[0])
            param_lst.append(q2_bin_params[l][name])
            error_lst.append(q2_bin_errors[l][name])

        # Plot the data with error bars
        axs.errorbar(q2, param_lst, yerr=error_lst,
                    fmt='o', color=colors[0], markersize=8, capsize=5,
                    capthick=2, label=name)

        # Customize the plot
        axs.set_title(f"{name} vs Q²")
        axs.grid(True, linestyle='--', alpha=0.7)
        axs.legend()

        # Add labels
        fig.tight_layout()
        fig.text(0.5, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', size=14)
        fig.text(0.0001, 0.5, f"{name}", ha='center', va='center', rotation='vertical', size=16)

        # Save figure
        pdf.savefig(fig, bbox_inches="tight")

    def create_fit_functions():
        """
        Create dictionary of fit functions for each parameter.
        Each function is defined with its parameters and implementation.

        Returns:
        --------
        dict : Dictionary containing:
            'function': the fitting function
            'param_names': list of parameter names
            'initial_guess': initial values for the fit
            'param_bounds': bounds for each parameter (optional)
        """
        def create_function_info(func, param_names, initial_guess=None, param_bounds=None):
            return {
                'function': func,
                'param_names': param_names,
                'initial_guess': initial_guess,
                'param_bounds': param_bounds
            }

        # Define your functions here
        def linear(x, m, b):
            """Linear function: f(x) = mx + b"""
            return m * x + b

        def quadratic(x, a, b, c):
            """Quadratic function: f(x) = ax² + bx + c"""
            return a * x**2 + b * x + c

        def cubic(x, a, b, c, d):
            """Cubic function: f(x) = ax^3 + bx^2 + cx + d"""
            return a * x**3 + b * x**2 + c * x + d

        def exponential(x, a, b, c):
            """Exponential function: f(x) = a*exp(bx)"""
            return a * np.exp(b * x) + c

        def power_law(x, a, b, c):
            """Power law with offset: f(x) = ax^b + c"""
            return a * x**b + c

        def woods_saxon(x, a, a0, width):
            """Woods-Saxon function for smooth damping"""
            return 1 / (1 + np.exp((a - a0) / width))

        def woods_saxon(x, a, a0, width):
            """Woods-Saxon function for smooth damping"""
            return 1 / (1 + np.exp((a - a0) / width))

        def sinusoidal(x, a, a0, width, b):
            """
            Sinusoidal function for smooth modulation.
            """
            return a * np.sin((x - a0) / width) + b
         
        fit_functions = {
            'linear': create_function_info(
                func=linear,
                param_names=['slope', 'intercept'],
                initial_guess=[1, 0],
                param_bounds=([-np.inf, -np.inf], [np.inf, np.inf])
            ),
            'quadratic': create_function_info(
                func=quadratic,
                param_names=['a', 'b', 'c'],
                initial_guess=[1, 1, 0],
                param_bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            ),
            'cubic': create_function_info(
                func=cubic,
                param_names=['a', 'b', 'c', 'd'],
                initial_guess=[1, 1, 1, 0],
                param_bounds=([-np.inf] * 4, [np.inf] * 4)
            ),
            'exponential': create_function_info(
                func=exponential,
                param_names=['amplitude', 'rate', 'c'],
                initial_guess=[1, 0.1, 0.0],
                param_bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            ),
            'power_law': create_function_info(
                func=power_law,
                param_names=['amplitude', 'exponent', 'offset'],
                initial_guess=[1, 1, 0],
                param_bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            ),
            'woods_saxon': create_function_info(
                func=woods_saxon,
                param_names=['a', 'a0', 'width'],
                initial_guess=[1, 0, 1],
                param_bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            ),
        }

        return fit_functions

    # Plotting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'

    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

    # Dictionary to store the best fit results for each parameter
    best_fit_results = {}

    # Get fit functions
    fit_functions = create_fit_functions()

    for name in param_names:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))
        q2 = []
        param_lst = []
        error_lst = []

        # Collect data for plotting
        for i, l in enumerate(res_df['Q2_labels'].unique()):
            q2.append(res_df['Q2'][res_df['Q2_labels'] == l].unique()[0])
            param_lst.append(q2_bin_params[l][name])
            error_lst.append(q2_bin_errors[l][name])

        # Convert to numpy arrays for fitting
        q2_array = np.array(q2)
        param_array = np.array(param_lst)
        error_array = np.array(error_lst)

        # Initialize variables to track the best fit
        best_reduced_chi2 = np.inf
        best_fit_info = None
        best_fit_values = None
        best_fit_errors = None

        for fit_name, fit_info in fit_functions.items():
            fit_func = fit_info['function']
            initial_guess = fit_info['initial_guess']
            param_bounds = fit_info.get('param_bounds', ([-np.inf] * len(initial_guess), [np.inf] * len(initial_guess)))

            try:
                # Perform the fit
                popt, pcov = curve_fit(fit_func, q2_array, param_array,
                                       sigma=error_array, absolute_sigma=True,
                                       p0=initial_guess, bounds=param_bounds)

                # Calculate fit errors
                perr = np.sqrt(np.diag(pcov))

                # Calculate chi-square
                residuals = param_array - fit_func(q2_array, *popt)
                chi2 = np.sum((residuals / error_array) ** 2)
                dof = len(q2_array) - len(popt)
                reduced_chi2 = chi2 / dof if dof > 0 else np.nan

                # Check if this fit is closer to the ideal reduced chi^2 of 1
                if abs(reduced_chi2 - 1) < abs(best_reduced_chi2 - 1):
                    best_reduced_chi2 = reduced_chi2
                    best_fit_info = {
                        'parameters': popt,
                        'errors': perr,
                        'covariance': pcov,
                        'chi2': chi2,
                        'reduced_chi2': reduced_chi2,
                        'dof': dof,
                        'function_type': fit_func.__name__
                    }
                    best_fit_values = fit_func(np.linspace(min(q2_array), max(q2_array), 100), *popt)
                    best_fit_errors = perr

            except RuntimeError:
                print(f"Fit failed for parameter {name} with function {fit_func.__name__}")
                continue

        # Save the best fit result for this parameter
        if best_fit_info:
            best_fit_results[name] = best_fit_info

            # Plot data points with error bars
            axs.errorbar(q2, param_lst, yerr=error_lst,
                         fmt='o', color=colors[0], markersize=8,
                         capsize=5, capthick=2, label=name)
            
            fit_info_text = f"Best Fit: {best_fit_info['function_type']}\n"
            fit_info_text += f"χ²/dof = {best_reduced_chi2:.2f}"
            
            # Plot best fit line
            q2_fit = np.linspace(min(q2_array), max(q2_array), 100)
            try:
                axs.plot(q2_fit, best_fit_values, color="r", label=fit_info_text, linestyle='dashed')
            except ValueError:
                print("")

        # Customize the plot
        axs.set_title(f"Parameter {name} vs Q²")
        axs.grid(True, linestyle='--', alpha=0.7)
        axs.legend()

        # Add labels
        fig.tight_layout()
        fig.text(0.5, 0.001, "$Q^2\\ ({GeV}^2)$", ha='center', va='center', size=14)
        fig.text(0.0001, 0.5, f"{name}", ha='center', va='center',
                 rotation='vertical', size=16)

        # Save figure
        pdf.savefig(fig, bbox_inches="tight")
        
    # Print best fit results
    for name, results in best_fit_results.items():
        print(f"\nBest results for parameter {name}:")
        print(f"Function type: {results['function_type']}")
        for i, param in enumerate(results['parameters']):
            print(f"Parameter {i}: {param:.6f} ± {results['errors'][i]:.6f}")
        print(f"χ²/dof: {results['reduced_chi2']:.3f}")
        
    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']

      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):

        k_params = k_fit_params[i]
        args = [q2]


        for j in range(len(gamma_fit_params)):

          gamma_params = gamma_fit_params[j]

          for ij in range(len(mass_fit_params)):
                    
              if (i, j, ij) not in chosen_fits:
                  # skip combinations that aren't desired
                  continue

              mass_params = mass_fit_params[ij]

              if i==1:
                # spline k
                k = k_fit_params[i](q2)
              else:
                k_args = args + [p for p in k_params]
                if i==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  k_args += [P for P in k_P_vals]
                k = fit_funcs_k[i](*k_args)

              if j==1:
                # spline gamma
                gamma = gamma_fit_params[j](q2)
              else:
                gamma_args = args + [p for p in gamma_params]
                if j==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  gamma_args += [P for P in gamma_P_vals]
                gamma = fit_funcs_gamma[j](*gamma_args)

              if ij==1:
                # spline mass
                mass = mass_fit_params[ij](q2)
              else:
                mass_args = args + [p for p in mass_params]
                if ij==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  mass_args += [P for P in mass_P_vals]
                mass = fit_funcs_mass[ij](*mass_args)
                
              # calculate fitted curve
              y = breit_wigner_res(w, mass, k, gamma)

              # try getting a chi squared for this curve for w_lims[i][0]<W<w_lims[i][1]
              W = res_df['W']
              y_cal = breit_wigner_res(W, mass, k, gamma)
              y_act = res_df['G1F1']
              y_act_err = res_df['G1F1.err']
              nu = abs(len(y_act)-3) # n points minus 3 fitted parameters (k, gamma, mass)
              chi2 = red_chi_sqr(y_cal, y_act, y_act_err, nu)

              axs[row, col].plot(w, y, markersize=m_size,
                                 label=f"$\chi_v^2$={chi2:.2f}",
                                 linestyle='dashed')
              
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_params["par_quad"]
      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)
      y_dis = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
            
      axs[row, col].plot(w_dis, y_dis, color="r", label=f"Quad DIS Fit, $\\beta$ = {beta_val:.4f}", linestyle="--")
      
      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)
      
    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")

    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
      
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)
                      k_err = calculate_param_error(fit_funcs_k[i], k_args, k_nucl_err)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)
                      gamma_err = calculate_param_error(fit_funcs_gamma[i], gamma_args, gamma_nucl_err)
                      
                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)
                      mass_err = calculate_param_error(fit_funcs_mass[i], mass_args, mass_nucl_err)

                  # Main analysis code structure
                  y = breit_wigner_res(w, mass, k, gamma)

                  # Chi-squared calculation for W in [w_lims[i][0], w_lims[i][1]]
                  W = res_df['W'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_cal = breit_wigner_res(W, mass, k, gamma)
                  y_act = res_df['G1F1'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_act_err = res_df['G1F1.err'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                                  
                  axs[row, col].plot(w, y, markersize=m_size,
                                    linestyle='dashed')

      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)

      # original DIS fit params x0, y0, c, beta
      #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
      #y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
      #                        quad2_dis_par[2], quad2_dis_par[3])
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_params["par_quad"]
      y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
      
      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon

                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                  # Recalculate k, gamma, mass, and Breit-Wigner as in earlier loop
                  # Calculate Breit-Wigner fit
                  w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                  # Calculate Breit-Wigner fit
                  k_new_new = ((0.25 * q2) / (1.0 + 1.55 * q2)) * np.exp(-q2 / (2 * 0.25))
                  y_bw = breit_wigner_res(w_res, mass, k, gamma)

                  bw_err = propagate_bw_error(
                      w_res, mass, mass_err, k, k_err, gamma, gamma_err
                  )

                  # Calculate DIS fit
                  y_dis = g1f1_quad_new_DIS(
                      [W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)],
                      *quad_new_dis_par
                  )
                  
                  dis_err = propagate_dis_error(
                      quad_fit_err
                  )  # Error propagation

                  # Calculate the complete fit
                  y_bw_bump =  breit_wigner_res(w_res, 1.55, k_new_new, 0.25)
                  y_transition = y_bw_bump + (y_bw - y_dis)
                  transition_err = propagate_transition_error(
                      w_res,
                      bw_err,
                      w_lims[i][0],
                      w_lims[i][1],
                  ) # Error propagation

                  # Ensure smooth transition to DIS
                  damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
                  damping_dis_err = damping_function_err(                      
                      w_res, w_dis_transition, w_dis_transition_err, damping_dis_width, damping_dis_width_err
                  )  # Error propagation
                  
                  y_complete = y_transition * damping_dis + y_dis
                                    
                  complete_err = propagate_complete_error(
                      w_res,
                      transition_err,
                      damping_dis_err,
                      dis_err,
                      w_lims[i][0],
                      w_lims[i][1],
                      w_dis_transition,
                      w_max
                  ) # Error propagation

                  axs[row, col].plot(
                      w_res,
                      damping_dis,
                      color="red",
                      linestyle="-.",
                      label=f"damping_dis",
                  )
                  
      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")        

    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
      
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)
                      k_err = calculate_param_error(fit_funcs_k[i], k_args, k_nucl_err)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)
                      gamma_err = calculate_param_error(fit_funcs_gamma[i], gamma_args, gamma_nucl_err)
                      
                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)
                      mass_err = calculate_param_error(fit_funcs_mass[i], mass_args, mass_nucl_err)
                      
                  # Calculate Breit-Wigner fit
                  y = breit_wigner_res(w, mass, k, gamma)

                  bw_err = propagate_bw_error(
                      w_res, mass, mass_err, k, k_err, gamma, gamma_err
                  )

                  # Chi-squared calculation for W in [w_lims[i][0], w_lims[i][1]]
                  W = res_df['W'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_cal = breit_wigner_res(W, mass, k, gamma)
                  y_act = res_df['G1F1'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_act_err = res_df['G1F1.err'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
            
                  axs[row, col].plot(w, y, markersize=m_size,
                                    linestyle='dashed')

      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)

      # original DIS fit params x0, y0, c, beta
      #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
      #y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
      #                        quad2_dis_par[2], quad2_dis_par[3])
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_params["par_quad"]
      y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
      
      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon

                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                  # Recalculate k, gamma, mass, and Breit-Wigner as in earlier loop
                  # Calculate Breit-Wigner fit
                  w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                  # Calculate Breit-Wigner fit
                  k_new_new = ((0.25 * q2) / (1.0 + 1.55 * q2)) * np.exp(-q2 / (2 * 0.25))
                  y_bw = breit_wigner_res(w_res, mass, k, gamma)

                  bw_err = propagate_bw_error(
                      w_res, mass, mass_err, k, k_err, gamma, gamma_err
                  )

                  # Calculate DIS fit
                  y_dis = g1f1_quad_new_DIS(
                      [W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)],
                      *quad_new_dis_par
                  )
                  
                  dis_err = propagate_dis_error(
                      quad_fit_err
                  )  # Error propagation

                  # Calculate the complete fit
                  y_bw_bump =  breit_wigner_res(w_res, 1.55, k_new_new, 0.25)
                  y_transition = y_bw_bump + (y_bw - y_dis)
                  transition_err = propagate_transition_error(
                      w_res,
                      bw_err,
                      w_lims[i][0],
                      w_lims[i][1],
                  ) # Error propagation

                  # Ensure smooth transition to DIS
                  damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
                  damping_dis_err = damping_function_err(                      
                      w_res, w_dis_transition, w_dis_transition_err, damping_dis_width, damping_dis_width_err
                  )  # Error propagation
                  
                  y_complete = y_transition * damping_dis + y_dis
                                    
                  complete_err = propagate_complete_error(
                      w_res,
                      transition_err,
                      damping_dis_err,
                      dis_err,
                      w_lims[i][0],
                      w_lims[i][1],
                      w_dis_transition,
                      w_max
                  ) # Error propagation
                  
                  axs[row, col].plot(
                      w_res,
                      y_transition,
                      color="green",
                      linestyle="-.",
                      label=f"y_transition",
                  )

                  axs[row, col].plot(
                      w_res,
                      (1 - damping_dis) * y_dis,
                      color="cyan",
                      linestyle="-.",
                      label=f"(1-damping_dis)*y_dis",
                  )
                  
                  axs[row, col].plot(
                      w_res,
                      y_transition * damping_dis,
                      color="red",
                      linestyle="-.",
                      label=f"y_transition * damping_dis",
                  )
                  
                  axs[row, col].plot(
                      w_res,
                      y_complete,
                      color="blue",
                      linestyle="solid",
                      label=f"y_complete",
                  )

                  axs[row, col].plot(
                      w_res,
                      y_dis,
                      color="purple",
                      linestyle=":",
                      label=f"y_dis",
                  )

                  
      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")    
    
    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
      
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)
                      k_err = calculate_param_error(fit_funcs_k[i], k_args, k_nucl_err)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)
                      gamma_err = calculate_param_error(fit_funcs_gamma[i], gamma_args, gamma_nucl_err)
                      
                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)
                      mass_err = calculate_param_error(fit_funcs_mass[i], mass_args, mass_nucl_err)
                      
                  # Calculate Breit-Wigner fit
                  y = breit_wigner_res(w, mass, k, gamma)

                  bw_err = propagate_bw_error(
                      w_res, mass, mass_err, k, k_err, gamma, gamma_err
                  )

                  # Chi-squared calculation for W in [w_lims[i][0], w_lims[i][1]]
                  W = res_df['W'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_cal = breit_wigner_res(W, mass, k, gamma)
                  y_act = res_df['G1F1'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
                  y_act_err = res_df['G1F1.err'][res_df['Q2_labels'] == l][res_df['W'] <= w_lims[i][1]][res_df['W'] >= w_lims[i][0]]
            
                  axs[row, col].plot(w, y, markersize=m_size,
                                    linestyle='dashed')

      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)

      # original DIS fit params x0, y0, c, beta
      #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
      #y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
      #                        quad2_dis_par[2], quad2_dis_par[3])
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_params["par_quad"]
      y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
      
      axs[row, col].plot(w_dis, y_dis_new, color="violet", label=f"Quad DIS Fit, $\\beta$ = {beta_val:.4f}", linestyle="--")

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon

                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                  # Recalculate k, gamma, mass, and Breit-Wigner as in earlier loop
                  # Calculate Breit-Wigner fit
                  w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                  # Calculate Breit-Wigner fit
                  k_new_new = ((0.25 * q2) / (1.0 + 1.55 * q2)) * np.exp(-q2 / (2 * 0.25))
                  y_bw = breit_wigner_res(w_res, mass, k, gamma)

                  bw_err = propagate_bw_error(
                      w_res, mass, mass_err, k, k_err, gamma, gamma_err
                  )

                  # Calculate DIS fit
                  y_dis = g1f1_quad_new_DIS(
                      [W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)],
                      *quad_new_dis_par
                  )
                  
                  dis_err = propagate_dis_error(
                      quad_fit_err
                  )  # Error propagation

                  # Calculate the complete fit
                  y_bw_bump =  breit_wigner_res(w_res, 1.55, k_new_new, 0.25)
                  y_transition = y_bw_bump + (y_bw - y_dis)
                  transition_err = propagate_transition_error(
                      w_res,
                      bw_err,
                      w_lims[i][0],
                      w_lims[i][1],
                  ) # Error propagation

                  # Ensure smooth transition to DIS
                  damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
                  damping_dis_err = damping_function_err(                      
                      w_res, w_dis_transition, w_dis_transition_err, damping_dis_width, damping_dis_width_err
                  )  # Error propagation
                  
                  y_complete = y_transition * damping_dis + y_dis

                  interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
                  y_complete_interpolated = interp_func(res_df['W'][res_df['Q2_labels']==l])

                  nu = abs(len(y_complete_interpolated)-len([w_dis_transition, damping_dis_width]))
                  chi2 = red_chi_sqr(y_complete_interpolated, res_df['G1F1'][res_df['Q2_labels']==l], res_df['G1F1.err'][res_df['Q2_labels']==l], nu)
                                    
                  complete_err = propagate_complete_error(
                      w_res,
                      transition_err,
                      damping_dis_err,
                      dis_err,
                      w_lims[i][0],
                      w_lims[i][1],
                      w_dis_transition,
                      w_max
                  ) # Error propagation

                  '''
                  def moving_average(data, window_size):
                      window_size = min(window_size, len(data))
                      if window_size % 2 == 0:
                          window_size -= 1
                      if window_size < 3:
                          window_size = 3
                      return np.convolve(data, np.ones(window_size)/window_size, mode='same')

                  window_size = 1001
                  complete_err = moving_average(complete_err, window_size)
                  '''
                  
                  axs[row, col].plot(
                      w_res,
                      y_complete,
                      color="blue",
                      linestyle="solid",
                      label=f"$\chi_v^2$={chi2:.2f}",
                  )
                  axs[row, col].fill_between(
                      w_res,
                      y_complete - complete_err,
                      y_complete + complete_err,
                      color="blue",
                      alpha=0.3,
                      label="Fit Error",
                  )

                  # Define the transformation functions properly
                  def wrapper_w_to_x(w):
                      """ Transform W to x_Bj for constant q2. """
                      return W_to_x(w, np.full_like(w, q2))

                  def wrapper_x_to_w(x):
                      """ Transform x to W for constant q2. """
                      return x_to_W(x, np.full_like(x, q2))

                  # Add secondary x-axis for x_Bj
                  ax2 = axs[row, col].secondary_xaxis(
                      'top',
                      functions=(wrapper_w_to_x, wrapper_x_to_w)  # Just pass the function references
                  )
                  ax2.set_xlabel("x")  # Label for the secondary axis

      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")
    
