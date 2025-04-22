#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-21 17:16:54 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit, differential_evolution, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json

##################################################################################################################################################
from functions import (
    red_chi_sqr,
    breit_wigner_res,
    breit_wigner_bump,
    W_to_x,
    quad_nucl_curve_k, quad_nucl_curve_gamma, quad_nucl_curve_mass, 
    g1f1_quad_fullx_DIS,
    damping_function,    
    k_gamma_mass_loop,
    k_new_new
)

##################################################################################################################################################
def fit_dis_transition(
        w_min, w_max, res_df, dis_fit_params,
        k_nucl_par, k_nucl_err,
        gamma_nucl_par, gamma_nucl_err,
        mass_nucl_par, mass_nucl_err,
        k_P_vals, gamma_P_vals, mass_P_vals,
        w_lims,
        pdf
):
    """
    Searches for best-fit parameters (w_dis_transition, damping_dis_width)
    for each Q² bin, then saves or loads those results from CSV.
    Also fits those bin-by-bin results to multiple candidate functions
    (linear, quadratic, etc.) to return the best continuous parameterization.
    Finally, the global candidate function is used to propagate uncertainties
    and update the per-bin parameters for use in the later global fit.

    Now each best-fit function includes 'eval_func(q2)'
    -> (value, error) from parameter covariance.

    Returns
    -------
    q2_bin_params : dict
        Nested dictionary of best-fit parameters for each Q² bin, e.g.:
        q2_bin_params[Q2_label]["w_dis_transition"]
        q2_bin_params[Q2_label]["damping_dis_width"]

    q2_bin_errors : dict
        Nested dictionary of parameter uncertainties for each Q² bin, e.g.:
        q2_bin_errors[Q2_label]["w_dis_transition"]
        q2_bin_errors[Q2_label]["damping_dis_width"]

    best_fit_results : dict
        Dictionary of best-fit functional forms for each parameter
        ('w_dis_transition' and 'damping_dis_width') across Q²,
        including 'eval_func(q2) -> (value, error)'.
    """

    ###########################################################################
    # (1) Define the function that optimizes w_dis_transition, damping_dis_width
    ###########################################################################
    def optimize_parameters_for_q2_bin(
        res_df, l,
        k_nucl_par, gamma_nucl_par, mass_nucl_par,
        k_P_vals, gamma_P_vals, mass_P_vals,
        w_lims
    ):
        """
        Differential-evolution + local refine to find w_dis_transition and
        damping_dis_width that minimize the mean chi² across the resonance+DIS model
        for the specified Q² bin (l).
        """

        def objective_function(params):
            w_dis_transition, damping_dis_width = params
            # Extract this bin's Q²:
            q2 = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]

            k_fit_params     = [k_nucl_par]
            gamma_fit_params = [gamma_nucl_par]
            mass_fit_params  = [mass_nucl_par]
            fit_funcs_k      = [quad_nucl_curve_k]
            fit_funcs_gamma  = [quad_nucl_curve_gamma]
            fit_funcs_mass   = [quad_nucl_curve_mass]
            fit_names        = ["New"]
            chosen_fits = [(0, 0, 0)]
            
            # Evaluate model vs data and accumulate chi²:
            chi_squared_values = []
            w_res = np.linspace(w_min, w_max, 1000)
            
            for (ii, jj, ijj, k, k_err, gamma, gamma_err, mass, mass_err) in k_gamma_mass_loop(
                    q2, w_res, k_fit_params, gamma_fit_params, mass_fit_params,
                    fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
                    k_P_vals, gamma_P_vals, mass_P_vals,
                    k_nucl_err, gamma_nucl_err, mass_nucl_err
            ):           
                # Breit-Wigner piece
                y_bw = breit_wigner_res(w_res, mass, k, gamma)
                
                # DIS piece
                x_dis = W_to_x(w_res, np.full_like(w_res, q2))
                quad_new_dis_par = dis_fit_params["par_quad"]  # e.g. (a, b, c, d)
                y_dis = g1f1_quad_fullx_DIS([x_dis, np.full_like(w_res, q2)], *quad_new_dis_par)

                # Additional example bump
                k_new = k_new_new(q2)
                y_bw_bump = breit_wigner_bump(w_res, 1.55, k_new, 0.25)

                # Transition
                y_transition = y_bw_bump + (y_bw - y_dis)

                # Damping and final
                damping_dis = damping_function(w_res, w_dis_transition, damping_dis_width)
                y_complete  = y_transition * damping_dis + y_dis

                # Compare to data
                W_data = res_df['W'][res_df['Q2_labels'] == l]
                y_data = res_df['G1F1'][res_df['Q2_labels'] == l]
                y_err  = res_df['G1F1.err'][res_df['Q2_labels'] == l]

                f_interp = interp1d(
                    w_res, y_complete,
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                model_at_data = f_interp(W_data)

                dof  = abs(len(model_at_data) - len(params))
                chi2 = red_chi_sqr(model_at_data, y_data, y_err, dof)
                chi_squared_values.append(chi2)
            return np.mean(chi_squared_values)

        # Bounds for (w_dis_transition, damping_dis_width):
        bounds = [
            (1.5, 1.9),  # w_dis_transition
            (0.1, 1.0)   # damping_dis_width
        ]

        # Global search
        result = differential_evolution(
            objective_function, bounds,
            strategy='best1bin',
            popsize=15,
            maxiter=50000
        )

        # Local refine
        refined_result = minimize(
            objective_function,
            result.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-8}
        )

        def compute_uncertainties(opt_result, objective_func, epsilon=1e-5, bounds=None, delta=1.0, max_iter=50, step_factor=1.1):
            """
            Compute parameter uncertainties using a central-difference numerical Hessian.
            If Hessian inversion fails, fallback to a profile likelihood method.

            Parameters:
              opt_result: The optimization result object (must have .x)
              objective_func: Function to compute the objective
              epsilon: Step size for Hessian finite differences
              bounds: Optional bounds for parameters (unused in fallback here)
              delta: The increase in the objective function used to define the 1σ interval
              max_iter: Maximum iterations for the profile likelihood search
              step_factor: Multiplicative factor for stepping in the profile search

            Returns:
              uncertainties: 1σ uncertainties for each parameter
              success: Boolean indicating if the Hessian-based method was successful
            """
            params = opt_result.x
            n_params = len(params)
            hessian = np.zeros((n_params, n_params))
            f0 = objective_func(params)

            # Compute Hessian diagonal
            for i in range(n_params):
                h = np.zeros(n_params)
                h[i] = epsilon
                f_plus = objective_func(params + h)
                f_minus = objective_func(params - h)
                hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (epsilon ** 2)

            # Compute Hessian off-diagonals
            for i in range(n_params):
                for j in range(i+1, n_params):
                    h_i = np.zeros(n_params)
                    h_j = np.zeros(n_params)
                    h_i[i] = epsilon
                    h_j[j] = epsilon
                    f_pp = objective_func(params + h_i + h_j)
                    f_pm = objective_func(params + h_i - h_j)
                    f_mp = objective_func(params - h_i + h_j)
                    f_mm = objective_func(params - h_i - h_j)
                    hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon ** 2)
                    hessian[j, i] = hessian[i, j]

            # Attempt to invert the Hessian
            try:
                reg = 1e-8 * np.eye(n_params)
                covariance = np.linalg.inv(hessian + reg)
                uncertainties = np.sqrt(np.diag(covariance))
                if np.all(np.isreal(uncertainties)) and np.all(np.isfinite(uncertainties)):
                    return uncertainties, True
            except np.linalg.LinAlgError:
                pass

            # Fallback: Profile likelihood method.
            def profile_uncertainty(i, f0, delta=delta):
                # Search in the downward direction
                lower_val = params[i]
                for _ in range(max_iter):
                    lower_val /= step_factor
                    test_params = params.copy()
                    test_params[i] = lower_val
                    if objective_func(test_params) > f0 + delta:
                        break
                # Search in the upward direction
                upper_val = params[i]
                for _ in range(max_iter):
                    upper_val *= step_factor
                    test_params = params.copy()
                    test_params[i] = upper_val
                    if objective_func(test_params) > f0 + delta:
                        break
                uncert = max(params[i] - lower_val, upper_val - params[i])
                return uncert

            uncertainties = np.array([profile_uncertainty(i, f0) for i in range(n_params)])
            return uncertainties, False

        uncertainties, _ = compute_uncertainties(refined_result, objective_function, bounds=bounds)
        return refined_result.x, uncertainties

    ###########################################################################
    # (2) Check for existing CSV or run optimization
    ###########################################################################
    full_results_csv = "../fit_data/full_results.csv"
    param_names = ['w_dis_transition', 'damping_dis_width']

    if not os.path.exists(full_results_csv):
        print(f"\nFile '{full_results_csv}' does not exist. Finding best parameters!")
        full_results = {}

        for l in res_df['Q2_labels'].unique():
            best_params, param_uncertainties = optimize_parameters_for_q2_bin(
                res_df, l,
                k_nucl_par, gamma_nucl_par, mass_nucl_par,
                k_P_vals, gamma_P_vals, mass_P_vals,
                w_lims
            )
            full_results[l] = {
                'params': best_params,
                'errors': param_uncertainties
            }

        params_df = pd.DataFrame(
            {label: data['params'] for label, data in full_results.items()},
            index=param_names
        )
        errors_df = pd.DataFrame(
            {label: data['errors'] for label, data in full_results.items()},
            index=param_names
        )

        params_df.to_csv(full_results_csv, index=True)
        errors_df.to_csv(full_results_csv.replace('.csv', '_errors.csv'), index=True)
        print("Results and uncertainties saved to CSV files.")
    else:
        print(f"\nFile '{full_results_csv}' exists. Loading variables from CSV.")

    ###########################################################################
    # (3) Load final parameters from CSV
    ###########################################################################
    q2_bin_params = {}
    q2_bin_errors = {}

    full_results_df = pd.read_csv(full_results_csv, index_col=0)
    full_errors_df = pd.read_csv(full_results_csv.replace('.csv', '_errors.csv'), index_col=0)

    print("CSV Columns:", full_results_df.columns)
    print("CSV Index:", full_results_df.index)

    for l in res_df['Q2_labels'].unique():
        if l in full_results_df.columns:
            q2_bin_params[l] = {}
            q2_bin_errors[l] = {}

            for name in param_names:
                if name in full_results_df.index:
                    q2_bin_params[l][name] = full_results_df.at[name, l]
                    q2_bin_errors[l][name] = full_errors_df.at[name, l]
                else:
                    print(f"Warning: Parameter '{name}' not found in the CSV index.")

            print(f"\nBest Parameters for Q2 bin {l}:")
            for p_name in param_names:
                val = q2_bin_params[l][p_name]
                err = q2_bin_errors[l][p_name]
                print(f"{p_name}: {val} ± {err}")
        else:
            print(f"Warning: Q2 label '{l}' not found in the CSV columns.")

    print("\nVariables successfully loaded from the CSV.")

    ###########################################################################
    # (4) Plot: bin-by-bin results for each parameter
    ###########################################################################
    with open("config.json", "r") as f:
        config = json.load(f)
    
    for name in param_names:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))

        q2_vals = []
        param_vals = []
        param_errs = []

        for i, l in enumerate(res_df['Q2_labels'].unique()):
            q2_this_bin = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]
            q2_vals.append(q2_this_bin)
            param_vals.append(q2_bin_params[l][name])
            param_errs.append(q2_bin_errors[l][name])

        axs.errorbar(
            q2_vals, param_vals, yerr=param_errs,
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],
            label=name
        )

        axs.set_title(f"{name} vs Q²", fontsize=config["font_sizes"]["title"])

        if config["grid"]["enabled"]:
            axs.grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

        axs.legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

        fig.tight_layout()
        fig.text(0.5, 0.001, "$Q^2\\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
        fig.text(0.0001, 0.5, f"{name}", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])

        pdf.savefig(fig, bbox_inches="tight")

    ###########################################################################
    # (5) Define candidate functions for global fit propagation
    ###########################################################################
    def inverse(x, a, b, c):
        return a + b/(x+c)

    def create_function_info(func, param_names, initial_guess=None, param_bounds=None):
        return {
            'function': func,
            'param_names': param_names,
            'initial_guess': initial_guess,
            'param_bounds': param_bounds
        }

    # Restore your original candidate functions with data-driven limits
    fit_functions = {
        'inverse_damp': create_function_info(
            func=inverse,
            param_names=['a', 'b', 'c'],
            initial_guess=[0.105, 0, 0.5],
            param_bounds=([0.100, -np.inf, 0.4], [0.110, np.inf, 0.6])
        ),
        'inverse_trans': create_function_info(
            func=inverse,
            param_names=['a', 'b', 'c'],
            initial_guess=[1.73, 0, 0.5],
            param_bounds=([1.70, -np.inf, 0.4], [1.75, np.inf, 0.6])
        )
    }

    ###########################################################################
    # (6) Numerical gradient for error propagation
    ###########################################################################
    def numerical_gradient(f, x, params, eps=1e-8):
        grad = []
        params = np.array(params, dtype=float)
        for i in range(len(params)):
            h_i = eps * max(abs(params[i]), 1.0)
            params_up = params.copy()
            params_dn = params.copy()
            params_up[i] += h_i
            params_dn[i] -= h_i
            f_up = f(x, *params_up)
            f_dn = f(x, *params_dn)
            deriv = (f_up - f_dn) / (2.0 * h_i)
            grad.append(deriv)
        return np.array(grad)

    def build_eval_func(f, popt, pcov):
        def _eval_func(x):
            x_arr = np.atleast_1d(x)
            values = []
            errors = []
            for xx in x_arr:
                val = f(xx, *popt)
                grad = numerical_gradient(f, xx, popt)
                var = grad @ pcov @ grad.T
                err = np.sqrt(var) if var > 0 else np.nan
                values.append(val)
                errors.append(err)
            if np.isscalar(x):
                return values[0], errors[0]
            else:
                return np.array(values), np.array(errors)
        return _eval_func

    ###########################################################################
    # (7) Fit bin-by-bin points to candidate functions (global fit propagation)
    ###########################################################################
    best_fit_results = {}
    
    for name in param_names:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))

        q2_vals = []
        param_vals = []
        param_errs = []

        for l in res_df['Q2_labels'].unique():
            q2_this_bin = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]
            q2_vals.append(q2_this_bin)
            param_vals.append(q2_bin_params[l][name])
            param_errs.append(q2_bin_errors[l][name])

        q2_array = np.array(q2_vals, dtype=float)
        param_array = np.array(param_vals, dtype=float)
        error_array = np.array(param_errs, dtype=float)

        best_reduced_chi2 = np.inf
        best_fit_info = None

        # Try each candidate function
        for fit_name, func_info in fit_functions.items():
            fit_func = func_info['function']
            initial_guess = func_info['initial_guess']
            param_bounds = func_info.get('param_bounds', ([-np.inf] * len(initial_guess), [np.inf] * len(initial_guess)))

            try:
                popt, pcov = curve_fit(
                    fit_func,
                    q2_array,
                    param_array,
                    sigma=error_array,
                    absolute_sigma=True,
                    p0=initial_guess,
                    bounds=param_bounds
                )
                perr = np.sqrt(np.diag(pcov))
                residuals = param_array - fit_func(q2_array, *popt)
                chi2 = np.sum((residuals / error_array) ** 2)
                dof = len(q2_array) - len(popt)
                reduced_chi2 = chi2 / dof if dof > 0 else np.nan

                if abs(reduced_chi2 - 1.0) < abs(best_reduced_chi2 - 1.0):
                    best_reduced_chi2 = reduced_chi2
                    best_fit_info = {
                        'function_type': fit_name,
                        'parameters': popt,
                        'errors': perr,
                        'covariance': pcov,
                        'chi2': chi2,
                        'reduced_chi2': reduced_chi2,
                        'dof': dof,
                    }
            except RuntimeError:
                continue

        axs.errorbar(
            q2_array, param_array, yerr=error_array,
            fmt=config["marker"]["type"],
            color=config["colors"]["scatter"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            linewidth=config["error_bar"]["line_width"],
            ecolor=config["colors"]["error_bar"],
            label=name
        )

        if best_fit_info is not None:
            chosen_func = fit_functions[best_fit_info['function_type']]['function']
            popt = best_fit_info['parameters']
            pcov = best_fit_info['covariance']

            best_fit_info['eval_func'] = build_eval_func(chosen_func, popt, pcov)

            q2_fit = np.linspace(min(q2_array), max(q2_array), 200)
            vals_plot, errs_plot = best_fit_info['eval_func'](q2_fit)
            fit_label = (
                f"Best Fit: {best_fit_info['function_type']}\n"
                f"χ²/dof = {best_fit_info['reduced_chi2']:.2f}"
            )
            axs.plot(q2_fit, vals_plot, color="r", linestyle='dashed', label=fit_label)

            best_fit_results[name] = best_fit_info

        axs.set_title(f"Parameter '{name}' vs Q² (Global Candidate Fit)", fontsize=config["font_sizes"]["title"])

        if config["grid"]["enabled"]:
            axs.grid(
                True, linestyle=config["grid"]["line_style"],
                linewidth=config["grid"]["line_width"], alpha=config["grid"]["alpha"],
                color=config["colors"]["grid"]
            )

        axs.legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

        fig.tight_layout()
        fig.text(0.5, 0.001, "$Q^2\\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])
        fig.text(0.0001, 0.5, f"{name}", ha='center', va='center', rotation='vertical', fontsize=config["font_sizes"]["y_axis"])
        pdf.savefig(fig, bbox_inches="tight")

    ###########################################################################
    # (8) Global Fit Propagation: Update bin-by-bin parameters with the global fit
    ###########################################################################
    for name, results in best_fit_results.items():
        for l in res_df['Q2_labels'].unique():
            q2_value = res_df['Q2'][res_df['Q2_labels'] == l].unique()[0]
            value, error = results['eval_func'](q2_value)
            q2_bin_params[l][name] = value
            q2_bin_errors[l][name] = error

    ###########################################################################
    # (9) Print final best fits
    ###########################################################################
    for name, results in best_fit_results.items():
        print(f"\n===== Best results for parameter '{name}' =====")
        print(f"Chosen function: {results['function_type']}")
        print(f"χ²/dof: {results['reduced_chi2']:.3f} (chi2={results['chi2']:.3f}, dof={results['dof']})")
        for i, param_val in enumerate(results['parameters']):
            print(f"  Param {i}: {param_val:.6f} ± {results['errors'][i]:.6f}")

    ###########################################################################
    # (10) Return everything
    ###########################################################################
    return best_fit_results, q2_bin_params, q2_bin_errors
