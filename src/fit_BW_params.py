#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-03-28 11:16:58 trottar"
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

from functions import (
    fit_with_dynamic_params,
    get_quad_nucl_curve_k,
    normalize_bw_k_curve_mode,
    quad_nucl_curve_k,
    quad_nucl_curve_gamma,
    quad_nucl_curve_mass,
)
from utility import prefix_generated_output_name, project_path, src_path

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


def _plot_experiment_points(ax, delta_par_df, value_column, error_column, experiment_styles, config):
    for label in delta_par_df["Experiment"].dropna().unique():
        experiment_rows = delta_par_df[delta_par_df["Experiment"] == label]
        style = experiment_styles[label]
        ax.errorbar(
            experiment_rows["Q2"],
            experiment_rows[value_column],
            yerr=experiment_rows[error_column],
            fmt=style["marker"],
            linestyle="none",
            color=style["color"],
            ecolor=style["color"],
            markersize=config["marker"]["size"],
            capsize=config["error_bar"]["cap_size"],
            capthick=config["error_bar"]["cap_thick"],
            elinewidth=config["error_bar"]["line_width"],
            markeredgecolor=config["marker"]["edge_color"],
            markeredgewidth=max(0.5, config["marker"]["edge_width"] / 2.0),
            label=label,
        )


def _plot_experiment_triplet(axs, delta_par_df, experiment_styles, config):
    _plot_experiment_points(axs[0], delta_par_df, "k", "k.err", experiment_styles, config)
    _plot_experiment_points(axs[1], delta_par_df, "gamma", "gamma.err", experiment_styles, config)
    _plot_experiment_points(axs[2], delta_par_df, "M", "M.err", experiment_styles, config)


def _set_high_q2_triplet_limits(axs, delta_par_df, q2, k_nucl, gamma_nucl, mass_nucl):
    high_q2_start = 2.5
    q2_max = float(np.nanmax(q2))
    x_max = q2_max + 0.15

    for ax in axs:
        ax.set_xlim(high_q2_start, x_max)

    region_mask_curve = q2 >= high_q2_start
    region_mask_data = delta_par_df["Q2"] >= high_q2_start

    def _tight_ylim(curve_values, data_values, data_errors, fallback_pad):
        values = []
        if np.any(region_mask_curve):
            curve_region = np.asarray(curve_values)[region_mask_curve]
            values.append(curve_region)
        if np.any(region_mask_data):
            data_region = np.asarray(data_values)[region_mask_data]
            err_region = np.asarray(data_errors)[region_mask_data]
            values.extend([data_region - err_region, data_region + err_region])
        if not values:
            return None
        merged = np.concatenate(values)
        finite = merged[np.isfinite(merged)]
        if finite.size == 0:
            return None
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        span = y_max - y_min
        pad = fallback_pad if span <= 0 else max(fallback_pad, 0.2 * span)
        return (y_min - pad, y_max + pad)

    k_ylim = _tight_ylim(k_nucl, delta_par_df["k"], delta_par_df["k.err"], fallback_pad=0.0015)
    gamma_ylim = _tight_ylim(gamma_nucl, delta_par_df["gamma"], delta_par_df["gamma.err"], fallback_pad=0.03)
    mass_ylim = _tight_ylim(mass_nucl, delta_par_df["M"], delta_par_df["M.err"], fallback_pad=0.01)

    if k_ylim is not None:
        axs[0].set_ylim(*k_ylim)
    if gamma_ylim is not None:
        axs[1].set_ylim(*gamma_ylim)
    if mass_ylim is not None:
        axs[2].set_ylim(*mass_ylim)


def _build_artifact_path(filename, dataset_tag):
    dated_filename = prefix_generated_output_name(filename)
    if dataset_tag == "legacy":
        return project_path("fit_data", dated_filename)

    tagged_dir = project_path("fit_data", dataset_tag)
    os.makedirs(tagged_dir, exist_ok=True)
    return os.path.join(tagged_dir, dated_filename)


def _curve_cache_suffix(bw_k_curve_mode):
    if bw_k_curve_mode == "fixed_zero":
        return "_fixed_zero_v12"
    return ""


def _swap_k_curve_tag(dataset_tag, from_mode, to_mode):
    current = f"_k_{from_mode}_"
    replacement = f"_k_{to_mode}_"
    if current in dataset_tag:
        return dataset_tag.replace(current, replacement, 1)

    current_suffix = f"_k_{from_mode}"
    replacement_suffix = f"_k_{to_mode}"
    if dataset_tag.endswith(current_suffix):
        return dataset_tag[: -len(current_suffix)] + replacement_suffix

    return dataset_tag


def _parse_fit_results_payload(fit_results_csv):
    fit_results_df = pd.read_csv(fit_results_csv)

    def parse_list(value):
        if pd.isna(value):
            return []
        value = re.sub(r'\s+', ',', value.strip())
        value = value.replace('[,', '[').replace(',]', ']')
        try:
            return ast.literal_eval(value)
        except Exception:
            print(f"Error parsing value: {value}")
            return []

    columns_to_parse = ["Best Fit Parameters", "P Value Uncertainties", "Parameter Uncertainties"]
    for col in columns_to_parse:
        fit_results_df[col] = fit_results_df[col].apply(parse_list)

    results = {}
    for _, row in fit_results_df.iterrows():
        parameter = row["Parameter"]
        results[parameter] = {
            "Best Fit Parameters": row["Best Fit Parameters"],
            "Best P Values": ast.literal_eval(row["Best P Values"]),
            "Chi-Squared": row["Chi-Squared"],
            "Parameter Uncertainties": row["Parameter Uncertainties"],
            "P Value Uncertainties": row["P Value Uncertainties"],
        }

    return results


def _prepare_k_fit_dataframe(delta_par_df, bw_k_curve_mode):
    k_fit_df = delta_par_df.copy()
    excluded_k_row = None

    if bw_k_curve_mode == "fixed_zero":
        excluded_index = k_fit_df["Q2"].idxmax()
        excluded_k_row = k_fit_df.loc[[excluded_index]].copy()
        excluded_record = excluded_k_row.iloc[0]
        print(
            "[fit_BW_params] fixed_zero mode: reusing the tuned k-fit solution and replacing only the high-Q2 continuation. "
            f"Highest-Q2 point kept out of the fixed_zero diagnostic chi2: "
            f"Q2={excluded_record['Q2']:.3f}, k={excluded_record['k']:.5f}."
        )

    if k_fit_df.empty:
        raise RuntimeError("No resonance rows remain for the BW k-fit after applying the selected k-curve mode.")

    return k_fit_df, excluded_k_row


def fit_BW_params(
    q2,
    delta_par_df,
    pdf,
    dataset_tag="legacy",
    bw_k_curve_mode="non_tune",
    quad_nucl_curve_k_func=None,
):

    bw_k_curve_mode = normalize_bw_k_curve_mode(bw_k_curve_mode)
    if quad_nucl_curve_k_func is None:
        quad_nucl_curve_k_func = get_quad_nucl_curve_k(bw_k_curve_mode)
    k_reference_curve_mode = "tune" if bw_k_curve_mode == "fixed_zero" else bw_k_curve_mode
    quad_nucl_curve_k_fit_func = get_quad_nucl_curve_k(k_reference_curve_mode)

    delta_par_df = delta_par_df.copy()
    if "Experiment" not in delta_par_df.columns:
        delta_par_df["Experiment"] = delta_par_df.get("Label", "resonance data")
    if "Label" not in delta_par_df.columns:
        delta_par_df["Label"] = delta_par_df["Experiment"]

    for err_col in ("k.err", "gamma.err", "M.err"):
        if err_col in delta_par_df.columns:
            delta_par_df[err_col] = delta_par_df[err_col].abs()

    finite_mask = (
        np.isfinite(delta_par_df["Q2"])
        & np.isfinite(delta_par_df["k"])
        & np.isfinite(delta_par_df["gamma"])
        & np.isfinite(delta_par_df["M"])
        & np.isfinite(delta_par_df["k.err"])
        & np.isfinite(delta_par_df["gamma.err"])
        & np.isfinite(delta_par_df["M.err"])
        & (delta_par_df["k.err"] > 0)
        & (delta_par_df["gamma.err"] > 0)
        & (delta_par_df["M.err"] > 0)
    )
    dropped_count = int((~finite_mask).sum())
    if dropped_count:
        print(f"[fit_BW_params] Dropping {dropped_count} non-finite resonance rows before fitting.")
    delta_par_df = delta_par_df.loc[finite_mask].reset_index(drop=True)
    if delta_par_df.empty:
        raise RuntimeError("No finite resonance Breit-Wigner rows remain for BW parameter fitting.")

    k_fit_df, excluded_k_row = _prepare_k_fit_dataframe(delta_par_df, bw_k_curve_mode)
    if excluded_k_row is not None:
        excluded_index = int(excluded_k_row.index[0])
        k_chi2_df = delta_par_df.drop(index=excluded_index).reset_index(drop=True)
    else:
        k_chi2_df = delta_par_df
    curve_cache_suffix = _curve_cache_suffix(bw_k_curve_mode)
    fit_results_csv = _build_artifact_path(f"fit_results{curve_cache_suffix}.csv", dataset_tag)
    tune_reference_dataset_tag = _swap_k_curve_tag(dataset_tag, "fixed_zero", "tune")
    tune_reference_fit_results_csv = _build_artifact_path("fit_results.csv", tune_reference_dataset_tag)

    #k_lb = [-1e10, -1e10, -1e10, -1e-10]
    #k_ub = [1e10, 1e10, 1e10, 1e-10]
    #k_lb = [-1e10, -1e10, -1e10, -1e10]
    #k_ub = [1e10, 1e10, 1e10, 1e10]
    k_lb = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
    k_ub = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10]
    k_bounds = Bounds(lb=k_lb, ub=k_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    k_p_vals_initial = [P0, P1, P2, Y1]

    #gamma_lb = [-1e10, -1e10, -1e10, 0.0]
    #gamma_ub = [1e10, 1e10, 1e10, 1e10]
    gamma_lb = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
    gamma_ub = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]
    gamma_bounds = Bounds(lb=gamma_lb, ub=gamma_ub)        
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    gamma_p_vals_initial = [P0, P1, P2, Y1]

    #mass_lb = [0.0, -1e10, 0.0, 0.0]
    #mass_ub = [1e10, 1e10, 1e10, 2.0]
    mass_lb = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
    mass_ub = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]    
    mass_bounds = Bounds(lb=mass_lb, ub=mass_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    mass_p_vals_initial = [P0, P1, P2, Y1]                

    def quad_nucl_curve_gamma_wrapper(x, a, b, c, d, e, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_gamma(x, a, b, c, d, e, y0, P0, P1, P2, Y1)
    def quad_nucl_curve_k_wrapper(x, a, b, c, d, e, f, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_k_fit_func(x, a, b, c, d, e, f, y0, P0, P1, P2, Y1)
    def quad_nucl_curve_mass_wrapper(x, a, b, c, d, e, y0):
      """
      quadratic * nucl potential form
      x: independent data
      a, b, c: quadratic curve parameters
      y0: term to have curve end at a constant value
      """  
      return quad_nucl_curve_mass(x, a, b, c, d, e, y0, P0, P1, P2, Y1)
    
    if not os.path.exists(fit_results_csv):
        print(f"\n\nFile '{fit_results_csv}' does not exist. Finding best fits!")
    
        # Initialize an empty list to store results
        fit_results = []

        # Perform fits for k, gamma, and mass
        print("-"*35)
        print(f"K Quad-Nucl Potential Fit Params [{bw_k_curve_mode}]")
        print("-"*35)
        if bw_k_curve_mode == "fixed_zero" and os.path.exists(tune_reference_fit_results_csv):
            print(f"[fit_BW_params] Loading tuned k reference from {tune_reference_fit_results_csv}")
            tune_results = _parse_fit_results_payload(tune_reference_fit_results_csv)
            tune_k_results = tune_results["k"]
            k_best_params = tune_k_results["Best Fit Parameters"]
            k_best_p_vals = tune_k_results["Best P Values"]
            k_best_chi2 = tune_k_results["Chi-Squared"]
            k_param_uncertainties = tune_k_results["Parameter Uncertainties"]
            k_p_val_uncertainties = tune_k_results["P Value Uncertainties"]
        else:
            k_best_params, k_best_p_vals, k_best_chi2, k_param_uncertainties, k_p_val_uncertainties = fit_with_dynamic_params(
                "k",
                x_data=k_fit_df["Q2"],
                y_data=k_fit_df["k"],
                y_err=k_fit_df["k.err"],
                param_bounds=k_bounds,
                p_vals_initial=k_p_vals_initial,
                fit_function=quad_nucl_curve_k_wrapper,
                N=3,
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
            N=3,
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
            N=3,
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

        print(f"Results saved to {fit_results_csv}")

    else:
        print(f"\n\nFile '{fit_results_csv}' exists. Loading variables from CSV.")
        fit_results = _parse_fit_results_payload(fit_results_csv)
        k_best_params = fit_results["k"]["Best Fit Parameters"]
        k_best_p_vals = fit_results["k"]["Best P Values"]
        k_best_chi2 = fit_results["k"]["Chi-Squared"]
        k_param_uncertainties = fit_results["k"]["Parameter Uncertainties"]
        k_p_val_uncertainties = fit_results["k"]["P Value Uncertainties"]

        gamma_best_params = fit_results["gamma"]["Best Fit Parameters"]
        gamma_best_p_vals = fit_results["gamma"]["Best P Values"]
        gamma_best_chi2 = fit_results["gamma"]["Chi-Squared"]
        gamma_param_uncertainties = fit_results["gamma"]["Parameter Uncertainties"]
        gamma_p_val_uncertainties = fit_results["gamma"]["P Value Uncertainties"]

        mass_best_params = fit_results["mass"]["Best Fit Parameters"]
        mass_best_p_vals = fit_results["mass"]["Best P Values"]
        mass_best_chi2 = fit_results["mass"]["Chi-Squared"]
        mass_param_uncertainties = fit_results["mass"]["Parameter Uncertainties"]
        mass_p_val_uncertainties = fit_results["mass"]["P Value Uncertainties"]

        print("Variables successfully loaded from the CSV.")


    # Unpack the results
    print("k Parameters")
    print("-"*50)
    k_nucl_par = k_best_params
    k_P_vals = k_best_p_vals
    k_nucl_chi2 = k_best_chi2
    print("Best fit parameters (a, b, c, d, e, f, y0):", k_nucl_par)
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
    print("Best fit parameters (a, b, c, d, e, y0):", gamma_nucl_par)
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
    print("Best fit parameters (a, b, c, d, e, y0):", mass_nucl_par)
    print("Best P values (P0, P1, P2, Y1):", mass_P_vals)
    print("Best chi-squared:", mass_nucl_chi2)
    print("Parameters uncertainties:", mass_param_uncertainties)    
    print("P value uncertainties:", mass_p_val_uncertainties)
    print("\n")

    # k
    k_nucl_args = [q2] + [p for p in k_nucl_par] + [P for P in k_P_vals]
    k_nucl = quad_nucl_curve_k_func(*k_nucl_args)
    k_nucl_err = [p for p in k_param_uncertainties] + [p for p in k_p_val_uncertainties]
    if bw_k_curve_mode == "fixed_zero":
        k_chi2_values = quad_nucl_curve_k_func(
            k_chi2_df["Q2"].to_numpy(dtype=np.float64),
            *k_nucl_par,
            *k_P_vals
        )
        k_ndf = max(1, len(k_chi2_df) - (len(k_nucl_par) + len(k_P_vals)))
        k_nucl_chi2 = float(np.sum(((k_chi2_df["k"].to_numpy(dtype=np.float64) - k_chi2_values) / k_chi2_df["k.err"].to_numpy(dtype=np.float64)) ** 2) / k_ndf)
        print(f"[fit_BW_params] fixed_zero diagnostic chi2 excludes the highest-Q2 k point and is recomputed as {k_nucl_chi2:.2f}.")
    # gamma
    gamma_nucl_args = [q2] + [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
    gamma_nucl = quad_nucl_curve_gamma(*gamma_nucl_args)
    gamma_nucl_err = [p for p in gamma_param_uncertainties] + [p for p in gamma_p_val_uncertainties]
    # mass
    mass_nucl_args = [q2] + [p for p in mass_nucl_par] + [P for P in mass_P_vals]
    mass_nucl = quad_nucl_curve_mass(*mass_nucl_args)
    mass_nucl_err = [p for p in mass_param_uncertainties] + [p for p in mass_p_val_uncertainties]

    # Load configuration
    with open(src_path("config.json"), "r") as f:
        config = json.load(f)
    experiment_styles = _build_experiment_styles(delta_par_df["Experiment"].dropna().unique())
    
    # plot M, k, gamma vs Q2 from variable M fit
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # maintain distinct colors between plots by keeping track of the index in the color map
    color_index = 0
    
    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))
    
    def find_param_errors(i, var_name):
        x_data = delta_par_df["Q2"]
        fit_model = None
        display_model = None
        if var_name == "k":
            x_data = k_fit_df["Q2"]
            true_params = [p for p in k_nucl_par] + [P for P in k_P_vals]
            y_data = k_fit_df["k"]
            y_err = k_fit_df["k.err"]
            y_nucl = k_nucl
            bounds = (k_lb + [P-(1e-6) for P in k_P_vals], k_ub + [P+(1e-6) for P in k_P_vals])
            fit_model = quad_nucl_curve_k_fit_func
            display_model = quad_nucl_curve_k_func
        elif var_name == "mass":
            true_params = [p for p in mass_nucl_par] + [P for P in mass_P_vals]
            y_data = delta_par_df["M"]
            y_err = delta_par_df["M.err"]
            y_nucl = mass_nucl            
            bounds = (mass_lb + [P-(1e-6) for P in mass_P_vals], mass_ub + [P+(1e-6) for P in mass_P_vals])
            fit_model = quad_nucl_curve_mass
            display_model = quad_nucl_curve_mass
        elif var_name == "gamma":
            true_params = [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
            y_data = delta_par_df["gamma"]
            y_err = delta_par_df["gamma.err"]
            y_nucl = gamma_nucl
            bounds = (gamma_lb + [P-(1e-6) for P in gamma_P_vals], gamma_ub + [P+(1e-6) for P in gamma_P_vals])
            fit_model = quad_nucl_curve_gamma
            display_model = quad_nucl_curve_gamma
            
        else:
            print("ERROR: Invalid variable name!")
            return

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        y_err = np.array(y_err)

        # Perform the initial fit
        popt, pcov = curve_fit(
            fit_model, x_data, y_data, p0=true_params, sigma=y_err, bounds=bounds, absolute_sigma=True
        )

        # Define file names for saving/loading bootstrap results
        params_filename = _build_artifact_path(f"bootstrap_{var_name}_params{curve_cache_suffix}.npy", dataset_tag)
        fits_data_filename = _build_artifact_path(f"bootstrap_{var_name}_fits_data{curve_cache_suffix}.npy", dataset_tag)
        fits_q2_filename = _build_artifact_path(f"bootstrap_{var_name}_fits_q2{curve_cache_suffix}.npy", dataset_tag)
        
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
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in k_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in k_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    elif var_name == "gamma":
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in gamma_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in gamma_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    elif var_name == "mass":
                        lb_tmp = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in mass_P_vals]
                        ub_tmp = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in mass_P_vals]
                        bounds=(lb_tmp, ub_tmp)
                    else:
                        bounds=bounds
                    boot_popt, _ = curve_fit(
                        fit_model, 
                        x_bootstrap, 
                        y_bootstrap, 
                        p0=true_params,
                        bounds=bounds,
                        sigma=y_err,
                        absolute_sigma=True
                    )                    
                    bootstrap_params[b] = boot_popt
                    bootstrap_fits_data[b] = display_model(x_data, *boot_popt)
                    bootstrap_fits_q2[b] = display_model(q2, *boot_popt)
                except RuntimeError:
                    bootstrap_params[b] = popt
                    bootstrap_fits_data[b] = display_model(x_data, *popt)
                    bootstrap_fits_q2[b] = display_model(q2, *popt)

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
                (display_model(x, *(params + epsilon * np.eye(len(params))[i])) - 
                 display_model(x, *(params - epsilon * np.eye(len(params))[i]))) / 
                (2 * epsilon) for i in range(len(params))
            ]).T

        # Compute fit and error bars
        fit = display_model(x_data, *popt)
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
        q2_fit = display_model(q2, *popt)
        q2_err = np.std(bootstrap_fits_q2, axis=0)
        window_size = min(len(q2) // 3, 15)
        smoothed_q2_err = moving_average(q2_err, window_size)
                        
        # Plot
        if var_name == "k":
            _plot_experiment_points(axs[i], delta_par_df, "k", "k.err", experiment_styles, config)
        elif var_name == "gamma":
            _plot_experiment_points(axs[i], delta_par_df, "gamma", "gamma.err", experiment_styles, config)
        else:
            _plot_experiment_points(axs[i], delta_par_df, "M", "M.err", experiment_styles, config)

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
    plt.close(fig)

    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # plot all the parameters vs Q2
    _plot_experiment_triplet(axs, delta_par_df, experiment_styles, config)
        
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
    plt.close(fig)

    # plot the fits with the data in the high-Q2 region
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    _plot_experiment_triplet(axs, delta_par_df, experiment_styles, config)

    axs[0].plot(q2, k_nucl, label="New Fit $\chi_v^2$=" + f"{k_nucl_chi2:.2f}", color=config["colors"]["fit"])
    axs[1].plot(q2, gamma_nucl, label="New Fit $\chi_v^2$=" + f"{gamma_nucl_chi2:.2f}", color=config["colors"]["fit"])
    axs[2].plot(q2, mass_nucl, label="New Fit $\chi_v^2$=" + f"{mass_nucl_chi2:.2f}", color=config["colors"]["fit"])

    axs[0].set_ylabel("k", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_ylabel("$\Gamma$", fontsize=config["font_sizes"]["y_axis"])
    axs[2].set_ylabel("M", fontsize=config["font_sizes"]["y_axis"])

    axs[0].legend(fontsize=config["font_sizes"]["legend"])
    axs[1].legend(fontsize=config["font_sizes"]["legend"])
    axs[2].legend(fontsize=config["font_sizes"]["legend"])

    _set_high_q2_triplet_limits(axs, delta_par_df, q2, k_nucl, gamma_nucl, mass_nucl)

    axs[0].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[1].axhline(y=0, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])
    axs[2].axhline(y=1.232, color=config["colors"]["grid"], linestyle='--', alpha=config["grid"]["alpha"])

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', fontsize=config["font_sizes"]["x_axis"])

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # plot all the parameters vs Q2
    _plot_experiment_triplet(axs, delta_par_df, experiment_styles, config)
        
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

    # Low Q2 plot range
    axs[0].set_xlim(0.0, 0.5)
    axs[1].set_xlim(0.0, 0.5)
    axs[2].set_xlim(0.0, 0.5)
    
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
    plt.close(fig)
    
    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # plot all the parameters vs Q2
    _plot_experiment_triplet(axs, delta_par_df, experiment_styles, config)
        
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

    # Low Q2 plot range
    axs[0].set_xlim(0.0, 0.1)
    axs[1].set_xlim(0.0, 0.1)
    axs[2].set_xlim(0.0, 0.1)
    
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
    plt.close(fig)
    return {
        "k params" : {
            "curve_mode" : bw_k_curve_mode,
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
