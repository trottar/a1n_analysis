#! /usr/bin/python

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dis_fit_models import (
    evaluate_dis_fit,
    get_dis_fit_model_config,
    get_dis_fit_model_keys,
    normalize_dis_fit_model,
)
from utility import project_path, safe_tabulate as tabulate, src_path


def _build_artifact_path(filename, dataset_tag):
    if dataset_tag == "legacy":
        return project_path("fit_data", filename)

    tagged_dir = project_path("fit_data", dataset_tag)
    os.makedirs(tagged_dir, exist_ok=True)
    return os.path.join(tagged_dir, filename)


def _save_dis_fit_summary(dataset_tag, dis_df, dis_fit_results, requested_model_key):
    summary_path = _build_artifact_path("dis_fit_summary.json", dataset_tag)
    payload = {
        "dataset_tag": dataset_tag,
        "requested_model_key": requested_model_key,
        "selected_model_key": dis_fit_results["model_key"],
        "model_display_name": dis_fit_results["model_display_name"],
        "curve_label": dis_fit_results["curve_label"],
        "n_points": int(len(dis_df)),
        "x_range": [
            float(np.min(dis_df["X"])),
            float(np.max(dis_df["X"])),
        ],
        "q2_range": [
            float(np.min(dis_df["Q2"])),
            float(np.max(dis_df["Q2"])),
        ],
        "parameter_names": list(dis_fit_results["parameter_names"]),
        "par_quad": np.asarray(dis_fit_results["par_quad"], dtype=float).tolist(),
        "par_err_quad": np.asarray(dis_fit_results["par_err_quad"], dtype=float).tolist(),
        "cov_quad": np.asarray(dis_fit_results["cov_quad"], dtype=float).tolist(),
        "corr_quad": np.asarray(dis_fit_results["corr_quad"], dtype=float).tolist(),
        "chi2_quad": float(dis_fit_results["chi2_quad"]),
        "beta_val": float(dis_fit_results["beta_val"]),
        "residual_summary": {
            "mean": float(np.mean(dis_fit_results["residuals"])),
            "std": float(np.std(dis_fit_results["residuals"])),
            "min": float(np.min(dis_fit_results["residuals"])),
            "max": float(np.max(dis_fit_results["residuals"])),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"[get_dis_fit] Saved DIS fit summary to {summary_path}")


def _save_dis_fit_comparison(dataset_tag, requested_model_key, fit_results, failed_results):
    comparison_json_path = _build_artifact_path("dis_fit_model_comparison.json", dataset_tag)
    comparison_csv_path = _build_artifact_path("dis_fit_model_comparison.csv", dataset_tag)

    successful_payload = []
    for rank, result in enumerate(sorted(fit_results, key=lambda item: item["chi2_quad"]), start=1):
        successful_payload.append(
            {
                "rank": rank,
                "model_key": result["model_key"],
                "model_display_name": result["model_display_name"],
                "curve_label": result["curve_label"],
                "chi2_quad": float(result["chi2_quad"]),
                "beta_val": float(result["beta_val"]),
                "parameter_names": list(result["parameter_names"]),
                "par_quad": np.asarray(result["par_quad"], dtype=float).tolist(),
                "par_err_quad": np.asarray(result["par_err_quad"], dtype=float).tolist(),
            }
        )

    payload = {
        "dataset_tag": dataset_tag,
        "requested_model_key": requested_model_key,
        "successful_results": successful_payload,
        "failed_results": failed_results,
    }

    with open(comparison_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    csv_rows = []
    for item in successful_payload:
        csv_rows.append(
            {
                "status": "success",
                "rank": item["rank"],
                "model_key": item["model_key"],
                "model_display_name": item["model_display_name"],
                "chi2_quad": item["chi2_quad"],
                "beta_val": item["beta_val"],
                "parameter_names": ", ".join(item["parameter_names"]),
                "parameters": json.dumps(item["par_quad"]),
                "parameter_errors": json.dumps(item["par_err_quad"]),
            }
        )
    for item in failed_results:
        csv_rows.append(
            {
                "status": "failed",
                "rank": None,
                "model_key": item["model_key"],
                "model_display_name": item["model_display_name"],
                "chi2_quad": None,
                "beta_val": None,
                "parameter_names": "",
                "parameters": "",
                "parameter_errors": item["error"],
            }
        )

    pd.DataFrame(csv_rows).to_csv(comparison_csv_path, index=False)
    print(f"[get_dis_fit] Saved DIS fit comparison to {comparison_json_path}")
    print(f"[get_dis_fit] Saved DIS fit comparison table to {comparison_csv_path}")


def _covariance_to_correlation(cov_matrix):
    std_devs = np.sqrt(np.diag(cov_matrix))
    return cov_matrix / np.outer(std_devs, std_devs)


def _randomize_init(params_init, bounds):
    params_init = np.asarray(params_init, dtype=float)
    if bounds is None:
        scale = np.where(np.abs(params_init) > 1.0e-6, np.abs(params_init), 1.0)
        return params_init + np.random.normal(loc=0.0, scale=scale, size=params_init.size)

    return np.array(
        [
            np.random.uniform(low, high)
            for low, high in zip(bounds[0], bounds[1])
        ],
        dtype=float,
    )


def optimize_init_params(func, x, y, y_err, params_init, bounds, n_tries=1000):
    best_chi2 = np.inf
    best_params = np.asarray(params_init, dtype=float)

    for _ in range(n_tries):
        try:
            random_init = _randomize_init(params_init, bounds)
            if bounds is None:
                params, _ = curve_fit(func, x, y, p0=random_init, sigma=y_err, maxfev=50000)
            else:
                params, _ = curve_fit(func, x, y, p0=random_init, sigma=y_err, bounds=bounds, maxfev=50000)

            y_fit = func(x, *params)
            nu = len(y) - len(params)
            chi2 = np.sum(((y - y_fit) / y_err) ** 2) / nu if nu > 0 else np.inf

            if abs(chi2 - 1.0) < abs(best_chi2 - 1.0):
                best_chi2 = chi2
                best_params = params
        except Exception:
            continue

    return best_params


def fit_new(func, x, y, y_err, params_init, param_names, constr=None, silent=False, optimize=True):
    if optimize:
        params_init = optimize_init_params(func, x, y, y_err, params_init, constr)

    if constr is None:
        params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, maxfev=50000)
    else:
        params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, bounds=constr, maxfev=50000)

    param_sigmas = [np.sqrt(covariance[i][i]) for i in range(len(params))]
    table = [[f"{params[i]:.5f} ± {param_sigmas[i]:.5f}" for i in range(len(params))]]

    nu = len(y) - len(param_names)
    y_fit = func(x, *params)
    chi_2 = np.sum(((y - y_fit) / y_err) ** 2) / nu if nu > 0 else np.inf

    if not silent:
        print(tabulate(table, param_names, tablefmt="fancy_grid"))
        print(f"$\\chi_v^2$ = {chi_2:.2f}")

    return params, covariance, param_sigmas, chi_2


def _print_fit_summary(dis_fit_result):
    print("\n\n", "-" * 25)
    print(f"Best-fit DIS parameters for {dis_fit_result['model_display_name']}:")
    for name, value, error in zip(
        dis_fit_result["parameter_names"],
        dis_fit_result["par_quad"],
        dis_fit_result["par_err_quad"],
    ):
        print(f"{name}:  {value:.4e} ± {error:.4e}")

    print("Covariance matrix:")
    for row in dis_fit_result["cov_quad"]:
        print(" ".join(f"{val:6.2e}" for val in row))

    print("\nCorrelation matrix:")
    for row in dis_fit_result["corr_quad"]:
        print(" ".join(f"{val:6.2e}" for val in row))
    print("-" * 25, "\n\n")


def _fit_single_dis_model(model_key, indep_data, dis_df, x_dense, q2_dense):
    config = get_dis_fit_model_config(model_key)

    print(f"[get_dis_fit] Trying DIS model '{model_key}' ({config['display_name']})")
    params, covariance, param_sigmas, chi2_quad = fit_new(
        config["func"],
        indep_data,
        dis_df["G1F1"],
        dis_df["G1F1.err"],
        config["init"],
        config["param_names"],
        constr=config["bounds"],
    )

    corr_quad = _covariance_to_correlation(covariance)
    beta_val = float(params[config["beta_index"]])
    fit_vals = config["func"]([x_dense, q2_dense], *params)
    residuals = (
        dis_df["G1F1"] - config["func"]([dis_df["X"], dis_df["Q2"]], *params)
    ) / dis_df["G1F1.err"]

    result = {
        "model_key": model_key,
        "model_display_name": config["display_name"],
        "curve_label": config["curve_label"],
        "parameter_names": list(config["param_names"]),
        "partials": list(config["partials"]),
        "beta_index": config["beta_index"],
        "comparison_color": config["comparison_color"],
        "par_quad": params,
        "cov_quad": covariance,
        "corr_quad": corr_quad,
        "par_err_quad": param_sigmas,
        "chi2_quad": chi2_quad,
        "beta_val": beta_val,
        "fit_vals": fit_vals,
        "residuals": residuals,
    }
    _print_fit_summary(result)
    return result


def _plot_active_dis_fit(dis_fit_result, dis_df, x_dense, q2_dense, pdf):
    with open(src_path("config.json"), "r") as handle:
        config = json.load(handle)

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})

    axs[0].errorbar(
        dis_df["X"],
        dis_df["G1F1"],
        yerr=dis_df["G1F1.err"],
        fmt=config["marker"]["type"],
        color=config["colors"]["scatter"],
        label="Data",
        markersize=config["marker"]["size"],
        capsize=config["error_bar"]["cap_size"],
        capthick=config["error_bar"]["cap_thick"],
        linewidth=config["error_bar"]["line_width"],
        ecolor=config["colors"]["error_bar"],
    )
    axs[0].plot(
        x_dense,
        dis_fit_result["fit_vals"],
        "r-",
        label=(
            f"{dis_fit_result['model_display_name']} Fit "
            f"($\\chi^2_{{red}}$ = {dis_fit_result['chi2_quad']:.2f})"
        ),
        linewidth=config["error_bar"]["line_width"],
    )

    axs[0].set_xlabel("x", fontsize=config["font_sizes"]["x_axis"])
    axs[0].set_ylabel("$g_1F_1$", fontsize=config["font_sizes"]["y_axis"])
    axs[0].set_ylim(-0.04, 0.04)
    axs[0].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

    if config["grid"]["enabled"]:
        axs[0].grid(
            True,
            linestyle=config["grid"]["line_style"],
            linewidth=config["grid"]["line_width"],
            alpha=config["grid"]["alpha"],
            color=config["colors"]["grid"],
        )

    axs[1].scatter(
        dis_df["X"],
        dis_fit_result["residuals"],
        color=config["colors"]["scatter"],
        s=config["marker"]["size"] * 2,
    )
    axs[1].axhline(y=0, color=config["colors"]["error_band"], linestyle="-", alpha=0.5)
    axs[1].set_xlabel("x", fontsize=config["font_sizes"]["x_axis"])
    axs[1].set_ylabel("Residuals ($\\sigma$)", fontsize=config["font_sizes"]["y_axis"])

    if config["grid"]["enabled"]:
        axs[1].grid(
            True,
            linestyle=config["grid"]["line_style"],
            linewidth=config["grid"]["line_width"],
            alpha=config["grid"]["alpha"],
            color=config["colors"]["grid"],
        )

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_dis_fit_comparison(fit_results, dis_df, x_dense, q2_dense, pdf):
    with open(src_path("config.json"), "r") as handle:
        config = json.load(handle)

    ordered_results = sorted(fit_results, key=lambda item: item["chi2_quad"])
    best_model_key = ordered_results[0]["model_key"]

    fig, axs = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={"height_ratios": [3, 1]})

    axs[0].errorbar(
        dis_df["X"],
        dis_df["G1F1"],
        yerr=dis_df["G1F1.err"],
        fmt=config["marker"]["type"],
        color=config["colors"]["scatter"],
        label="Data",
        markersize=config["marker"]["size"],
        capsize=config["error_bar"]["cap_size"],
        capthick=config["error_bar"]["cap_thick"],
        linewidth=config["error_bar"]["line_width"],
        ecolor=config["colors"]["error_bar"],
    )

    for result in ordered_results:
        line_width = 2.5 if result["model_key"] == best_model_key else 1.5
        axs[0].plot(
            x_dense,
            result["fit_vals"],
            color=result["comparison_color"],
            linewidth=line_width,
            label=f"{result['model_key']}: $\\chi^2_{{red}}$={result['chi2_quad']:.2f}",
        )

    axs[0].set_xlabel("x", fontsize=config["font_sizes"]["x_axis"])
    axs[0].set_ylabel("$g_1^{^{3}He}/F_1^{^{3}He}$", fontsize=config["font_sizes"]["y_axis"])
    axs[0].set_ylim(-0.04, 0.04)
    axs[0].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

    if config["grid"]["enabled"]:
        axs[0].grid(
            True,
            linestyle=config["grid"]["line_style"],
            linewidth=config["grid"]["line_width"],
            alpha=config["grid"]["alpha"],
            color=config["colors"]["grid"],
        )

    model_labels = [result["model_key"] for result in ordered_results]
    chi2_values = [result["chi2_quad"] for result in ordered_results]
    bar_colors = [result["comparison_color"] for result in ordered_results]
    axs[1].bar(model_labels, chi2_values, color=bar_colors)
    axs[1].set_ylabel("$\\chi^2_{red}$", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_xlabel("DIS model", fontsize=config["font_sizes"]["x_axis"])

    if config["grid"]["enabled"]:
        axs[1].grid(
            True,
            axis="y",
            linestyle=config["grid"]["line_style"],
            linewidth=config["grid"]["line_width"],
            alpha=config["grid"]["alpha"],
            color=config["colors"]["grid"],
        )

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def get_dis_fit(indep_data, dis_df, q2_interp, x_dense, q2_dense, pdf, dataset_tag="legacy", dis_fit_model="fullx"):
    requested_model_key = normalize_dis_fit_model(dis_fit_model)
    candidate_model_keys = (
        get_dis_fit_model_keys() if requested_model_key == "all" else [requested_model_key]
    )

    fit_results = []
    failed_results = []
    for model_key in candidate_model_keys:
        try:
            fit_results.append(
                _fit_single_dis_model(model_key, indep_data, dis_df, x_dense, q2_dense)
            )
        except Exception as exc:
            display_name = get_dis_fit_model_config(model_key)["display_name"]
            failed_results.append(
                {
                    "model_key": model_key,
                    "model_display_name": display_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[get_dis_fit] DIS model '{model_key}' failed with "
                f"{type(exc).__name__}: {exc}"
            )
            if requested_model_key != "all":
                raise

    if not fit_results:
        raise RuntimeError("No DIS fit model converged successfully.")

    if requested_model_key == "all":
        selected_result = min(fit_results, key=lambda item: item["chi2_quad"])
        print(
            "[get_dis_fit] Selected best reduced-chi2 DIS model for downstream stages: "
            f"{selected_result['model_key']} "
            f"($\\chi^2_{{red}}$={selected_result['chi2_quad']:.2f})"
        )
    else:
        selected_result = fit_results[0]

    _plot_active_dis_fit(selected_result, dis_df, x_dense, q2_dense, pdf)
    if len(fit_results) > 1:
        _plot_dis_fit_comparison(fit_results, dis_df, x_dense, q2_dense, pdf)
        _save_dis_fit_comparison(dataset_tag, requested_model_key, fit_results, failed_results)

    _save_dis_fit_summary(dataset_tag, dis_df, selected_result, requested_model_key)
    selected_result["comparison_results"] = fit_results
    selected_result["failed_comparison_results"] = failed_results
    selected_result["requested_model_key"] = requested_model_key
    return selected_result
