#! /usr/bin/python

import json
import os
import time
from datetime import datetime

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
from utility import prefix_generated_output_name, project_path, safe_tabulate as tabulate, src_path


def _build_artifact_path(filename, dataset_tag):
    dated_filename = prefix_generated_output_name(filename)
    if dataset_tag == "legacy":
        return project_path("fit_data", dated_filename)

    tagged_dir = project_path("fit_data", dataset_tag)
    os.makedirs(tagged_dir, exist_ok=True)
    return os.path.join(tagged_dir, dated_filename)


def _save_dis_fit_summary(dataset_tag, summary_payload):
    summary_path = _build_artifact_path("dis_fit_summary.json", dataset_tag)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)
    print(f"[get_dis_fit] Saved DIS fit summary to {summary_path}")


def _save_dis_fit_comparison(dataset_tag, requested_model_key, fit_results, failed_results, source_group=None):
    comparison_json_path = _build_artifact_path("dis_fit_model_comparison.json", dataset_tag)
    comparison_csv_path = _build_artifact_path("dis_fit_model_comparison.csv", dataset_tag)

    successful_payload = []
    for rank, result in enumerate(sorted(fit_results, key=_dis_fit_ranking_key), start=1):
        successful_payload.append(
            {
                "rank": rank,
                "model_key": result["model_key"],
                "model_display_name": result["model_display_name"],
                "curve_label": result["curve_label"],
                "chi2_quad": float(result["chi2_quad"]),
                "chi2_distance_from_unity": float(result["chi2_distance_from_unity"]),
                "beta_val": float(result["beta_val"]),
                "parameter_names": list(result["parameter_names"]),
                "par_quad": np.asarray(result["par_quad"], dtype=float).tolist(),
                "par_err_quad": np.asarray(result["par_err_quad"], dtype=float).tolist(),
            }
        )

    payload = {
        "dataset_tag": dataset_tag,
        "source_group": source_group,
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
                "chi2_distance_from_unity": item["chi2_distance_from_unity"],
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
                "chi2_distance_from_unity": None,
                "beta_val": None,
                "parameter_names": "",
                "parameters": "",
                "parameter_errors": item["error"],
            }
        )

    pd.DataFrame(csv_rows).to_csv(comparison_csv_path, index=False)
    print(f"[get_dis_fit] Saved DIS fit comparison to {comparison_json_path}")
    print(f"[get_dis_fit] Saved DIS fit comparison table to {comparison_csv_path}")


def _save_dis_fit_timing_summary(dataset_tag, timing_rows):
    timing_path = _build_artifact_path("dis_fit_timing_summary.csv", dataset_tag)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
    print(f"[get_dis_fit] Saved DIS fit timing summary to {timing_path}")


def _covariance_to_correlation(cov_matrix, parameter_names):
    covariance = np.asarray(cov_matrix, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square before conversion to a correlation matrix.")
    if covariance.shape[0] != len(parameter_names):
        raise ValueError(
            "Covariance matrix dimension does not match the number of fit parameters."
        )

    n_params = covariance.shape[0]
    correlation = np.full((n_params, n_params), np.nan, dtype=float)
    variances = np.diag(covariance)
    std_devs = np.sqrt(np.where(np.isfinite(variances) & (variances >= 0.0), variances, np.nan))

    for row_index in range(n_params):
        for column_index in range(n_params):
            denominator = std_devs[row_index] * std_devs[column_index]
            if np.isfinite(denominator) and denominator > 0.0:
                correlation[row_index, column_index] = covariance[row_index, column_index] / denominator

    for row_index in range(n_params):
        for column_index in range(row_index + 1, n_params):
            left_value = correlation[row_index, column_index]
            right_value = correlation[column_index, row_index]
            if np.isfinite(left_value) and np.isfinite(right_value):
                sym_value = 0.5 * (left_value + right_value)
            elif np.isfinite(left_value):
                sym_value = left_value
            elif np.isfinite(right_value):
                sym_value = right_value
            else:
                sym_value = np.nan
            correlation[row_index, column_index] = sym_value
            correlation[column_index, row_index] = sym_value

    for diag_index in range(n_params):
        if np.isfinite(std_devs[diag_index]) and std_devs[diag_index] > 0.0:
            correlation[diag_index, diag_index] = 1.0
        else:
            correlation[diag_index, diag_index] = np.nan

    correlation[np.isinf(correlation)] = np.nan
    return correlation


def _dis_fit_ranking_key(result):
    return (float(result["chi2_distance_from_unity"]), float(result["chi2_quad"]))


def _matrix_warnings(dis_fit_result):
    warnings = []
    undefined_uncertainties = [
        name
        for name, error in zip(dis_fit_result["parameter_names"], dis_fit_result["par_err_quad"])
        if not np.isfinite(error)
    ]
    if undefined_uncertainties:
        warnings.append(
            "Undefined parameter uncertainties for: " + ", ".join(undefined_uncertainties)
        )

    corr_matrix = np.asarray(dis_fit_result["corr_quad"], dtype=float)
    off_diagonal_mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    if np.any(~np.isfinite(corr_matrix[off_diagonal_mask])):
        warnings.append("One or more off-diagonal correlations are undefined (NaN).")
    return warnings


def _serialize_dis_fit_result(dis_fit_result, *, selected_model_key):
    return {
        "model_key": dis_fit_result["model_key"],
        "model_display_name": dis_fit_result["model_display_name"],
        "curve_label": dis_fit_result["curve_label"],
        "functional_form_text": dis_fit_result["functional_form_text"],
        "functional_form_latex": dis_fit_result["functional_form_latex"],
        "parameter_names": list(dis_fit_result["parameter_names"]),
        "par_quad": np.asarray(dis_fit_result["par_quad"], dtype=float).tolist(),
        "par_err_quad": np.asarray(dis_fit_result["par_err_quad"], dtype=float).tolist(),
        "cov_quad": np.asarray(dis_fit_result["cov_quad"], dtype=float).tolist(),
        "corr_quad": np.asarray(dis_fit_result["corr_quad"], dtype=float).tolist(),
        "chi2_total": float(dis_fit_result["chi2_total"]),
        "chi2_quad": float(dis_fit_result["chi2_quad"]),
        "ndf": int(dis_fit_result["ndf"]),
        "chi2_distance_from_unity": float(dis_fit_result["chi2_distance_from_unity"]),
        "beta_val": float(dis_fit_result["beta_val"]),
        "runtime_sec": float(dis_fit_result.get("runtime_sec", np.nan)),
        "selected_downstream": bool(dis_fit_result["model_key"] == selected_model_key),
        "warnings": _matrix_warnings(dis_fit_result),
    }


def _build_dis_fit_summary_payload(
    dataset_tag,
    dis_df,
    selected_result,
    fit_results,
    failed_results,
    requested_model_key,
    *,
    source_group=None,
    run_metadata=None,
):
    payload = {
        "dataset_tag": dataset_tag,
        "source_group": source_group,
        "requested_model_key": requested_model_key,
        "selected_model_key": selected_result["model_key"],
        "selected_downstream_model": selected_result["model_key"],
        "model_display_name": selected_result["model_display_name"],
        "curve_label": selected_result["curve_label"],
        "functional_form_text": selected_result["functional_form_text"],
        "functional_form_latex": selected_result["functional_form_latex"],
        "n_points": int(len(dis_df)),
        "x_range": [
            float(np.min(dis_df["X"])),
            float(np.max(dis_df["X"])),
        ],
        "q2_range": [
            float(np.min(dis_df["Q2"])),
            float(np.max(dis_df["Q2"])),
        ],
        "parameter_names": list(selected_result["parameter_names"]),
        "par_quad": np.asarray(selected_result["par_quad"], dtype=float).tolist(),
        "par_err_quad": np.asarray(selected_result["par_err_quad"], dtype=float).tolist(),
        "cov_quad": np.asarray(selected_result["cov_quad"], dtype=float).tolist(),
        "corr_quad": np.asarray(selected_result["corr_quad"], dtype=float).tolist(),
        "chi2_total": float(selected_result["chi2_total"]),
        "chi2_quad": float(selected_result["chi2_quad"]),
        "ndf": int(selected_result["ndf"]),
        "chi2_distance_from_unity": float(selected_result["chi2_distance_from_unity"]),
        "beta_val": float(selected_result["beta_val"]),
        "residual_summary": {
            "mean": float(np.mean(selected_result["residuals"])),
            "std": float(np.std(selected_result["residuals"])),
            "min": float(np.min(selected_result["residuals"])),
            "max": float(np.max(selected_result["residuals"])),
        },
        "warnings": _matrix_warnings(selected_result),
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_metadata": run_metadata or {},
        "models": [
            _serialize_dis_fit_result(result, selected_model_key=selected_result["model_key"])
            for result in _ordered_fit_results(fit_results)
        ],
        "failed_results": failed_results,
    }
    return payload


def _matrix_to_markdown(matrix, labels):
    header = "| | " + " | ".join(labels) + " |"
    separator = "|" + "---|" * (len(labels) + 1)
    rows = [header, separator]
    for row_label, row_values in zip(labels, np.asarray(matrix, dtype=float)):
        formatted_values = [
            "nan" if not np.isfinite(value) else f"{value:.6g}"
            for value in row_values
        ]
        rows.append(f"| {row_label} | " + " | ".join(formatted_values) + " |")
    return "\n".join(rows)


def _save_dis_fit_report_files(
    dataset_tag,
    summary_payload,
):
    report_path = _build_artifact_path("dis_fit_report.md", dataset_tag)
    parameters_path = _build_artifact_path("dis_fit_parameters.csv", dataset_tag)
    covariance_path = _build_artifact_path("dis_fit_covariance.csv", dataset_tag)
    correlations_path = _build_artifact_path("dis_fit_correlations.csv", dataset_tag)

    parameter_rows = []
    covariance_rows = []
    correlation_rows = []
    report_lines = [
        "# DIS Fit Report",
        "",
        f"- Generated at: {summary_payload['generated_at']}",
        f"- Requested DIS_FIT_MODEL: {summary_payload['requested_model_key']}",
        f"- Selected downstream model: {summary_payload['selected_model_key']}",
        f"- Dataset tag: {summary_payload['dataset_tag']}",
        f"- Source group: {summary_payload.get('source_group') or 'none'}",
        "",
        "## Run Metadata",
        "",
    ]

    for key, value in (summary_payload.get("run_metadata") or {}).items():
        if isinstance(value, (list, tuple)):
            rendered_value = ", ".join(str(item) for item in value)
        else:
            rendered_value = value
        report_lines.append(f"- {key}: {rendered_value}")

    for model_payload in summary_payload["models"]:
        report_lines.extend(
            [
                "",
                f"## Model `{model_payload['model_key']}`",
                "",
                f"- Display name: {model_payload['model_display_name']}",
                f"- Selected downstream: {model_payload['selected_downstream']}",
                f"- Total chi2: {model_payload['chi2_total']:.6g}",
                f"- NDF: {model_payload['ndf']}",
                f"- Reduced chi2: {model_payload['chi2_quad']:.6g}",
                f"- |chi2_red - 1|: {model_payload['chi2_distance_from_unity']:.6g}",
                "",
                "### Functional Form",
                "",
                f"- Text: `{model_payload['functional_form_text']}`",
                f"- LaTeX: `${model_payload['functional_form_latex']}$",
                "",
                "### Parameters",
                "",
                "| Parameter | Value | Uncertainty |",
                "|---|---:|---:|",
            ]
        )
        for parameter_index, (parameter_name, parameter_value, parameter_error) in enumerate(
            zip(
                model_payload["parameter_names"],
                model_payload["par_quad"],
                model_payload["par_err_quad"],
            )
        ):
            report_lines.append(
                f"| {parameter_name} | {parameter_value:.6g} | "
                f"{'nan' if not np.isfinite(parameter_error) else f'{parameter_error:.6g}'} |"
            )
            parameter_rows.append(
                {
                    "model_key": model_payload["model_key"],
                    "selected_downstream": model_payload["selected_downstream"],
                    "parameter_index": parameter_index,
                    "parameter_name": parameter_name,
                    "value": parameter_value,
                    "uncertainty": parameter_error,
                    "chi2_total": model_payload["chi2_total"],
                    "chi2_red": model_payload["chi2_quad"],
                    "ndf": model_payload["ndf"],
                    "functional_form_text": model_payload["functional_form_text"],
                }
            )

        report_lines.extend(
            [
                "",
                "### Covariance Matrix",
                "",
                _matrix_to_markdown(model_payload["cov_quad"], model_payload["parameter_names"]),
                "",
                "### Correlation Matrix",
                "",
                _matrix_to_markdown(model_payload["corr_quad"], model_payload["parameter_names"]),
            ]
        )

        if model_payload["warnings"]:
            report_lines.extend(["", "### Warnings", ""])
            for warning in model_payload["warnings"]:
                report_lines.append(f"- {warning}")

        parameter_names = model_payload["parameter_names"]
        covariance_matrix = np.asarray(model_payload["cov_quad"], dtype=float)
        correlation_matrix = np.asarray(model_payload["corr_quad"], dtype=float)
        for row_index, row_name in enumerate(parameter_names):
            for column_index, column_name in enumerate(parameter_names):
                covariance_rows.append(
                    {
                        "model_key": model_payload["model_key"],
                        "row_index": row_index,
                        "row_parameter": row_name,
                        "column_index": column_index,
                        "column_parameter": column_name,
                        "value": covariance_matrix[row_index, column_index],
                    }
                )
                correlation_rows.append(
                    {
                        "model_key": model_payload["model_key"],
                        "row_index": row_index,
                        "row_parameter": row_name,
                        "column_index": column_index,
                        "column_parameter": column_name,
                        "value": correlation_matrix[row_index, column_index],
                    }
                )

    if summary_payload["failed_results"]:
        report_lines.extend(["", "## Failed Models", ""])
        for failure in summary_payload["failed_results"]:
            report_lines.append(
                f"- `{failure['model_key']}` ({failure['model_display_name']}): {failure['error']}"
            )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    pd.DataFrame(parameter_rows).to_csv(parameters_path, index=False)
    pd.DataFrame(covariance_rows).to_csv(covariance_path, index=False)
    pd.DataFrame(correlation_rows).to_csv(correlations_path, index=False)
    print(f"[get_dis_fit] Saved DIS fit report to {report_path}")
    return report_path, parameters_path, covariance_path, correlations_path


ALL_FIT_LINE_STYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
]


def _ordered_fit_results(fit_results):
    return sorted(fit_results, key=_dis_fit_ranking_key)


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
    print(f"|chi2_red - 1| = {dis_fit_result['chi2_distance_from_unity']:.4f}")
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

    corr_quad = _covariance_to_correlation(covariance, config["param_names"])
    beta_val = float(params[config["beta_index"]])
    fit_vals = config["func"]([x_dense, q2_dense], *params)
    residuals = (
        dis_df["G1F1"] - config["func"]([dis_df["X"], dis_df["Q2"]], *params)
    ) / dis_df["G1F1.err"]
    ndf = max(len(dis_df) - len(config["param_names"]), 0)
    chi2_total = float(chi2_quad * ndf) if ndf > 0 else np.nan

    result = {
        "model_key": model_key,
        "model_display_name": config["display_name"],
        "curve_label": config["curve_label"],
        "functional_form_text": config["functional_form_text"],
        "functional_form_latex": config["functional_form_latex"],
        "parameter_names": list(config["param_names"]),
        "partials": list(config["partials"]),
        "beta_index": config["beta_index"],
        "comparison_color": config["comparison_color"],
        "par_quad": params,
        "cov_quad": covariance,
        "corr_quad": corr_quad,
        "par_err_quad": param_sigmas,
        "chi2_total": chi2_total,
        "chi2_quad": chi2_quad,
        "chi2_distance_from_unity": abs(float(chi2_quad) - 1.0),
        "beta_val": beta_val,
        "fit_vals": fit_vals,
        "residuals": residuals,
        "ndf": ndf,
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

    ordered_results = _ordered_fit_results(fit_results)
    best_model_key = ordered_results[0]["model_key"]
    sort_idx = np.argsort(np.asarray(dis_df["X"], dtype=float))
    x_sorted = np.asarray(dis_df["X"], dtype=float)[sort_idx]

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
        style_index = ordered_results.index(result) % len(ALL_FIT_LINE_STYLES)
        line_width = 2.3 if result["model_key"] == best_model_key else 1.6
        axs[0].plot(
            x_dense,
            result["fit_vals"],
            color=result["comparison_color"],
            linestyle=ALL_FIT_LINE_STYLES[style_index],
            linewidth=line_width,
            label=(
                f"{result['model_key']}: $\\chi^2_{{red}}$={result['chi2_quad']:.2f}"
            ),
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

    for result in ordered_results:
        style_index = ordered_results.index(result) % len(ALL_FIT_LINE_STYLES)
        residuals_sorted = np.asarray(result["residuals"], dtype=float)[sort_idx]
        axs[1].plot(
            x_sorted,
            residuals_sorted,
            color=result["comparison_color"],
            linestyle=ALL_FIT_LINE_STYLES[style_index],
            linewidth=1.5,
            label=result["model_key"],
        )

    axs[1].axhline(y=0.0, color=config["colors"]["error_band"], linestyle="-", alpha=0.5)
    axs[1].set_ylabel("Residuals ($\\sigma$)", fontsize=config["font_sizes"]["y_axis"])
    axs[1].set_xlabel("x", fontsize=config["font_sizes"]["x_axis"])
    axs[1].legend(fontsize=config["font_sizes"]["legend"], frameon=config["legend"]["frame_on"])

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


def fit_dis_model_suite(indep_data, dis_df, x_dense, q2_dense, dis_fit_model="fullx", source_group=None):
    requested_model_key = normalize_dis_fit_model(dis_fit_model)
    candidate_model_keys = (
        get_dis_fit_model_keys() if requested_model_key == "all" else [requested_model_key]
    )
    active_source_group = (
        source_group
        or dis_df.attrs.get("source_group_metadata", {}).get("source_group")
        or ""
    )
    n_points = int(len(dis_df))

    fit_results = []
    failed_results = []
    timing_rows = []
    for model_key in candidate_model_keys:
        start_time = time.perf_counter()
        try:
            fit_result = _fit_single_dis_model(model_key, indep_data, dis_df, x_dense, q2_dense)
            runtime_sec = time.perf_counter() - start_time
            ndf = max(n_points - len(fit_result["parameter_names"]), 0)
            fit_result["ndf"] = ndf
            fit_result["runtime_sec"] = runtime_sec
            fit_results.append(fit_result)
            timing_rows.append(
                {
                    "source_group": active_source_group,
                    "model_key": model_key,
                    "n_points": n_points,
                    "fit_status": "success",
                    "chi2": float(fit_result["chi2_quad"] * ndf) if ndf > 0 else np.nan,
                    "ndf": ndf,
                    "chi2_red": float(fit_result["chi2_quad"]),
                    "runtime_sec": runtime_sec,
                    "selected_by_all": False,
                }
            )
        except Exception as exc:
            runtime_sec = time.perf_counter() - start_time
            display_name = get_dis_fit_model_config(model_key)["display_name"]
            failed_results.append(
                {
                    "model_key": model_key,
                    "model_display_name": display_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            timing_rows.append(
                {
                    "source_group": active_source_group,
                    "model_key": model_key,
                    "n_points": n_points,
                    "fit_status": "failed",
                    "chi2": np.nan,
                    "ndf": np.nan,
                    "chi2_red": np.nan,
                    "runtime_sec": runtime_sec,
                    "selected_by_all": False,
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
        selected_result = min(fit_results, key=_dis_fit_ranking_key)
        print(
            "[get_dis_fit] Selected closest-to-unity reduced-chi2 DIS model for downstream stages: "
            f"{selected_result['model_key']} "
            f"($\\chi^2_{{red}}$={selected_result['chi2_quad']:.2f}, "
            f"|$\\chi^2_{{red}}$-1|={selected_result['chi2_distance_from_unity']:.2f})"
        )
    else:
        selected_result = fit_results[0]

    for row in timing_rows:
        if row["fit_status"] != "success":
            continue
        row["selected_by_all"] = (
            requested_model_key == "all"
            and row["model_key"] == selected_result["model_key"]
        )

    selected_result["comparison_results"] = fit_results
    selected_result["failed_comparison_results"] = failed_results
    selected_result["requested_model_key"] = requested_model_key
    selected_result["timing_rows"] = timing_rows
    selected_result["source_group"] = active_source_group
    return {
        "selected_result": selected_result,
        "fit_results": fit_results,
        "failed_results": failed_results,
        "timing_rows": timing_rows,
        "requested_model_key": requested_model_key,
        "source_group": active_source_group,
    }


def get_dis_fit(
    indep_data,
    dis_df,
    q2_interp,
    x_dense,
    q2_dense,
    pdf,
    dataset_tag="legacy",
    dis_fit_model="fullx",
    source_group=None,
    run_metadata=None,
):
    suite = fit_dis_model_suite(
        indep_data,
        dis_df,
        x_dense,
        q2_dense,
        dis_fit_model=dis_fit_model,
        source_group=source_group,
    )
    selected_result = suite["selected_result"]
    fit_results = suite["fit_results"]
    failed_results = suite["failed_results"]
    requested_model_key = suite["requested_model_key"]
    active_source_group = suite["source_group"]

    if requested_model_key == "all" and len(fit_results) > 1:
        _plot_dis_fit_comparison(fit_results, dis_df, x_dense, q2_dense, pdf)
        _save_dis_fit_comparison(
            dataset_tag,
            requested_model_key,
            fit_results,
            failed_results,
            source_group=active_source_group,
        )
    else:
        _plot_active_dis_fit(selected_result, dis_df, x_dense, q2_dense, pdf)

    summary_payload = _build_dis_fit_summary_payload(
        dataset_tag,
        dis_df,
        selected_result,
        fit_results,
        failed_results,
        requested_model_key,
        source_group=active_source_group,
        run_metadata=run_metadata,
    )
    _save_dis_fit_summary(dataset_tag, summary_payload)
    _save_dis_fit_report_files(dataset_tag, summary_payload)
    _save_dis_fit_timing_summary(dataset_tag, suite["timing_rows"])
    return selected_result
