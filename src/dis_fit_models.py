#! /usr/bin/python

import numpy as np

from functions import (
    g1f1_cubic_DIS,
    g1f1_cubic_alpha_DIS,
    g1f1_quad_alpha_DIS,
    g1f1_quad2_DIS,
    g1f1_quad_DIS,
    g1f1_quad_fullx_DIS,
    partial_a2,
    partial_a3,
    partial_a_fullx,
    partial_b2,
    partial_b3,
    partial_b_fullx,
    partial_beta2,
    partial_beta3,
    partial_beta4,
    partial_beta_fullx,
    partial_c2,
    partial_c3,
    partial_c4,
    partial_c_fullx,
    partial_d3,
    partial_d_fullx,
    partial_sigma_fullx,
    partial_x0,
    partial_x0_fullx,
    partial_y0,
    partial_alpha_fullx,
)


def partial_alpha_new(x, q2, par):
    alpha, a, b, c, beta = par
    poly = a + b * x + c * x * x
    return np.log(x) * (x ** alpha) * poly * (1.0 + beta / q2)


def partial_a_new(x, q2, par):
    alpha, a, b, c, beta = par
    return (x ** alpha) * (1.0 + beta / q2)


def partial_b_new(x, q2, par):
    alpha, a, b, c, beta = par
    return (x ** (alpha + 1.0)) * (1.0 + beta / q2)


def partial_c_new(x, q2, par):
    alpha, a, b, c, beta = par
    return (x ** (alpha + 2.0)) * (1.0 + beta / q2)


def partial_beta_new(x, q2, par):
    alpha, a, b, c, beta = par
    poly = a + b * x + c * x * x
    return (x ** alpha) * poly / q2


partial_alpha_quad_alpha = partial_alpha_new
partial_a_quad_alpha = partial_a_new
partial_b_quad_alpha = partial_b_new
partial_c_quad_alpha = partial_c_new
partial_beta_quad_alpha = partial_beta_new


def partial_alpha_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    poly = a + b * x + c * x * x + d * x * x * x
    return np.log(x) * (x ** alpha) * poly * (1.0 + beta / q2)


def partial_a_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    return (x ** alpha) * (1.0 + beta / q2)


def partial_b_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    return (x ** (alpha + 1.0)) * (1.0 + beta / q2)


def partial_c_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    return (x ** (alpha + 2.0)) * (1.0 + beta / q2)


def partial_d_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    return (x ** (alpha + 3.0)) * (1.0 + beta / q2)


def partial_beta_cubic_alpha(x, q2, par):
    alpha, a, b, c, d, beta = par
    poly = a + b * x + c * x * x + d * x * x * x
    return (x ** alpha) * poly / q2


DIS_FIT_MODEL_REGISTRY = {
    "fullx": {
        "func": g1f1_quad_fullx_DIS,
        "param_names": ["alpha", "a", "b", "c", "beta", "d", "x0", "sigma"],
        "init": [0.66084205, -0.23606144, -1.25499178, 2.65987975, 0.09666789, 0.0, 0.2, 0.35],
        "bounds": (
            [-np.inf, -np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.1, 0.2],
            [np.inf, np.inf, 0.0, np.inf, np.inf, np.inf, 0.3, 0.5],
        ),
        "partials": [
            partial_alpha_fullx,
            partial_a_fullx,
            partial_b_fullx,
            partial_c_fullx,
            partial_beta_fullx,
            partial_d_fullx,
            partial_x0_fullx,
            partial_sigma_fullx,
        ],
        "beta_index": 4,
        "display_name": "Full-x Quadratic",
        "curve_label": "Full-x DIS Fit",
        "comparison_color": "tab:red",
    },
    "quad_alpha": {
        "func": g1f1_quad_alpha_DIS,
        "param_names": ["alpha", "a", "b", "c", "beta"],
        "init": [0.66084205, -0.23606144, -1.25499178, 2.65987975, 0.09666789],
        "bounds": (
            [-np.inf, -np.inf, -np.inf, 0.0, -np.inf],
            [np.inf, np.inf, 0.0, np.inf, np.inf],
        ),
        "partials": [
            partial_alpha_quad_alpha,
            partial_a_quad_alpha,
            partial_b_quad_alpha,
            partial_c_quad_alpha,
            partial_beta_quad_alpha,
        ],
        "beta_index": 4,
        "display_name": "Power-Law Quadratic",
        "curve_label": "Power-Law Quadratic Alpha DIS Fit",
        "comparison_color": "tab:orange",
    },
    "cubic_alpha": {
        "func": g1f1_cubic_alpha_DIS,
        "param_names": ["alpha", "a", "b", "c", "d", "beta"],
        "init": [0.66084205, -0.23606144, -1.25499178, 2.65987975, -0.22, 0.09666789],
        "bounds": (
            [-np.inf, -np.inf, -np.inf, 0.0, -np.inf, -np.inf],
            [np.inf, np.inf, 0.0, np.inf, np.inf, np.inf],
        ),
        "partials": [
            partial_alpha_cubic_alpha,
            partial_a_cubic_alpha,
            partial_b_cubic_alpha,
            partial_c_cubic_alpha,
            partial_d_cubic_alpha,
            partial_beta_cubic_alpha,
        ],
        "beta_index": 5,
        "display_name": "Power-Law Cubic",
        "curve_label": "Power-Law Cubic Alpha DIS Fit",
        "comparison_color": "tab:brown",
    },
    "quad2": {
        "func": g1f1_quad2_DIS,
        "param_names": ["x0", "y0", "c", "beta"],
        "init": [0.16424, -0.02584, 0.16632, 0.11059],
        "bounds": (
            [0.12, -0.05, 0.10, 0.105],
            [0.20, 0.0, 0.20, 0.115],
        ),
        "partials": [partial_x0, partial_y0, partial_c4, partial_beta4],
        "beta_index": 3,
        "display_name": "Constrained Quadratic",
        "curve_label": "Constrained Quadratic DIS Fit",
        "comparison_color": "tab:green",
    },
    "quad": {
        "func": g1f1_quad_DIS,
        "param_names": ["a", "b", "c", "beta"],
        "init": [-0.03, -0.02, 0.30, 0.10],
        "bounds": None,
        "partials": [partial_a2, partial_b2, partial_c2, partial_beta2],
        "beta_index": 3,
        "display_name": "Quadratic",
        "curve_label": "Quadratic DIS Fit",
        "comparison_color": "tab:blue",
    },
    "cubic": {
        "func": g1f1_cubic_DIS,
        "param_names": ["a", "b", "c", "d", "beta"],
        "init": [-0.03, -0.02, 0.30, -0.22, 0.10],
        "bounds": None,
        "partials": [partial_a3, partial_b3, partial_c3, partial_d3, partial_beta3],
        "beta_index": 4,
        "display_name": "Cubic",
        "curve_label": "Cubic DIS Fit",
        "comparison_color": "tab:purple",
    },
}


DIS_FIT_MODEL_ORDER = ("fullx", "quad_alpha", "cubic_alpha", "quad2", "quad", "cubic")

DIS_FIT_MODEL_ALIASES = {
    "full_x": "fullx",
    "quad_alpha": "quad_alpha",
    "quadnew": "quad_alpha",
    "quad_new": "quad_alpha",
    "cubicalpha": "cubic_alpha",
    "cubic_alpha": "cubic_alpha",
    "cubicnew": "cubic_alpha",
    "cubic_new": "cubic_alpha",
    "new": "quad_alpha",
    "quadratic": "quad",
    "constrained_quadratic": "quad2",
}


def normalize_dis_fit_model(model_key):
    normalized = str(model_key).strip().lower()
    normalized = DIS_FIT_MODEL_ALIASES.get(normalized, normalized)
    if normalized == "all":
        return normalized
    if normalized not in DIS_FIT_MODEL_REGISTRY:
        supported = ", ".join(list(DIS_FIT_MODEL_ORDER) + ["all"])
        raise ValueError(
            f"Unsupported DIS_FIT_MODEL '{model_key}'. Expected one of: {supported}."
        )
    return normalized


def get_dis_fit_model_keys():
    return list(DIS_FIT_MODEL_ORDER)


def get_dis_fit_model_config(model_key):
    normalized = normalize_dis_fit_model(model_key)
    if normalized == "all":
        raise ValueError("DIS_FIT_MODEL='all' does not map to a single model config.")
    return DIS_FIT_MODEL_REGISTRY[normalized]


def evaluate_dis_fit(dis_fit_params, x, q2):
    config = get_dis_fit_model_config(dis_fit_params["model_key"])
    x_array = np.asarray(x, dtype=np.double)
    q2_array = np.asarray(q2, dtype=np.double)
    return config["func"]([x_array, q2_array], *dis_fit_params["par_quad"])


def derive_dis_fit_tag(dataset_tag, dis_fit_model):
    normalized = normalize_dis_fit_model(dis_fit_model)
    if normalized == "fullx":
        return dataset_tag
    if normalized == "all":
        return f"{dataset_tag}_disfit_all"
    return f"{dataset_tag}_disfit_{normalized}"
