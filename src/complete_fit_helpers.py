#! /usr/bin/python

import numpy as np

from dis_fit_models import evaluate_dis_fit
from functions import (
    breit_wigner_bump,
    breit_wigner_res,
    damping_function,
    k_gamma_mass_loop,
    k_new_new,
    quad_nucl_curve_gamma,
    quad_nucl_curve_k,
    quad_nucl_curve_mass,
    x_to_W,
)


def _safe_transition_eval(best_fit_results, parameter_key, q2):
    try:
        return best_fit_results[parameter_key]["eval_func"](q2)
    except Exception:
        return np.nan, np.nan


def evaluate_complete_fit_from_x(
    q2,
    x_values,
    dis_fit_params,
    dis_transition_fit,
    k_nucl_par,
    k_nucl_err,
    gamma_nucl_par,
    gamma_nucl_err,
    mass_nucl_par,
    mass_nucl_err,
    k_P_vals,
    gamma_P_vals,
    mass_P_vals,
    *,
    w_values=None,
    quad_nucl_curve_k_func=quad_nucl_curve_k,
):
    x_array = np.asarray(x_values, dtype=np.double)
    q2_value = float(q2)
    q2_array = np.full_like(x_array, q2_value, dtype=np.double)
    if w_values is None:
        w_array = x_to_W(x_array, q2_array)
    else:
        w_array = np.asarray(w_values, dtype=np.double)

    best_fit_results = dis_transition_fit[0]
    w_dis_transition, w_dis_transition_err = _safe_transition_eval(
        best_fit_results,
        "w_dis_transition",
        q2_value,
    )
    damping_dis_width, damping_dis_width_err = _safe_transition_eval(
        best_fit_results,
        "damping_dis_width",
        q2_value,
    )

    k_fit_params = [k_nucl_par]
    gamma_fit_params = [gamma_nucl_par]
    mass_fit_params = [mass_nucl_par]
    fit_funcs_k = [quad_nucl_curve_k_func]
    fit_funcs_gamma = [quad_nucl_curve_gamma]
    fit_funcs_mass = [quad_nucl_curve_mass]

    (
        _ii,
        _jj,
        _ijj,
        k_value,
        k_err,
        gamma_value,
        gamma_err,
        mass_value,
        mass_err,
    ) = next(
        k_gamma_mass_loop(
            q2_value,
            w_array,
            k_fit_params,
            gamma_fit_params,
            mass_fit_params,
            fit_funcs_k,
            fit_funcs_gamma,
            fit_funcs_mass,
            k_P_vals,
            gamma_P_vals,
            mass_P_vals,
            k_nucl_err,
            gamma_nucl_err,
            mass_nucl_err,
        )
    )

    y_bw = breit_wigner_res(w_array, mass_value, k_value, gamma_value)
    y_dis = evaluate_dis_fit(dis_fit_params, x_array, q2_array)
    y_bw_bump = breit_wigner_bump(w_array, 1.55, k_new_new(q2_value), 0.25)
    y_transition = y_bw_bump + (y_bw - y_dis)
    damping_dis = damping_function(w_array, w_dis_transition, damping_dis_width)
    y_complete = np.nan_to_num(y_transition * damping_dis + y_dis, nan=0.0)

    return {
        "Q2": q2_array,
        "X": x_array,
        "W": w_array,
        "y_dis": y_dis,
        "y_bw": y_bw,
        "y_bw_bump": y_bw_bump,
        "y_transition": y_transition,
        "damping_dis": damping_dis,
        "y_complete": y_complete,
        "w_dis_transition": w_dis_transition,
        "w_dis_transition_err": w_dis_transition_err,
        "damping_dis_width": damping_dis_width,
        "damping_dis_width_err": damping_dis_width_err,
        "k": k_value,
        "k_err": k_err,
        "gamma": gamma_value,
        "gamma_err": gamma_err,
        "mass": mass_value,
        "mass_err": mass_err,
    }
