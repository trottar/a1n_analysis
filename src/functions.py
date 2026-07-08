#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-04-21 17:27:54 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#

import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.optimize import Bounds
from scipy.optimize import minimize, differential_evolution
from scipy.stats import chi2
import inspect
    
##################################################################################################################################################
# Importing utility functions

from utility import safe_tabulate as tabulate
	
#from utility import custom_map

##################################################################################################################################################

def x_to_W(x, Q2):
  """ convert x to W for constant Q2"""
  Mp = 0.93870319 # average nucleon mass in 3He
  return np.sqrt(Mp**2 + Q2*((1/x)-1))

def W_to_x(W, Q2):
  """ convert W to x for constant Q2"""
  Mp = 0.93870319 # average nucleon mass in 3He
  return Q2/(W**2 + Q2 - Mp**2)

#TODO (DONE): reformulate this function so k_new is the peak height
def breit_wigner_res(w, M, k, gamma):
  """fit for constant Q2"""
  #return k/((w*w - M*M)**2 + M*M*gamma*gamma)
  # RLT (9/23/2024): Updated k_new=k/(M^2*gamma^2)
  gamma = abs(gamma)
  k_new = k*((M**2)*(gamma**2))
  return k_new/((w*w - M*M)**2 + M*M*gamma*gamma)

def breit_wigner_wrapper(M_test):
  """for fitting with constant M"""
  def temp_func(w, k, gamma, M=M_test):
    return breit_wigner_res(w, M, k, gamma)
  return temp_func

# Positive bump
def breit_wigner_bump(w, M, k, gamma):
  """fit for constant Q2"""
  gamma = abs(gamma)
  return k/((w*w - M*M)**2 + M*M*gamma*gamma)

def breit_wigner_bump_wrapper(M_test):
  """for fitting with constant M"""
  def temp_func(w, k, gamma, M=M_test):
    return breit_wigner_bump(w, M, k, gamma)
  return temp_func


def lin_curve(x, a, b):
  """simple line function for fitting"""
  return a*x + b

def quad_curve(x, a, b, c):
  """quadratic fit function"""
  return a + b*x + c*x**2

#HERE
VALID_BW_K_CURVE_MODES = {"fixed_zero", "non_tune", "smooth_zero", "tune"}


def normalize_bw_k_curve_mode(mode):
  normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
  if normalized in {"", "default", "current"}:
    normalized = "non_tune"
  if normalized in {"fixedzero", "zero_fixed"}:
    normalized = "fixed_zero"
  if normalized in {"smoothzero", "zero_smooth", "spline_zero", "pchip_zero", "smooth"}:
    normalized = "smooth_zero"
  if normalized not in VALID_BW_K_CURVE_MODES:
    supported = ", ".join(sorted(VALID_BW_K_CURVE_MODES))
    raise ValueError(f"Unsupported BW k-curve mode '{mode}'. Expected one of: {supported}.")
  return normalized


_SMOOTH_ZERO_TAIL_CONFIG = None


def clear_smooth_zero_tail_config():
  global _SMOOTH_ZERO_TAIL_CONFIG
  _SMOOTH_ZERO_TAIL_CONFIG = None


def set_smooth_zero_tail_config(config):
  global _SMOOTH_ZERO_TAIL_CONFIG
  if config is None:
    _SMOOTH_ZERO_TAIL_CONFIG = None
  else:
    _SMOOTH_ZERO_TAIL_CONFIG = dict(config)


def _get_smooth_zero_tail_config():
  if _SMOOTH_ZERO_TAIL_CONFIG is None:
    return None
  return dict(_SMOOTH_ZERO_TAIL_CONFIG)


def k_curve_non_tune(x, a, b, c, d, f, e, x0=0.1, k=100):
  """Current non-tuned k(Q^2) model used in the BW chain."""
  d = 0
  x = np.asarray(x, dtype=np.float64)  # Ensure array compatibility

  # Define both models
  linear_part = f + e * x
  nonlinear_part = -a * np.exp(-x/b) + (c / x) # chi2 = 4.1 Bounds(lb=[-1e10, -1e10, -1e10, -1e10], ub=[1e10, 1e10, 1e10, 1e10])

  # Sigmoid-based smooth step function
  s = 1 / (1 + np.exp(-k * (x - x0)))  # s ~ 0 for x << x0, s ~ 1 for x >> x0

  return (1 - s) * linear_part + s * nonlinear_part

def theta_func(x, x_points=[1, 3, 5], theta_points=[-np.pi/2, 0, np.pi/2]):
    """
    Interpolates theta for a given x (Q²) value based on three control points.
    
    Parameters:
      x : float or np.array
          Q² value(s) in GeV².
      x_points : list of float
          Q² control points where the theta values are defined.
      theta_points : list of float
          Theta values corresponding to the control points.
    
    Returns:
      float or np.array: Interpolated theta value(s).
    """
    return np.interp(x, x_points, theta_points)

def k_curve_tune(x, a, b, c, d, f, e):
    """
    Tuned piecewise k(Q²) model with a high-Q² sine modulation.
    """
    x = np.asarray(x, dtype=np.float64)
    k_val = np.empty_like(x)
    
    # --- Low Q² Region: x ≤ 0.1 (Linear) ---
    mask_low = x <= 0.1
    k_val[mask_low] = f + e * x[mask_low]
    
    # --- Mid Q² Region: 0.1 < x ≤ 2.75 (Blend between linear and exponential) ---
    mask_mid = (x > 0.1) & (x <= 2.75)
    lin_mid = f + e * x[mask_mid]
    exp_mid = (a + c / x[mask_mid]) * np.exp(-x[mask_mid] / d)
    weight_mid = (x[mask_mid] - 0.1) / (2.75 - 0.1)
    k_val[mask_mid] = (1 - weight_mid) * lin_mid + weight_mid * exp_mid
    
    # --- High Q² Region: x > 2.75 (Exponential fall off with sine modulation) ---
    mask_high = x > 2.75
    exp_high = (a + c / x[mask_high]) * np.exp(-x[mask_high] / d)
    theta = np.interp(x[mask_high], [2.75, 5.0, 9.0], [0, np.pi/2, np.pi])
    sine_var = b * np.sin(theta)
    weight_sine = np.clip((x[mask_high] - 2.75) / (5.0 - 2.75), 0, 1)
    k_val[mask_high] = exp_high + weight_sine * sine_var

    return k_val


def _sample_curve_value_and_slope(curve_func, q2_anchor, curve_args, step=1e-3):
    """Sample a curve value and a numerical slope at the anchor point."""
    q2_anchor = float(q2_anchor)
    q2_minus = max(1e-6, q2_anchor - step)
    q2_plus = q2_anchor + step
    y_anchor = float(np.asarray(curve_func(np.array([q2_anchor], dtype=np.float64), *curve_args), dtype=np.float64)[0])
    y_minus = float(np.asarray(curve_func(np.array([q2_minus], dtype=np.float64), *curve_args), dtype=np.float64)[0])
    y_plus = float(np.asarray(curve_func(np.array([q2_plus], dtype=np.float64), *curve_args), dtype=np.float64)[0])
    slope_anchor = (y_plus - y_minus) / (q2_plus - q2_minus)
    return y_anchor, slope_anchor


def _bridge_core(lam, delta, delta_zero):
    endpoint = np.exp(-lam * delta_zero)
    numerator = np.exp(-lam * delta) - endpoint
    denominator = 1.0 - endpoint
    if denominator == 0.0:
        return np.nan
    return numerator / denominator


def _resolve_fixed_zero_exponential_bridge(
    y_start,
    slope_start,
    y_match,
    q2_exp_start,
    q2_match,
    q2_zero,
    fallback_power=2.0,
):
    """
    Resolve a smooth exponential-like bridge from q2_exp_start to q2_zero.

    The bridge is:
      y(Q²) = y_start * g(Q²)^power

    where:
      g(Q²) = [exp(-lambda * (Q² - q2_exp_start)) - exp(-lambda * L)] / [1 - exp(-lambda * L)]
      L = q2_zero - q2_exp_start

    This guarantees:
      y(q2_exp_start) = y_start
      y(q2_zero) = 0
      y'(q2_zero) = 0      for power > 1

    `lambda` and `power` are chosen to match:
      the original slope at q2_exp_start
      the sampled curve value at q2_match
    """
    delta_match = float(q2_match - q2_exp_start)
    delta_zero = float(q2_zero - q2_exp_start)
    fallback_lambda = max(1.0 / max(delta_zero, 1e-6), 1e-6)

    if (
        not np.isfinite(y_start)
        or not np.isfinite(slope_start)
        or not np.isfinite(y_match)
        or delta_match <= 0.0
        or delta_zero <= 0.0
        or y_start == 0.0
        or slope_start <= 0.0
    ):
        return fallback_lambda, fallback_power

    ratio = abs(y_match / y_start) if y_start != 0.0 else np.nan
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        return fallback_lambda, fallback_power

    def power_from_lambda(lam):
        endpoint = np.exp(-lam * delta_zero)
        denominator = 1.0 - endpoint
        if denominator <= 0.0:
            return np.nan
        return slope_start * denominator / (-y_start * lam)

    def constraint(lam):
        p = power_from_lambda(lam)
        g_match = _bridge_core(lam, delta_match, delta_zero)
        if not np.isfinite(p) or not np.isfinite(g_match) or p <= 1.0 or g_match <= 0.0 or g_match >= 1.0:
            return np.nan
        return np.log(ratio) - p * np.log(g_match)

    lambda_grid = np.logspace(-4, 2, 400)
    bracket = None
    prev_lam = None
    prev_val = None
    for lam in lambda_grid:
        val = constraint(lam)
        if not np.isfinite(val):
            continue
        if prev_val is not None and prev_val * val <= 0.0:
            bracket = (prev_lam, lam)
            break
        prev_lam = lam
        prev_val = val

    if bracket is None:
        return fallback_lambda, fallback_power

    lam_lo, lam_hi = bracket
    val_lo = constraint(lam_lo)
    for _ in range(80):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        val_mid = constraint(lam_mid)
        if not np.isfinite(val_mid):
            lam_hi = lam_mid
            continue
        if val_lo * val_mid <= 0.0:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid
            val_lo = val_mid

    lam = 0.5 * (lam_lo + lam_hi)
    power = power_from_lambda(lam)
    if not np.isfinite(power) or power <= 1.0:
        return fallback_lambda, fallback_power

    return lam, power


def _apply_fixed_zero_high_q2_strategy(
    x,
    curve_func,
    curve_args,
    q2_exp_start=1.0,
    q2_match=2.75,
    q2_zero=4.0,
    q2_blend_start=0.9,
    q2_blend_end=1.05,
):
    """
    Keep the original branch below q2_exp_start, then replace the remainder
    with one smooth exponential-like bridge that approaches zero by q2_zero.
    """
    x = np.asarray(x, dtype=np.float64)
    base_curve = np.asarray(curve_func(x, *curve_args), dtype=np.float64)
    fixed_curve = np.array(base_curve, copy=True)

    y_anchor, slope_anchor = _sample_curve_value_and_slope(curve_func, q2_exp_start, curve_args)
    y_match = float(np.asarray(curve_func(np.array([q2_match], dtype=np.float64), *curve_args), dtype=np.float64)[0])
    lam, power = _resolve_fixed_zero_exponential_bridge(
        y_anchor,
        slope_anchor,
        y_match,
        q2_exp_start,
        q2_match,
        q2_zero,
    )

    delta_zero = q2_zero - q2_exp_start

    bridge_curve = np.array(base_curve, copy=True)
    mask_bridge = (x >= q2_exp_start) & (x < q2_zero)
    if np.any(mask_bridge):
        core = _bridge_core(lam, x[mask_bridge] - q2_exp_start, delta_zero)
        bridge_curve[mask_bridge] = y_anchor * np.power(np.clip(core, 0.0, None), power)
    bridge_curve[x >= q2_zero] = 0.0

    blend_start = min(q2_blend_start, q2_blend_end)
    blend_end = max(q2_blend_start, q2_blend_end)
    if blend_end > blend_start:
        mask_blend = (x >= blend_start) & (x <= blend_end)
        if np.any(mask_blend):
            t = (x[mask_blend] - blend_start) / (blend_end - blend_start)
            smoothstep = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
            fixed_curve[mask_blend] = (
                (1.0 - smoothstep) * base_curve[mask_blend]
                + smoothstep * bridge_curve[mask_blend]
            )

    mask_post_blend = (x > blend_end) & (x < q2_zero)
    if np.any(mask_post_blend):
        fixed_curve[mask_post_blend] = bridge_curve[mask_post_blend]

    fixed_curve[x >= q2_zero] = 0.0
    return fixed_curve


def _apply_smooth_zero_high_q2_strategy(
    x,
    curve_func,
    curve_args,
    q2_smooth_start=2.75,
    q2_zero=4.0,
    q2_blend_start=2.35,
    q2_blend_end=2.75,
    q2_peak=None,
    y_smooth_start=None,
    y_peak=None,
):
    """
    Preserve the tuned low/mid-Q² behavior and replace only the high-Q² tail
    with a two-sided smooth tail: one smoothstep rise into the high-Q² anchor
    point and one smoothstep fall to zero at q2_zero.
    """
    x = np.asarray(x, dtype=np.float64)
    base_curve = np.asarray(curve_func(x, *curve_args), dtype=np.float64)
    smooth_curve = np.array(base_curve, copy=True)
    bridge_curve = np.array(base_curve, copy=True)

    tail_config = _get_smooth_zero_tail_config()
    if tail_config is not None:
        q2_smooth_start = float(tail_config.get("q2_smooth_start", q2_smooth_start))
        q2_zero = float(tail_config.get("q2_zero", q2_zero))
        q2_blend_start = float(tail_config.get("q2_blend_start", q2_blend_start))
        q2_blend_end = float(tail_config.get("q2_blend_end", q2_blend_end))
        q2_peak = tail_config.get("q2_peak", q2_peak)
        y_smooth_start = tail_config.get("y_smooth_start", y_smooth_start)
        y_peak = tail_config.get("y_peak", y_peak)

    if q2_peak is None:
        q2_peak = q2_smooth_start + 0.75 * (q2_zero - q2_smooth_start)
    q2_peak = float(q2_peak)
    q2_zero = float(q2_zero)
    q2_smooth_start = float(q2_smooth_start)

    if not np.isfinite(q2_peak) or q2_peak <= q2_smooth_start or q2_peak >= q2_zero:
        q2_peak = q2_smooth_start + 0.75 * (q2_zero - q2_smooth_start)

    base_y_start, _ = _sample_curve_value_and_slope(curve_func, q2_smooth_start, curve_args)
    if y_smooth_start is None or not np.isfinite(y_smooth_start):
        y_smooth_start = base_y_start
    else:
        y_smooth_start = float(y_smooth_start)

    if y_peak is None or not np.isfinite(y_peak):
        y_peak = float(np.asarray(curve_func(np.array([q2_peak], dtype=np.float64), *curve_args), dtype=np.float64)[0])
    else:
        y_peak = float(y_peak)

    mask_rise = (x >= q2_smooth_start) & (x <= q2_peak)
    if np.any(mask_rise):
        rise_width = max(q2_peak - q2_smooth_start, 1e-12)
        t_rise = np.clip((x[mask_rise] - q2_smooth_start) / rise_width, 0.0, 1.0)
        s_rise = 6.0 * t_rise**5 - 15.0 * t_rise**4 + 10.0 * t_rise**3
        bridge_curve[mask_rise] = y_smooth_start + (y_peak - y_smooth_start) * s_rise

    mask_fall = (x > q2_peak) & (x < q2_zero)
    if np.any(mask_fall):
        fall_width = max(q2_zero - q2_peak, 1e-12)
        t_fall = np.clip((x[mask_fall] - q2_peak) / fall_width, 0.0, 1.0)
        s_fall = 6.0 * t_fall**5 - 15.0 * t_fall**4 + 10.0 * t_fall**3
        bridge_curve[mask_fall] = y_peak * (1.0 - s_fall)

    bridge_curve[x >= q2_zero] = 0.0

    blend_start = min(q2_blend_start, q2_blend_end)
    blend_end = max(q2_blend_start, q2_blend_end)
    if blend_end > blend_start:
        mask_blend = (x >= blend_start) & (x <= blend_end)
        if np.any(mask_blend):
            t = (x[mask_blend] - blend_start) / (blend_end - blend_start)
            smoothstep = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
            smooth_curve[mask_blend] = (
                (1.0 - smoothstep) * base_curve[mask_blend]
                + smoothstep * bridge_curve[mask_blend]
            )

    mask_post_blend = (x > blend_end) & (x < q2_zero)
    if np.any(mask_post_blend):
        smooth_curve[mask_post_blend] = bridge_curve[mask_post_blend]

    smooth_curve[x >= q2_zero] = 0.0
    return smooth_curve


def k_curve_fixed_zero(x, a, b, c, d, f, e):
    """
    Fixed-zero k(Q²) model.

    This keeps the original behavior below Q²≈1 and replaces the higher-Q²
    continuation with one smooth exponential bridge that reaches zero at Q²=4.
    """
    return _apply_fixed_zero_high_q2_strategy(
        x,
        k_curve_tune,
        (a, b, c, d, f, e),
    )


def k_curve_smooth_zero(x, a, b, c, d, f, e):
    """
    Smooth-zero k(Q²) model.

    This keeps the tuned low/mid-Q² behavior and replaces only the high-Q²
    continuation with a monotone cubic bridge that decays smoothly to zero.
    """
    return _apply_smooth_zero_high_q2_strategy(
        x,
        k_curve_tune,
        (a, b, c, d, f, e),
    )


# Backward-compatible default alias.
k_curve = k_curve_non_tune

#HERE
def gamma_curve(x, a, b, c, d, f, x0=0.1, k=100):
  """function"""
  x = np.asarray(x, dtype=np.float64)  # Ensure array compatibility

  # Define both models
  linear_part = d + f * x
  nonlinear_part = -a * np.exp(-x/b) - (c / x) # chi2 = 1.9 Bounds(lb=[-1e10, -1e10, -1e10, -1e10], ub=[1e10, 1e10, 1e10, 1e10])

  # Sigmoid-based smooth step function
  s = 1 / (1 + np.exp(-k * (x - x0)))  # s ~ 0 for x << x0, s ~ 1 for x >> x0

  return (1 - s) * linear_part + s * nonlinear_part

#HERE
def mass_curve(x, a, b, c, d, e, x0=0.1, k=100):
  """function"""
  x = np.asarray(x, dtype=np.float64)  # Ensure array compatibility

  # Define both models
  linear_part = d + e * x
  nonlinear_part = 1.232 - a * np.exp(-x / b) - (c / x)

  # Sigmoid-based smooth step function
  s = 1 / (1 + np.exp(-k * (x - x0)))  # s ~ 0 for x << x0, s ~ 1 for x >> x0

  return (1 - s) * linear_part + s * nonlinear_part

def quadconstr_curve(x, x0, y0, c):
  """quadratic fit function to constrain minimum"""
  return c*(x-x0)**2 + y0

def cubic_curve(x, a, b, c, d):
  """cubic fit function"""
  return a + b*x + c*x**2 + d*x**3

def exp_curve(x, a, b, c):
  """exponential fit function
  y = a * Exp[b * x] + c
  """
  return a*np.exp(b*x) + c

def cub_exp_curve(x, c0, c1, c2, c3, e0, e1):
  """cubic * exponential curve fit function"""
  return cubic_curve(x, c0, c1, c2, c3) * exp_curve(x, e0, e1, c=0)

def quadconstr_exp_curve(x, x0, y0, c, e0, e1):
  """cubic * exponential curve fit function"""
  return quadconstr_curve(x, x0, y0, c) * exp_curve(x, e0, e1, c=0)

def nucl_potential(x, p0, p1, p2, y0):
  """
  nuclear potential form from Xiaochao
  p0: depth of potential
  p1: jump point of potential
  p2: width of potential
  y0: constant value after jump
  """
  return y0 + p0/(1.0 + np.exp((x-p1)/p2))

# HERE
def quad_nucl_curve_gamma(x, a, b, c, d, e, y0, p0, p1, p2, y1):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  c: quadratic parameter
  y0: constant value term for after decay
  p0: depth of nucl potential
  p1: jump point of nucl potential
  p2: width of nucl potential
  y1: final constant value of nuclear potential
  """
  return gamma_curve(x, a, b, c, d, e) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
# HERE
def quad_nucl_curve_k_non_tune(x, a, b, c, d, e, f, y0, p0, p1, p2, y1):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  y0: constant value term for after decay
  p0: depth of nucl potential
  p1: jump point of nucl potential
  p2: width of nucl potential
  y1: final constant value of nuclear potential
  """
  return k_curve_non_tune(x, a, b, c, d, e, f) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0


def quad_nucl_curve_k_tune(x, a, b, c, d, e, f, y0, p0, p1, p2, y1):
  """
  Tuned quadratic * nucl potential k(Q^2) form.
  """
  return k_curve_tune(x, a, b, c, d, e, f) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0


def quad_nucl_curve_k_fixed_zero(x, a, b, c, d, e, f, y0, p0, p1, p2, y1):
  """
  Fixed-zero quadratic * nucl potential k(Q^2) form.
  """
  return _apply_fixed_zero_high_q2_strategy(
    x,
    quad_nucl_curve_k_tune,
    (a, b, c, d, e, f, y0, p0, p1, p2, y1),
  )


def quad_nucl_curve_k_smooth_zero(x, a, b, c, d, e, f, y0, p0, p1, p2, y1):
  """
  Smooth-zero quadratic * nucl potential k(Q^2) form.
  """
  return _apply_smooth_zero_high_q2_strategy(
    x,
    quad_nucl_curve_k_tune,
    (a, b, c, d, e, f, y0, p0, p1, p2, y1),
  )


def get_quad_nucl_curve_k(mode="non_tune"):
  normalized = normalize_bw_k_curve_mode(mode)
  if normalized == "tune":
    return quad_nucl_curve_k_tune
  if normalized == "fixed_zero":
    return quad_nucl_curve_k_fixed_zero
  if normalized == "smooth_zero":
    return quad_nucl_curve_k_smooth_zero
  return quad_nucl_curve_k_non_tune


# Backward-compatible default alias.
quad_nucl_curve_k = quad_nucl_curve_k_non_tune
# HERE
def quad_nucl_curve_mass(x, a, b, c, d, e, y0, p0, p1, p2, y1):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  y0: constant value term for after decay
  p0: depth of nucl potential
  p1: jump point of nucl potential
  p2: width of nucl potential
  y1: final constant value of nuclear potential
  """
  return mass_curve(x, a, b, c, d, e) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
  
def lin_nucl_curve(x, a, b):
  """
  linear * nucl potential form
  x: independent data
  a, b: linear curve parameters
  """
  return lin_curve(x, a, b)

def red_chi_sqr(y_calc, y_obs, y_err, nu):
  """
  calculates reduced chi squared of fit (nu = n observations - m fitted parameters)
  """
  return np.sum(np.square((y_obs-y_calc)/y_err))/nu

def fit_with_dynamic_params(var_name, x_data, y_data, y_err, param_bounds, p_vals_initial, fit_function, N=10, 
                             population_size=15, max_iterations=15000, mutation_range=(0.4, 1.6), 
                             recombination_rate=0.8, strategy='best1bin', tolerance=1e-8):
    """
    Enhanced phase space search function for parameter fitting with more flexible optimization strategies.
    
    Args:
        var_name (str): Name of the variable being fit
        x_data (array): Independent variable data
        y_data (array): Dependent variable data
        y_err (array): Errors in dependent variable
        param_bounds (object): Bounds for parameters
        p_vals_initial (array): Initial p-value guesses
        fit_function (callable): Function to fit data
        N (int, optional): Number of global search iterations. Defaults to 10.
        population_size (int, optional): Size of differential evolution population. Defaults to 15.
        max_iterations (int, optional): Maximum iterations for differential evolution. Defaults to 15000.
        mutation_range (tuple, optional): Range for mutation factor. Defaults to (0.4, 1.6).
        recombination_rate (float, optional): Crossover probability. Defaults to 0.8.
        strategy (str, optional): Differential evolution strategy. Defaults to 'best1bin'.
        tolerance (float, optional): Optimization tolerance. Defaults to 1e-8.
    
    Returns:
        tuple: Optimized parameters, p-values, reduced chi-squared, parameter uncertainties, p-value uncertainties
    """
    
    # Determine number of parameters and p-values
    num_params = len(param_bounds.lb)
    num_P_vals = len(p_vals_initial)
    
    # Inspect the fit function to understand its signature
    sig = inspect.signature(fit_function)
    param_names = list(sig.parameters.keys())
    
    def chi_squared(params):
        """Calculate reduced chi-squared for the given parameters."""
        # Prepare arguments for the fit function
        fit_args = []
        
        # Always start with x_data
        fit_args.append(x_data)
        
        # Add model parameters
        fit_args.extend(params[:num_params])
        
        # Add p-values
        fit_args.extend(p_vals_initial)
        
        # Trim arguments to match function signature if needed
        fit_args = fit_args[:len(param_names)]
        
        model = np.asarray(fit_function(*fit_args), dtype=np.float64)
        if not np.all(np.isfinite(model)):
            return np.inf

        safe_y_err = np.where(np.abs(y_err) > 0, y_err, np.nan)
        residuals = (y_data - model) / safe_y_err
        if not np.all(np.isfinite(residuals)):
            return np.inf

        chi2_value = np.sum(residuals ** 2)
        degrees_of_freedom = max(1, len(y_data) - num_params)
        
        return chi2_value / degrees_of_freedom
    
    def optimize_fit(bounds):
        """
        Perform differential evolution optimization with more robust parameters.
        
        Args:
            bounds (list): Parameter bounds for optimization
        
        Returns:
            tuple: Best reduced chi-squared and corresponding parameters
        """
        print("Performing advanced differential evolution optimization...")
        
        result = differential_evolution(
            chi_squared, 
            bounds=bounds, 
            popsize=population_size, 
            mutation=mutation_range, 
            recombination=recombination_rate, 
            strategy=strategy, 
            tol=tolerance, 
            maxiter=max_iterations,
            seed=np.random.randint(2**32 - 1)  # Ensure reproducibility with random seed
        )
        
        print(f"Optimization result - Success: {result.success}, Message: {result.message}")
        return result.fun, result.x
    
    # Prepare bounds for optimization
    search_bounds = [(lb, ub) for lb, ub in zip(param_bounds.lb, param_bounds.ub)]
    
    print(f"Parameter bounds: {search_bounds}")
    
    # Multiple global search iterations
    best_reduced_chi_squared = np.inf
    best_params = None
    
    for i in range(N):
        print(f"Global Search Iteration {i + 1}/{N}")
        
        current_reduced_chi_squared, current_params = optimize_fit(search_bounds)
        print(f"Iteration {i + 1} - Reduced chi-squared: {current_reduced_chi_squared:.5f}")

        if best_params is None and current_params is not None and np.all(np.isfinite(current_params)):
            best_reduced_chi_squared = current_reduced_chi_squared
            best_params = current_params
            continue

        if (
            current_params is not None
            and np.all(np.isfinite(current_params))
            and abs(current_reduced_chi_squared-1) < abs(best_reduced_chi_squared-1)
        ):
            best_reduced_chi_squared = current_reduced_chi_squared
            best_params = current_params

    if best_params is None:
        fallback_params = []
        for lb, ub in search_bounds:
            if np.isfinite(lb) and np.isfinite(ub):
                fallback_params.append(0.5 * (lb + ub))
            elif np.isfinite(lb):
                fallback_params.append(lb)
            elif np.isfinite(ub):
                fallback_params.append(ub)
            else:
                fallback_params.append(0.0)
        best_params = np.array(fallback_params, dtype=np.float64)
        best_reduced_chi_squared = np.inf
        print(
            f"Warning: no valid optimization result found for '{var_name}'. "
            "Using fallback parameters and NaN uncertainties."
        )
    
    def compute_uncertainties(best_solution):
        """
        Compute parameter uncertainties using advanced numerical differentiation
        with regularization to stabilize the Hessian inversion.

        Args:
            best_solution (array): Best-fit parameters

        Returns:
            array: Parameter uncertainties
        """
        if best_solution is None:
            return np.full(num_params, np.nan)

        best_solution = np.asarray(best_solution, dtype=np.float64)
        if best_solution.size != num_params or not np.all(np.isfinite(best_solution)):
            return np.full(num_params, np.nan)

        degrees_of_freedom = max(1, len(y_data) - num_params)
        delta_chi2 = chi2.ppf(0.68, degrees_of_freedom) - chi2.ppf(0.32, degrees_of_freedom)

        def perturb_params(params, index, step):
            """Create a perturbed parameter set."""
            perturbed = params.copy()
            perturbed[index] += step
            return perturbed

        def compute_hessian(params, regularization_strength=1e-8):
            """Numerically compute Hessian matrix with regularization."""
            hess = np.zeros((len(params), len(params)))
            #step_sizes = [max(1e-6 * abs(p), 1e-8) for p in params]  # Adaptive step size
            step_sizes = [max(1e-8 * abs(p), 1e-10) for p in params]  # Adaptive step size

            for i in range(len(params)):
                for j in range(len(params)):
                    # Compute central differences
                    params_ij_plus = params.copy()
                    params_ij_minus = params.copy()

                    params_ij_plus[i] += step_sizes[i]
                    params_ij_plus[j] += step_sizes[j]

                    params_ij_minus[i] -= step_sizes[i]
                    params_ij_minus[j] -= step_sizes[j]

                    base_value = chi_squared(params)
                    hess[i, j] = (
                        chi_squared(params_ij_plus)
                        - chi_squared(perturb_params(params, i, step_sizes[i]))
                        - chi_squared(perturb_params(params, j, step_sizes[j]))
                        + base_value
                    ) / (4 * step_sizes[i] * step_sizes[j])

            # Regularize the Hessian by adding a small value to the diagonal
            regularization = regularization_strength * np.eye(len(params))
            hess += regularization
            return hess

        try:
            # Compute regularized Hessian
            hessian = compute_hessian(best_solution)

            # Check condition number and issue a warning if unstable
            condition_number = np.linalg.cond(hessian)
            if condition_number > 1e12:
                print("Warning: Hessian is ill-conditioned. Regularization applied.")

            # Invert the regularized Hessian
            covariance = np.linalg.inv(hessian)

            # Compute uncertainties from the diagonal elements of covariance matrix
            uncertainties = np.sqrt(np.abs(np.diag(covariance) * delta_chi2))

        except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError):
            # If inversion fails, return NaNs for all uncertainties
            uncertainties = np.full(num_params, np.nan)

        return uncertainties
    
    # Compute parameter uncertainties
    # Use the p_vals_initial as fixed for this optimization
    param_uncertainties = compute_uncertainties(best_params)
    
    return (best_params, p_vals_initial, best_reduced_chi_squared, 
            param_uncertainties, np.zeros_like(p_vals_initial))

def fit(func, x, y, y_err, params_init, param_names, constr=None, silent=False):
  """
  func: functional form to fit data with
  x: independent data
  y: dependent data
  y_err: uncertainties in dependent data
  params_init: a list of initial guesses for the parameters
  param_names: a list of the parameter names as strings
  constr: constraints on parameters - 2 tuple of bounds
  return: the parameters, covariance matrix, list of the
          uncertainties in the parameters, and chi_squared of the fit
  """
  # no specified bounds for params
  if constr is None:
    constr = ([-np.inf for x in param_names],
              [np.inf for x in param_names])

  params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, bounds=constr, maxfev = 50000)
  param_sigmas = [np.sqrt(covariance[i][i]) for i in range(len(params))]
  table = [
    [f"{params[i]:.5f} ± {param_sigmas[i]:.5f}" for i in range(len(params))]
    ]

  # get reduced chi squared of fit
  nu = len(y) - len(param_names)
  args = [x] # list of args to give to func to get fitted curve
  args += [p for p in params]
  y_fit = func(*args)
  chi_2 = red_chi_sqr(y_fit, y, y_err, nu)

  if not silent:
    print(tabulate(table, param_names, tablefmt="fancy_grid"))
    print(f"$\chi_v^2$ = {chi_2:.2f}")

  return params, covariance, param_sigmas, chi_2

def weighted_avg(y, w=1):
  """
  y: one dimensional array to average
  w: weights - array of ones if no weights are provided
     if using error as weight, w=1/error
  """
  if type(w) is int:
    w = np.ones(y.size)

  return np.sum(y*w)/np.sum(w)

#######################################################################

def partial_k(w, M, k, gamma):
    """
    Partial derivative wrt k, holding M and gamma fixed.
    f(w) = (k*M^2*gamma^2) / Denominator
    => ∂f/∂k = (M^2 * gamma^2) / Denominator
    """
    numerator = (M**2) * (gamma**2)
    denominator = (w**2 - M**2)**2 + (M**2)*(gamma**2)
    return numerator / denominator

def partial_gamma(w, M, k, gamma):
    """
    Partial derivative wrt gamma, holding M and k fixed.
    
    f(w) = [k * (M^2 * gamma^2)] / D
    D = (w^2 - M^2)^2 + M^2 * gamma^2
    
    Using quotient rule:
      ∂f/∂gamma = [ (∂N/∂gamma)*D - N*(∂D/∂gamma ) ] / D^2
      where N = k*M^2*gamma^2
    """
    denominator = (w**2 - M**2)**2 + M**2 * gamma**2
    denominator_squared = denominator**2

    # ∂N/∂gamma = k * M^2 * 2 gamma = 2*k*M^2*gamma
    numerator_1 = 2 * k * (M**2) * gamma  # This multiplies D / D^2 => /D

    # ∂D/∂gamma = ∂/∂gamma [M^2 gamma^2] = 2 M^2 gamma
    # => N * (2 M^2 gamma) => k*M^2*gamma^2 * 2 M^2 gamma
    numerator_2 = k * (M**2) * (gamma**2) * (2 * M**2 * gamma)

    # Final difference
    return (numerator_1 / denominator) - (numerator_2 / denominator_squared)

def partial_mass(w, M, k, gamma):
    """
    Partial derivative wrt M, holding gamma and k fixed.

    f(w) = [k*(M^2 * gamma^2)] / D
    D = (w^2 - M^2)^2 + (M^2)*(gamma^2)
    """
    denominator = (w**2 - M**2)**2 + (M**2)*(gamma**2)
    denominator_squared = denominator**2

    # dN/dM = k * 2 M gamma^2
    numerator_1 = k * (2 * M * (gamma**2))

    # dD/dM = -4 M (w^2 - M^2) + 2 M gamma^2
    term1 = -4 * M * (w**2 - M**2)
    term2 = 2 * M * (gamma**2)
    dD_dM = term1 + term2

    # Combine via quotient rule
    # ( numerator_1 * denominator ) - [N * dD_dM]
    # where N = k * (M^2 * gamma^2)
    N = k * (M**2) * (gamma**2)

    return (numerator_1 / denominator) - ((N * dD_dM) / denominator_squared)

def partial_damp_W_transition(W, W_transition, width):
    """Partial derivative with respect to W_transition"""
    return (1/(width * (1 + np.exp((W - W_transition)/width))**2) * 
            np.exp((W - W_transition)/width))

def partial_damp_width(W, W_transition, width):
    """Partial derivative with respect to width"""
    return ((W - W_transition)/(width**2 * (1 + np.exp((W - W_transition)/width))**2) * 
            np.exp((W - W_transition)/width))

#######################################################################

# determine uncertainty in the fit
# partial functions for quadratic and cubic forms
# 2 means for quad form, 3 means for cubic form

def partial_a2(x, q2, par):
  """1 + beta/Q2"""
  return 1 + par[3]/q2

def partial_a3(x, q2, par):
  """1 + beta/Q2"""
  return 1 + par[4]/q2

def partial_b2(x, q2, par):
  """X*(1 + beta/Q2)"""
  return x*(1 + par[3]/q2)

def partial_b3(x, q2, par):
  """X*(1 + beta/Q2)"""
  return x*(1 + par[4]/q2)

def partial_c2(x, q2, par):
  """X^2*(1 + beta/Q2)"""
  return x*x*(1 + par[3]/q2)

def partial_c3(x, q2, par):
  """X^2*(1 + beta/Q2)"""
  return x*x*(1 + par[4]/q2)

def partial_beta2(x, q2, par):
  """1/Q^2*(a+bx+cx^2)"""
  return (1/q2)*(par[0]+par[1]*x+par[2]*x*x)

def partial_beta3(x, q2, par):
  """1/Q^2*(a+bx+cx^2)"""
  return (1/q2)*(par[0]+par[1]*x+par[2]*x*x+par[3]*x*x*x)

def partial_d3(x, q2, par):
  """x^3 * (1 + beta/Q2)"""
  return x*x*x*(1 + par[4]/q2)

# partials for constrained quadratic form
def partial_x0(x, q2, par):
  """[-2c * (x - x0)](1 + beta/Q2)"""
  return (-2*par[2] * (x - par[0]))*(1 + par[3]/q2)

def partial_y0(x, q2, par):
  """(1 + beta/Q2)"""
  return (1 + par[3]/q2)

def partial_c4(x, q2, par):
  """(x-x0)^2(1 + beta/Q2)"""
  return (x-par[0])**2 * (1 + par[3]/q2)

def partial_beta4(x, q2, par):
  """(c(x-x0)^2 + y0)(1/Q2)"""
  return (par[2]*(x-par[0])**2 + par[1]) * (1/q2)

def partial_alpha(x, q2, par):
  """(c(x-x0)^2 + y0)(1/Q2)"""
  return (par[2]*(x-par[0])**2 + par[1]) * (1/q2)

####
'''
# Partials of Table F.1 from XZ's thesis
def partial_alpha_new(x, q2, par):
    """Partial derivative with respect to alpha."""
    #alpha, a, b, c, d, beta = par
    alpha, a, b, c, beta = par
    
    #poly = a + b*x + c*x**2 + d*x**3
    poly = a + b*x + c*x**2
    term = np.log(x) * x**alpha
    return term * poly * (1 + beta/q2)

def partial_a_new(x, q2, par):
    """Partial derivative with respect to a."""
    #alpha, a, b, c, d, beta = par
    alpha, a, b, c, beta = par
    
    return x**alpha * (1 + beta/q2)

def partial_b_new(x, q2, par):
    """Partial derivative with respect to b."""
    #alpha, a, b, c, d, beta = par
    alpha, a, b, c, beta = par
    
    return x**(alpha + 1) * (1 + beta/q2)

def partial_c_new(x, q2, par):
    """Partial derivative with respect to c."""
    #alpha, a, b, c, d, beta = par
    alpha, a, b, c, beta = par
    
    return x**(alpha + 2) * (1 + beta/q2)

def partial_d_new(x, q2, par):
    """Partial derivative with respect to d."""
    alpha, a, b, c, d, beta = par
    
    return x**(alpha + 3) * (1 + beta/q2)

def partial_beta_new(x, q2, par):
    """Partial derivative with respect to beta."""
    #alpha, a, b, c, d, beta = par
    alpha, a, b, c, beta = par
    
    #poly = a + b*x + c*x**2 + d*x**3
    poly = a + b*x + c*x**2
    return x**alpha * poly / q2
'''

# Partials of Modified Table F.1 from XZ's thesis for full x range
import numpy as np

def partial_alpha_fullx(x, q2, par):
    """∂/∂α [ x^α · (quad + gauss) · (1 + β/q2) ]"""
    α, a, b, c, β, d, x0, σ = par
    quad     = a + b*x + c*x**2
    gauss    = np.exp(-0.5*((x - x0)/σ)**2)
    base     = quad + d*gauss
    prefactor= x**α * np.log(x)
    return prefactor * base * (1.0 + β/q2)

def partial_a_fullx(x, q2, par):
    """∂/∂a"""
    α, a, b, c, β, d, x0, σ = par
    gauss = np.exp(-0.5*((x - x0)/σ)**2)
    return x**α * (1.0 + β/q2)

def partial_b_fullx(x, q2, par):
    """∂/∂b"""
    α, a, b, c, β, d, x0, σ = par
    return x**(α+1) * (1.0 + β/q2)

def partial_c_fullx(x, q2, par):
    """∂/∂c"""
    α, a, b, c, β, d, x0, σ = par
    return x**(α+2) * (1.0 + β/q2)

def partial_beta_fullx(x, q2, par):
    """∂/∂β"""
    α, a, b, c, β, d, x0, σ = par
    quad  = a + b*x + c*x**2
    gauss = np.exp(-0.5*((x - x0)/σ)**2)
    return x**α * (quad + d*gauss) / q2

def partial_d_fullx(x, q2, par):
    """∂/∂d"""
    α, a, b, c, β, d, x0, σ = par
    quad  = a + b*x + c*x**2
    gauss = np.exp(-0.5*((x - x0)/σ)**2)
    return x**α * gauss * (1.0 + β/q2)

def partial_x0_fullx(x, q2, par):
    """∂/∂x0"""
    α, a, b, c, β, d, x0, σ = par
    quad      = a + b*x + c*x**2
    Δ         = x - x0
    gauss     = np.exp(-0.5*(Δ/σ)**2)
    dgauss_dx0= gauss * (Δ/σ**2)
    return x**α * d * dgauss_dx0 * (1.0 + β/q2)

def partial_sigma_fullx(x, q2, par):
    """∂/∂σ"""
    α, a, b, c, β, d, x0, σ = par
    quad       = a + b*x + c*x**2
    Δ          = x - x0
    gauss      = np.exp(-0.5*(Δ/σ)**2)
    dgauss_dσ  = gauss * (Δ**2/σ**3)
    return x**α * d * dgauss_dσ * (1.0 + β/q2)
####

# from Xiaochao's thesis
def g1f1_quad_DIS(x_q2, a, b , c, beta):
  return (a+b*x_q2[0]+c*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

# from Xiaochao's thesis
def g1f1_quad_DIS(x_q2, a, b , c, beta):
  return (a+b*x_q2[0]+c*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

# different form for quadratic to constrain the minimum
# y = c*(x-x0)^2 + y0 where (x0, y0) is the minimum
def g1f1_quad2_DIS(x_q2, x0, y0, c, beta):
  return (c*(x_q2[0]-x0)**2+y0)*(1+(beta/x_q2[1]))

# Canonical alpha-prefactor quadratic DIS form
def g1f1_quad_alpha_DIS(x_q2, alpha, a, b, c, beta):
    return (x_q2[0]**alpha) * (a + b*x_q2[0] + c*x_q2[0]*x_q2[0]) * (1+(beta/x_q2[1]))


# Canonical alpha-prefactor cubic DIS form
def g1f1_cubic_alpha_DIS(x_q2, alpha, a, b, c, d, beta):
    return (x_q2[0]**alpha) * (
        a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0]
    ) * (1 + (beta / x_q2[1]))


# Backward-compatible aliases for historical *_new naming.
def g1f1_quad_new_DIS(x_q2, alpha, a, b, c, beta):
    return g1f1_quad_alpha_DIS(x_q2, alpha, a, b, c, beta)


def g1f1_cubic_new_DIS(x_q2, alpha, a, b, c, d, beta):
    return g1f1_cubic_alpha_DIS(x_q2, alpha, a, b, c, d, beta)

# Modified Table F.1 from XZ's thesis with fit for full x range
def g1f1_quad_fullx_DIS(x_q2, alpha, a, b, c, beta, d, x0, sigma):
    """
    Fit g1/F1(x,Q2) with a power‐law × quadratic shape × (1 + beta/Q2)
    plus a Gaussian dip at low x.

    Parameters
    ----------
    x_q2 : array‐like, shape (2,)
        [ x , Q2 ]
    alpha : float
        low‑x power exponent
    a, b, c : float
        coefficients of the quadratic polynomial in x
    beta : float
        controls the 1/Q2 correction
    d : float (< 0 for a dip)
        amplitude of the Gaussian dip
    x0 : float
        center of the dip in x
    sigma : float
        width of the dip in x

    Returns
    -------
    g1f1 : float
    """
    x, Q2 = x_q2
    quad = a + b*x + c*x**2
    gauss_dip = d * np.exp(-0.5*((x - x0)/sigma)**2)
    return (x**alpha) * (quad + gauss_dip) * (1.0 + beta / Q2)

# guess form for downward trend at high x - a cubic!
def g1f1_cubic_DIS(x_q2, a, b , c, d, beta):
  return (a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

def damping_function(W, W_transition, width):
    """Woods-Saxon function for smooth damping"""
    return 1 / (1 + np.exp((W - W_transition) / width))

def k_new_new(q2):
    """
    k_new_new(q2) = [0.25 * q2 / (1 + 1.55 * q2)] * exp(-q2 / (2 * 0.25)) * ((1.55**2)*(0.25**2))
    """
    return (0.25*q2 / (1.0 + 1.55*q2)) * np.exp(-q2 / (2.0*0.25)) * ((1.55**2)*(0.25**2))

##############################################################

import numpy as np
from scipy.ndimage import gaussian_filter1d

def dk_new_new_dq2(q2):
    """
    Analytic derivative of k_new wrt (q^2) treating (q^2) as the variable.
    k_new(q2) = [0.25*q2/(1+1.55*q2)] * exp(-q2/(2*0.25))
    Returns a float or array depending on whether q2 is float or array-like.
    """
    A, B = 0.25, 1.55

    f1 = A * q2 / (1 + B * q2)
    df1 = A / (1 + B * q2) ** 2

    f2 = np.exp(-q2 / (2.0 * 0.25))  
    df2 = -2.0 * f2  

    return f2 * df1 + f1 * df2

def k_new_new_err(q2, dq2):
    """
    Compute uncertainty in k_new using error propagation.
    """
    return np.abs(dk_new_new_dq2(q2)) * dq2

def fit_error(x, q2, par, par_sigmas, pcorr, partials):
    """
    Propagate uncertainties for a multi-parameter fit.
    """
    npar = len(par)
    y_err = np.zeros(len(x))

    for i in range(npar):
        for j in range(npar):
            cov_ij = pcorr[i][j] * par_sigmas[i] * par_sigmas[j]
            y_err += partials[i](x, q2, par) * partials[j](x, q2, par) * cov_ij

    return np.sqrt(y_err)

def propagate_bw_error(w, mass, mass_err, k, k_err, gamma, gamma_err):
    """
    Propagate errors for the Breit-Wigner resonance formula.
    """
    dfdM = partial_mass(w, mass, k, gamma)
    dfdK = partial_k(w, mass, k, gamma)
    dfdGamma = partial_gamma(w, mass, k, gamma)

    return np.sqrt(
        (dfdM * mass_err) ** 2 +
        (dfdK * k_err) ** 2 +
        (dfdGamma * gamma_err) ** 2
    )

def damping_function_err(w, w_transition, w_transition_err, damping_width, damping_width_err):
    """
    Propagate errors for the damping function.
    """
    dfdTrans = partial_damp_W_transition(w, w_transition, damping_width)
    dfdWidth = partial_damp_width(w, w_transition, damping_width)

    return np.sqrt(
        (dfdTrans * w_transition_err) ** 2 +
        (dfdWidth * damping_width_err) ** 2
    )

def propagate_dis_error(fit_errs):
    """
    Propagate errors for the DIS fit.
    """
    return np.nan_to_num(fit_errs)  # Handle NaN values robustly

def propagate_transition_error(w, bw_err, bw_bump_err, w_res_min, w_res_max, w_dis_transition):
    """
    Propagate errors for the transition between Breit-Wigner and DIS regions.
    """
    if len(w) != len(bw_err):
        raise ValueError("All input lists must have the same length.")

    propagated_errors = np.zeros(len(w))

    for i in range(len(w)):
        if w[i] < w_res_max:
            alpha, beta = 0.75, 0.25
        elif w_res_max <= w[i] <= w_dis_transition:
            alpha, beta = 0.0, 1.0
        else:
            alpha, beta = 0.0, 0.0

        propagated_errors[i] = np.sqrt(
            (alpha * bw_err[i]) ** 2 +
            (beta * bw_bump_err[i]) ** 2
        )

    return propagated_errors

import numpy as np

def calculate_fit_residuals(y_complete, y_data, y_data_err):
    """
    Compute residuals of the total fit (y_complete) with experimental data.

    Parameters:
    - y_complete: Fit values at each W
    - y_data: Experimental G1/F1 values
    - y_data_err: Measurement uncertainties for G1/F1

    Returns:
    - residuals: Absolute difference between fit and data (y_data - y_complete)
    - normalized_residuals: Residuals scaled by measurement uncertainty (chi-like)
    """

    # Ensure arrays are the same length
    if len(y_complete) != len(y_data) or len(y_complete) != len(y_data_err):
        raise ValueError("Input arrays must have the same length.")

    # Compute absolute residuals
    residuals = y_data - y_complete

    # Compute normalized residuals (scaled by experimental uncertainty)
    normalized_residuals = residuals / np.maximum(y_data_err, 1e-8)  # Prevent division by zero

    return residuals, normalized_residuals

def propagate_complete_error(w, transition_err, damping_dis_err, dis_err, w_res_min, w_res_max, w_dis_transition, w_max):
    """
    Propagate errors across all fit regions.
    """
    if not (len(w) == len(transition_err) == len(damping_dis_err) == len(dis_err)):
        raise ValueError("Input lists must have the same length.")

    propagated_errors = np.zeros(len(w))

    for i in range(len(w)):
        if w[i] <= w_res_max:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        elif w_res_max < w[i] <= w_dis_transition:
            alpha, beta, gamma = 0.25, 0.75, 0.0
        elif w_dis_transition < w[i] <= w_max:
            alpha, beta, gamma = 0.0, 0.25, 0.75
        else:
            alpha, beta, gamma = 0.0, 0.0, 1.0

        propagated_errors[i] = np.sqrt(
            (alpha * transition_err[i]) ** 2 +
            (beta * damping_dis_err[i]) ** 2 +
            (gamma * dis_err[i]) ** 2
        )

    def moving_average(data, window_size):
        window_size = min(window_size, len(data))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            window_size = 3
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    smoothed_errors = moving_average(propagated_errors, min(len(propagated_errors) // 3, 1000))

    return smoothed_errors

def calculate_param_error(fit_func, args, param_errs):
    """
    Compute propagated error using numerical differentiation.
    """
    err_squared = 0.0

    for idx in range(len(param_errs)):
        if param_errs[idx] == 0:
            continue

        args_plus, args_minus = args.copy(), args.copy()
        args_plus[idx] += param_errs[idx]
        args_minus[idx] -= param_errs[idx]

        f_plus, f_minus = fit_func(*args_plus), fit_func(*args_minus)
        partial_derivative = (f_plus - f_minus) / (2 * param_errs[idx])

        err_squared += (partial_derivative * param_errs[idx]) ** 2

    return np.sqrt(err_squared)

#########################################################################

###############################################################################
# Helper: unify the triple-nested loop for (k, gamma, mass) in one place
###############################################################################
def k_gamma_mass_loop(
    q2, w_array, 
    k_fit_params, gamma_fit_params, mass_fit_params, 
    fit_funcs_k, fit_funcs_gamma, fit_funcs_mass,
    k_P_vals, gamma_P_vals, mass_P_vals,
    k_nucl_err, gamma_nucl_err, mass_nucl_err
):
    """
    Yields (i, j, ij, k, k_err, gamma, gamma_err, mass, mass_err) for each combination 
    in the triple-nested loop, *only* for the 'chosen_fits' index = (0,0,0).

    We replicate the same code used in each block to compute:
     - k, gamma, mass
     - k_err, gamma_err, mass_err
    Exactly as in the original repeated loops.

    Because your code typically sets:
        chosen_fits = [(0,0,0)]
    we preserve that. If you'd like to skip combinations, do so below.
    """
    chosen_fits = [(0, 0, 0)]  # only do the Quad,Quad,Quad combination

    for i in range(len(k_fit_params)):
        k_params = k_fit_params[i]
        args_k = [q2]
        for j in range(len(gamma_fit_params)):
            gamma_params = gamma_fit_params[j]
            args_gamma = [q2]
            for ij in range(len(mass_fit_params)):
                if (i, j, ij) not in chosen_fits:
                    continue  # skip combos that aren't desired

                mass_params = mass_fit_params[ij]
                args_mass = [q2]

                # -------------- k --------------
                if i == 1:
                    # spline k
                    k = k_fit_params[i](q2)
                    k_err = 0.0  # or a spline-based error if needed
                else:
                    k_args = args_k + [p for p in k_params]
                    if i == 0:
                        k_args += [P for P in k_P_vals]
                    k = fit_funcs_k[i](*k_args)
                    k_err = calculate_param_error(fit_funcs_k[i], k_args, k_nucl_err)

                # -------------- gamma --------------
                if j == 1:
                    # spline gamma
                    gamma = gamma_fit_params[j](q2)
                    gamma_err = 0.0
                else:
                    gamma_args = args_gamma + [p for p in gamma_params]
                    if j == 0:
                        gamma_args += [P for P in gamma_P_vals]
                    gamma = fit_funcs_gamma[j](*gamma_args)
                    gamma_err = calculate_param_error(fit_funcs_gamma[j], gamma_args, gamma_nucl_err)

                # -------------- mass --------------
                if ij == 1:
                    # spline mass
                    mass = mass_fit_params[ij](q2)
                    mass_err = 0.0
                else:
                    mass_args = args_mass + [p for p in mass_params]
                    if ij == 0:
                        mass_args += [P for P in mass_P_vals]
                    mass = fit_funcs_mass[ij](*mass_args)
                    mass_err = calculate_param_error(fit_funcs_mass[ij], mass_args, mass_nucl_err)

                yield (i, j, ij, k, k_err, gamma, gamma_err, mass, mass_err)
