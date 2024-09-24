#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2024-09-24 03:30:09 trottar"
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
from tabulate import tabulate

def x_to_W(x, Q2):
  """ convert x to W for constant Q2"""
  Mp = 0.93870319 # average nucleon mass in 3He
  return np.sqrt(Mp**2 + Q2*((1/x)+1))

def W_to_x(W, Q2):
  """ convert W to x for constant Q2"""
  Mp = 0.93870319 # average nucleon mass in 3He
  return Q2/(W**2 + Q2 - Mp**2)

#TODO (DONE): reformulate this function so k_new is the peak height
def breit_wigner_res(w, M, k, gamma):
  """fit for constant Q2"""
  #return k/((w*w - M*M)**2 + M*M*gamma*gamma)
  # RLT (9/23/2024): Updated k_new=k/(M^2*gamma^2)
  k_new = k/((M**2)*(gamma**2))
  return k_new/((w*w - M*M)**2 + M*M*gamma*gamma)

def breit_wigner_wrapper(M_test):
  """for fitting with constant M"""
  def temp_func(w, k, gamma, M=M_test):
    return breit_wigner_res(w, M, k, gamma)
  return temp_func

def lin_curve(x, a, b):
  """simple line function for fitting"""
  return a*x + b

def quad_curve(x, a, b, c):
  """quadratic fit function"""
  return a + b*x + c*x**2

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


def quad_nucl_curve(x, a, b, c, y0, p0, p1, p2, y1):
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
  return quad_curve(x, a, b, c) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0

# from Xiaochao's thesis
def g1f1_quad_DIS(x_q2, a, b , c, beta):
  return (a+b*x_q2[0]+c*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

# different form for quadratic to constrain the minimum
# y = c*(x-x0)^2 + y0 where (x0, y0) is the minimum
def g1f1_quad2_DIS(x_q2, x0, y0, c, beta):
  return (c*(x_q2[0]-x0)**2+y0)*(1+(beta/x_q2[1]))

# guess form for downward trend at high x - a cubic!
def g1f1_cubic_DIS(x_q2, a, b , c, d, beta):
  return (a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

def red_chi_sqr(y_calc, y_obs, y_err, nu):
  """
  calculates reduced chi squared of fit (nu = n observations - m fitted parameters)
  """
  return np.sum(np.square((y_obs-y_calc)/y_err))/nu

def fit_with_dynamic_params(x_data, y_data, y_err, initial_params, param_bounds, p_vals_initial, p_vals_bounds, fit_function, N=10, global_opt=False):
    """
    Enhanced fitting function to find optimal parameters for a given model.

    :param x_data: Independent variable data
    :param y_data: Dependent variable data
    :param y_err: Error in y_data
    :param initial_params: Initial guess for the model parameters
    :param param_bounds: Bounds for model parameters
    :param p_vals_initial: Initial guess for P-values
    :param p_vals_bounds: Bounds for P-values
    :param fit_function: The function to fit (e.g., quad_nucl_curve_constp)
    :param N: Number of iterations to perform the fit
    :param global_opt: Use global optimization if set to True
    :return: Best fit parameters, best P-values, and best reduced chi-squared value
    """
    
    # Calculate the number of total parameters
    num_params = len(initial_params) + len(p_vals_initial)

    def chi_squared(params):
        P_vals = params[:len(p_vals_initial)]
        model_params = params[len(p_vals_initial):]
        model = fit_function(x_data, *model_params)
        
        residuals = (y_data - model) / y_err
        chi2 = np.sum(residuals ** 2)
        degrees_of_freedom = len(y_data) - num_params
        
        # Return reduced chi-squared, handling potential divide by zero
        return chi2 / degrees_of_freedom if degrees_of_freedom > 0 else np.inf

    def optimize_fit(p_vals):
        combined_bounds = p_vals_bounds + [(lb, ub) for lb, ub in zip(param_bounds.lb, param_bounds.ub)]
        
        if global_opt:
            result = differential_evolution(chi_squared, bounds=combined_bounds)
        else:
            result = minimize(chi_squared, x0=np.concatenate([p_vals, initial_params]), 
                              bounds=combined_bounds, method='L-BFGS-B',
                              options={'ftol': 1e-6, 'gtol': 1e-6})
        
        return result.fun, result.x[len(p_vals_initial):]  # Return reduced chi-squared and model parameters

    best_params = None
    best_p_vals = None
    best_reduced_chi_squared = np.inf

    for i in range(N):
        print(f"Iteration {i + 1}/{N}")
        
        # Use structured initialization instead of random offsets
        #p_vals_initial_random = p_vals_initial + np.random.uniform(-0.5, 0.5, size=len(p_vals_initial))
        p_vals_initial_random = p_vals_initial
        
        # Optimize
        result = minimize(lambda params: optimize_fit(params)[0], 
                          x0=p_vals_initial_random,
                          bounds=p_vals_bounds, method='L-BFGS-B')
        
        current_p_vals = result.x
        current_reduced_chi_squared, current_params = optimize_fit(current_p_vals)

        print(f"Iteration {i + 1} - Reduced chi-squared: {current_reduced_chi_squared}, Parameters: {current_params}, P-Values: {current_p_vals}")
        
        if abs(current_reduced_chi_squared - 1) < abs(best_reduced_chi_squared - 1):
            best_reduced_chi_squared = current_reduced_chi_squared
            best_params = current_params
            best_p_vals = current_p_vals

    return best_params, best_p_vals, best_reduced_chi_squared

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

  params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, bounds=constr, maxfev = 10000)
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

# from Xiaochao's thesis
def g1f1_quad_DIS(x_q2, a, b , c, beta):
  return (a+b*x_q2[0]+c*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

# different form for quad to constrain minimum
# y = c*(x-x0)^2 + y0 where (x0, y0) is the minimum
def g1f1_quad2_DIS(x_q2, x0, y0, c, beta):
  return (c*(x_q2[0]-x0)**2+y0)*(1+(beta/x_q2[1]))

# guess form for downward trend at high x
def g1f1_cubic_DIS(x_q2, a, b , c, d, beta):
  return (a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))



