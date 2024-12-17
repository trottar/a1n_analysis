#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2024-12-17 10:46:34 trottar"
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
from tabulate import tabulate
import inspect
    
##################################################################################################################################################
# Importing utility functions

#from utility import custom_map

##################################################################################################################################################

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
  k_new = k*((M**2)*(gamma**2))
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

#HERE
def k_curve(x, a, b, c, d, e, f):
  """function"""
  return -a * np.exp(-x/b) + (c / x) # chi2 = 4.1 Bounds(lb=[-1e10, -1e10, -1e10, -1e10], ub=[1e10, 1e10, 1e10, 0.0])
  #return -a * np.exp(-x/b) + (c / (d * x + e * x**2 + f * x**3)) # chi2 = 4.1 Bounds(lb=[-1e10, -1e10, -1e10, -1e10], ub=[1e10, 1e10, 1e10, 0.0])
  #return (-a / (1 + b * x**c)) * np.exp(-x / d) + (e / x) # chi2 = XX Bounds(lb=[-1e10, -1e10, -1e10, -1e10], ub=[1e10, 1e10, 1e10, 1e10])
#HERE
def gamma_curve(x, a, b, c):
  """function"""
  return a / (1 + (x / b))**(c)  # chi2 = 2.1 Bounds(lb=[-1e10, -1e10, -1e10, 0.0], ub=[1e10, 1e10, 1e10, 0.3])
#HERE
def mass_curve(x, a, b, c):
  """function"""
  return 1.232 - a * np.exp(-x/b) - (c / x)  # chi2 = 4.2 Bounds(lb=[0.0, -1e10, 0.0, 0.0], ub=[1e10, 1e10, 1e10, 2.0])
  
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
  return gamma_curve(x, a, b, c) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
# HERE
def quad_nucl_curve2(x, a, b, c, d, e, f, y0, p0, p1, p2, y1):
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
  return k_curve(x, a, b, c, d, e, f) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
# HERE
def quad_nucl_curve3(x, a, b, c, y0, p0, p1, p2, y1):
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
  return mass_curve(x, a, b, c) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
  
def lin_nucl_curve(x, a, b):
  """
  linear * nucl potential form
  x: independent data
  a, b: linear curve parameters
  """
  return lin_curve(x, a, b)

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

'''
def fit_with_dynamic_params(var_name, x_data, y_data, y_err, param_bounds, p_vals_initial, fit_function, N=10):

    num_P_vals = len(p_vals_initial)
    
    if var_name == "gamma" or var_name == "k" or var_name == "mass":
        num_params = len(p_vals_initial)
    else:
        num_params = len(p_vals_initial)-1
        
    def chi_squared(params):
        P_vals = params[:num_params]
        
        model_params = params[num_P_vals:]

        #print("!!!!!!!",params)
        #print(P_vals)
        #print(model_params)
        
        model = fit_function(x_data, *model_params, *P_vals)
        
        residuals = (y_data - model) / y_err
        chi2 = np.sum(residuals ** 2)
        degrees_of_freedom = len(y_data) - num_P_vals
        
        return chi2 / degrees_of_freedom if degrees_of_freedom > 0 else np.inf
    
    def optimize_fit(combined_bounds):

        print("Using differential evolution...")
        result = differential_evolution(chi_squared, bounds=combined_bounds, popsize=10, 
                                        mutation=(0.5, 1.5), recombination=0.7, 
                                        strategy='best1bin', tol=1e-7, maxiter=10000)

        print("Optimization result:", result)
        return result.fun, result.x
    
    best_reduced_chi_squared = np.inf
    best_params = None

    # Print the combined bounds to ensure they're correct
    combined_bounds = [(lb, ub) for lb, ub in zip(param_bounds.lb, param_bounds.ub)]
    print(f"Combined bounds: {combined_bounds}")
    
    for i in range(N):
        print(f"Iteration {i + 1}/{N}")
        
        current_reduced_chi_squared, current_params = optimize_fit(combined_bounds)
        print(f"Iteration {i + 1} - Reduced chi-squared: {current_reduced_chi_squared:.5f}")

        current_p_vals = p_vals_initial
        
        if current_reduced_chi_squared < best_reduced_chi_squared:
            best_reduced_chi_squared = current_reduced_chi_squared
            best_params = current_params
            best_p_vals = current_p_vals
    
    final_params = best_params[:num_params]
    final_p_vals = best_p_vals[:num_P_vals]
    
    # Calculate confidence intervals
    dof = len(y_data) - num_P_vals
    delta_chi2 = chi2.ppf(0.68, dof) - chi2.ppf(0.32, dof)
    hessian = np.zeros((num_P_vals, num_params))
    step = 1e-5
    for i in range(num_P_vals):
        for j in range(num_params):
            params_plus_i = best_p_vals.copy()
            params_plus_i[i] += step
            params_plus_j = best_p_vals.copy()
            params_plus_j[j] += step
            params_plus_ij = best_p_vals.copy()
            params_plus_ij[i] += step
            params_plus_ij[j] += step
            
            f = chi_squared(best_p_vals)
            f_i = chi_squared(params_plus_i)
            f_j = chi_squared(params_plus_j)
            f_ij = chi_squared(params_plus_ij)
            
            hessian[i, j] = (f_ij - f_i - f_j + f) / (step * step)
    
    try:
        covariance = np.linalg.inv(hessian)
        uncertainties = np.sqrt(np.diag(covariance) * delta_chi2)
    except np.linalg.LinAlgError:
        uncertainties = np.full(num_P_vals, np.nan)
    
    param_uncertainties = uncertainties[num_P_vals:]
    p_val_uncertainties = uncertainties[:num_params]
    
    return final_params, final_p_vals, best_reduced_chi_squared, param_uncertainties, p_val_uncertainties
'''

def fit_with_dynamic_params(var_name, x_data, y_data, y_err, param_bounds, p_vals_initial, fit_function, N=10, 
                             population_size=15, max_iterations=50000, mutation_range=(0.4, 1.6), 
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
        
        model = fit_function(*fit_args)
        
        residuals = (y_data - model) / y_err
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
        
        if abs(current_reduced_chi_squared-1) < abs(best_reduced_chi_squared-1):
            best_reduced_chi_squared = current_reduced_chi_squared
            best_params = current_params
    
    # Advanced uncertainty estimation with more robust method
    def compute_uncertainties(best_solution):
        """
        Compute parameter uncertainties using advanced numerical differentiation.
        
        Args:
            best_solution (array): Best-fit parameters
        
        Returns:
            array: Parameter uncertainties
        """
        degrees_of_freedom = max(1, len(y_data) - num_params)
        delta_chi2 = chi2.ppf(0.68, degrees_of_freedom) - chi2.ppf(0.32, degrees_of_freedom)
        
        def perturb_params(params, index, step):
            """Create a perturbed parameter set."""
            perturbed = params.copy()
            perturbed[index] += step
            return perturbed
        
        def compute_hessian(params):
            """Numerically compute Hessian matrix."""
            hess = np.zeros((len(params), len(params)))
            step = 1e-4
            
            for i in range(len(params)):
                for j in range(len(params)):
                    # Compute second-order mixed partial derivatives
                    base_value = chi_squared(params)
                    
                    # Numerical approximation of mixed partial derivative
                    params_i_plus = perturb_params(params, i, step)
                    params_j_plus = perturb_params(params, j, step)
                    params_ij_plus = perturb_params(params_i_plus, j, step)
                    
                    mixed_derivative = (chi_squared(params_ij_plus) - chi_squared(params_i_plus) 
                                         - chi_squared(params_j_plus) + base_value) / (step**2)
                    
                    hess[i, j] = mixed_derivative
            
            return hess
        
        try:
            hessian = compute_hessian(best_solution)
            covariance = np.linalg.inv(hessian)
            uncertainties = np.sqrt(np.abs(np.diag(covariance) * delta_chi2))
        except (np.linalg.LinAlgError, ValueError):
            uncertainties = np.full(len(best_solution), np.nan)
        
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

def residual_function(W, a, b, c, W_transition):
    """A simple polynomial function to fit the residual"""
    return a * (W - W_transition)**2 + b * (W - W_transition) + c

def damping_function(W, W_transition, width):
    """Woods-Saxon function for smooth damping"""
    return 1 / (1 + np.exp((W - W_transition) / width))

