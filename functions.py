#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2025-01-14 16:23:11 trottar"
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
def k_curve(x, a, b, c):
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
def quad_nucl_curve_gamma(x, a, b, c, y0, p0, p1, p2, y1):
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
def quad_nucl_curve_k(x, a, b, c, y0, p0, p1, p2, y1):
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
  return k_curve(x, a, b, c) * nucl_potential(x, p0, p1, p2, y1) + np.ones(x.size)*y0
# HERE
def quad_nucl_curve_mass(x, a, b, c, y0, p0, p1, p2, y1):
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

def red_chi_sqr(y_calc, y_obs, y_err, nu):
  """
  calculates reduced chi squared of fit (nu = n observations - m fitted parameters)
  """
  return np.sum(np.square((y_obs-y_calc)/y_err))/nu

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
    
    def compute_uncertainties(best_solution):
        """
        Compute parameter uncertainties using advanced numerical differentiation
        with regularization to stabilize the Hessian inversion.

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

        except (np.linalg.LinAlgError, ValueError):
            # If inversion fails, return NaNs for all uncertainties
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

def partial_k(w, M, k, gamma):
    """
    Partial derivative of the Breit-Wigner resonance formula with respect to k.
    """
    numerator = M**2 * gamma**2
    denominator = ((w**2 - M**2)**2 + M**2 * gamma**2)
    return numerator / denominator


def partial_gamma(w, M, k, gamma):
    """
    Partial derivative of the Breit-Wigner resonance formula with respect to gamma.
    """
    numerator_1 = k * 2 * M**2 * gamma
    numerator_2 = k * (M**2 * gamma**2) * 2 * M**2 * gamma
    denominator = ((w**2 - M**2)**2 + M**2 * gamma**2)
    denominator_squared = denominator**2
    return (numerator_1 / denominator) - (numerator_2 / denominator_squared)


def partial_mass(w, M, k, gamma):
    """
    Partial derivative of the Breit-Wigner resonance formula with respect to M.
    """
    numerator_1 = k * (2 * M * gamma**2)
    term1 = -4 * M * (w**2 - M**2)
    term2 = 2 * M * gamma**2
    numerator_2 = k * (M**2 * gamma**2) * (term1 + term2)
    denominator = ((w**2 - M**2)**2 + M**2 * gamma**2)
    denominator_squared = denominator**2
    return (numerator_1 / denominator) - (numerator_2 / denominator_squared)

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

# Table F.1 from XZ's thesis
#def g1f1_quad_new_DIS(x_q2, alpha, a, b, c, d, beta):
#    return (x_q2[0]**alpha) * (a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0]) * (1+(beta/x_q2[1]))
def g1f1_quad_new_DIS(x_q2, alpha, a, b, c, beta):
    return (x_q2[0]**alpha) * (a + b*x_q2[0] + c*x_q2[0]*x_q2[0]) * (1+(beta/x_q2[1]))


# guess form for downward trend at high x - a cubic!
def g1f1_cubic_DIS(x_q2, a, b , c, d, beta):
  return (a + b*x_q2[0] + c*x_q2[0]*x_q2[0] + d*x_q2[0]*x_q2[0]*x_q2[0])*(1+(beta/x_q2[1]))

def residual_function(W, a, b, c, W_transition):
    """A simple polynomial function to fit the residual"""
    return a * (W - W_transition)**2 + b * (W - W_transition) + c

def damping_function(W, W_transition, width):
    """Woods-Saxon function for smooth damping"""
    return 1 / (1 + np.exp((W - W_transition) / width))

##############################################################

def fit_error(x, q2, par, par_sigmas, pcov, partials):
  """
  Equation F.5 from Xiaochao's thesis
  x: X array
  q2: Q2 array
  par: fit parameters
  par_sigmas: list of errors in parameters
  pcov: covariance matrix
  partials: list of partial functions for the fit function
  return: array of errors in fitted points
  """
  # initialize fit variance array
  y_err = np.zeros(len(x))

  for i in range(len(par)):
    y_err += (partials[i](x,q2,par)**2) * par_sigmas[i]**2 * pcov[i][i]

    for j in range(i+1, len(par)):
      if i != j:
          y_err += 2 * (partials[i](x,q2,par) * partials[j](x,q2,par)) * (par_sigmas[i] * par_sigmas[j]) * pcov[i][j]

  return np.sqrt(y_err)

# Function to propagate errors for Breit-Wigner fit
def propagate_bw_error(w, mass, mass_err, k, k_err, gamma, gamma_err):
    """
    Propagate errors for Breit-Wigner fit.

    Parameters:
    - w: W values (array)
    - mass: Mass value from fit
    - mass_err: Error in mass
    - k: k value from fit
    - k_err: Error in k
    - gamma: Gamma value from fit
    - gamma_err: Error in gamma

    Returns:
    - Propagated error for Breit-Wigner fit
    """
    # Partial derivatives for each parameter (mass, k, gamma)
    d_m = (np.gradient(breit_wigner_res(w, mass + mass_err, k, gamma)) - 
           np.gradient(breit_wigner_res(w, mass - mass_err, k, gamma))) / (2 * mass_err)
    d_k = (np.gradient(breit_wigner_res(w, mass, k + k_err, gamma)) - 
           np.gradient(breit_wigner_res(w, mass, k - k_err, gamma))) / (2 * k_err)
    d_gamma = (np.gradient(breit_wigner_res(w, mass, k, gamma + gamma_err)) - 
               np.gradient(breit_wigner_res(w, mass, k, gamma - gamma_err))) / (2 * gamma_err)
    # Propagate errors
    propagated_error = np.sqrt((d_m * mass_err)**2 + (d_k * k_err)**2 + (d_gamma * gamma_err)**2)
    return propagated_error

def damping_function_err(w, w_transition, w_transition_err, damping_width, damping_width_err):
    """
    Calculate error for the damping function using Jacobian.
    
    Parameters:
    - w: W values (array)
    - w_transition: Transition value for W
    - damping_width: Damping resolution width
    
    Returns:
    - Error for damping function
    """

    # Partial derivatives for each parameter (mass, k, gamma)
    dw_trans = (np.gradient(damping_function(w, w_transition + w_transition_err, damping_width)) - 
                 np.gradient(damping_function(w, w_transition - w_transition_err, damping_width))) / (2 * w_transition_err)
    dwidth = (np.gradient(damping_function(w, w_transition, damping_width + damping_width_err)) - 
               np.gradient(damping_function(w, w_transition, damping_width - damping_width_err))) / (2 * damping_width_err)
    
    # Propagate errors
    propagated_error = np.sqrt((dw_trans * w_transition_err)**2 + (dwidth * damping_width_err)**2)
    return propagated_error

def propagate_residual_error(w_res, popt, pcov, residual_function, w_dis_region):
    """
    Propagate errors for the residual fit using Jacobian matrix.
    
    Parameters:
    - w_res: W_res values (array) [shape (1000,)]
    - popt: Optimal parameters from curve fitting [shape (3,)]
    - pcov: Covariance matrix from curve fitting [shape (3, 3)]
    - residual_function: Residual function used in the fit
    - w_dis_region: Region parameter for the residual function
    
    Returns:
    - Propagated errors for residual fit at each w_res point [shape (1000,)]
    """
    
    # Compute the Jacobian matrix (keep original error calculation)
    def jacobian(x, params):
        epsilon = np.sqrt(np.finfo(float).eps)
        return np.array([
            (residual_function(x, *(params + epsilon * np.eye(len(params))[i]), w_dis_region) - 
             residual_function(x, *(params - epsilon * np.eye(len(params))[i]), w_dis_region)) / 
            (2 * epsilon) for i in range(len(params))
        ]).T

    # Compute fit and error bars
    fit = residual_function(w_res, *popt, w_dis_region)
    J = jacobian(w_res, popt)
    fit_var = np.sum(J @ pcov * J, axis=1)
    fit_err = np.sqrt(fit_var)    

    return fit_err

def propagate_dis_error(fit_errs):
    """
    Propagate errors for the DIS fit, handling NaN values in dx and fit_errs.
    
    Parameters:
    - x: x values (array)
    - q2: Q^2 values (array)
    - fit_params: Fit parameters
    - fit_errs: Fit parameter errors (array)
    
    Returns:
    - Propagated errors for DIS fit
    """

    return fit_errs

def propagate_transition_error(w, bw_err, residual_err, damping_res_err, w_res_min, w_res_max, w_dis_transition):
    """
    Propagate errors for the transition between Breit-Wigner and DIS with region-dependent scaling factors.
    The sum of alpha and beta is 1.0 for each region.

    Parameters:
    - w: List of W values (in the region [w_res_min, w_dis_transition])
    - bw_err: List of errors in Breit-Wigner fit
    - residual_err: List of errors in residual fit
    - damping_res_err: List of errors in damping_res function
    - w_res_min: Minimum value of W for the Breit-Wigner region
    - w_res_max: Maximum value of W for the Breit-Wigner region
    - w_dis_transition: W value marking the start of the DIS transition region
    
    Returns:
    - List of propagated errors for the transition
    """
    # Ensure all input lists are of the same length
    if not (len(w) == len(bw_err) == len(residual_err) == len(damping_res_err)):
        raise ValueError("All input lists must have the same length.")
    
    # Initialize the total propagated errors list
    propagated_errors = []

    # Loop through each W value and calculate the propagated error
    for i in range(len(w)):

        # Region 1: [w_res_min, w_res_max], where alpha = 1.0 and beta = 0.0
        if w[i] >= w_res_min and w[i] <= w_res_max:
            alpha, beta = 1.0, 0.0
            error = np.sqrt((alpha * bw_err[i])**2)

        # Region 2: [w_res_max, w_dis_transition], where beta = 1.0 and alpha = 0.0
        elif w[i] > w_res_max and w[i] <= w_dis_transition:
            alpha, beta = 0.0, 1.0
            error = np.sqrt(beta * (residual_err[i]**2 + damping_res_err[i]**2))

        # Append the error to the results list
        propagated_errors.append(error)
    
    return propagated_errors

def propagate_complete_error(w, transition_err, damping_dis_err, dis_err, w_res_min, w_dis_transition, w_dis_region, w_max):
    """
    Propagate errors for the complete fit over a range of W values, including region-dependent scaling factors.
    The sum of alpha, beta, and gamma is 1.0 for each region.

    Parameters:
    - w: List of W values (in the region [w_res_min, w_max])
    - transition_err: List of transition errors
    - damping_dis_err: List of damping DIS errors
    - dis_err: List of DIS errors
    - w_res_min: Minimum value of W for the transition region
    - w_dis_region: W value that separates the different regions
    - w_max: Maximum value of W for the DIS region
    
    Returns:
    - List of propagated errors for the complete fit over all W values
    """
    # Ensure all input lists are of the same length
    if not (len(w) == len(transition_err) == len(damping_dis_err) == len(dis_err)):
        raise ValueError("All input lists must have the same length.")
    
    # Initialize the total propagated errors list
    propagated_errors = []

    # Loop through each W value and calculate the propagated error
    for i in range(len(w)):

        # Region-dependent scaling factors
        if w[i] >= w_res_min and w[i] <= w_dis_transition:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        elif w[i] >= w_dis_transition and w[i] <= w_dis_region:
            alpha, beta, gamma = 0.0, 1.0, 0.0
        else:
            alpha, beta, gamma = 0.0, 0.0, 1.0

        # Compute the propagated error for the current W value
        error = np.sqrt(
            (alpha * transition_err[i])**2 +
            (beta * damping_dis_err[i])**2 +
            (gamma * dis_err[i])**2
        )

        # Append the error to the results list
        propagated_errors.append(error)
    
    return propagated_errors

def calculate_param_error(fit_func, args, param_errs):
    """
    Calculate propagated error for k using numerical differentiation.

    Parameters:
    - fit_func: Function for k (e.g., fit_funcs_k[i])
    - args: Complete list of arguments passed to the function
    - param_errs: Errors in the parameters (same length as args)

    Returns:
    - Propagated error in k
    """
    err_squared = 0.0

    for idx in range(len(param_errs)):
        # Skip args without defined uncertainties
        if param_errs[idx] == 0:
            continue

        # Perturb the current parameter by its uncertainty
        args_plus = args.copy()
        args_minus = args.copy()
        args_plus[idx] += param_errs[idx]
        args_minus[idx] -= param_errs[idx]

        # Compute the partial derivative numerically
        f_plus = fit_func(*args_plus)
        f_minus = fit_func(*args_minus)
        partial_derivative = (f_plus - f_minus) / (2 * param_errs[idx])

        # Add the squared contribution of this term
        err_squared += (partial_derivative * param_errs[idx]) ** 2

    return np.sqrt(err_squared)
