#!/usr/bin/env python
# coding: utf-8

# # Imports and Loading Data
# check for TODO comments to help you get started

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib import colormaps
import pandas as pd
from scipy.interpolate import griddata, interp1d
from scipy.optimize import curve_fit, Bounds, differential_evolution, minimize
from tabulate import tabulate
from scipy.interpolate import splrep, BSpline
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import probplot
from numpy.linalg import LinAlgError
import re
import ast
import os

#from google.colab import drive
#drive.mount('/mnt/drive') # mount google drive - MUST AGREE TO POPUPS

##################################################################################################################################################
# Importing utility functions

from utility import show_pdf_with_evince, most_common_combination

##################################################################################################################################################

# Redefine W-range
w_min = 1.1
w_max = 3.0

# Initial resonance region range (optimized later on)
w_res_min = 1.1
w_res_max = 1.4

##################################################################################################################################################

# In[2]:


# # fix Mingyu W.cal
# # mingyu_df = mingyu_df.drop(columns=["W"])
# mingyu_df["W.cal"] = W_cal(mingyu_df["Q2"], mingyu_df["x"])
# mingyu_df.head(10)
# mingyu_df.to_csv(dir + 'mingyu_g1f1_g2f1_dis.csv', index=False)


# In[3]:


# Load csv files into data frames
dir = 'data/'
e06014_df = pd.read_csv(dir + 'dflay_e06014.csv')
e94010_df = pd.read_csv(dir + 'e94010.csv')
e97110_df = pd.read_csv(dir + 'e97110.csv')
psolva1a2_df = pd.read_csv(dir + 'psolv_e01012_a1a2.csv')
psolvg1g2_df = pd.read_csv(dir + 'psolv_e01012_g1g2.csv')
zheng_df = pd.read_csv(dir + 'zheng_thesis_pub_e99117.csv')
hermes_df = pd.read_csv(dir + 'hermes_2000.csv')
e142_df = pd.read_csv(dir + 'slac_e142.csv')
e154_df = pd.read_csv(dir + 'slac_e154.csv')
e97103_df = pd.read_csv(dir + 'kramer_e97103.csv')

mingyu_df = pd.read_csv(dir + 'mingyu_g1f1_g2f1_dis.csv') # mingyu thesis DIS


# Saikat's data tables for interpolation
# caldata = pd.read_csv(dir + 'saikat_tables/XZ_table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11.csv') #  0.1<Q2<15.0 GeV2
# caldata = pd.read_csv(dir + 'saikat_tables/table_3He_JAM_smeared_kpsv_onshell_ipol1_ipolres1_IA14_SF23_AC11.csv') #  0.001<Q2<5.0 GeV2

# combined g1f1, g2f1, a1, a2 tables
g1f1_df = pd.read_csv(dir + 'g1f1_comb.csv')
g2f1_df = pd.read_csv(dir + 'g2f1_comb.csv')
a1_df = pd.read_csv(dir + 'a1_comb.csv')
a2_df = pd.read_csv(dir + 'a2_comb.csv')


# # Useful Functions

from functions import x_to_W, W_to_x, breit_wigner_res, breit_wigner_wrapper, lin_curve, quad_curve, quadconstr_curve, \
    cubic_curve, exp_curve, cub_exp_curve, quadconstr_exp_curve, nucl_potential, quad_nucl_curve_gamma, quad_nucl_curve_k, quad_nucl_curve_mass, g1f1_quad_DIS, \
    g1f1_quad2_DIS, g1f1_cubic_DIS, fit, red_chi_sqr, fit_with_dynamic_params, weighted_avg, partial_a2, partial_a3, partial_b2, \
    partial_b3, partial_c2, partial_c3, partial_beta2, partial_beta3, partial_d3, partial_x0, partial_y0, partial_c4, partial_beta4, partial_alpha, \
    residual_function, damping_function, g1f1_quad_new_DIS, partial_alpha_new, partial_a_new, partial_b_new, partial_c_new, partial_d_new, partial_beta_new, \
    fit_error, propagate_bw_error, damping_function_err, propagate_dis_error, propagate_residual_error, propagate_transition_error, propagate_complete_error, calculate_param_error

def quad_nucl_curve_constp_gamma(x, a, b, c, y0):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  y0: term to have curve end at a constant value
  """  
  return quad_nucl_curve_gamma(x, a, b, c, y0, P0, P1, P2, Y1)
def quad_nucl_curve_constp_k(x, a, b, c, y0):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  y0: term to have curve end at a constant value
  """  
  return quad_nucl_curve_k(x, a, b, c, y0, P0, P1, P2, Y1)
def quad_nucl_curve_constp_mass(x, a, b, c, y0):
  """
  quadratic * nucl potential form
  x: independent data
  a, b, c: quadratic curve parameters
  y0: term to have curve end at a constant value
  """  
  return quad_nucl_curve_mass(x, a, b, c, y0, P0, P1, P2, Y1)

def lin_nucl_curve_constp(x, a, b):
  """
  linear * nucl potential form
  x: independent data
  a, b, c: linear curve parameters
  y0: term to have curve end at a constant value
  """  
  return lin_nucl_curve(x, a, b)

# # Model Building

# ## Fit $g_1/F_1$ Resonance data with Breit-Wigner distribution
# 
# TODO (DONE): substitute $k_{new}=k/(M^2 * \Gamma^2)$
# 
# \\
# $g_1/F_1 = \frac{k}{(W^2-M^2)^2 + M^2  \Gamma^2}$
# 
# $k$, $\Gamma$ are functions of $Q^2$
# 
# $M$ is the mass of the peak (1232 MeV)

# ### Make dataframe with resonance data and assign labels for each Q2 bin

# combine Mingyu data and g1f1_df
temp_df = pd.DataFrame(
    {
        "Q2": mingyu_df["Q2"],
        "W": mingyu_df["W.cal"],
        "X": mingyu_df["x"],
        "G1F1": mingyu_df["g1F1_3He"],
        "G1F1.err": mingyu_df["g1f1.err"],
        "Label": ["Mingyu" for x in range(len(mingyu_df["Q2"]))],
    }
)

# temp_df.head()

dis_df = g1f1_df
#dis_df = pd.concat([temp_df, g1f1_df], ignore_index=True) # add Mingyu data
print(dis_df.head(100))


# In[18]:


# make dataframe of DIS values (W>2 && Q2>1)
dis_df = dis_df[dis_df['W']>2.0]
dis_df = dis_df[dis_df['Q2']>1.0]

dis_df.head(100)

# independent variable data to feed to curve fit, X and Q2
indep_data = [dis_df['X'], dis_df['Q2']]

# Create a PdfPages object to manage the PDF file
with PdfPages("plots/g1f1_fits.pdf") as pdf:

    # fit the g1f1 DIS data with constrained quadratic form
    '''
    quad_new_init = [0.66084205, -0.23606144,  1.25499178, -2.65987975,  0.09666789]
    quad_new_constr = None
    
    '''
    quad_new_init = [0.66084205, -0.23606144,  -1.25499178, 2.65987975,  0.09666789]
    quad_new_constr = ([-np.inf, -np.inf, -np.inf, 0.0, -np.inf],
                    [np.inf, np.inf, 0.0, np.inf, np.inf]) # min and max bounds on alpha, a, b, c, and beta
    #'''

    quad_param_names = ["alpha", "a", "b", "c", "beta"]
    
    def optimize_init_params(func, x, y, y_err, params_init, bounds, n_tries=1000):
        """
        Optimize initial parameters by trying multiple starting points
        """
        best_chi2 = np.inf
        best_params = params_init

        # Generate random starting points within bounds
        for _ in range(n_tries):
            try:
                random_init = np.array([
                    np.random.uniform(low, high) 
                    for low, high in zip(bounds[0], bounds[1])
                ])
                params, _ = curve_fit(func, x, y, p0=random_init, sigma=y_err, 
                                    bounds=bounds, maxfev=50000)

                # Calculate chi2 for this fit
                y_fit = func(x, *params)
                chi2 = np.sum(((y - y_fit) / y_err) ** 2) / (len(y) - len(params))

                if abs(chi2 - 1) < abs(best_chi2 - 1):
                    best_chi2 = chi2
                    best_params = params
            except:
                continue

        return best_params

    def fit_new(func, x, y, y_err, params_init, param_names, constr=None, silent=False, optimize=True):
        """
        Enhanced fitting function with parameter optimization
        """
        if constr is None:
            constr = ([-np.inf for _ in param_names],
                     [np.inf for _ in param_names])

        # Optimize initial parameters if requested
        if optimize:
            params_init = optimize_init_params(func, x, y, y_err, params_init, constr)

        # Perform final fit with optimized initial parameters
        params, covariance = curve_fit(func, x, y, p0=params_init, sigma=y_err, 
                                     bounds=constr, maxfev=50000)

        param_sigmas = [np.sqrt(covariance[i][i]) for i in range(len(params))]
        table = [[f"{params[i]:.5f} ± {param_sigmas[i]:.5f}" 
                  for i in range(len(params))]]

        # Calculate reduced chi squared
        nu = len(y) - len(param_names)
        y_fit = func(x, *params)
        chi_2 = np.sum(((y - y_fit) / y_err) ** 2) / nu

        if not silent:
            print(tabulate(table, param_names, tablefmt="fancy_grid"))
            print(f"$\chi_v^2$ = {chi_2:.2f}")

        return params, covariance, param_sigmas, chi_2

    par_quad, cov_quad, par_err_quad, chi2_quad = fit_new(g1f1_quad_new_DIS, indep_data,
                                                          dis_df['G1F1'],
                                                          dis_df['G1F1.err'],
                                                          quad_new_init,
                                                          quad_param_names,
                                                          constr=quad_new_constr)
    #["alpha", "a", "b", "c", "d", "beta"])

    def covariance_to_correlation(cov_matrix):
        # Calculate the standard deviations (square root of variances)
        std_devs = np.sqrt(np.diag(cov_matrix))

        # Create the correlation matrix
        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

        return correlation_matrix

    corr_quad = covariance_to_correlation(cov_quad)
                                                      
    print("\n\n", "-"*25)
    print("Best set of new DIS parameters with uncertainties:")
    for names, param, error in zip(quad_param_names, par_quad, par_err_quad):
        print(f"{names}:  {param:.4e} ± {error:.4e}")
        if names == "beta":
            beta_val = param
    # Print covariance matrix
    print("Covariance matrix:")
    for row in cov_quad:
        print(" ".join(f"{val:6.2e}" for val in row))
    # Print correlation matrix
    print("\nCorrelation matrix:")
    for row in corr_quad:
        print(" ".join(f"{val:6.2e}" for val in row))
    print("-"*25, "\n\n")

    dis_fit_init = par_quad
    
    fig, axs = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    q2_interp = interp1d(dis_df['X'].values, dis_df['Q2'].values, kind='linear')
    x_dense = np.linspace(dis_df['X'].min(), dis_df['X'].max(), 10000)
    #q2_dense = q2_interp(x_dense)
    q2_dense = np.full(x_dense.size, 5.0) # array of q2 = 5.0 GeV^2
    fit_vals = g1f1_quad_new_DIS([x_dense, q2_dense], *par_quad)
    
    axs[0].errorbar(dis_df['X'], dis_df['G1F1'], yerr=dis_df['G1F1.err'], 
                   fmt='o', color='black', label='Data', markersize=4)
    axs[0].plot(x_dense, fit_vals, 'r-', label=f'Fit ($\chi^2_{{red}}$ = {chi2_quad:.2f})')
    
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('$g_1F_1$')
    axs[0].legend()
    axs[0].grid(True)
    
    residuals = (dis_df['G1F1'] - g1f1_quad_new_DIS([dis_df['X'], dis_df['Q2']], *par_quad)) / dis_df['G1F1.err']
    axs[1].scatter(dis_df['X'], residuals, color='black', s=20)
    axs[1].axhline(y=0, color='r', linestyle='-', alpha=0.5)
    
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Residuals ($\\sigma$)')
    axs[1].grid(True)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")

    ###

    # ## Fit $g_1/F_1$ DIS data with neutron form from Xiaochao's thesis
    # $g_1/F_1 = (a+bx+cx^2)(1+β/Q^2)$
    # 
    # Downward trend cubic form
    # $g_1/F_1 = (a+bx+cx^2+dx^3)(1+β/Q^2)$

    # In[17]:

    # In[19]:


    # independent variable data to feed to curve fit, X and Q2
    #indep_data = [dis_df['X'], dis_df['Q2']]

    '''
    # fit g1f1 DIS data with cubic form
    # initial guess for a,b,c,d,beta
    cubic_dis_init = (-.03, -.02, 0.3, -.22, 0.1)
    par_cub, cov_cub, par_err_cub, chi2_cub = fit(g1f1_cubic_DIS, indep_data,
                                                  dis_df['G1F1'],
                                                  dis_df['G1F1.err'],
                                                  cubic_dis_init,
                                                  ["a", "b", "c", "d", "beta"])
    '''

    '''
    # fit the g1f1 DIS data with constrained quadratic form
    quad2_init = [0.16424, -.02584, 0.16632, 0.11059]
    quad2_constr = ([0.12, -.05, 0.10, 0.105],
                    [0.20, -.00, 0.20, 0.115]) # min and max bounds on x0, y0, c, and beta
    par_quad, cov_quad, par_err_quad, chi2_quad = fit(g1f1_quad2_DIS, indep_data,
                                                      dis_df['G1F1'],
                                                      dis_df['G1F1.err'],
                                                      quad2_init,
                                                      ["x0", "y0", "c", "beta"],
                                                      constr=quad2_constr)
    '''
    
    # In[20]:


    # Generate fitted curve using the fitted parameters for constant q2
    x = np.linspace(0,1.0,1000, dtype=np.double)
    q2 = np.full(x.size, 5.0) # array of q2 = 5.0 GeV^2

    #args3 = [[x, q2]] + [p for p in par_cub]
    #args2 = [[x, q2]] + [p for p in par_quad]

    args_new = [[x, q2]] + [p for p in dis_fit_init]    

    #cubic_fit_curve = g1f1_cubic_DIS(*args3)
    #quad_fit_curve = g1f1_quad2_DIS(*args2)
    #quad_fit_curve = g1f1_quad_new_DIS(*args2)

    quad_new_fit_curve = g1f1_quad_new_DIS(*args_new)

    # In[21]:

    # # list of partials for parameters (index 0 is for a -> index 3 is for beta)
    partials2 = [partial_a2, partial_b2, partial_c2, partial_beta2]

    # list of partials for parameters (index 0 is for a -> index 4 is for beta)
    partials3 = [partial_a3, partial_b3, partial_c3, partial_d3, partial_beta3]

    # list of partials for constrained quadratic form
    partials4 = [partial_x0, partial_y0, partial_c4, partial_beta4]
    
    # Table F.1 from XZ's thesis
    #partials_new = [partial_alpha_new, partial_a_new, partial_b_new, partial_c_new, partial_d_new, partial_beta_new]
    partials_new = [partial_alpha_new, partial_a_new, partial_b_new, partial_c_new, partial_beta_new]

    # In[22]:


    #cubic_fit_err = fit_error(x, q2, par_cub, par_err_cub, cov_cub, partials3)

    # quad_fit_err = fit_error(x, q2, par_quad, par_err_quad, cov_quad, partials2)
    ##quad_fit_err = fit_error(x, q2, par_quad, par_err_quad, cov_quad, partials_new)
    quad_fit_err = fit_error(x, q2, par_quad, par_err_quad, corr_quad, partials_new)

    # In[23]:


    # make figure
    fig, (ax1) = plt.subplots(1, 1, figsize=(18,10))

    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    colors = ("red", "darkorange", "limegreen",
              "darkslategray", "darkblue", "rebeccapurple",
              "darkmagenta")

    # plot w/ labels
    for i,l in enumerate(dis_df['Label'].unique()):
      ax1.errorbar(dis_df['X'][dis_df['Label']==l],
                    dis_df['G1F1'][dis_df['Label']==l],
                    yerr=dis_df['G1F1.err'][dis_df['Label']==l],
                    fmt=m_type, color=colors[i], markersize=m_size, capsize=cap_size,
                    label=l, capthick=cap_thick)


    # plot fit and fit error
    #ax1.plot(x, cubic_fit_curve, label="Cubic Fit, $Q^2=5\ {GeV}^2$" + f" $\chi_v^2={chi2_cub:.2f}$")
    #ax1.fill_between(x, cubic_fit_curve-cubic_fit_err, cubic_fit_curve+cubic_fit_err, alpha=0.5)
    ax1.plot(x, quad_new_fit_curve, label="Quadratic Fit, $Q^2=5\ {GeV}^2$" + f" $\chi_v^2={chi2_quad:.2f}$", color="darkred")
    ax1.fill_between(x, quad_new_fit_curve-quad_fit_err, quad_new_fit_curve+quad_fit_err, alpha=0.5, color="darkred")
    ax1.axhline(y=0, color="black", linestyle="dashed")

    ax1.legend()
    fig.tight_layout()
    fig.text(0.53, 0.001, "X", ha='center', va='center')
    fig.text(0.001, 0.56, '$g_1^{^{3}He}/F_1^{^{3}He}$', ha='center', va='center', rotation='vertical')

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")
    
    # In[5]:

    # make dataframe of Resonance values (1<W<2)
    res_df = g1f1_df[g1f1_df['W']<2.0]
    res_df = res_df[res_df['W']>1.0]        
    
    # drop Flay data
    res_df = res_df.drop(res_df[res_df.Label == "Flay E06-014 (2014)"].index)

    # drop Kramer data
    res_df = res_df.drop(res_df[res_df.Label == 'Kramer E97-103 (2003)'].index)

    q2_labels = []
    # go through each experiment and divide into q2 bins
    for name in res_df['Label'].unique():
      data = res_df[res_df['Label']==name]
      if name == "Flay E06-014 (2014)":
        # not using Flay data
        continue
        # # split Flay data into buckets
        # n_bins = 2
        # q2_ranges = np.linspace(data['Q2'].min(), data['Q2'].max(), n_bins+1)
        # print(q2_ranges)
        # for i in range(n_bins):
        #   max = q2_ranges[i+1]
        #   min = q2_ranges[i]
        #   bucket = data[data["Q2"]<=max]
        #   bucket = bucket[bucket["Q2"]>=min]
        #   mean_q2 = bucket['Q2'].mean()
        #   q2_labels += [f"{name} ${min:.2f} \leq Q^2 \leq {max:.2f}\ GeV^2$" for x in range(len(bucket))]

      else:
        for q2 in data['Q2'].unique():
          q2_labels += [f"{name} $Q^2={q2}\ GeV^2$" for x in range(len(data[data['Q2']==q2]))]
          print(name, q2, len(data[data['Q2']==q2]))


    res_df['Q2_labels'] = q2_labels
    # res_df = res_df.drop(labels=['Q2 Buckets'])
    # res_df.head(1000)
    n_bins = len(res_df['Q2_labels'])


    # ### Plot each Q2 bin

    # In[7]:


    # formatting variables
    m_size = 6
    cap_size = 2
    cap_thick = 1
    m_type = '.'
    
    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

    # make figure
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//4 + 1
    fig, axs = plt.subplots(num_plots//4 + 1, 4, figsize=(20,n_rows*5))

    # plot resonance w/ labels
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//4
      col = i%4
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                  res_df['G1F1'][res_df['Q2_labels']==l],
                  yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                  fmt=m_type, color=colors[i], markersize=m_size, capsize=cap_size,
                  label=l, capthick=cap_thick)

      axs[row,col].legend()
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.1)

      fig.tight_layout()
      fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center')
      fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")


    # ### Resonance Fitting and Plotting Functions

    # In[8]:


    ## Plotting Function
    def plot_res_fits(w_bounds, M, region_name, p_df):
      # formatting variables
      m_size = 6
      cap_size = 2
      cap_thick = 1
      m_type = '.'
      
      colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
                 "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
                "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

      # make figure
      n_col = 5
      num_plots = len(res_df['Q2_labels'].unique())
      n_rows = num_plots//n_col + 1
      fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6,n_rows*6))

      # make fit curves and plot with data
      for i,l in enumerate(res_df['Q2_labels'].unique()):
        row = i//n_col
        col = i%n_col

        # params for fit with 3 parameters M, k, gamma (variable mass)
        params = [p_df[p_df['Label']==l][f"{param_names[j]}"].unique()[0] for j in range(n_params)]

        # params for fit with 2 parameters k, gamma (fixed mass)
        params_constm = [p_df[p_df['Label']==l][f"{param_names[j+1]}_constM"].unique()[0] for j in range(n_params-1)]

        # Generate fitted curve using the fitted parameters
        w = np.linspace(w_bounds[i][0], w_bounds[i][1], 1000, dtype=np.double)

        # make fitted curve for all three parameter fit
        if 0 not in params:
          fit = breit_wigner_res(w, params[0], params[1], params[2])
          axs[row, col].plot(w, fit, color=colors[1], markersize=m_size,
                            label="Fit")

        # make fitted curve for two parameter fit (k, gamma)
        if 0 not in params_constm:
          fit_constm = breit_wigner_res(w, M, params_constm[0], params_constm[1])
          axs[row, col].plot(w, fit_constm, color=colors[2], markersize=m_size,
                            label=f"Fit M={M}", linestyle='dashed')


        # plot the data
        axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                      res_df['G1F1'][res_df['Q2_labels']==l],
                      yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                      fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                      capthick=cap_thick)

        axs[row,col].legend()
        # set axes limits
        axs[row,col].axhline(0, color="black", linestyle="--")
        axs[row,col].set_ylim(-.15,0.1)
        axs[row,col].set_xlim(0.9,2.1)
        axs[row,col].set_title(l)

      fig.tight_layout()
      fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
      fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")


    ## Fitting Function
    def fit_breit_wigner(pdf, w_bounds, M, region_name):
      """
      w_bounds: list of bounds of W for fitting for each bin (ex: [(w min, w max),...])
      M: actual mass of resonance
      init: initial guesses for M, k, gamma
      region_name: unique name for resonance region that is being fitted

      returns a new dataframe with the parameters
      """
      # lists to make dataframe
      q2_list = []
      label_list = []
      exp_list = []
      # lists to hold k, gamma and their errors for constant M fits
      constM_par_lists = [[],[]]
      constM_par_err_lists = [[],[]]
      # lists to hold M, k, gamma and their errors
      par_lists = [[],[],[]]
      par_err_lists = [[],[],[]]

      # go through each experiment for constant Q2 and do a fit
      for i, name in enumerate(res_df['Q2_labels'].unique()):
        n_points = len(res_df['W'][res_df['Q2_labels']==name])
        init = (m_init[i], k_init[i], gamma_init[i])

        label_list.append(name) # add name to list of labels

        # get Q2 for this bin
        if "Flay" in name:
          q2 = res_df['Q2'][res_df['Q2_labels']==name].mean()
        else:
          q2 = res_df['Q2'][res_df['Q2_labels']==name].unique()[0]
        q2_list.append(q2)

        # add experiment name to list
        exp_list.append(res_df['Label'][res_df['Q2_labels']==name].unique()[0])

        # chop off data outside w_min and w_max
        w = res_df['W'][res_df['Q2_labels']==name][res_df['W']<w_bounds[i][1]][res_df['W']>w_bounds[i][0]]
        g1f1 = res_df['G1F1'][res_df['Q2_labels']==name][res_df['W']<w_bounds[i][1]][res_df['W']>w_bounds[i][0]]
        g1f1_err = res_df['G1F1.err'][res_df['Q2_labels']==name][res_df['W']<w_bounds[i][1]][res_df['W']>w_bounds[i][0]]

        # default bounds for parameters M, k, gamma
        par_bounds = ([-np.inf, -np.inf, -np.inf],
                        [np.inf, np.inf, np.inf])

        if "Solvg." in name:
          # bound gamma
          par_bounds = ([-np.inf, -np.inf, .15],
                        [np.inf, np.inf, .4])

        try:
            # fit for (M, k, gamma) and get parameters and covariance matrix
            params, pcov, perr, chi2 = fit(breit_wigner_res, w, g1f1, g1f1_err, init, ["M", "k", "gamma"], par_bounds, silent=True)
        except Exception as e:
          print(f"{name} Fit Failed\n   {e}")
          params = np.zeros(n_params)
          pcov = np.zeros((n_params, n_params))

        # add parameters and their errors to lists for M, k, gamma
        for j in range(len(params)):
          par_lists[j].append(params[j])
          par_err_lists[j].append(perr[j])

        try:
          # try fitting with constant M
          params_constm, pcov_constm, perr_constm, chi2_constm = fit(breit_wigner_wrapper(M), w, g1f1, g1f1_err, init[1:], ["k", "gamma"], [par_bounds[0][1:],par_bounds[1][1:]], silent=True)
        except Exception as e:
          print(f"{name} Constant M Fit Failed\n   {e}")
          params_constm = np.zeros(n_params-1)
          pcov_constm = np.zeros((n_params-1, n_params-1))

        # add parameters and their errors to lists for k & gamma for constant M fits
        for j in range(len(params_constm)):
          constM_par_lists[j].append(params_constm[j])
          constM_par_err_lists[j].append(perr_constm[j])

        table = [
            [f"{params[j]:.5f} ± {perr[j]:.5f}" for j in range(n_params)] + [f"{chi2:.2f}"],
            [f"{M}"] + [f"{params_constm[j]:.5f} ± {perr_constm[j]:.5f}" for j in range(n_params - 1)] + [f"{chi2_constm:.2f}"]
            ]

        # Print the table
        print(f"{name} Fit Parameters")
        # header is param_names
        print(tabulate(table, param_names + ["$\chi_v^2$"], tablefmt="fancy_grid"))

        # Add diagnostic plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f"Diagnostic Plots for {name}")

        # Plot 1: Data and fitted curve
        axs[0, 0].errorbar(w, g1f1, yerr=g1f1_err, fmt='o', label='Data')
        w_fine = np.linspace(w.min(), w.max(), 1000)
        axs[0, 0].plot(w_fine, breit_wigner_res(w_fine, *params), 'r-', label='Fit')
        axs[0, 0].set_xlabel('W')
        axs[0, 0].set_ylabel('G1F1')
        axs[0, 0].legend()
        axs[0, 0].set_title('Data and Fitted Curve')

        # Plot 2: Residuals (with normalization by error)
        residuals = g1f1 - breit_wigner_res(w, *params)  # Raw residuals
        normalized_residuals = residuals / g1f1_err  # Normalized by error

        # Plot normalized residuals
        axs[0, 1].scatter(w, normalized_residuals)
        axs[0, 1].axhline(y=0, color='r', linestyle='--', label='Zero Residual')
        axs[0, 1].set_xlabel('W')
        axs[0, 1].set_ylabel('Normalized Residual')
        axs[0, 1].set_title('Normalized Residuals vs W')
        axs[0, 1].legend()

        # Plot 3: Q-Q plot of normalized residuals        
        (osm, osr), _ = probplot(normalized_residuals)
        axs[1, 0].plot(osm, osr, 'o')
        axs[1, 0].plot(osm, osm, 'r--')
        axs[1, 0].set_xlabel('Theoretical Quantiles')
        axs[1, 0].set_ylabel('Sample Quantiles')
        axs[1, 0].set_title('Q-Q Plot of Normalized Residuals')

        # Plot 4: Reduced Chi-squared contributions
        chi_squared = (normalized_residuals ** 2)  # Now using normalized residuals
        dof = len(w) - n_params  # Degrees of freedom
        reduced_chi_squared = chi_squared / dof

        # Plot reduced chi-squared contributions
        axs[1, 1].scatter(w, reduced_chi_squared)
        axs[1, 1].axhline(y=1, color='r', linestyle='--', label='χ²ᵣ = 1')
        axs[1, 1].set_xlabel('W')
        axs[1, 1].set_ylabel('Reduced χ² Contribution')
        axs[1, 1].set_title('Reduced χ² Contributions per Data Point')
        axs[1, 1].legend()

        # Adjust y-axis to show the line at unity clearly
        y_max = max(2, max(reduced_chi_squared) * 1.1)  # Ensure visibility up to χ²ᵣ = 2
        axs[1, 1].set_ylim(0, y_max)

        plt.tight_layout()
        # Save figures
        pdf.savefig(fig, bbox_inches="tight")

      # make lists into dataframe
      params_df = pd.DataFrame({"Q2": q2_list,
                                "Experiment": exp_list,
                                "Label": label_list,
                                "k_constM": constM_par_lists[0],
                                "k_constM.err": constM_par_err_lists[0],
                                "gamma_constM": constM_par_lists[1],
                                "gamma_constM.err": constM_par_err_lists[1],
                                "M": par_lists[0],
                                "M.err": par_err_lists[0],
                                "k": par_lists[1],
                                "k.err": par_err_lists[1],
                                "gamma": [abs(par) for par in par_lists[2]],
                                "gamma.err": par_err_lists[2]})

      # plot
      plot_res_fits(w_bounds=w_bounds, M=M, region_name=region_name, p_df=params_df)

      return params_df

    # ### Fit and Plot 1232 MeV "Delta" Resonance - outputs fit parameters for each bin and plots them

    # In[9]:

    param_names = ["M", "k", "gamma"]
    n_params = len(param_names)

    ## fit W=1.232 resonance
    # initial guesses for k and M
    k_init = [-.025, -.06, -.01, .02,
              -.1, -.08, -.08, -.07,
              -.06, -.06, -.05, -.2,
              -.2, -.15, -.14, -.13,
              -.13, -0.1, .01]

    m_init = [1.3, 1.35, 1.2, 1.25,
              1.23, 1.23, 1.23, 1.23,
              1.2, 1.22, 1.22, 1.2,
              1.22, 1.25, 1.3, 1.3,
              1.3, 1.3, 1.5]

    gamma_init = [0.1, 0.3, 0.1, 0.1,
              0.1, 0.1, 0.1, 0.1,
              0.1, 0.1, 0.1, 0.1,
              0.1, 0.2, 0.2, 0.2,
              0.2, 0.1, 0.1]

    '''
    w_lims = [(1.0, 1.4), (1.0, 1.4), (1.0, 1.4), (1.0, 1.4),
              (1.1, 1.4), (1.1, 1.4), (1.0, 1.4), (1.1, 1.35),
              (1.05, 1.4), (1.05, 1.4), (1.0, 1.5), (1.05, 1.5),
              (1.0, 1.45), (1.0, 1.5), (1.0, 1.5), (1.0, 1.5),
              (1.0, 1.5), (1.0, 1.65), (1.0, 1.8)]
    '''

    # RLT (10/16/2024)
    w_lims = [(1.125, 1.4), (1.125, 1.4), (1.100, 1.4), (1.100, 1.4),
              (1.100, 1.4), (1.100, 1.4), (1.100, 1.4), (1.100, 1.35),
              (1.085, 1.4), (1.085, 1.4), (1.085, 1.5), (1.100, 1.5),
              (1.100, 1.45), (1.100, 1.5), (1.100, 1.5), (1.100, 1.5),
              (1.100, 1.5), (1.100, 1.65), (1.100, 1.8)]
    
    gamma_bounds = ()

    delta_par_df = fit_breit_wigner(pdf, w_bounds=w_lims, M=1.232, region_name="1232MeV")
    # delta_par_df.head(50)


    # ### Delta peak visible in constant M fits for E94-010 and E97-110 - plot k and γ from these fits vs $Q^2$
    # add k's from Solvg. constrained gamma fits

    # In[10]:


    # # delta_par_df.head(100)

    # # reorder by increasing Q2
    # df = delta_par_df.sort_values(by=['Q2'])
    # df.head(20)


    # In[11]:


    # plot M, k, gamma vs Q2 from variable M fit
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # maintain distinct colors between plots by keeping track of the index in the color map
    color_index = 0

    # plot all the parameters vs Q2
    for i, exp_name in enumerate(delta_par_df["Experiment"].unique()):
      axs[0].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["k"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["k.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)

      axs[1].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)
      
      axs[2].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["M"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["M.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)

    axs[0].set_ylabel("k")
    axs[1].set_ylabel("$\Gamma$")
    axs[2].set_ylabel("M")

    axs[0].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")

    
    # Averaging points for spline - ignore EXCEPT for part adding fictitious points at high Q2

    # In[12]:

    # x = list(res_df["Q2"].unique())
    # k_unique = []
    # k_err_unique = []
    # gamma_unique = []
    # gamma_err_unique = []

    # # go through each unique Q2 value and do weighted average of the points
    # # weight = 1/err
    # for Q2 in x:
    #   # average k's and their errors
    #   k_avg = weighted_avg(df[df["Q2"]==Q2]["k"], w=1/df[df["Q2"]==Q2]["k.err"])
    #   k_unique.append(k_avg)
    #   k_avg_err = (1/len(df[df["Q2"]==Q2]["k.err"])) * np.sqrt(np.sum(df[df["Q2"]==Q2]["k.err"]**2))
    #   k_err_unique.append(k_avg_err)

    #   # average gammas and their errors
    #   gamma_avg = weighted_avg(df[df["Q2"]==Q2]["gamma"], w=(1/df[df["Q2"]==Q2]["gamma.err"]))
    #   gamma_unique.append(gamma_avg)
    #   gamma_avg_err = (1/len(df[df["Q2"]==Q2]["gamma.err"])) * np.sqrt(np.sum(df[df["Q2"]==Q2]["gamma.err"]**2))
    #   gamma_err_unique.append(gamma_avg_err)

    #   # print(Q2)
    #   # print(f"  {k_avg:.5f}±{k_avg_err:.5f}")
    #   # print(f"  {gamma_avg:.5f}±{gamma_avg_err:.5f}")

    # add fictitious end points to control behavior at high Q2 - TODO (DONE): determine adequate value for gamma from E94-010 points and apply to delta par df 
    # x0 = 4.0
    # for i in range(7):
    #   x.append(x0+i)
    #   k_unique.append(0.0)
    #   k_err_unique.append(.001)
    #   gamma_unique.append(0.2)
    #   gamma_err_unique.append(.1)

    
    # RLT (9/23/2024)

    x = list(delta_par_df["Q2"].unique())
    k_unique = []
    k_err_unique = []
    gamma_unique = []
    gamma_err_unique = []
    mass_unique = []
    mass_err_unique = []    
    
    #   # go through each unique Q2 value and do weighted average of the points
    for Q2 in x:
        #   # average k's and their errors
        k_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["k"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])
        k_unique.append(k_avg)
        k_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["k.err"]**2))
        k_err_unique.append(k_avg_err)

        #   # average gammas and their errors
        gamma_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["gamma"], w=(1/delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]))
        gamma_unique.append(gamma_avg)
        gamma_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["gamma.err"]**2))
        gamma_err_unique.append(gamma_avg_err)

        mass_avg = weighted_avg(delta_par_df[delta_par_df["Q2"]==Q2]["M"], w=1/delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])
        mass_unique.append(mass_avg)
        mass_avg_err = (1/len(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"])) * np.sqrt(np.sum(delta_par_df[delta_par_df["Q2"]==Q2]["M.err"]**2))
        mass_err_unique.append(mass_avg_err)
        
    #   # print(Q2)
    #   # print(f"  {k_avg:.5f}±{k_avg_err:.5f}")
    #   # print(f"  {gamma_avg:.5f}±{gamma_avg_err:.5f}")
    
    x0 = 4.0
    for i in range(7):
      x.append(x0+i)
      k_unique.append(0.0)
      k_err_unique.append(k_avg_err)
      gamma_unique.append(0.25)
      gamma_err_unique.append(gamma_avg_err)
      mass_unique.append(1.232)
      mass_err_unique.append(mass_avg_err)

    # In[13]:

    # try spline fit of the points for k and gamma
    N = len(k_unique)
    #print(N-np.sqrt(2*N), N+np.sqrt(2*N)) #ideal s range
    sk = 4.0
    sg = 4.0
    sm = 4.0
    tck_k = splrep(x=x, y=k_unique, w=1/np.array(k_err_unique), s=sk, k=3)
    tck_gamma = splrep(x=x, y=gamma_unique, w=1/np.array(gamma_err_unique), s=sg, k=3)
    tck_mass = splrep(x=x, y=mass_unique, w=1/np.array(mass_err_unique), s=sm, k=3)

    # Generate fitted curves using the fitted parameters
    #q2 = np.linspace(0.0, delta_par_df["Q2"].max()+3.0, 1000, dtype=np.double)
    q2 = np.linspace(0.1, delta_par_df["Q2"].max()+3.0, 1000, dtype=np.double)

    # plot k and gamma vs Q2 from variable M fit

    '''
    
    fig, axs = plt.subplots(1, 3, figsize=(18,10))
    k_list = []
    k_err_list = []
    gamma_list = []
    gamma_err_list = []
    mass_list = []
    mass_err_list = []
    q2_list = []
    label_list = []

    # maintain distinct colors between plots by keeping track of the index in the color map
    color_index = 0

    # # drop Q2 = 3.3 GeV2 fit for Solvg. since k shouldn't be positive
    # delta_par_df = delta_par_df.drop(delta_par_df[delta_par_df.Label == "Solvg. E01-012 (2006) $Q^2=3.3\ GeV^2$"].index)

    # plot all the parameters vs Q2
    for i, exp_name in enumerate(delta_par_df["Experiment"].unique()):
      axs[0].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["k"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["k.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)

      axs[1].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["gamma.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)

      axs[2].errorbar(delta_par_df[delta_par_df["Experiment"]==exp_name]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==exp_name]["M"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==exp_name]["M.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=exp_name, capthick=cap_thick)
      
    axs[0].errorbar(x[0:], k_unique[0:], yerr=k_err_unique[0:], fmt=m_type,
                    markersize=m_size, capsize=cap_size,
                    label="Combined for Spline", capthick=cap_thick, alpha=0.35)

    axs[1].errorbar(x[0:], gamma_unique[0:], yerr=gamma_err_unique[0:], fmt=m_type,
                    markersize=m_size, capsize=cap_size,
                    label="Combined for Spline", capthick=cap_thick, alpha=0.35)

    axs[2].errorbar(x[0:], mass_unique[0:], yerr=mass_err_unique[0:], fmt=m_type,
                    markersize=m_size, capsize=cap_size,
                    label="Combined for Spline", capthick=cap_thick, alpha=0.35)
    
    axs[0].set_ylabel("k")
    axs[1].set_ylabel("$\Gamma$")
    axs[2].set_ylabel("M")

    axs[0].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0, color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)
    # axs[0].set_xlim(-.1, 1.0)

    # # plot splines - run next cell before uncommenting this part
    axs[0].plot(q2, BSpline(*tck_k)(q2), '-', label=f'Spline s={sk}')
    axs[1].plot(q2, BSpline(*tck_gamma)(q2), '-', label=f'Spline s={sg}')
    axs[2].plot(q2, BSpline(*tck_mass)(q2), '-', label=f'Spline s={sg}')
    
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")
    '''

    # A bunch of different fits I tried... just keep Quad*Nucl Potential (I say nuclear potential but it's Woods-Saxon). TODO (DONE): fit M, k, gamma from the floating Delta fits

    # In[14]:

    '''
    # fit k, gamma, mass with line
    print("K Linear Fit Params")
    k_lin_par, k_lin_cov, k_lin_err, k_lin_chi2 = fit(lin_curve, delta_par_df["Q2"],
                                                      delta_par_df["k"],
                                                      delta_par_df["k.err"],
                                                      params_init=(0,1),
                                                      param_names=["a", "b"])
    print("Gamma Linear Fit Params")
    gamma_lin_par, gamma_lin_cov, gamma_lin_err, gamma_lin_chi2 = fit(lin_curve, delta_par_df["Q2"],
                                                              delta_par_df["gamma"],
                                                              delta_par_df["gamma.err"],
                                                              params_init=(0,1),
                                                              param_names=["a", "b"])
    print("Mass Linear Fit Params")
    mass_lin_par, mass_lin_cov, mass_lin_err, mass_lin_chi2 = fit(lin_curve, delta_par_df["Q2"],
                                                              delta_par_df["M"],
                                                              delta_par_df["M.err"],
                                                              params_init=(0,1),
                                                              param_names=["a", "b"])


    # fit k and gamma with quadratic curve
    print("K Quadratic Fit Params")
    k_quad_par, k_quad_cov, k_quad_err, k_quad_chi2 = fit(quad_curve, delta_par_df["Q2"],
                                                          delta_par_df["k"],
                                                          delta_par_df["k.err"],
                                                          params_init=(0,0,0),
                                                          param_names=["a", "b", "c"])
    print("Gamma Quadratic Fit Params")
    gamma_quad_par, gamma_quad_cov, gamma_quad_err, gamma_quad_chi2 = fit(quad_curve, delta_par_df["Q2"],
                                                                  delta_par_df["gamma"],
                                                                  delta_par_df["gamma.err"],
                                                                  params_init=(0,0,0),
                                                                  param_names=["a", "b", "c"])


    # fit k and gamma with cubic curve
    print("K Cubic Fit Params")
    k_cub_par, k_cub_cov, k_cub_err, k_cub_chi2 = fit(cubic_curve, delta_par_df["Q2"],
                                                      delta_par_df["k"], delta_par_df["k.err"],
                                                      params_init=(0,0,0,0),
                                                      param_names=["a", "b", "c", "d"])
    print("Gamma Cubic Fit Params")
    gamma_cub_par, gamma_cub_cov, gamma_cub_err, gamma_cub_chi2 = fit(cubic_curve, delta_par_df["Q2"],
                                                              delta_par_df["gamma"],
                                                              delta_par_df["gamma.err"],
                                                              params_init=(0,0,0,0),
                                                              param_names=["a", "b", "c", "d"])

    # fit k and gamma with exponential curve
    print("K Exponential Fit Params")
    try:
      k_exp_par, k_exp_cov, k_exp_err, k_exp_chi2 = fit(exp_curve, delta_par_df["Q2"],
                                                        delta_par_df["k"], delta_par_df["k.err"],
                                                        params_init=(0,0,0),
                                                        param_names=["a", "b", "c"])
    except Exception:
      print(" Fit failed")


    print("Gamma Exponential Fit Params")
    gamma_exp_par, gamma_exp_cov, gamma_exp_err, gamma_exp_chi2 = fit(exp_curve, delta_par_df["Q2"],
                                                              delta_par_df["gamma"],
                                                              delta_par_df["gamma.err"],
                                                              params_init=(0,0,0),
                                                              param_names=["a", "b", "c"])

    # fit k with constrained quadratic * exp curve
    constraints = ([.4, -.01, -np.inf, -np.inf, -np.inf],
                   [.6, -.005, np.inf, np.inf, np.inf])
    print("k Quadratic-Exponential Fit Params")
    k_quadexp_par, k_quadexp_cov, k_quadexp_err, k_quadexp_chi2 = fit(quadconstr_exp_curve, delta_par_df["Q2"],
                                                                      delta_par_df["k"], delta_par_df["k.err"],
                                                                      params_init=(0.5,-.007,1,1,-1),
                                                                      param_names=["x0", "y0", "c", "a", "b"])


    # fit k with cubic * exp curve
    print("k Cubic-Exponential Fit Params")
    k_cubexp_par, k_cubexp_cov, k_cubexp_err, k_cubexp_chi2 = fit(cub_exp_curve, delta_par_df["Q2"],
                                                                  delta_par_df["k"], delta_par_df["k.err"],
                                                                  params_init=(-.03,-.008,.006,-.001,.95,-1),
                                                                  param_names=["a0", "b0", "c0", "d0", "a1", "b1"])
    '''

    '''
    # bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf, -np.inf],
    #                 ub=[np.inf, np.inf, np.inf, np.inf])
    bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf, 0.0004],
                    ub=[np.inf, np.inf, np.inf, 0.0008]) # bounds for a, b, c, y0
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    k_P_vals = [P0, P1, P2, Y1]
    k_nucl_par, k_nucl_cov, k_nucl_err, k_nucl_chi2 = fit(quad_nucl_curve_constp, delta_par_df["Q2"],
                                                          delta_par_df["k"], delta_par_df["k.err"],
                                                          params_init=(-.003, -.003, .002, 0.0007),
                                                          param_names=["a", "b", "c", "y0"],
                                                          constr=bounds)
    print("Gamma Quad-Nucl Potential Fit Params")
    # bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf, -np.inf],
    #                 ub=[np.inf, np.inf, np.inf, np.inf])
    bounds = Bounds(lb=[-np.inf, -np.inf, -np.inf, 0.0],
                    ub=[np.inf, np.inf, np.inf, 0.3]) # bounds for a, b, c, y0
    P0 = 1.3
    P1 = 2.0
    P2 = 0.5
    Y1 = 0.0
    gamma_P_vals = [P0, P1, P2, Y1]
    gamma_nucl_par, gamma_nucl_cov, gamma_nucl_err, gamma_nucl_chi2 = fit(quad_nucl_curve_constp, delta_par_df["Q2"],
                                                                  delta_par_df["gamma"],
                                                                  delta_par_df["gamma.err"],
                                                                  params_init=(0.17, 0.24, -.08, 0.1),
                                                                  param_names=["a", "b", "c", "y0"],
                                                                  constr=bounds)
    '''

    fit_results_csv = "fit_results.csv"

    k_lb = [-1e10, -1e10, -1e10, -1e-10]
    k_ub = [1e10, 1e10, 1e10, 1e-10]
    k_bounds = Bounds(lb=k_lb, ub=k_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    k_p_vals_initial = [P0, P1, P2, Y1]

    gamma_lb = [-1e10, -1e10, -1e10, 0.0]
    gamma_ub = [1e10, 1e10, 1e10, 0.3]
    gamma_bounds = Bounds(lb=gamma_lb, ub=gamma_ub)        
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    gamma_p_vals_initial = [P0, P1, P2, Y1]

    mass_lb = [0.0, -1e10, 0.0, 0.0]
    mass_ub = [1e10, 1e10, 1e10, 2.0]
    mass_bounds = Bounds(lb=mass_lb, ub=mass_ub)
    P0 = 0.7
    P1 = 1.7
    P2 = 0.3
    Y1 = 0.0
    mass_p_vals_initial = [P0, P1, P2, Y1]                
    
    if not os.path.exists(fit_results_csv):
        print(f"\n\nFile '{fit_results_csv}' does not exist. Finding best fits!")
    
        # Initialize an empty list to store results
        fit_results = []

        # Perform fits for k, gamma, and mass
        print("-"*35)
        print("K Quad-Nucl Potential Fit Params")
        print("-"*35)
        k_best_params, k_best_p_vals, k_best_chi2, k_param_uncertainties, k_p_val_uncertainties = fit_with_dynamic_params(
            "k",
            x_data=delta_par_df["Q2"],
            y_data=delta_par_df["k"],
            y_err=delta_par_df["k.err"],
            param_bounds=k_bounds,
            p_vals_initial=k_p_vals_initial,
            fit_function=quad_nucl_curve_constp_k,
            N=10,
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
            fit_function=quad_nucl_curve_constp_gamma,
            N=10,
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
            fit_function=quad_nucl_curve_constp_mass,
            N=10,
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

        print("Results saved to fit_results.csv")

    else:
        print(f"\n\nFile '{fit_results_csv}' exists. Loading variables from CSV.")

        # Load the CSV file
        fit_results_df = pd.read_csv(fit_results_csv)

        # Define a function to parse list-like strings with irregular formatting
        def parse_list(value):
            if pd.isna(value):  # Handle NaN values
                return []
            # Remove extra spaces and split by spaces or commas
            value = re.sub(r'\s+', ',', value.strip())  # Replace spaces with commas
            value = value.replace('[,', '[').replace(',]', ']')  # Clean up misplaced commas
            try:
                return ast.literal_eval(value)
            except Exception:
                print(f"Error parsing value: {value}")
                return []

        # Apply parsing to the relevant columns
        columns_to_parse = ["Best Fit Parameters", "P Value Uncertainties", "Parameter Uncertainties"]
        for col in columns_to_parse:
            fit_results_df[col] = fit_results_df[col].apply(parse_list)

        # Extract variables from the parsed dataframe
        for _, row in fit_results_df.iterrows():
            if row["Parameter"] == "k":
                k_best_params = row["Best Fit Parameters"]
                k_best_p_vals = ast.literal_eval(row["Best P Values"])  # Parse normally formatted column
                k_best_chi2 = row["Chi-Squared"]
                k_param_uncertainties = row["Parameter Uncertainties"]
                k_p_val_uncertainties = row["P Value Uncertainties"]

            elif row["Parameter"] == "gamma":
                gamma_best_params = row["Best Fit Parameters"]
                gamma_best_p_vals = ast.literal_eval(row["Best P Values"])
                gamma_best_chi2 = row["Chi-Squared"]
                gamma_param_uncertainties = row["Parameter Uncertainties"]
                gamma_p_val_uncertainties = row["P Value Uncertainties"]

            elif row["Parameter"] == "mass":
                mass_best_params = row["Best Fit Parameters"]
                mass_best_p_vals = ast.literal_eval(row["Best P Values"])
                mass_best_chi2 = row["Chi-Squared"]
                mass_param_uncertainties = row["Parameter Uncertainties"]                
                mass_p_val_uncertainties = row["P Value Uncertainties"]

        print("Variables successfully loaded from the CSV.")


    # Unpack the results
    print("k Parameters")
    print("-"*50)
    k_nucl_par = k_best_params
    k_P_vals = k_best_p_vals
    k_nucl_chi2 = k_best_chi2
    print("Best fit parameters (a, b, c, y0):", k_nucl_par)
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
    print("Best fit parameters (a, b, c, y0):", gamma_nucl_par)
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
    print("Best fit parameters (a, b, c, y0):", mass_nucl_par)
    print("Best P values (P0, P1, P2, Y1):", mass_P_vals)
    print("Best chi-squared:", mass_nucl_chi2)
    print("Parameters uncertainties:", mass_param_uncertainties)    
    print("P value uncertainties:", mass_p_val_uncertainties)
    print("\n")
    
    '''    
    k_lin = lin_curve(q2, k_lin_par[0], k_lin_par[1])
    gamma_lin = lin_curve(q2, gamma_lin_par[0], gamma_lin_par[1])
    mass_lin = lin_curve(q2, mass_lin_par[0], mass_lin_par[1])    

    k_quad = quad_curve(q2, k_quad_par[0], k_quad_par[1], k_quad_par[2])
    gamma_quad = quad_curve(q2, gamma_quad_par[0], gamma_quad_par[1], gamma_quad_par[2])

    k_cub = cubic_curve(q2, k_cub_par[0], k_cub_par[1], k_cub_par[2], k_cub_par[3])
    gamma_cub = cubic_curve(q2, gamma_cub_par[0], gamma_cub_par[1], gamma_cub_par[2], gamma_cub_par[3])

    k_exp = exp_curve(q2, k_exp_par[0], k_exp_par[1], k_exp_par[2])
    gamma_exp = exp_curve(q2, gamma_exp_par[0], gamma_exp_par[1], gamma_exp_par[2])

    k_cubexp = cub_exp_curve(q2, k_cubexp_par[0], k_cubexp_par[1], k_cubexp_par[2], k_cubexp_par[3], k_cubexp_par[4], k_cubexp_par[5])

    k_quadexp = quadconstr_exp_curve(q2, k_quadexp_par[0], k_quadexp_par[1], k_quadexp_par[2], k_quadexp_par[3], k_quadexp_par[4])
    '''

    k_nucl_args = [q2] + [p for p in k_nucl_par] + [P for P in k_P_vals]
    k_nucl = quad_nucl_curve_k(*k_nucl_args)
    k_nucl_err = [p for p in k_param_uncertainties] + [p for p in k_p_val_uncertainties]
    gamma_nucl_args = [q2] + [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
    gamma_nucl = quad_nucl_curve_gamma(*gamma_nucl_args)
    gamma_nucl_err = [p for p in gamma_param_uncertainties] + [p for p in gamma_p_val_uncertainties]
    mass_nucl_args = [q2] + [p for p in mass_nucl_par] + [P for P in mass_P_vals]
    mass_nucl = quad_nucl_curve_mass(*mass_nucl_args)
    mass_nucl_err = [p for p in mass_param_uncertainties] + [p for p in mass_p_val_uncertainties]
    
    #for i, label in enumerate(delta_par_df["Experiment"].unique()):

    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    #'''

    def find_param_errors(i, var_name):
        x_data = delta_par_df["Q2"]
        if var_name == "k":
            true_params = [p for p in k_nucl_par] + [P for P in k_P_vals]
            y_data = delta_par_df["k"]
            y_err = delta_par_df["k.err"]
            y_nucl = k_nucl
            bounds = (k_lb + [P-(1e-6) for P in k_P_vals], k_ub + [P+(1e-6) for P in k_P_vals])
            model = quad_nucl_curve_k
        elif var_name == "mass":
            true_params = [p for p in mass_nucl_par] + [P for P in mass_P_vals]
            y_data = delta_par_df["M"]
            y_err = delta_par_df["M.err"]
            y_nucl = mass_nucl            
            bounds = (mass_lb + [P-(1e-6) for P in mass_P_vals], mass_ub + [P+(1e-6) for P in mass_P_vals])
            model = quad_nucl_curve_mass
        elif var_name == "gamma":
            true_params = [p for p in gamma_nucl_par] + [P for P in gamma_P_vals]
            y_data = delta_par_df["gamma"]
            y_err = delta_par_df["gamma.err"]
            y_nucl = gamma_nucl
            bounds = (gamma_lb + [P-(1e-6) for P in gamma_P_vals], gamma_ub + [P+(1e-6) for P in gamma_P_vals])
            model = quad_nucl_curve_gamma
        else:
            print("ERROR: Invalid variable name!")
            return

        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        y_err = np.array(y_err)

        # Perform the initial fit
        popt, pcov = curve_fit(
            model, x_data, y_data, p0=true_params, sigma=y_err, bounds=bounds, absolute_sigma=True
        )
        
        # Bootstrap parameters
        n_bootstrap = 1000
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
                    lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in k_P_vals]
                    ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in k_P_vals]
                    bounds=(lb_tmp, ub_tmp)
                elif var_name == "gamma":
                    lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in k_P_vals]
                    ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in k_P_vals]
                    bounds=(lb_tmp, ub_tmp)
                elif var_name == "mass":
                    lb_tmp = [-1e10, -1e10, -1e10, -1e10] + [P-(1e-6) for P in k_P_vals]
                    ub_tmp = [1e10, 1e10, 1e10, 1e10] + [P+(1e-6) for P in k_P_vals]
                    bounds=(lb_tmp, ub_tmp)
                else:
                    bounds=bounds
                boot_popt, _ = curve_fit(
                    model, 
                    x_bootstrap, 
                    y_bootstrap, 
                    p0=true_params,
                    bounds=bounds,
                    sigma=y_err,
                    absolute_sigma=True
                )                    
                bootstrap_params[b] = boot_popt
                bootstrap_fits_data[b] = model(x_data, *boot_popt)
                bootstrap_fits_q2[b] = model(q2, *boot_popt)
            except RuntimeError:
                bootstrap_params[b] = popt
                bootstrap_fits_data[b] = model(x_data, *popt)
                bootstrap_fits_q2[b] = model(q2, *popt)

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
                (model(x, *(params + epsilon * np.eye(len(params))[i])) - 
                 model(x, *(params - epsilon * np.eye(len(params))[i]))) / 
                (2 * epsilon) for i in range(len(params))
            ]).T

        # Compute fit and error bars
        fit = model(x_data, *popt)
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
        q2_fit = model(q2, *popt)
        q2_err = np.std(bootstrap_fits_q2, axis=0)
        window_size = min(len(q2) // 3, 10)
        smoothed_q2_err = moving_average(q2_err, window_size)
            
        # Plot
        axs[i].errorbar(x_data, y_data, yerr=y_err, fmt='o', label='Data')
        axs[i].plot(x_data, fit, label='Curve_fit', color='red')
        axs[i].plot(q2, q2_fit, label='Extrapolation', color='purple', linestyle='--')
        axs[i].plot(q2, y_nucl, label='Diff. Ev.', color='blue')
        axs[i].fill_between(q2, 
                           q2_fit - smoothed_q2_err,
                           q2_fit + smoothed_q2_err,
                           alpha=0.5, color="darkred")

    for i, var_name in enumerate(["k", "gamma", "mass"]):
        find_param_errors(i, var_name)

    axs[0].set_ylabel("k")
    axs[1].set_ylabel("$\Gamma$")
    axs[2].set_ylabel("M")    

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_ylim(-.12, 0.02)
    axs[1].set_ylim(-.5, 0.5)
    axs[2].set_ylim(1.1, 1.5)
    axs[0].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0, color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")
    
    # Assuming you have:
    # k_nucl: best-fit parameters
    # k_nucl_err: parameter uncertainties
    # fit_function: the function used for fitting
    
    # plot the fits with the data
    fig, axs = plt.subplots(1, 3, figsize=(18,10))

    # plot all the parameters vs Q2
    for i, label in enumerate(delta_par_df["Experiment"].unique()):
        axs[0].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==label]["k"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==label]["k.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=label, capthick=cap_thick)

        axs[1].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                        delta_par_df[delta_par_df["Experiment"]==label]["gamma"],
                        yerr=delta_par_df[delta_par_df["Experiment"]==label]["gamma.err"], fmt=m_type,
                        color=colors[i], markersize=m_size, capsize=cap_size,
                        label=label, capthick=cap_thick)

        axs[2].errorbar(delta_par_df[delta_par_df["Experiment"]==label]["Q2"],
                      delta_par_df[delta_par_df["Experiment"]==label]["M"],
                      yerr=delta_par_df[delta_par_df["Experiment"]==label]["M.err"], fmt=m_type,
                      color=colors[i], markersize=m_size, capsize=cap_size,
                      label=label, capthick=cap_thick)
        
    axs[0].plot(q2, k_nucl, label="New Fit $\chi_v^2$=" + f"{k_nucl_chi2:.2f}", color='red')

    axs[1].plot(q2, gamma_nucl, label="New Fit $\chi_v^2$=" + f"{gamma_nucl_chi2:.2f}", color='red')

    axs[2].plot(q2, mass_nucl, label="New Fit $\chi_v^2$=" + f"{mass_nucl_chi2:.2f}", color='red')

    # plot splines    
    q2 = np.linspace(0.0, delta_par_df["Q2"].max()+3.0, 1000, dtype=np.double)

    '''
    k_spline = BSpline(*tck_k)
    gamma_spline = BSpline(*tck_gamma)
    mass_spline = BSpline(*tck_mass)

    axs[0].plot(q2, k_spline(q2), '-', label=f'Spline s={sk}')
    axs[1].plot(q2, gamma_spline(q2), '-', label=f'Spline s={sg}')
    axs[2].plot(q2, mass_spline(q2), '-', label=f'Spline s={sm}')
    '''
    
    fig.tight_layout()

    axs[0].set_ylabel("k")
    axs[1].set_ylabel("$\Gamma$")
    axs[2].set_ylabel("M")

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_ylim(-.12, 0.02)
    axs[1].set_ylim(-.5, 0.5)
    axs[2].set_ylim(1.1, 1.5)
    axs[0].axhline(y=0,color="black", linestyle='--', alpha=0.5)
    axs[1].axhline(y=0, color="black", linestyle='--', alpha=0.5)
    axs[2].axhline(y=1.232, color="black", linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.text(0.53, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center')

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")

    # Use these functions for k and gamma to plot the delta peaks of the data

    w = np.linspace(w_res_min, w_res_max, 1000, dtype=np.double)

    def optimize_parameters_for_q2_bin(res_df, l, k_nucl_par, gamma_nucl_par, mass_nucl_par, k_P_vals, gamma_P_vals, mass_P_vals, w_lims):
        def objective_function(params):

            w_dis_transition, w_dis_region, damping_res_width, damping_dis_width = params

            q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

            # List to store chi-squared values for each Q2 label
            chi_squared_values = []

            k_fit_params = [k_nucl_par] #, k_spline]
            gamma_fit_params = [gamma_nucl_par] #, gamma_spline]
            mass_fit_params = [mass_nucl_par] #, mass_spline]      
            fit_funcs_k = [quad_nucl_curve_k] #, BSpline]
            fit_funcs_gamma = [quad_nucl_curve_gamma] #, BSpline]
            fit_funcs_mass = [quad_nucl_curve_mass] #, BSpline]
            fit_names = ["New"] # , f"Spline, s={sk}"]

            chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

            for i in range(len(k_fit_params)):
                k_params = k_fit_params[i]
                args = [q2]

                for j in range(len(gamma_fit_params)):

                  gamma_params = gamma_fit_params[j]

                  for ij in range(len(mass_fit_params)):

                      if (i, j, ij) not in chosen_fits:
                          # skip combinations that aren't desired
                          continue

                      mass_params = mass_fit_params[ij]

                      if i==1:
                        # spline k
                        k = k_fit_params[i](q2)
                      else:
                        k_args = args + [p for p in k_params]
                        if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                        k = fit_funcs_k[i](*k_args)

                      if j==1:
                        # spline gamma
                        gamma = gamma_fit_params[j](q2)
                      else:
                        gamma_args = args + [p for p in gamma_params]
                        if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                        gamma = fit_funcs_gamma[j](*gamma_args)

                      if ij==1:
                        # spline mass
                        mass = mass_fit_params[ij](q2)
                      else:
                        mass_args = args + [p for p in mass_params]
                        if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                        mass = fit_funcs_mass[ij](*mass_args)

                      # calculate fitted curve
                      y = breit_wigner_res(w, mass, k, gamma)

                      # try getting a chi squared for this curve for w_res_min<W<w_res_max
                      W = res_df['W'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
                      y_cal = breit_wigner_res(W, mass, k, gamma)
                      y_act = res_df['G1F1'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
                      y_act_err = res_df['G1F1.err'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              
            w_dis = np.linspace(2.0,3.0,1000)
            q2_array = np.ones(w_dis.size)*q2
            x_dis = W_to_x(w_dis, q2_array)

            # original DIS fit params x0, y0, c, beta
            #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
            #y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
            #                      quad2_dis_par[2], quad2_dis_par[3])

            # Table F.1 from XZ's thesis
            quad_new_dis_par = dis_fit_init
            y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)


            for i in range(len(k_fit_params)):

              k_params = k_fit_params[i]
              args = [q2]

              for j in range(len(gamma_fit_params)):

                gamma_params = gamma_fit_params[j]

                for ij in range(len(mass_fit_params)):

                    if (i, j, ij) not in chosen_fits:
                        # skip combinations that aren't desired
                        continue

                    mass_params = mass_fit_params[ij]

                    if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                    else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                    if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                    else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                    if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                    else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                        # add constant parameters P0, P1, P2 for Woods-Saxon
                        mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                    w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                    # Calculate Breit-Wigner fit
                    y_bw = breit_wigner_res(w_res, mass, k, gamma)
                    damping = damping_function(w_res, w_dis_transition, damping_res_width)
                    y_bw_damped = y_bw * damping              

                    # Calculate DIS fit
                    #y_dis = g1f1_quad2_DIS([W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)], quad2_dis_par[0], quad2_dis_par[1],
                    #               quad2_dis_par[2], quad2_dis_par[3])
                    y_dis = g1f1_quad_new_DIS([W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)], *quad_new_dis_par)

                    # Fit residual function
                    w_fit = w_res[(w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)]
                    residual_fit = y_dis[(w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)]\
                                   - y_bw_damped[(w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)]
                    popt, pcov = curve_fit(lambda W, a, b, c: residual_function(W, a, b, c, w_dis_region), w_fit, residual_fit, p0=[0, 0, 0])

                    # Calculate the complete fit
                    y_dis_transition = y_bw_damped + (1 - damping) * (y_dis - residual_function(w_res, *popt, w_dis_region))

                    # Ensure smooth transition to DIS
                    dis_transition = damping_function(w_res, w_dis_region, damping_dis_width)                
                    y_complete = y_dis_transition * dis_transition + y_dis * (1 - dis_transition)

                    interp_func = interp1d(w_res, y_complete, kind='linear', bounds_error=False, fill_value="extrapolate")
                    y_complete_interpolated = interp_func(res_df['W'][res_df['Q2_labels']==l])
                    
                    nu = abs(len(y_act)-len(params))
                    chi2 = red_chi_sqr(y_complete_interpolated, res_df['G1F1'][res_df['Q2_labels']==l], res_df['G1F1.err'][res_df['Q2_labels']==l], nu)
                    
                    chi_squared_values.append(chi2)
                  
            # Return mean chi-squared across all Q2 labels
            return np.mean(chi_squared_values)
    
        # Define parameter bounds
        bounds = [
            (1.4, 1.6),   # w_dis_transition
            (1.6, 1.9),   # w_dis_region
            (0.01, 0.1),  # damping_res_width
            (0.05, 0.2)   # damping_dis_width
        ]

        # Run differential evolution for this Q2 bin
        result = differential_evolution(
            objective_function, 
            bounds, 
            strategy='best1bin', 
            popsize=15,
            maxiter=50000
        )

        def estimate_parameter_uncertainties(result, objective_function, param_bounds=None):
            """
            Estimate parameter uncertainties using multiple methods and fallback options.

            Parameters:
            -----------
            result : OptimizeResult
                The optimization result from differential evolution or other optimizer
            objective_function : callable
                The objective function used in optimization
            param_bounds : list of tuples, optional
                Parameter bounds used in optimization, for uncertainty scaling

            Returns:
            --------
            uncertainties : np.ndarray
                Estimated uncertainties for each parameter
            method_used : str
                The method successfully used to estimate uncertainties
            """

            optimal_params = result.x
            n_params = len(optimal_params)

            # Initialize default uncertainties based on parameter bounds
            if param_bounds is not None:
                default_uncertainties = np.array([
                    (high - low) * 0.01  # 1% of parameter range
                    for low, high in param_bounds
                ])
            else:
                default_uncertainties = np.abs(optimal_params) * 0.01  # 1% of parameter value

            def try_hessian_method():
                """Attempt to estimate errors using Hessian matrix"""
                try:
                    # Refine solution and compute Hessian
                    refined_result = minimize(
                        objective_function, 
                        optimal_params, 
                        method='BFGS',
                        options={'return_hessian': True}
                    )

                    if not refined_result.success:
                        return None, "Hessian refinement failed to converge"

                    hessian = refined_result.hess

                    # Check if Hessian is positive definite
                    try:
                        # Compute inverse Hessian
                        hess_inv = np.linalg.inv(hessian)

                        # Check for negative diagonal elements
                        if np.any(np.diag(hess_inv) <= 0):
                            return None, "Hessian has negative diagonal elements"

                        uncertainties = np.sqrt(np.diag(hess_inv))

                        # Check for unreasonably large uncertainties
                        if np.any(uncertainties > 100 * np.abs(optimal_params)):
                            return None, "Unreasonably large uncertainties from Hessian"

                        return uncertainties, None

                    except LinAlgError:
                        return None, "Singular Hessian matrix"

                except Exception as e:
                    return None, f"Hessian computation failed: {str(e)}"

            def try_finite_difference_method():
                """Estimate errors using finite differences"""
                try:
                    # Small perturbation size
                    eps = 1e-5

                    # Compute numerical second derivatives
                    uncertainties = np.zeros(n_params)

                    for i in range(n_params):
                        # Create perturbation vectors
                        h = np.zeros(n_params)
                        h[i] = eps

                        # Compute second derivative
                        f_plus = objective_function(optimal_params + h)
                        f_minus = objective_function(optimal_params - h)
                        f_center = objective_function(optimal_params)

                        second_deriv = (f_plus + f_minus - 2 * f_center) / (eps * eps)

                        if second_deriv > 0:  # Check for positive curvature
                            uncertainties[i] = np.sqrt(1.0 / second_deriv)
                        else:
                            return None, "Negative curvature in finite difference"

                    return uncertainties, None

                except Exception as e:
                    return None, f"Finite difference computation failed: {str(e)}"

            # Try methods in order of preference
            methods = [
                (try_hessian_method, "Hessian"),
                (try_finite_difference_method, "Finite Difference")
            ]

            for method_func, method_name in methods:
                uncertainties, error_msg = method_func()
                if uncertainties is not None:
                    return uncertainties, method_name
                else:
                    print(f"Warning: {method_name} method failed: {error_msg}")

            # Fallback to default uncertainties if all methods fail
            print("Warning: Using fallback uncertainty estimation")
            return default_uncertainties, "Default"

        errors, method = estimate_parameter_uncertainties(
            result, 
            objective_function,
            param_bounds=bounds
        )
        
        return result.x, errors
        
    full_results_csv = "full_results.csv"

    # Assuming optimize_parameters_for_q2_bin now returns both best_params and uncertainties
    param_names = ['w_dis_transition', 'w_dis_region', 'damping_res_width', 'damping_dis_width']

    if not os.path.exists(full_results_csv):
        print(f"\n\nFile '{full_results_csv}' does not exist. Finding best parameters!")

        # Initialize a dictionary to store results for all Q2 bins
        full_results = {}

        # Iterate through unique Q2 labels
        for l in res_df['Q2_labels'].unique():
            # Optimize parameters for this specific Q2 bin
            best_params, param_uncertainties = optimize_parameters_for_q2_bin(
                res_df, l, k_nucl_par, gamma_nucl_par, mass_nucl_par, 
                k_P_vals, gamma_P_vals, mass_P_vals, w_lims
            )

            # Store both parameters and uncertainties for this Q2 label
            full_results[l] = {
                'params': best_params,
                'errors': param_uncertainties
            }

        # Create separate DataFrames for parameters and uncertainties
        params_df = pd.DataFrame({label: data['params'] for label, data in full_results.items()}, index=param_names)
        errors_df = pd.DataFrame({label: data['errors'] for label, data in full_results.items()}, index=param_names)

        # Save results to CSV files
        params_df.to_csv(full_results_csv, index=True)
        errors_df.to_csv(full_results_csv.replace('.csv', '_errors.csv'), index=True)

        print("Results and uncertainties saved to CSV files")

    else:
        print(f"\n\nFile '{full_results_csv}' exists. Loading variables from CSV.")

    # Initialize dictionaries to store parameters and errors for each Q2 bin
    q2_bin_params = {}
    q2_bin_errors = {}

    # Load both CSV files
    full_results_df = pd.read_csv(full_results_csv, index_col=0)
    full_errors_df = pd.read_csv(full_results_csv.replace('.csv', '_errors.csv'), index_col=0)

    # Display the columns and index for debugging purposes
    print("CSV Columns:", full_results_df.columns)
    print("CSV Index:", full_results_df.index)

    # Iterate through each unique Q2 label
    for l in res_df['Q2_labels'].unique():
        if l in full_results_df.columns:
            # Initialize dictionaries for this Q2 bin
            q2_bin_params[l] = {}
            q2_bin_errors[l] = {}

            # Extract parameters and errors for this Q2 bin
            for name in param_names:
                if name in full_results_df.index:
                    q2_bin_params[l][name] = full_results_df.at[name, l]
                    q2_bin_errors[l][name] = full_errors_df.at[name, l]
                else:
                    print(f"Warning: Parameter '{name}' not found in the CSV index.")

            # Print parameters and errors for this Q2 bin
            print(f"\nBest Parameters for Q2 bin {l}:")
            for p_name in param_names:
                print(f"{p_name}: {q2_bin_params[l][p_name]} ± {q2_bin_errors[l][p_name]}")
        else:
            print(f"Warning: Q2 label '{l}' not found in the CSV columns.")

    print("\nVariables successfully loaded from the CSV.")

    # Plotting code
    colors = ("dimgrey", "maroon", "saddlebrown", "red", "darkorange", "darkolivegreen",
              "limegreen", "darkslategray", "cyan", "steelblue", "darkblue", "rebeccapurple",
              "darkmagenta", "indigo", "crimson", "sandybrown", "orange", "teal", "mediumorchid")

    for name in param_names:
        fig, axs = plt.subplots(1, 1, figsize=(15, 15))

        q2 = []
        param_lst = []
        error_lst = []

        # Collect data for plotting
        for i, l in enumerate(res_df['Q2_labels'].unique()):
            q2.append(res_df['Q2'][res_df['Q2_labels']==l].unique()[0])
            param_lst.append(q2_bin_params[l][name])
            error_lst.append(q2_bin_errors[l][name])

        # Plot the data with error bars
        axs.errorbar(q2, param_lst, yerr=error_lst,
                    fmt='o', color=colors[0], markersize=8, capsize=5,
                    capthick=2, label=name)

        # Customize the plot
        axs.set_title(f"Parameter {name} vs Q²")
        axs.grid(True, linestyle='--', alpha=0.7)
        axs.legend()

        # Add labels
        fig.tight_layout()
        fig.text(0.5, 0.001, "$Q^2\ ({GeV}^2)$", ha='center', va='center', size=14)
        fig.text(0.0001, 0.5, f"{name}", ha='center', va='center', rotation='vertical', size=16)

        # Save figure
        pdf.savefig(fig, bbox_inches="tight")
    
    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    best_combo = []    
    # make fit curves and plot with data|
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      w_dis_region = q2_bin_params[l]['w_dis_region']
      damping_res_width = q2_bin_params[l]['damping_res_width']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      w_dis_region_err = q2_bin_errors[l]['w_dis_region']
      damping_res_width_err = q2_bin_errors[l]['damping_res_width']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']
      
      best_chi = (-1, -1, -1, 10.0) # Initialize
      
      '''
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      fit_funcs = [quad_nucl_curve_gamma]
      fit_names = ["Quad*Woods-Saxon"]
      '''

      '''
      k_fit_params = [k_nucl_par, k_spline]
      gamma_fit_params = [gamma_nucl_par, gamma_spline]
      mass_fit_params = [mass_nucl_par, mass_spline]
      fit_funcs_k = [quad_nucl_curve_k, BSpline]
      fit_funcs_gamma = [quad_nucl_curve_gamma, BSpline]
      fit_funcs_mass = [quad_nucl_curve_mass, BSpline]
      #fit_names = ["Quad*Woods-Saxon", f"Spline, s={sk}"]
      #fit_names = ["Quad*W-S", f"Spline, s={sk}"]
      fit_names = ["New", f"Spline, s={sk}"]
      '''
      
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      #fit_names = ["Quad*Woods-Saxon"]
      #fit_names = ["Quad*W-S"]
      fit_names = ["New"]
      
      # use the 4 following lines if you want to plot all the other types of fits I tried (don't forget to uncomment the other fits first!)
      # k_fit_params = [k_lin_par, k_quad_par, k_cub_par, k_nucl_par]
      # gamma_fit_params = [gamma_lin_par, gamma_quad_par, gamma_cub_par, gamma_nucl_par]
      # fit_funcs = [lin_curve, quad_curve, cubic_curve, quad_nucl_curve_gamma]
      # fit_names = ["Linear", "Quadratic", "Cubic", "Quad*Woods-Saxon"]


      # select desired k, gamma fits to be used by index
      # [(i, j),...] where i=index for k fit, j=index for gamma fit
      # # 0 = linear, 1 = quadratic, 2 = cubic, 3=quad-nucl_potential
      # 0 = quad-nucl, 1 = spline
      chosen_fits = [(0, 0, 0), (1, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1), (0, 1, 0)]

      # have a 3 fits for k,  gamma, mass -> 24 possible combinations
      for i in range(len(k_fit_params)):

        k_params = k_fit_params[i]
        args = [q2]

        for j in range(len(gamma_fit_params)):

          gamma_params = gamma_fit_params[j]

          for ij in range(len(mass_fit_params)):
                    
              if (i, j, ij) not in chosen_fits:
                  # skip combinations that aren't desired
                  continue

              mass_params = mass_fit_params[ij]

              if i==1:
                # spline k
                args = [q2]
                k = k_fit_params[i](q2)
              else:
                k_args = args + [p for p in k_params]
                if i==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  k_args += [P for P in k_P_vals]
                k = fit_funcs_k[i](*k_args)

              if j==1:
                # spline gamma
                args = [q2]
                gamma = gamma_fit_params[j](q2)
              else:
                gamma_args = args + [p for p in gamma_params]
                if j==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  gamma_args += [P for P in gamma_P_vals]
                gamma = fit_funcs_gamma[j](*gamma_args)

              if ij==1:
                # spline mass
                args = [q2]
                mass = mass_fit_params[ij](q2)
              else:
                mass_args = args + [p for p in mass_params]
                if ij==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  mass_args += [P for P in mass_P_vals]
                mass = fit_funcs_mass[ij](*mass_args)
                
              # calculate fitted curve
              y = breit_wigner_res(w, mass, k, gamma)

              # try getting a chi squared for this curve for w_res_min<W<w_res_max
              W = res_df['W'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              y_cal = breit_wigner_res(W, mass, k, gamma)
              y_act = res_df['G1F1'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              y_act_err = res_df['G1F1.err'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              nu = abs(len(y_act)-3) # n points minus 3 fitted parameters (k, gamma, mass)
              chi2 = red_chi_sqr(y_cal, y_act, y_act_err, nu)

              if abs(chi2-1) < abs(best_chi[-1]-1):
                  best_chi = (i, j, ij, chi2)
              
              axs[row, col].plot(w, y, markersize=m_size,
                                 label=f"$k$ {fit_names[i]} | $\Gamma$ {fit_names[j]} | M {fit_names[ij]} | $\chi_v^2$={chi2:.2f}",
                                 #label=f"$k$ {i}, $\Gamma$ {j} M {ij} $\chi_v^2$={chi2:.2f}",
                                 linestyle='dashed')
              
      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)

      # Save best combination for k, gamma, and mass
      best_combo.append(best_chi)

      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.1)
      axs[row,col].set_title(l)

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figures
    pdf.savefig(fig,bbox_inches="tight")    
    
    most_common = most_common_combination(best_combo)
    
    # ### Try extending DIS Fit to resonance region and see how it looks

    # In[16]:


    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      w_dis_region = q2_bin_params[l]['w_dis_region']
      damping_res_width = q2_bin_params[l]['damping_res_width']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      w_dis_region_err = q2_bin_errors[l]['w_dis_region']
      damping_res_width_err = q2_bin_errors[l]['damping_res_width']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']

      '''
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      fit_funcs = [quad_nucl_curve_gamma]
      fit_names = ["Quad*Woods-Saxon"]
      '''

      '''
      k_fit_params = [k_nucl_par, k_spline]
      gamma_fit_params = [gamma_nucl_par, gamma_spline]
      mass_fit_params = [mass_nucl_par, mass_spline]      
      fit_funcs_k = [quad_nucl_curve_k, BSpline]
      fit_funcs_gamma = [quad_nucl_curve_gamma, BSpline]
      fit_funcs_mass = [quad_nucl_curve_mass, BSpline]
      #fit_names = ["Quad*Woods-Saxon", f"Spline, s={sk}"]
      fit_names = ["New", f"Spline, s={sk}"]
      '''

      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      #fit_names = ["Quad*Woods-Saxon"]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      #chosen_fits = [most_common]
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):

        k_params = k_fit_params[i]
        args = [q2]


        for j in range(len(gamma_fit_params)):

          gamma_params = gamma_fit_params[j]

          for ij in range(len(mass_fit_params)):
                    
              if (i, j, ij) not in chosen_fits:
                  # skip combinations that aren't desired
                  continue

              mass_params = mass_fit_params[ij]

              if i==1:
                # spline k
                k = k_fit_params[i](q2)
              else:
                k_args = args + [p for p in k_params]
                if i==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  k_args += [P for P in k_P_vals]
                k = fit_funcs_k[i](*k_args)

              if j==1:
                # spline gamma
                gamma = gamma_fit_params[j](q2)
              else:
                gamma_args = args + [p for p in gamma_params]
                if j==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  gamma_args += [P for P in gamma_P_vals]
                gamma = fit_funcs_gamma[j](*gamma_args)

              if ij==1:
                # spline mass
                mass = mass_fit_params[ij](q2)
              else:
                mass_args = args + [p for p in mass_params]
                if ij==0:
                  # add constant parameters P0, P1, P2 for Woods-Saxon
                  mass_args += [P for P in mass_P_vals]
                mass = fit_funcs_mass[ij](*mass_args)
                
              # calculate fitted curve
              y = breit_wigner_res(w, mass, k, gamma)

              # try getting a chi squared for this curve for w_res_min<W<w_res_max
              W = res_df['W'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              y_cal = breit_wigner_res(W, mass, k, gamma)
              y_act = res_df['G1F1'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              y_act_err = res_df['G1F1.err'][res_df['Q2_labels']==l][res_df['W']<=w_res_max][res_df['W']>=w_res_min]
              nu = abs(len(y_act)-3) # n points minus 3 fitted parameters (k, gamma, mass)
              chi2 = red_chi_sqr(y_cal, y_act, y_act_err, nu)

              axs[row, col].plot(w, y, markersize=m_size,
                                 label=f"$k$ {fit_names[i]} | $\Gamma$ {fit_names[j]} | M {fit_names[ij]} | $\chi_v^2$={chi2:.2f}",
                                 #label=f"$k$ {i}, $\Gamma$ {j} M {ij} $\chi_v^2$={chi2:.2f}",
                                 linestyle='dashed')
              
      # extend DIS model to W=2 GeV and plot
      # original DIS fit params x0, y0, c, beta
      #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_init
      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)
      #y_dis = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
      #                      quad2_dis_par[2], quad2_dis_par[3])
      y_dis = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
            
      axs[row, col].plot(w_dis, y_dis, color="r", label=f"Quad DIS Fit, $\\beta$ = {beta_val:.4f}", linestyle="--")

      '''
      # quadratic DIS fit using mingyu fit beta value
      quad2_dis_par = [0.16424, -.02584, 0.16632, 0.04469]
      y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
                            quad2_dis_par[2], quad2_dis_par[3])
      axs[row, col].plot(w_dis, y_dis_new, color="violet", label="Quad DIS Fit, $\\beta$ = 0.04469", linestyle="--")
      '''

      '''
      # mingyu cubic DIS fit params a, b, c, d, beta
      cube_dis_par = [-.0171, -.16611, 0.68161, -.56782, 0.04469]
      y_cube_dis = g1f1_cubic_DIS([x_dis, q2_array], cube_dis_par[0], cube_dis_par[1],
                            cube_dis_par[2], cube_dis_par[3], cube_dis_par[4])
      axs[row, col].plot(w_dis, y_cube_dis, color="b", label="Mingyu Cubic DIS Fit", linestyle="--")
      '''      
      
      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)
      
    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")

    # make figure
    n_col = 5
    num_plots = len(res_df['Q2_labels'].unique())
    n_rows = num_plots//n_col + 1
    fig, axs = plt.subplots(num_plots//n_col + 1, n_col, figsize=(n_col*6.5,n_rows*6))

    # make fit curves and plot with data
    for i,l in enumerate(res_df['Q2_labels'].unique()):
      row = i//n_col
      col = i%n_col

      q2 = res_df['Q2'][res_df['Q2_labels']==l].unique()[0]

      # Read all parameters for this specific Q2 bin
      w_dis_transition = q2_bin_params[l]['w_dis_transition']
      w_dis_region = q2_bin_params[l]['w_dis_region']
      damping_res_width = q2_bin_params[l]['damping_res_width']
      damping_dis_width = q2_bin_params[l]['damping_dis_width']      

      w_dis_transition_err = q2_bin_errors[l]['w_dis_transition']
      w_dis_region_err = q2_bin_errors[l]['w_dis_region']
      damping_res_width_err = q2_bin_errors[l]['damping_res_width']
      damping_dis_width_err = q2_bin_errors[l]['damping_dis_width']

      '''
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      fit_funcs = [quad_nucl_curve_gamma]
      fit_names = ["Quad*Woods-Saxon"]
      '''

      '''
      k_fit_params = [k_nucl_par, k_spline]
      gamma_fit_params = [gamma_nucl_par, gamma_spline]
      mass_fit_params = [mass_nucl_par, mass_spline]      
      fit_funcs_k = [quad_nucl_curve_k, BSpline]
      fit_funcs_gamma = [quad_nucl_curve_gamma, BSpline]
      fit_funcs_mass = [quad_nucl_curve_mass, BSpline]
      #fit_names = ["Quad*Woods-Saxon", f"Spline, s={sk}"]
      fit_names = ["New", f"Spline, s={sk}"]
      '''
      
      k_fit_params = [k_nucl_par]
      gamma_fit_params = [gamma_nucl_par]
      mass_fit_params = [mass_nucl_par]
      fit_funcs_k = [quad_nucl_curve_k]
      fit_funcs_gamma = [quad_nucl_curve_gamma]
      fit_funcs_mass = [quad_nucl_curve_mass]
      #fit_names = ["Quad*Woods-Saxon"]
      fit_names = ["New"]
      
      # select desired k, gamma fits to be used by index
      # [(i, j, l),...] where i=index for k fit, j=index for gamma fit, l=index for M fit
      #chosen_fits = [most_common]
      chosen_fits = [(0, 0, 0)] # k Quad, gamma Quad, M Quad

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)
                      k_err = calculate_param_error(fit_funcs_k[i], k_args, k_nucl_err)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)
                      gamma_err = calculate_param_error(fit_funcs_gamma[i], gamma_args, gamma_nucl_err)
                      
                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)
                      mass_err = calculate_param_error(fit_funcs_mass[i], mass_args, mass_nucl_err)
                      
                  # Calculate Breit-Wigner fit
                  y = breit_wigner_res(w, mass, k, gamma)
                  bw_err = propagate_bw_error(w, mass, mass_err, k, k_err, gamma, gamma_err)  # Error propagation

                  # Chi-squared calculation for W in [w_res_min, w_res_max]
                  W = res_df['W'][res_df['Q2_labels'] == l][res_df['W'] <= w_res_max][res_df['W'] >= w_res_min]
                  y_cal = breit_wigner_res(W, mass, k, gamma)
                  y_act = res_df['G1F1'][res_df['Q2_labels'] == l][res_df['W'] <= w_res_max][res_df['W'] >= w_res_min]
                  y_act_err = res_df['G1F1.err'][res_df['Q2_labels'] == l][res_df['W'] <= w_res_max][res_df['W'] >= w_res_min]
                  nu = abs(len(y_act) - 3)  # Number of points minus 3 fit parameters
                  chi2 = red_chi_sqr(y_cal, y_act, y_act_err, nu)
            
                  axs[row, col].plot(w, y, markersize=m_size,
                                    label=f"$k$ {fit_names[i]} | $\Gamma$ {fit_names[j]} | M {fit_names[ij]} | $\chi_v^2$={chi2:.2f}",
                                    linestyle='dashed')

      w_dis = np.linspace(2.0,3.0,1000)
      q2_array = np.ones(w_dis.size)*q2
      x_dis = W_to_x(w_dis, q2_array)

      # original DIS fit params x0, y0, c, beta
      #quad2_dis_par = [0.16424, -.02584, 0.16632, 0.11059]
      #y_dis_new = g1f1_quad2_DIS([x_dis, q2_array], quad2_dis_par[0], quad2_dis_par[1],
      #                        quad2_dis_par[2], quad2_dis_par[3])
      # Table F.1 from XZ's thesis
      quad_new_dis_par = dis_fit_init
      y_dis_new = g1f1_quad_new_DIS([x_dis, q2_array], *quad_new_dis_par)
      
      axs[row, col].plot(w_dis, y_dis_new, color="violet", label=f"Quad DIS Fit, $\\beta$ = {beta_val:.4f}", linestyle="--")

      for i in range(len(k_fit_params)):
          k_params = k_fit_params[i]
          args = [q2]

          for j in range(len(gamma_fit_params)):
              gamma_params = gamma_fit_params[j]

              for ij in range(len(mass_fit_params)):
                  if (i, j, ij) not in chosen_fits:
                      # skip combinations that aren't desired
                      continue

                  mass_params = mass_fit_params[ij]

                  if i==1:
                      # spline k
                      k = k_fit_params[i](q2)
                  else:
                      k_args = args + [p for p in k_params]
                      if i==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          k_args += [P for P in k_P_vals]
                      k = fit_funcs_k[i](*k_args)

                  if j==1:
                      # spline gamma
                      gamma = gamma_fit_params[j](q2)
                  else:
                      gamma_args = args + [p for p in gamma_params]
                      if j==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon
                          gamma_args += [P for P in gamma_P_vals]
                      gamma = fit_funcs_gamma[j](*gamma_args)

                  if ij==1:
                      # spline mass
                      mass = mass_fit_params[ij](q2)
                  else:
                      mass_args = args + [p for p in mass_params]
                      if ij==0:
                          # add constant parameters P0, P1, P2 for Woods-Saxon

                          mass_args += [P for P in mass_P_vals]
                      mass = fit_funcs_mass[ij](*mass_args)

                  # Recalculate k, gamma, mass, and Breit-Wigner as in earlier loop
                  # Calculate Breit-Wigner fit
                  w_res = np.linspace(w_min, w_max, 1000, dtype=np.double)

                  y_bw = breit_wigner_res(w_res, mass, k, gamma)
                  bw_err = propagate_bw_error(w_res, mass, mass_err, k, k_err, gamma, gamma_err)  # Error propagation
                  damping = damping_function(w_res, w_dis_transition, damping_res_width)
                  damping_res_err = damping_function_err(w_res, w_dis_transition, w_dis_transition_err, damping_res_width, damping_res_width_err)  # Error propagation
                  y_bw_damped = y_bw * damping

                  # Calculate DIS fit
                  y_dis = g1f1_quad_new_DIS(
                      [W_to_x(w_res, np.full_like(w_res, q2)), np.full_like(w_res, q2)],
                      *quad_new_dis_par
                  )
                  
                  #dis_fit_err = fit_error(W_to_x(w_res, np.full_like(w_res, q2)), q2, par_quad, par_err_quad, corr_quad, partials_new)
                  dis_err = propagate_dis_error(
                      quad_fit_err
                      #dis_fit_err
                  )  # Error propagation

                  # Fit residual function
                  w_fit = w_res[(w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)]
                  residual_fit = y_dis[
                      (w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)
                  ] - y_bw_damped[
                      (w_res >= w_dis_transition - 0.05) & (w_res <= w_dis_transition + 0.05)
                  ]
                  popt, pcov = curve_fit(
                      lambda W, a, b, c: residual_function(W, a, b, c, w_dis_region),
                      w_fit,
                      residual_fit,
                      p0=[0, 0, 0],
                  )

                  residual_err = propagate_residual_error(w_res, popt, pcov, residual_function, w_dis_region)

                  # Calculate the complete fit
                  y_dis_transition = y_bw_damped + (1 - damping) * (
                      y_dis - residual_function(w_res, *popt, w_dis_region)
                  )
                  transition_err = propagate_transition_error(
                      w_res,
                      bw_err,
                      residual_err,
                      damping_res_err,
                      w_res_min,
                      w_res_max,
                      w_dis_transition
                  ) # Error propagation

                  # Ensure smooth transition to DIS
                  dis_transition = damping_function(w_res, w_dis_region, damping_dis_width)
                  damping_dis_err = damping_function_err(w_res, w_dis_region, w_dis_region_err, damping_dis_width, damping_dis_width_err)  # Error propagation
                  
                  y_complete = y_dis_transition * dis_transition + y_dis * (1 - dis_transition)
                                    
                  complete_err = propagate_complete_error(
                      w_res,
                      transition_err,
                      damping_dis_err,
                      dis_err,
                      w_res_min,
                      w_dis_transition,
                      w_dis_region,
                      w_max
                  ) # Error propagation
                  
                  #print("!!!!!!!!!!!!",complete_err)
                  axs[row, col].plot(
                      w_res,
                      y_complete,
                      color="blue",
                      linestyle="solid",
                      label=f"Complete Fit",
                  )
                  axs[row, col].fill_between(
                      w_res,
                      y_complete - complete_err,
                      y_complete + complete_err,
                      color="blue",
                      alpha=0.3,
                      label="Fit Error",
                  )

                  x_res = W_to_x(w_res, np.full_like(w_res, q2))
                  
                  # Create a twin axis that shares the y-axis
                  ax2 = axs[row, col].twiny()

                  # Plot the same data against x_res (but make it invisible)
                  ax2.plot(x_res, y_complete, alpha=0)

                  ax2.set_xlabel(r"$x_{Bj}$")

                  # This should force the x-axis limits to match your x_res range
                  ax2.set_xlim(min(x_res), max(x_res))

      # plot the data
      axs[row, col].errorbar(res_df['W'][res_df['Q2_labels']==l],
                    res_df['G1F1'][res_df['Q2_labels']==l],
                    yerr=res_df['G1F1.err'][res_df['Q2_labels']==l],
                    fmt=m_type, color=colors[0], markersize=m_size, capsize=cap_size,
                    capthick=cap_thick)
      
      axs[row,col].legend()
      # set axes limits
      axs[row,col].axhline(0, color="black", linestyle="--")
      axs[row,col].set_ylim(-.15,0.1)
      axs[row,col].set_xlim(0.9,2.5)
      axs[row,col].set_title(l)

    fig.tight_layout()
    fig.text(0.5, 0.001, "W (GeV)", ha='center', va='center', size = 14)
    fig.text(0.0001, 0.5, "$g_1^{3He}/F_1^{3He}$", ha='center', va='center', rotation='vertical', size=16)

    # Save figure
    pdf.savefig(fig,bbox_inches="tight")
    
show_pdf_with_evince("plots/g1f1_fits.pdf")
    
