# Minimum Distance calculations: model moments and optimization

import numpy as np
import pandas as pd
import numdifftools as nd
from numpy.linalg import inv
from scipy.optimize import minimize
from pathlib import Path
from tools import vech_indices, cov_omega_theta

def implied_cov_permshk_continuous(params, T):
  '''
  Calculates the covariance matrix for permanent shocks in continuous time
  '''
  # read in the parameters
  var_perm          = params[0] # variance of permanent shock
  
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) #/* Income */

  for j in range(T):
    cov_y[j,j] = 2.0/3.0*var_perm 
  for j in range(1,T):
    cov_y[j-1,j] = 1.0/6.0*var_perm 
    cov_y[j,j-1] = cov_y[j-1,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_cov_transhk_continuous(params, T):
  '''
  Calculates the covariance matrix for a transitory shock with NO persistence
  '''
  # read in the parameters
  var_puretran          = params[0] # variance of the purely transitory shock
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) 
  for j in range(T):
    cov_y[j,j] = 2.0*var_puretran
  for j in range(1,T):
    cov_y[j-1,j] = -var_puretran
    cov_y[j,j-1] = cov_y[j-1,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_cov_expdecayshk_continuous(params, T):
  '''
  Calculates the covariance matrix for an exponentially decaying stochastic 
  process, time aggregated in continuous time
  '''
  # read in the parameters
  var_expdecayshk   = params[0] # variance of the exponential decay shock - this is the variance of the unexpected change in the time aggregated process over a year
  omega             = params[1] # decay parameter of the process
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) #/* Income */
  # pre-calculate covariance matrix of a stochasic process with shocks
  # that decay at rate omega
  cov_omega = cov_omega_theta(omega, omega)
  var_omega    = cov_omega[2]
  cov_omega_1  = cov_omega[3]
  cov_omega_2  = cov_omega[4]
 
  for j in range(T):
    cov_y[j,j] = var_expdecayshk*var_omega
  for j in range(1,T):
    cov_y[j-1,j] = var_expdecayshk*cov_omega_1
    cov_y[j,j-1] = cov_y[j-1,j]
  for M in range(2,T):
    for j in range(M,T):
      cov_y[j-M,j] = var_expdecayshk*np.exp(-(M-2)*omega)*cov_omega_2
      cov_y[j,j-M] = cov_y[j-M,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_inc_cov_discrete(impact_matrix):
    '''
    Calculates the covariance matrix for discrete process defined by
    the "impact_matrix"
    '''
    T = impact_matrix.shape[0]
    K = impact_matrix.shape[1]
    cov_y = np.zeros((T,T))
    for k in range(K):
        for i in range(T):
            for j in range(i+1):
                cov_y[i,i-j] += impact_matrix[i,k]*impact_matrix[i-j,k]
    for i in range(T):
        for j in range(i+1):
            cov_y[i-j,i] = cov_y[i,i-j]
    vech_indicesT = vech_indices(T)
    cov_y_vec = cov_y[vech_indicesT]
    return cov_y_vec

def impact_matrix_bonus(var_bonus, T, num_months=12, var_weights = None):
    '''
      Calculates the "impact matrix" for a purely transitory (bonus) shock
      Each row is a year [0, 1,..., T] 
      Each column is a 'month' in each year [0,1,...,11,...,12*(T+1)-1]
      Each element represents the income *change* in that row (year) from a shock that 
      occurs in the column (month-year).
      In this simple case, a shock that occurs in a month-year will impact the income
      change in that year positively by the size of the income shock, and the income 
      change in the following year negatively the same amount.
    '''
    if var_weights is None:
        var_weights = np.array([1.0/num_months]*num_months)
    if len(var_weights)!=num_months:
        return "var_weight must be the same length as there are num_months"
    var_weights = var_weights/np.sum(var_weights)  # Normalize
    K = (T+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        year = np.floor(k/(num_months*1.0)).astype(int)-1  # start with shocks that happen in year -1, as these impact the change in income in period 0 (negaitively)
        month = k%num_months # this is the modulo operator
        if year>=0:
            impact_matrix[year,k]   +=  (var_weights[month]*var_bonus)**0.5
        if year<=T-2:
            impact_matrix[year+1,k] += -(var_weights[month]*var_bonus)**0.5
    return impact_matrix

def impact_matrix_tran(var_tran, omega, T, num_months=12, pre_periods=10, var_weights = None):
    '''
      Calculates the "impact matrix" for a purely transitory (bonus) shock
      Each row is a year [0, 1,..., T] 
      Each column is a 'month' in each year [0,1,...,11,...,12*(T+1)-1]
      Each element represents the income *change* in that row (year) from a shock that 
      occurs in the column (month-year).
      In this case, a shock that occurs in a month-year will impact income
      in that year, according to how many months are left, and the years following
      in an exponentially decaying fashion. We cut the impact off at "pre_periods"
       years before the first year, so that shocks that happen "pre_periods" years
      before the first year have no impact on income change in any of the T years measured
    '''
    if var_weights is None:
        var_weights = np.array([1.0/num_months]*num_months)
    if len(var_weights)!=num_months:
        return "var_weight must be the same length as there are num_months"
    var_weights = var_weights/np.sum(var_weights)  # Normalize
    
    first_year_income = np.mean(np.exp(-omega*np.array(range(num_months))/(num_months*1.0))) # sum of income received in the year following the shock
    K = (T+pre_periods+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        month = k%num_months # this is the modulo operator
        for i in range(K-k):
            year = np.floor((k+i)/(num_months*1.0)).astype(int) -1 - pre_periods
            if year>=0:
                impact_matrix[year,k] += ((var_tran*var_weights[month])**0.5)/(first_year_income*num_months)*np.exp(-omega*i/(num_months*1.0))
            if year>=-1 and year<=T-2:
                impact_matrix[year+1,k] += -((var_tran*var_weights[month])**0.5)/(first_year_income*num_months)*np.exp(-omega*i/(num_months*1.0))
    return impact_matrix

def impact_matrix_perm(var_perm, T, num_months=12, var_weights = None):
    '''
      Calculates the "impact matrix" for a permanent shock
      Each row is a year [0, 1,..., T] 
      Each column is a 'month' in each year [0,1,...,11,...,12*(T+1)-1]
      Each element represents the income *change* in that row (year) from a shock that 
      occurs in the column (month-year).
      In this case, a shock that occurs in a month-year will impact the income
      change in that year positively by the amount of time left for the income
      to arrive, and then in the following year the remaining time at the 
      beginning of the year, adding up to the full change in permanent income
    '''
    if var_weights is None:
        var_weights = np.array([1.0/num_months]*num_months)
    if len(var_weights)!=num_months:
        return "var_weight must be the same length as there are num_months"
    var_weights = var_weights/np.sum(var_weights)  # Normalize
    
    K = (T+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        year = np.floor(k/(num_months*1.0)).astype(int)-1  # start with shocks that happen in year -1, as these impact the change in income in period 0 
        month = k%num_months # this is the modulo operator
        if year>=0:
            impact_matrix[year,k]   +=  (var_weights[month]*var_perm)**0.5 * ((year+2)*num_months-k)/(1.0*num_months)
        if year<=T-2:
            impact_matrix[year+1,k] += -(var_weights[month]*var_perm)**0.5 * (k-(year+1)*num_months)/(1.0*num_months)
    return impact_matrix

def implied_inc_cov_composite(params,T):
    var_perm = params[0]
    var_tran = params[1]
    omega    = params[2]
    bonus    = params[3]
    rho      = params[4]
    if rho==0.0:
        perm_inc_cov = implied_cov_permshk_continuous([var_perm],T)
    else:
        perm_inc_cov = implied_cov_expdecayshk_continuous([var_perm,rho],T)
    bonus_inc_cov = implied_cov_transhk_continuous([var_tran*bonus],T)
    trandecay_inc_cov = implied_cov_expdecayshk_continuous([var_tran*(1-bonus),omega],T)
#    impact_tran = impact_matrix_tran(var_tran*(1-bonus),omega,T,num_months=5,pre_periods=10)
#    tran_inc_cov = implied_inc_cov_discrete(impact_tran, T, num_months=12)
    implied_inc_cov_composite = perm_inc_cov + bonus_inc_cov + trandecay_inc_cov
    return implied_inc_cov_composite

def parameter_estimation(empirical_moments, Omega, T, init_params, optimize_index=None, bounds=None):
  '''
  Estimates model parameters
  '''
  #fix certain parameters if required
  if (optimize_index is not None):
      optimize_params = init_params[optimize_index]
      fixed_params      = init_params[np.logical_not(optimize_index)]
      if (bounds is not None):
          all_bounds = bounds
          bounds = []
          for i in range(len(all_bounds)):
              if optimize_index[i]:
                  bounds += [all_bounds[i]]
  else:
    optimize_params = init_params
    optimize_index = np.array([True]*len(init_params), dtype=bool)
    fixed_params = np.array([])

  def implied_cov_limited_params(optimize_params, T, optimize_index, fixed_params):
    params = np.zeros(len(optimize_index))
    params[optimize_index] = optimize_params
    params[np.logical_not(optimize_index)] = fixed_params
    model_cov = implied_inc_cov_composite(params, T)
    return model_cov
  def objectiveFun(optimize_params, T, empirical_cov, weight_matrix, optimize_index, fixed_params):
    model_cov = implied_cov_limited_params(optimize_params, T, optimize_index, fixed_params)
    distance = np.dot(np.dot((model_cov-empirical_cov), weight_matrix), (model_cov-empirical_cov))
    return distance

  # Define the weight matrix as Equal Weight Minimum Distance
  weight_matrix = np.diag(np.diag(Omega)**(-1))
  #ret = objectiveFun(optimize_params, T, empirical_moments, weight_matrix,optimize_index, fixed_params)
  
  # Do minimization
  solved_objective = minimize(objectiveFun, optimize_params, args=(T, empirical_moments, weight_matrix, optimize_index, fixed_params), method='L-BFGS-B', bounds=bounds, options= {'disp': True})
  solved_params = solved_objective.x
  # Calculate standard errors
  fun_for_jacob = lambda params: implied_cov_limited_params(params, T, optimize_index, fixed_params)
  jacob = nd.Jacobian(fun_for_jacob)(solved_params)
  Sandwich1 = inv(np.dot(np.transpose(jacob),np.dot(weight_matrix,jacob)))
  Sandwich2 = np.dot(np.transpose(jacob),np.dot(weight_matrix,np.dot(Omega,np.dot(weight_matrix,jacob))))
  cov_params = np.dot(Sandwich1,np.dot(Sandwich2,Sandwich1))
  standard_errors = np.diag(cov_params)**0.5
  # Create output
  output_params = np.zeros(len(optimize_index))
  output_params[optimize_index] = solved_params
  output_params[np.logical_not(optimize_index)] = fixed_params
  
  output_se = np.zeros(len(optimize_index))
  output_se[optimize_index] = standard_errors
  output_se[np.logical_not(optimize_index)] = 0.0

  return output_params, output_se


# Function to estimate parameters for each subgroup for which we have moments
def parameter_estimation_by_subgroup(moments_BPP_dir,subgroup_stub,subgroup_names, T, init_params, optimize_index=None, bounds=None):
    subgroup_estimates = np.zeros((len(subgroup_names),5))
    subgroup_se = np.zeros((len(subgroup_names), 5))
    #Just doing income for now - remove other moments
    income_moments = np.array([[False]*2*T]*2*T, dtype=bool)
    income_moments[T:,T:] = True
    vech_indices2T = vech_indices(2*T)
    income_moments = income_moments[vech_indices2T]
    for i in range(len(subgroup_names)):
        this_empirical_moments_all = np.genfromtxt(Path(moments_BPP_dir,subgroup_stub+str(i+1)+"c_vector.txt"), delimiter=',')
        this_Omega_all          =    np.genfromtxt(Path(moments_BPP_dir,subgroup_stub+str(i+1)+"_omega.txt"), delimiter=',')
        this_empirical_moments_inc = this_empirical_moments_all[income_moments]
        this_Omega_inc = this_Omega_all[income_moments,:][:,income_moments]
        this_estimates, this_estimate_se = parameter_estimation(this_empirical_moments_inc, this_Omega_inc, T, init_params, optimize_index, bounds)
        subgroup_estimates[i,:] = this_estimates
        subgroup_se[i,:] = this_estimate_se
    return subgroup_estimates, subgroup_se




