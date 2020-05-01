# Minimum Distance calculations: model moments and optimization

import numpy as np
import pandas as pd
import numdifftools as nd
from numpy.linalg import inv
from scipy.optimize import minimize

def vech_indices(N):
    rows = [];
    columns = []
    for i in range(N):
        rows += range(i,N)
        columns += [i]*(N-i)
    return (np.array(rows), np.array(columns))

#This is the covariance of two stochastic processes, driven by the same underlying shocks, 
#but which decay at different np.exponential rates omega and theta
def cov_omega_theta(omega, theta):
  omegax = 1.0/(1.0-np.exp(-omega))
  thetax = 1.0/(1.0-np.exp(-theta))
  omth = omega + theta
  omthx = 1.0/(1.0-np.exp(-omth))
  #variance at T
  var = 2.0*thetax*omegax  \
          + thetax*omegax*(  ((2.0-np.exp(-theta))*(2.0-np.exp(-omega))+1.0)/(omth*omthx)     \
                           - (3.0-np.exp(-theta))/(theta*thetax) - (3.0-np.exp(-omega))/(omega*omegax)   )  \
            +1.0/(omth*omegax*thetax)
  # Covariance for moving the theta process up to T+1
  cov_1 = thetax*omegax*((2.0-np.exp(-theta))/(theta*thetax)    \
                          + 1.0/(omega*omegax) - (2.0-np.exp(-theta))/(omth*omthx)  \
                          -1.0 )    \
           - omegax/thetax*((2.0-np.exp(-omega))/(omth*omthx) - 1.0/(theta*thetax) )     \
           + np.exp(-theta)/(omegax*thetax*omth)
  # Covariance for moving the theta process up to T+2
  cov_2 = -omegax/thetax * (1.0/(theta*thetax) - 1.0/(omth*omthx))  \
          -omegax/thetax * (np.exp(-theta)*(2.0-np.exp(-omega))/(omth*omthx) - np.exp(-theta)/(theta*thetax) ) \
          +np.exp(-2.0*theta)/(omegax*thetax*omth)
  #cov_M = np.exp(-(M-2.0)*theta)*cov2
  # Covariance for moving the omega process up to T+1
  cov_m1 =  thetax*omegax*((2.0-np.exp(-omega))/(omega*omegax)     \
                           + 1.0/(theta*thetax) - (2.0-np.exp(-omega))/(omth*omthx)  \
                           -1.0 )     \
          - thetax/omegax*((2.0-np.exp(-theta))/(omth*omthx) - 1.0/(omega*omegax) ) \
          + np.exp(-omega)/(omegax*thetax*omth)
  # Covariance for moving the omega process up to T+2
  cov_m2 = -thetax/omegax * (1.0/(omega*omegax) - 1.0/(omth*omthx)) \
           -thetax/omegax * (np.exp(-omega)*(2.0-np.exp(-theta))/(omth*omthx) - np.exp(-omega)/(omega*omegax) ) \
           + np.exp(-2.0*omega)/(omegax*thetax*omth)
  #cov_mM = np.exp((2.0+M)*omega)*cov2
  cov =  np.array([cov_m2, cov_m1, var, cov_1, cov_2])
  return cov

def implied_inc_cov_continuous(params, T):
  '''
  Calculates the model implied income covariance matrix, based on a 
  continuous time model
  '''
  # read in the parameters
  var_perm          = params[0] # variance of permanent shock
  var_tran          = params[1] # variance of both slightly persistent and completely transitory shock
  omega             = params[2] # decay parameter of slightly persistent shock
  bonus             = params[3] # fraction of var_tran that is purely transitory (like a bonus)
  
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) #/* Income */
  # pre-calculate covariance matrix of a stochasic process with shocks
  # that decay at rate omega
  cov_omega = cov_omega_theta(omega, omega)
  cov_omega_m2 = cov_omega[0]
  cov_omega_m1 = cov_omega[1]
  var_omega    = cov_omega[2]
  cov_omega_1  = cov_omega[3]
  cov_omega_2  = cov_omega[4]
 
  #/* This is the covariance of Income */
  for j in range(T):
    cov_y[j,j] = 2.0/3.0*var_perm \
                   + var_tran*((1-bonus)*var_omega + 2.0*bonus)
  for j in range(1,T):
    cov_y[j-1,j] = 1.0/6.0*var_perm \
                     + var_tran*((1-bonus)*cov_omega_1 - bonus)
    cov_y[j,j-1] = cov_y[j-1,j]
  for M in range(2,T):
    for j in range(M,T):
      cov_y[j-M,j] = var_tran*(1-bonus)*cov_omega_2*np.exp(-(M-2)*omega)
      cov_y[j,j-M] = cov_y[j-M,j]
      
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_inc_cov_discrete(impact_matrix, T, sub_periods=12):
    cov_y = np.zeros((T,T))
    for k in range(impact_matrix.shape[1]):
        for i in range(T):
            for j in range(i+1):
                cov_y[i,i-j] += (impact_matrix[i+1,k] - impact_matrix[i,k])*(impact_matrix[i-j+1,k] - impact_matrix[i-j,k])
    for i in range(T):
        for j in range(i+1):
            cov_y[i-j,i] = cov_y[i,i-j]
    vech_indicesT = vech_indices(T)
    cov_y_vec = cov_y[vech_indicesT]
    return cov_y_vec

def impact_matrix_bonus(var_bonus, T, sub_periods=12):
    impact_matrix = np.zeros((T+1,(T+1)*sub_periods))
    for k in range(impact_matrix.shape[1]):
        impact_matrix[np.floor(k/(sub_periods*1.0)).astype(int),k] += (1.0/sub_periods*var_bonus)**0.5
    return impact_matrix

def impact_matrix_tran(var_tran, omega, T, sub_periods=12, pre_periods=10):
    # calc average income in current year from shock = approx (1-np.exp(-omega))/omega
    mean_income_flow = 0.0
    for k in range(sub_periods):
        mean_income_flow = np.sum(np.exp(-omega*range(k)/(sub_periods*1.0)))/sub_periods
    impact_matrix = np.zeros((T+1,(T+pre_periods+1)*sub_periods))
    num_shocks = impact_matrix.shape[1]
    for k in range(num_shocks):
        for i in range(num_shocks-k):
            index = np.floor((k+i)/(sub_periods*1.0)).astype(int) - pre_periods
            if index>=0:
                impact_matrix[index,k] += ((var_tran/sub_periods)**0.5)/(mean_income_flow*sub_periods)*np.exp(-omega*i/(sub_periods*1.0))
    return impact_matrix

def implied_inc_cov_composite(params,T):
    var_perm = params[0]
    var_tran = params[1]
    omega    = params[2]
    bonus    = params[3]
    rho      = params[4]
    perm_inc_cov = implied_inc_cov_continuous([0.0,var_perm,rho,0.0],T)
    #perm_inc_cov = implied_inc_cov_continuous([var_perm,0.0,omega,0.0],T)
    tran_inc_cov = implied_inc_cov_continuous([0.0,var_tran*(1-bonus),omega,0.0],T)
    
#    impact_tran = impact_matrix_tran(var_tran*(1-bonus),omega,T,sub_periods=5,pre_periods=10)
#    tran_inc_cov = implied_inc_cov_discrete(impact_tran, T, sub_periods=12)
        
    bonus_inc_cov = implied_inc_cov_continuous([0.0,var_tran*bonus,omega,1.0],T)
    implied_inc_cov_composite = perm_inc_cov + tran_inc_cov + bonus_inc_cov
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








