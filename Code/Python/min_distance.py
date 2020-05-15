# Minimum Distance calculations: model moments and optimization

import numpy as np
import pandas as pd
import numdifftools as nd
from numpy.linalg import inv
from scipy.optimize import minimize
from pathlib import Path
from tools import vech_indices, cov_omega_theta

def implied_cov_permshk_continuous(var_perm, T):
  '''
  Calculates the covariance matrix for permanent shocks in continuous time
  '''
  if isinstance(var_perm, (np.floating, float)):
      var_perm = np.ones(T+1)*var_perm # variance of permanent shock
  elif len(var_perm)!=T+1:
      return "Number of parameters must be equal to 1 or T+1"
  
  # Create covariance matrix
  cov_y  = np.zeros((T,T)) #/* Income */
  for j in range(T):
    cov_y[j,j] = 1.0/3.0*var_perm[j] + 1.0/3.0*var_perm[j+1]
  for j in range(1,T):
    cov_y[j-1,j] = 1.0/6.0*var_perm[j-1] 
    cov_y[j,j-1] = cov_y[j-1,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_cov_bonusshk_continuous(var_bonus, T):
  '''
  Calculates the covariance matrix for a transitory shock with NO persistence
  '''
  if isinstance(var_bonus, (np.floating, float)):
     var_bonus = var_bonus*np.ones(T+1)
  elif len(var_bonus)!=T+1:
     return "var_bonus must be a float or array of length T+1"
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) 
  for j in range(T):
    cov_y[j,j] = var_bonus[j-1] + var_bonus[j]
  for j in range(1,T):
    cov_y[j-1,j] = -var_bonus[j-1]
    cov_y[j,j-1] = cov_y[j-1,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_cov_expdecayshk_continuous_not_time_varying(var_expdecayshk, omega, T):
  '''
  Calculates the covariance matrix for an exponentially decaying stochastic 
  process, time aggregated in continuous time
  Doesn't allow time-varying parameters (easier to read the code!)
  '''
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) #/* Income */
  # pre-calculate covariance matrix of a stochasic process with shocks
  # that decay at rate omega
  cov_omega, components = cov_omega_theta(omega, omega)
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

def implied_cov_expdecayshk_continuous(var_expdecayshk, omega, T):
  '''
  Calculates the covariance matrix for an exponentially decaying stochastic 
  process, time aggregated in continuous time
  '''
  time_varying_omega = False
  if isinstance(var_expdecayshk, (np.floating, float)):
      var_expdecayshk = var_expdecayshk*np.ones(T+1)
  elif len(var_expdecayshk)!=T+1:
      return "var_expdecayshk must be a float or array of length T+1"
  if isinstance(omega, (np.floating, float)):
      omega = omega*np.ones(T+1)
  if len(omega)==T+1:
      time_varying_omega = (max(np.abs(np.diff(omega)))>0.0)
  else:
      return "omega must be a float or array of length T+1"
  # Set up covariance matrix, initialized to zero
  cov_y  = np.zeros((T,T)) #/* Income */
  # pre-calculate covariance matrix of a stochasic process with shocks
  # that decay at rate omega, will be updated if omega is time-varying
  cov_omega, components = cov_omega_theta(omega[0], omega[0])
  #first add in components of variance from shocks from before time starts
  for t in range(T):
      if t+2<=T-1:
          for m in range(T-t-2):
              cov_y[t,t+2+m] += var_expdecayshk[0]*components[0,3]*np.exp(-(2*t+m)*omega[0])
      if t+1<=T-1:
          cov_y[t,t+1] += var_expdecayshk[0]*components[1,3]*np.exp(-t*2*omega[0])
      cov_y[t,t]       += var_expdecayshk[0]*components[2,3]*np.exp(-t*2*omega[0])
  # Now add in compoments of variance for each time period (loop over k is for shock years, loop over t is the year for which we are calculating the variance/covariance)
  for k in np.array(range(T+2))-1:
      param_index = max(k,0) #Shocks before time -1 take same params at time -1
      # If omega is time-varying, then update the cov_omega results to reflect parameters from the shock year (K)
      if time_varying_omega:
          cov_omega, components = cov_omega_theta(omega[param_index], omega[param_index]) 
      # Now loop over the T years in which we measure variance/covariance, adding in the effect of shocks that originate at time k
      for t in range(T):
          if (t-k>=2):
              for m in range(T-t-2):
                  if t+2+m<=T-1:
                      cov_y[t,t+2+m] += var_expdecayshk[param_index]*components[0,2]*np.exp(-(2*(t-k-2)+m)*omega[param_index])
              if t+1<=T-1:
                  cov_y[t,t+1] += var_expdecayshk[param_index]*components[1,2]*np.exp(-2*(t-k-2)*omega[param_index])
              cov_y[t,t  ]     += var_expdecayshk[param_index]*components[2,2]*np.exp(-2*(t-k-2)*omega[param_index])
          if (t-k==1):
              for m in range(T-t-2):
                  if t+2+m<=T-1:
                      cov_y[t,t+2+m] += var_expdecayshk[param_index]*components[0,1]*np.exp(-m*omega[0])
              if t+1<=T-1:
                  cov_y[t,t+1] += var_expdecayshk[param_index]*components[1,1]
              cov_y[t,t  ]     += var_expdecayshk[param_index]*components[2,1]
          if (t-k==0):
              for m in range(T-t-2):
                  if t+2+m<=T-1:
                      cov_y[t,t+2+m] += var_expdecayshk[param_index]*components[0,0]*np.exp(-m*omega[0])
              if t+1<=T-1:
                  cov_y[t,t+1] += var_expdecayshk[param_index]*components[1,0]
              cov_y[t,t  ]     += var_expdecayshk[param_index]*components[2,0]
  # So far we've created an upper triangular matrix, reflect it along diagonal:
  for t in np.array(range(T)):
      for j in np.array(range(T-t-1))+1:
          cov_y[t+j,t] = cov_y[t,t+j]
  # Turn matrix into vector
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_inc_cov_monthly(impact_matrix):
    '''
    Calculates the covariance matrix for discrete monthly process defined by
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
    
    if isinstance(var_bonus, (np.floating, float)):
        var_bonus = var_bonus*np.ones(T+1)
    elif len(var_bonus)!=T+1:
        return "var_bonus must be a float or array of length T+1"
    
    K = (T+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        year = np.floor(k/(num_months*1.0)).astype(int)-1  # start with shocks that happen in year -1, as these impact the change in income in period 0 (negaitively)
        month = k%num_months # this is the modulo operator
        if year>=0:
            impact_matrix[year,k]   +=  (var_weights[month]*var_bonus[year+1])**0.5 # Note var_bonus goes from 0 to T (T+1 years), impact_matrix goes from ) to T-1 (T years))
        if year<=T-2:
            impact_matrix[year+1,k] += -(var_weights[month]*var_bonus[year+1])**0.5
    return impact_matrix

def impact_matrix_tran(var_tran, omega, T, num_months=12, pre_periods=10, var_weights = None):
    '''
      Calculates the "impact matrix" for an exponentially decaying shock shock
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
    
    if isinstance(var_tran, (np.floating, float)):
        var_tran = var_tran*np.ones(T+1)
    elif len(var_tran)!=T+1:
        return "var_tran must be a float or array of length T+1"
    if isinstance(omega, (np.floating, float)):
        omega = omega*np.ones(T+1)
        first_year_income = np.ones(T+1)*np.mean(np.exp(-omega[0]*np.array(range(num_months))/(num_months*1.0))) # sum of income received in the year following the shock
    elif len(omega)==T+1:
        for i in range(T+1):
            first_year_income = np.mean(np.exp(-omega[i]*np.array(range(num_months))/(num_months*1.0))) # sum of income received in the year following the shock
    else:
        return "omega must be a float or array of length T+1"
    
    K = (T+pre_periods+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        month = k%num_months # this is the modulo operator
        year_shock = max(0, np.floor(k/(num_months*1.0)).astype(int) - pre_periods) #var_tran is fixed at the year -1 value for all the pre-periods
        for i in range(K-k):
            year = np.floor((k+i)/(num_months*1.0)).astype(int) -1 - pre_periods
            if year>=0:
                impact_matrix[year,k] += ((var_tran[year_shock]*var_weights[month])**0.5)/(first_year_income[year_shock]*num_months)*np.exp(-omega[year_shock]*i/(num_months*1.0))
            if year>=-1 and year<=T-2:
                impact_matrix[year+1,k] += -((var_tran[year_shock]*var_weights[month])**0.5)/(first_year_income[year_shock]*num_months)*np.exp(-omega[year_shock]*i/(num_months*1.0))
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
    
    if isinstance(var_perm, (np.floating, float)):
        var_perm = var_perm*np.ones(T+1)
    elif len(var_perm)!=T+1:
        return "var_perm must be a float or array of length T+1"
    
    K = (T+1)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        year = np.floor(k/(num_months*1.0)).astype(int)-1  # start with shocks that happen in year -1, as these impact the change in income in period 0 
        month = k%num_months # this is the modulo operator
        if year>=0:
            impact_matrix[year,k]   += (var_weights[month]*var_perm[year+1])**0.5 * ((year+2)*num_months-k)/(1.0*num_months)
        if year<=T-2:
            impact_matrix[year+1,k] += (var_weights[month]*var_perm[year+1])**0.5 * (k-(year+1)*num_months)/(1.0*num_months)
    return impact_matrix

def impact_matrix_MA1(var_tran, theta, T, num_months=12, var_weights = None):
    '''
      Calculates the "impact matrix" for a "MA1" shock - in fact a shock in
      which income takes one value for 12 months, and then theta times that
      value for the second month. Replicates an MA1 process if all the shocks
      happen in the first month.
      Each row is a year [0, 1,..., T] 
      Each column is a 'month' in each year [0,1,...,11,...,12*(T+1)-1]
      Each element represents the income *change* in that row (year) from a shock that 
      occurs in the column (month-year).
    '''
    if var_weights is None:
        var_weights = np.array([1.0/num_months]*num_months)
    if len(var_weights)!=num_months:
        return "var_weight must be the same length as there are num_months"
    var_weights = var_weights/np.sum(var_weights)  # Normalize
    
    if isinstance(var_tran, (np.floating, float)):
        var_tran = var_tran*np.ones(T+1)
    elif len(var_tran)!=T+1:
        return "var_tran must be a float or array of length T+1"
    if isinstance(theta, (np.floating, float)):
        theta = theta*np.ones(T+1)
    elif len(theta)!=T+1:
        return "theta must be a float or array of length T+1"
    
    K = (T+3)*num_months
    impact_matrix = np.zeros((T,K))
    for k in range(K):
        month = k%num_months # this is the modulo operator
        year_shock = max(0, np.floor(k/(num_months*1.0)).astype(int) - 2) #var_tran is fixed at the year -1 value for all the pre-periods
        year = np.floor(k/(num_months*1.0)).astype(int)-3  # start with shocks that happen in year -1, as these impact the change in income in period 0 (negaitively)
        month = k%num_months # this is the modulo operator
        if year>=0:
            impact_matrix[year,k]   +=  (var_weights[month]*var_tran[year_shock])**0.5 * (num_months-month)/(1.0*num_months)
        if year>=-1 and year<=T-2:
            impact_matrix[year+1,k] +=  (var_weights[month]*var_tran[year_shock])**0.5 *(2*month-num_months)/(1.0*num_months)
            impact_matrix[year+1,k] +=  theta[year_shock] * (var_weights[month]*var_tran[year_shock])**0.5 *(num_months-month)/(1.0*num_months)
        if year>=-2 and year<=T-3:
            impact_matrix[year+2,k] +=  theta[year_shock] * (var_weights[month]*var_tran[year_shock])**0.5 *(2*month-num_months)/(1.0*num_months)
            impact_matrix[year+2,k] +=  -(var_weights[month]*var_tran[year_shock])**0.5 *month/(1.0*num_months)
        if year<=T-4:
            impact_matrix[year+3,k] +=  -theta[year_shock] * (var_weights[month]*var_tran[year_shock])**0.5 *month/(1.0*num_months)
    return impact_matrix

def implied_cov_permshk_annual(var_perm, T):
  '''
  Calculates the covariance matrix for permanent shocks that arrive in discrete,
  annual periods (like BPP 2008)
  '''
  if isinstance(var_perm, (np.floating, float)):
      var_perm = np.ones(T+1)*var_perm # variance of permanent shock
  elif len(var_perm)!=T+1:
      return "Number of parameters must be equal to 1 or T+1"
  
  # Create covariance matrix
  cov_y  = np.zeros((T,T)) #/* Income */
  for j in range(T):
    cov_y[j,j] = var_perm[j+1]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def implied_cov_MA1_annual(var_tran, theta, T):
  '''
  Calculates the covariance matrix for MA(1) shocks that arrive in discrete
  annual periods (like BPP 2008)
  '''
  if isinstance(var_tran, (np.floating, float)):
      var_tran = var_tran*np.ones(T+1)
  elif len(var_tran)!=T+1:
      return "var_tran must be a float or array of length T+1"
  if isinstance(theta, (np.floating, float)):
      theta = theta*np.ones(T+1)
  else:
      return "theta must be a float or array of length T+1"
  
  # Create covariance matrix
  cov_y  = np.zeros((T,T)) #/* Income */
  for j in range(T):
    cov_y[j,j] = var_tran[j+1]+(1-theta[j])**2*var_tran[j]+theta[max(j-1,0)]**2*var_tran[max(j-1,0)]
  for j in range(1,T):
    cov_y[j-1,j] = -(1-theta[j])*var_tran[j]+theta[max(j-1,0)]*(1-theta[max(j-1,0)])*var_tran[max(j-1,0)]
    cov_y[j,j-1] = cov_y[j-1,j]
  for j in range(2,T):
    cov_y[j-2,j]=-theta[max(j-1,0)]*var_tran[max(j-1,0)]
    cov_y[j,j-2] = cov_y[j-2,j]
  vech_indicesT = vech_indices(T)
  cov_y_vec=cov_y[vech_indicesT]
  return cov_y_vec

def composite_parameter_read(params,T):
    '''
    Reads in a vector of parameters and interprets it as parameters for 
    the standard composite model
    '''
    block_len = T+1
    if len(params)==4 or len(params)==5:
        var_perm = params[0]
        var_tran = params[1]
        omega    = params[2]
        bonus    = params[3]
        if len(params)==5:
            rho      = params[4]
        else:
            rho = 0.0
    elif len(params)==block_len*4 or len(params)==block_len*5:
        var_perm = params[0:block_len]
        var_tran = params[  block_len:2*block_len]
        omega    = params[2*block_len:3*block_len]
        bonus    = params[3*block_len:4*block_len]
        if len(params)==block_len*5:
            rho      = params[4*block_len:5*block_len]
            if np.array_equal(rho, np.zeros_like(rho)):
                rho = 0.0
        else:
            rho = 0.0
    else:
        return "params must be length 4,5, 4*(T+1) or 5*(T+1)" # If rho is missing, assume zero
    return var_perm, var_tran, omega, bonus, rho

def implied_inc_cov_composite_continuous(params,T):
    var_perm, var_tran, omega, bonus, rho = composite_parameter_read(params,T)
    if rho==0.0:
        perm_inc_cov = implied_cov_permshk_continuous(var_perm,T)
    else:
        perm_inc_cov = implied_cov_expdecayshk_continuous(var_perm,rho,T)
    bonus_inc_cov = implied_cov_bonusshk_continuous(var_tran*bonus,T)
    trandecay_inc_cov = implied_cov_expdecayshk_continuous(var_tran*(1-bonus),omega,T)
    implied_inc_cov_composite = perm_inc_cov + bonus_inc_cov + trandecay_inc_cov
    return implied_inc_cov_composite

def implied_inc_cov_composite_monthly(params,T, var_monthly_weights=None):
    var_perm, var_tran, omega, bonus, rho = composite_parameter_read(params,T)
    if (rho!=0.0):
        return "Monthly model cannot handle permanent shock decay"
    if (var_monthly_weights is not None):
        if len(var_monthly_weights)==2:
            var_perm_weights = var_monthly_weights[0]
            var_tran_weights = var_monthly_weights[1]
        else:
            var_perm_weights = var_monthly_weights
            var_tran_weights = var_monthly_weights
        num_months = len(var_perm_weights) 
    else:
        num_months = 12
        var_perm_weights = np.array([1.0]*num_months)
        var_tran_weights = np.array([1.0]*num_months)
            
    impact_tran = impact_matrix_tran(var_tran*(1-bonus),omega,T,num_months=num_months,var_weights=var_tran_weights)
    trandecay_inc_cov = implied_inc_cov_monthly(impact_tran)
    impact_bonus = impact_matrix_bonus(var_tran*bonus,T,num_months=num_months,var_weights=var_tran_weights)
    bonus_inc_cov = implied_inc_cov_monthly(impact_bonus)
    impact_perm = impact_matrix_perm(var_perm,T,num_months=num_months,var_weights=var_perm_weights)
    perm_inc_cov = implied_inc_cov_monthly(impact_perm)
    implied_inc_cov_composite = perm_inc_cov + bonus_inc_cov + trandecay_inc_cov
    return implied_inc_cov_composite

def implied_inc_cov_composite_MA1_monthly(params,T, var_monthly_weights=None):
    '''
    Same as implied_inc_cov_composite_monthly, except the exponential decay
    transitory component is replaced by an MA1 component
    '''
    var_perm, var_tran, theta, bonus, rho = composite_parameter_read(params,T)
    if (rho!=0.0):
        return "Monthly model cannot handle permanent shock decay"
    if (var_monthly_weights is not None):
        if len(var_monthly_weights)==2:
            var_perm_weights = var_monthly_weights[0]
            var_tran_weights = var_monthly_weights[1]
        else:
            var_perm_weights = var_monthly_weights
            var_tran_weights = var_monthly_weights
        num_months = len(var_perm_weights)  
    else:
        num_months = 12
        var_perm_weights = np.array([1.0]*num_months)
        var_tran_weights = np.array([1.0]*num_months)
              
    impact_tran = impact_matrix_MA1(var_tran*(1-bonus),theta,T,num_months,var_tran_weights)
    trandecay_inc_cov = implied_inc_cov_monthly(impact_tran)
    impact_bonus = impact_matrix_bonus(var_tran*bonus,T,num_months,var_tran_weights)
    bonus_inc_cov = implied_inc_cov_monthly(impact_bonus)
    impact_perm = impact_matrix_perm(var_perm,T,num_months,var_perm_weights)
    perm_inc_cov = implied_inc_cov_monthly(impact_perm)
    implied_inc_cov_composite = perm_inc_cov + bonus_inc_cov + trandecay_inc_cov
    return implied_inc_cov_composite

def implied_inc_cov_composite_annual(params,T):
    var_perm, var_tran, omega, bonus, rho = composite_parameter_read(params,T)
    if (rho!=0.0):
        return "Annual model cannot handle permanent shock decay"
    perm_inc_cov = implied_cov_permshk_annual(var_perm,T)
    bonus_inc_cov = implied_cov_bonusshk_continuous(var_tran*bonus,T) #annual model same as continuous time for bonus shocks
    MA1_inc_cov = implied_cov_MA1_annual(var_tran*(1-bonus),omega,T)
    implied_inc_cov_composite = perm_inc_cov + bonus_inc_cov + MA1_inc_cov
    return implied_inc_cov_composite

def model_covariance(params, T, model="PermTranBonus_continuous", var_monthly_weights=None):
    if (model=="PermTranBonus_continuous"): 
        model_cov = implied_inc_cov_composite_continuous(params, T)
    if (model=="PermTranBonus_annual"): 
        model_cov = implied_inc_cov_composite_annual(params, T)
    if (model=="PermTranBonus_monthly"): 
        model_cov = implied_inc_cov_composite_monthly(params, T, var_monthly_weights)
    if (model=="PermTranBonus_MA1_monthly"): 
        model_cov = implied_inc_cov_composite_MA1_monthly(params, T, var_monthly_weights)
    return model_cov

def parameter_estimation(empirical_moments, Omega, T, init_params, optimize_index=None, bounds=None, model="PermTranBonus_continuous", var_monthly_weights=None):
  '''
  Estimates model parameters
  '''
  #fix certain parameters if required
  if (optimize_index is not None):
      optimize_params = init_params[np.equal(optimize_index,range(len(optimize_index)))] # parameters to be optimized are only those that have their own index in "optimize_index"
      fixed_params      = init_params[np.equal(optimize_index,-1)] # parameters to be fixed have -1 as an entry in the optimize_index
      if (bounds is not None):
          all_bounds = bounds
          bounds = []
          for i in range(len(all_bounds)):
              if (optimize_index[i]==i):
                  bounds += [all_bounds[i]]
  else:
    optimize_params = init_params
    optimize_index = np.array(range(len(init_params)))
    fixed_params = np.array([])

  def implied_cov_limited_params(optimize_params, T, optimize_index, fixed_params,model,var_monthly_weights):
    fixed_index = np.equal(optimize_index,-1) # fixed parameters are indicated by -1 in optimize_index
    recover_index = np.array(range(len(optimize_index)))[np.equal(optimize_index,range(len(optimize_index)))] # index of each optimizing parameter in the original init_params vector
    params = np.zeros(len(optimize_index))
    params[recover_index] = optimize_params # Sets optimizing parameters equal to their entered value
    params[fixed_index]   = fixed_params # Sets fixed parameters equal to their entered value
    params[np.logical_not(fixed_index)] = params[optimize_index[np.logical_not(fixed_index)]] # Sets other parameters equal to the optimizing parameter chosen
    model_cov = model_covariance(params, T, model, var_monthly_weights)
    return model_cov
  def objectiveFun(optimize_params, T, empirical_cov, weight_matrix, optimize_index, fixed_params,model,var_monthly_weights):
    model_cov = implied_cov_limited_params(optimize_params, T, optimize_index, fixed_params,model,var_monthly_weights)
    distance = np.dot(np.dot((model_cov-empirical_cov), weight_matrix), (model_cov-empirical_cov))
    return distance

  # Define the weight matrix as Equal Weight Minimum Distance
  weight_matrix = np.diag(np.diag(Omega)**(-1))
  #ret = objectiveFun(optimize_params, T, empirical_moments, weight_matrix,optimize_index, fixed_params,model,var_monthly_weights)
  
  # Do minimization
  solved_objective = minimize(objectiveFun, optimize_params, args=(T, empirical_moments, weight_matrix, optimize_index, fixed_params,model,var_monthly_weights), method='L-BFGS-B', bounds=bounds, options= {'disp': 1})
  solved_params = solved_objective.x
  # Calculate standard errors
  fun_for_jacob = lambda params: implied_cov_limited_params(params, T, optimize_index, fixed_params,model,var_monthly_weights)
  jacob = nd.Jacobian(fun_for_jacob,step=0.00001)(solved_params)
  Sandwich1 = inv(np.dot(np.transpose(jacob),np.dot(weight_matrix,jacob)))
  Sandwich2 = np.dot(np.transpose(jacob),np.dot(weight_matrix,np.dot(Omega,np.dot(weight_matrix,jacob))))
  cov_params = np.dot(Sandwich1,np.dot(Sandwich2,Sandwich1))
  standard_errors = np.diag(cov_params)**0.5
  # Create output
  fixed_index = np.equal(optimize_index,-1) # fixed parameters are indicated by -1 in optimize_index
  recover_index = np.array(range(len(optimize_index)))[np.equal(optimize_index,range(len(optimize_index)))] # index of each optimizing parameter in the original init_params vector
  output_params = np.zeros(len(optimize_index))
  output_params[recover_index] = solved_params # Sets optimizing parameters equal to their entered value
  output_params[fixed_index]   = fixed_params # Sets fixed parameters equal to their entered value
  output_params[np.logical_not(fixed_index)] = output_params[optimize_index[np.logical_not(fixed_index)]] # Sets other parameters equal to the optimizing parameter chosen
  
  output_se = np.zeros(len(optimize_index))
  output_se[recover_index] = standard_errors
  output_se[fixed_index] = 0.0
  output_se[np.logical_not(fixed_index)] = output_se[optimize_index[np.logical_not(fixed_index)]]

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




