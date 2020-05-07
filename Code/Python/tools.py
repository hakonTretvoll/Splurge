'''
Tools that are used to calculate covariance matrices of time aggregated
exponential decay processes
'''
import numpy as np

def vech_indices(N):
    '''
    Returns the indices of the lower trianglular elements
    of an NxN matrix, to replicate functions in Matlab and R etc
    '''
    rows = [];
    columns = []
    for i in range(N):
        rows += range(i,N)
        columns += [i]*(N-i)
    return (np.array(rows), np.array(columns))

def expm1mx(x):
    '''
    Calculates exp(x) -1 - x to high precision
    '''
    if abs(x)>0.95:
        ret = np.expm1(x)-x
    else:
        shx2 = np.sinh(x/2.0)
        sh2x2 = shx2**2
        ret = 2.0 * sh2x2 + (2.0 * shx2 * (1 + sh2x2)**0.5 - x)
    return ret

def expm1mxm05x2(x):
    '''
    Calculates exp(x) -1 - x- 0.5x**2 to high precision
    '''
    if abs(x)>0.2:
        ret = expm1mx(x)-x**2/2.0
    else:
        ret = 0.0
        N=10
        for i in range(N):
            n = N-i+2
            n_factorial = 1
            for j in range(n):
                n_factorial *= (j+1)
            ret += x**n/n_factorial
    return ret
           
def cov_omega_theta(omega, theta):
  '''
    Calculates the covariance of two time aggregated exponential processes, 
    decaying at rates omega and theta
    Code is complicated by needing machine accuracy in many parts to get a 
    smooth function
  '''
  if (omega==0.0 and theta==0.0):
      cov_m2 = 0.0
      cov_m1 = 1.0/6.0
      cov_0    = 2.0/3.0
      cov_1  = 1.0/6.0
      cov_2  = 0.0
  else:
      expm1_om   = np.expm1(-omega)
      expm1_th   = np.expm1(-theta)
      expm1mx_om   = expm1mx(-omega)
      expm1mx_th   = expm1mx(-theta)
      expm1mxm05x2_om   = expm1mxm05x2(-omega)
      expm1mxm05x2_th   = expm1mxm05x2(-theta)
      omth = omega + theta
      expm1mx_omth = expm1mx(-omth)
      expm1mxm05x2_omth = expm1mxm05x2(-omth)
      if (omega==0.0):
          #covariance at T
          cov_0_T0 = -1/(expm1_th*theta**2)*(expm1mxm05x2_th + expm1mx_th*theta )
          cov_0_T1 = - (1-expm1_th)/expm1_th*( -expm1_th/theta +1/theta**2*(expm1mxm05x2_th + expm1mx_th*theta) -0.5  -0.5/(1-expm1_th))
          cov_0_Tinf = 0.0
          cov_0 = cov_0_T0 + cov_0_T1 + cov_0_Tinf
          # Covariance for moving the theta process up to T+1
          cov_1_T0 = -1/expm1_th *( (1-expm1_th)/theta**2*(1-np.exp(-theta) - theta*np.exp(-theta)) -0.5)
          cov_1_T1  = expm1_th*( -expm1_th/theta -0.5 +1/theta**2*(expm1mxm05x2_th + expm1mx_th*theta)) 
          cov_1_Tinf = 0.0
          cov_1 = cov_1_T0 + cov_1_T1 + cov_1_Tinf
          # Covariance for moving the theta process up to T+2
          cov_2_T0 = expm1_th*(0.5 - 1/theta**2*( expm1mxm05x2_th + expm1mx_th*theta ))
          cov_2_T1 = expm1_th*np.exp(-theta)*(-0.5 +1/theta*( -expm1_th  +1/theta*(expm1mxm05x2_th + expm1mx_th*theta ) ))
          cov_2_Tinf = 0.0
          cov_2 = cov_2_T0 + cov_2_T1 + cov_2_Tinf  
          #cov_M = np.exp(-(M-2.0)*theta)*cov2
          # Covariance for moving the omega process up to T+1
          cov_m1_T0 = -1.0/(expm1_th*theta)*( expm1mx_th - 1/theta*( expm1mxm05x2_th+ theta*expm1mx_th ))
          cov_m1_T1 = 0.0
          cov_m1_Tinf = 0.0
          cov_m1 = cov_m1_T0 + cov_m1_T1 + cov_m1_Tinf
          # Covariance for moving the omega process up to T+2
          cov_m2 = 0.0
      elif (theta==0.0):
          reverse_return = cov_omega_theta(theta, omega) # Note the self-call, be carefull!!
          cov_0 = reverse_return[2]
          cov_1 = reverse_return[1]
          cov_2 = reverse_return[0]
          cov_m1 = reverse_return[3]
          cov_m2 = reverse_return[4]
      else:        
          #covariance at T
          cov_0_T0 = 1.0/(expm1_om*expm1_th)*(              expm1mxm05x2_om/omega +              expm1mxm05x2_th/theta -                           expm1mxm05x2_omth/omth)
          cov_0_T1 = 1.0/(expm1_om*expm1_th)*( (1-expm1_om)*expm1mxm05x2_om/omega + (1-expm1_th)*expm1mxm05x2_th/theta - (1-expm1_om)*(1-expm1_th)*expm1mxm05x2_omth/omth + (1-omth/2)*(expm1_om*expm1_th)  + omega*expm1_th/2 + theta*expm1_om/2)
          cov_0_Tinf = expm1_om*expm1_th/omth
          cov_0 = cov_0_T0 + cov_0_T1 + cov_0_Tinf
          # Covariance for moving the theta process up to T+1
          cov_1_T0 = 1.0/(expm1_om*expm1_th)*( (1-expm1_th)*(  expm1mxm05x2_omth/omth - expm1mxm05x2_th/theta ) - expm1mxm05x2_om/omega - omega*expm1_th/2.0   )
          cov_1_T1  = - expm1_th/expm1_om*( -(1-expm1_om)*expm1mx_omth/omth - expm1_om + expm1mx_th/theta) 
          cov_1_Tinf = np.exp(-theta)*expm1_om*expm1_th/omth
          cov_1 = cov_1_T0 + cov_1_T1 + cov_1_Tinf
          # Covariance for moving the theta process up to T+2
          cov_2_T0 = -expm1_th/expm1_om * ( expm1mx_omth/omth - expm1mx_th/theta)
          cov_2_T1 = -expm1_th/expm1_om * np.exp(-theta)* ( expm1mx_th/theta - (1-expm1_om)*expm1mx_omth/omth - expm1_om)
          cov_2_Tinf = np.exp(-2.0*theta)*expm1_om*expm1_th/omth
          cov_2 = cov_2_T0 + cov_2_T1 + cov_2_Tinf
          
          # Covariance for moving the omega process up to T+1
          cov_m1_T0 = 1.0/(expm1_om*expm1_th)*( (1-expm1_om)*(  expm1mxm05x2_omth/omth - expm1mxm05x2_om/omega ) - expm1mxm05x2_th/theta - theta*expm1_om/2.0   )
          cov_m1_T1  = -expm1_om/expm1_th*( -(1-expm1_th)*expm1mx_omth/omth - expm1_th + expm1mx_om/omega) 
          cov_m1_Tinf = np.exp(-omega)*expm1_th*expm1_om/omth
          cov_m1 = cov_m1_T0 + cov_m1_T1 + cov_m1_Tinf
          # Covariance for moving the omega process up to T+2
          cov_m2_T0 = -expm1_om/expm1_th * ( expm1mx_omth/omth - expm1mx_om/omega)
          cov_m2_T1 = -expm1_om/expm1_th * np.exp(-omega)* ( expm1mx_om/omega - (1-expm1_th)*expm1mx_omth/omth - expm1_th)
          cov_m2_Tinf = np.exp(-2.0*omega)*expm1_th*expm1_om/omth
          cov_m2 = cov_m2_T0 + cov_m2_T1 + cov_m2_Tinf
  return np.array([cov_m2, cov_m1, cov_0, cov_1, cov_2])

## OLD CODE which is not high precision (results in discontinuities around omega=0 or theta=0)
## Depreciated, but useful to keep as the code more easily maps into the 
## actual integrals done
#def cov_omega_theta(omega, theta):
#  omegax = -1.0/np.expm1(-omega)
#  thetax = -1.0/np.expm1(-theta)
#  omth = omega + theta
#  omthx = -1.0/np.expm1(-omth)
#  #variance at T
#  var = 2.0*thetax*omegax  \
#          + thetax*omegax*(  ((2.0-np.exp(-theta))*(2.0-np.exp(-omega))+1.0)/(omth*omthx)     \
#                           - (3.0-np.exp(-theta))/(theta*thetax) - (3.0-np.exp(-omega))/(omega*omegax)   )  \
#            +1.0/(omth*omegax*thetax)
#  # Covariance for moving the theta process up to T+1
#  cov_1 = thetax*omegax*((2.0-np.exp(-theta))/(theta*thetax)    \
#                          + 1.0/(omega*omegax) - (2.0-np.exp(-theta))/(omth*omthx)  \
#                          -1.0 )    \
#           - omegax/thetax*((2.0-np.exp(-omega))/(omth*omthx) - 1.0/(theta*thetax) )     \
#           + np.exp(-theta)/(omegax*thetax*omth)
#  # Covariance for moving the theta process up to T+2
#  cov_2 = -omegax/thetax * (1.0/(theta*thetax) - 1.0/(omth*omthx))  \
#          -omegax/thetax * (np.exp(-theta)*(2.0-np.exp(-omega))/(omth*omthx) - np.exp(-theta)/(theta*thetax) ) \
#          +np.exp(-2.0*theta)/(omegax*thetax*omth)
#  #cov_M = np.exp(-(M-2.0)*theta)*cov2
#  # Covariance for moving the omega process up to T+1
#  cov_m1 =  thetax*omegax*((2.0-np.exp(-omega))/(omega*omegax)     \
#                           + 1.0/(theta*thetax) - (2.0-np.exp(-omega))/(omth*omthx)  \
#                           -1.0 )     \
#          - thetax/omegax*((2.0-np.exp(-theta))/(omth*omthx) - 1.0/(omega*omegax) ) \
#          + np.exp(-omega)/(omegax*thetax*omth)
#  # Covariance for moving the omega process up to T+2
#  cov_m2 = -thetax/omegax * (1.0/(omega*omegax) - 1.0/(omth*omthx)) \
#           -thetax/omegax * (np.exp(-omega)*(2.0-np.exp(-theta))/(omth*omthx) - np.exp(-omega)/(omega*omegax) ) \
#           + np.exp(-2.0*omega)/(omegax*thetax*omth)
#  #cov_mM = np.exp((2.0+M)*omega)*cov2
#  cov =  np.array([cov_m2, cov_m1, var, cov_1, cov_2])
#  return cov





