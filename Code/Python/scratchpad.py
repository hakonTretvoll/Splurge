
import numpy as np

# calculates exp(x) -1 - x to high precision
def expm1mx(x):
    if abs(x)>0.95:
        ret = np.expm1(x)-x
    else:
        shx2 = np.sinh(x/2.0)
        sh2x2 = shx2**2
        ret = 2.0 * sh2x2 + (2.0 * shx2 * (1 + sh2x2)**0.5 - x)
    return ret
# calculates exp(x) -1 - x- 0.5x**2 to high precision
def expm1mxm05x2(x):
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
  omegax = -1.0/np.expm1(-omega)
  thetax = -1.0/np.expm1(-theta)
  omth = omega + theta
  omthx = -1.0/np.expm1(-omth)
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
  return np.array([var, cov_1, cov_2])

            
def cov_omth_test(omega, theta):
  '''
    Calculates the covariance of two time aggregated exponential processes, 
    decaying at rates omega and theta
    Code is complicated by needing machine accuracy in many parts to get a 
    smooth function
  '''
  if (omega==0.0 and theta==0.0):
      cov_m2 = 0.0
      cov_m1 = 1.0/6.0
      var    = 2.0/3.0
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
          #variance at T
          var_T0 = -1/(expm1_th*theta**2)*(expm1mxm05x2_th + expm1mx_th*theta )
          var_T1 = -np.exp(-theta)/expm1_th*( expm1mx_th/theta + 1/theta**2*(expm1mxm05x2_th+theta*expm1mx_th) + 0.5*expm1_th )
          var_Tinf = 0.0
          var = var_T0 + var_T1 + var_Tinf
          # Covariance for moving the theta process up to T+1
          cov_1_T0 = -1/expm1_th *( -np.exp(-theta)/theta**2*(1-np.exp(-theta) - theta*np.exp(-theta)) -0.5)
          cov_1_T1  = 999
          cov_1_Tinf = 0.0
          cov_1 = cov_1_T0 + cov_1_T1 + cov_1_Tinf
      elif (theta==0.0):
          var_T0 = 999
          var_T1 = 999
          var_Tinf = 999
          var = var_T0 + var_T1 + var_Tinf
          
          cov_1 = 999
          cov_2 = 999
      else:
          omth = omega + theta
          expm1mx_omth = expm1mx(-omth)

          expm1mxm05x2_omth = expm1mxm05x2(-omth)
          
          #variance at T
          var_T0 = 1.0/(expm1_om*expm1_th)*(              expm1mxm05x2_om/omega +              expm1mxm05x2_th/theta -                           expm1mxm05x2_omth/omth)
          var_T1 = 1.0/(expm1_om*expm1_th)*( (1-expm1_om)*expm1mxm05x2_om/omega + (1-expm1_th)*expm1mxm05x2_th/theta - (1-expm1_om)*(1-expm1_th)*expm1mxm05x2_omth/omth + (1-omth/2)*(expm1_om*expm1_th)  + omega*expm1_th/2 + theta*expm1_om/2)
          var_Tinf = expm1_om*expm1_th/omth
          var = var_T0 + var_T1 + var_Tinf
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
  return np.array([var, cov_1, cov_2])

omega = 0.13
theta = 0.09
x1=cov_omega_theta(omega,theta)
x2=cov_omth_test(omega,theta)
xdiff=cov_omega_theta(omega,theta)-cov_omth_test(omega,theta)
[x1,x2,xdiff]


omega = 0.000000000000001
theta = 0.09
x1=cov_omega_theta(omega,theta)
x2=cov_omth_test(omega,theta)
xdiff=cov_omega_theta(omega,theta)-cov_omth_test(omega,theta)
[x1,x2,xdiff]

omega = 0.000000000000001
theta = 0.0000000000000001
x1=cov_omega_theta(omega,theta)
x2=cov_omth_test(omega,theta)
xdiff=cov_omega_theta(omega,theta)-cov_omth_test(omega,theta)
[x1,x2,xdiff]


omega = 0.0000000000000001
theta = 0.000000000000001
x1=cov_omega_theta(omega,theta)
x2=cov_omth_test(omega,theta)
xdiff=cov_omega_theta(omega,theta)-cov_omth_test(omega,theta)
[x1,x2,xdiff]









