# Estimates minimum distance for splurge model

require(numDeriv)
require(MCMCpack)

#This is the covariance of two stochastic processes, driven by the same underlying shocks, 
#but which decay at different exponential rates omega and theta
cov_omega_theta <-function(omega, theta){
  omegax = 1.0/(1.0-exp(-omega))
  thetax = 1.0/(1.0-exp(-theta))
  omth = omega + theta
  omthx = 1.0/(1.0-exp(-omth))

  var = 2.0*thetax*omegax 
        + thetax*omegax*( 
                 ((2.0-exp(-theta))*(2.0-exp(-omega))+1.0)/(omth*omthx) 
                 - (3.0-exp(-theta))/(theta*thetax) - (3.0-exp(-omega))/(omega*omegax)  
                  ) 
        + 1.0/(omth*omegax*thetax)
  # Covariance for moving the theta process up to T+1
  cov_1 = thetax*omegax*((2.0-exp(-theta))/(theta*thetax) 
                       + 1.0/(omega*omegax) - (2.0-exp(-theta))/(omth*omthx) 
                       -1.0 ) 
          - omegax/thetax*((2.0-exp(-omega))/(omth*omthx) - 1.0/(theta*thetax) ) 
          + exp(-theta)/(omegax*thetax*omth)
  # Covariance for moving the theta process up to T+2
  cov_2 = -omegax/thetax * (1.0/(theta*thetax) - 1.0/(omth*omthx)) 
         -omegax/thetax * (exp(-theta)*(2.0-exp(-omega))/(omth*omthx) - exp(-theta)/(theta*thetax) ) 
         + exp(-2.0*theta)/(omegax*thetax*omth)
#cov_M = exp(-(M-2.0)*theta)*cov2
  # Covariance for moving the omega process up to T+1
  cov_m1 = thetax*omegax*((2.0-exp(-omega))/(omega*omegax) 
                       + 1.0/(theta*thetax) - (2.0-exp(-omega))/(omth*omthx)
                       -1.0 ) 
          - thetax/omegax*((2.0-exp(-theta))/(omth*omthx) - 1.0/(omega*omegax) ) 
          + exp(-omega)/(omegax*thetax*omth)
  # Covariance for moving the omega process up to T+2
  cov_m2 = -thetax/omegax * (1.0/(omega*omegax) - 1.0/(omth*omthx)) 
                -thetax/omegax * (exp(-omega)*(2.0-exp(-theta))/(omth*omthx) - exp(-omega)/(omega*omegax) ) 
                + exp(-2.0*omega)/(omegax*thetax*omth)
#cov_mM = exp((2.0+M)*omega)*cov2
  cov =  c(cov_m2, cov_m1, var, cov_1, cov_2)
  return (cov)
}

########################################################################################################################################
# Implied covariance structure under the original BPP model (no time aggregation)
implied_cov_splurge <-function(params, taste, T) {
  
  # read in the parameters
  var_perm    <- params[1] 
  var_tran    <- params[2]
  phi         <- params[3] 
  psi         <- params[4]
  psi_tilde   <- params[5]
  omega       <- params[6]
  theta       <- params[7]
  var_c_error <- params[8] 
  if (taste) {
    varcsi    <- params[9] }
  else {
    varcsi    <- 0.0
  }
  # Set up covariance matricies, initialized to zero
  cov_y  <- matrix(0,nrow=T,ncol=T) #/* Income */
  cov_c <- matrix(0,nrow=T,ncol=T) #/* Consumption w/o measurement error */
  cov_c_error  <- matrix(0,nrow=T,ncol=T) #/* Consumption with measurement error*/
  cov_error <- matrix(0,nrow=T,ncol=T) #/* Measurement error of consumption */
  cov_y_c <- matrix(0,nrow=T,ncol=T) #/* Cov Income Consumption */
  cov   <- matrix(0,nrow=2*T,ncol=2*T)
  
  cov_omega = cov_omega_theta(omega, omega)
  cov_omega_m2 = cov_omega[1]
  cov_omega_m1 = cov_omega[2]
  var_omega    = cov_omega[3]
  cov_omega_1  = cov_omega[4]
  cov_omega_2  = cov_omega[5]
  
  cov_theta = cov_omega_theta(theta, theta)
  cov_theta_m2 = cov_theta[1]
  cov_theta_m1 = cov_theta[2]
  var_theta    = cov_theta[3]
  cov_theta_1  = cov_theta[4]
  cov_theta_2  = cov_theta[5]
  
  cov_om_th = cov_omega_theta(omega, theta)
  cov_om_th_m2 = cov_om_th[1]
  cov_om_th_m1 = cov_om_th[2]
  var_om_th    = cov_om_th[3]
  cov_om_th_1  = cov_om_th[4]
  cov_om_th_2  = cov_om_th[5]


  #/* This is the covariance of Income */
  for (j in 1:T){
    cov_y[j,j] = 2.0/3.0*var_perm 
                  + var_tran*var_omega
  }
  for (j in 2:T){
    cov_y[j-1,j] <- 1.0/6.0*var_perm 
                    + var_tran*cov_omega_1
    cov_y[j,j-1] <- cov_y[j-1,j]
  }
  for (M in 2:(T-1)){
    for (j in (M+1):T){
      cov_y[j-M,j]<- var_tran*cov_omega_2*exp(-(M-2)*omega)
      cov_y[j,j-M] <- cov_y[j-M,j]
    }
  }
  #/* This is the covariance of consumption */
  for (j in 1:T){
    cov_c[j,j] <- 2.0/3.0*phi^2*var_perm + var_tran*(  psi_tilde^2       *var_omega 
                                                     + 2.0*psi*psi_tilde *var_om_th 
                                                     + psi^2             *var_theta) + varcsi
  }
  for (j in 2:T){
    cov_c[j-1,j] <- 1.0/6.0*phi^2*var_perm + var_tran*(  psi_tilde^2   *cov_omega_1 
                                                       + psi*psi_tilde *cov_om_th_1 
                                                       + psi*psi_tilde *cov_om_th_m1 
                                                       + psi^2         *cov_theta_1)
    cov_c[j,j-1] <- cov_c[j-1,j]
  }
  for (M in 2:(T-1)){
    for (j in (M+1):T){
      cov_c[j-M,j]<- var_tran*(   (  psi^2         *cov_theta_2  
                                   + psi*psi_tilde *cov_om_th_2) *exp(-(M-2)*theta)
                                + (  psi*psi_tilde *cov_om_th_m2 
                                   + psi_tilde     *cov_omega_2) *exp(-(M-2)*omega) )
      cov_c[j,j-M] <- cov_c[j-M,j]
    }
  }
  for (j in 1:T){
    cov_error[j,j] <- 2.0*var_c_error
  }
  for (j in 2:T){
    cov_error[j-1,j]<- -var_c_error
    cov_error[j,j-1] <- cov_error[j-1,j]
  }
  cov_c_error<-cov_error+cov_c
  
  #/* This is the covariance of income and consumption */
  for (j in 1:T){
    cov_y_c[j,j] = 2.0/3.0*phi*var_perm + var_tran * (  psi_tilde *var_omega 
                                                      + psi       *var_om_th )
  }
  for (j in 2:T){
    cov_y_c[j-1,j] <- 1.0/6.0*phi*var_perm
                        + var_tran * (  psi_tilde *cov_omega_1
                                      + psi       *cov_om_th_m1 )
    cov_y_c[j,j-1] <- 1.0/6.0*phi*var_perm
                        + var_tran * (  psi_tilde *cov_omega_1
                                      + psi       *cov_om_th_1 )
  }
  for (M in 2:(T-1)){
    for (j in (M+1):T){
      cov_y_c[j-M,j] <- var_tran * (  psi_tilde *cov_omega_2
                                    + psi       *cov_om_th_m2 ) *exp(-(M-2)*omega)
      cov_y_c[j,j-M] <- var_tran * (  psi_tilde *cov_omega_2
                                      + psi     *cov_om_th_2  ) *exp(-(M-2)*theta)
    }
  }
  #/* Final matrix */
  cov[1:T,1:T]                 <-   cov_c_error
  cov[(T+1):(2*T),1:T]         <-   cov_y_c
  cov[1:T,(T+1):(2*T)]         <- t(cov_y_c)
  cov[(T+1):(2*T),(T+1):(2*T)] <-   cov_y
  cov_vec <- vech(cov)
  
  return (cov_vec)
}

# Estimates splurge parameters
splurge_parameter_estimation <- function(c_vector, Omega, T, taste=1){
  
  init_params <- matrix(0,nrow=8+taste,ncol=1)
  init_params[1] <- 0.003  # var_perm
  init_params[2] <- 0.003  # var_tran
  init_params[3] <- 0.9    # phi
  init_params[4] <- 0.3    # psi
  init_params[5] <- 0.2    # psi_tilde
  init_params[6] <- 5.0    # omega - exponential decay of transitory shock
  init_params[7] <- 0.4    # theta - exponential decay of consumption response to transitory shock
  init_params[8] <- 0.06   # var_c_error  
  if (taste){
    init_params[9] <- 0.01 #variance of taste shocks
  }
  
  objectiveFun <-function(params, taste, T, empirical_cov, weight_matrix){
    model_cov <- implied_cov_splurge(params, taste, T)
    distance <- (model_cov-empirical_cov) %*% weight_matrix %*% (model_cov-empirical_cov)
    return (distance)
  }
  
  # Define the weight matrix as Equal Weight Minimum Distance
  weight_matrix <- diag(diag(Omega)^(-1))
  weight_matrix <- diag(length(c_vector))
  
  ret <- objectiveFun(init_params, taste, T, c_vector, weight_matrix)
  
  solved_objective <- nlm(objectiveFun, init_params, taste, T, c_vector, weight_matrix, iterlim = 1000)
  solved_params <- solved_objective$estimate
  jacob <- jacobian(implied_cov_splurge, solved_params, taste=taste,T=T)
  
  Sandwich1 <- solve(t(jacob) %*% weight_matrix %*% jacob)
  Sandwich2 <- t(jacob) %*% weight_matrix %*% Omega %*% weight_matrix %*% jacob
  cov_params <- Sandwich1 %*% Sandwich2 %*% Sandwich1
  standard_errors <- diag(cov_params)^0.5
  
  # read solution
  var_perm    <- solved_params[1] 
  var_tran    <- solved_params[2]
  phi         <- solved_params[3] 
  psi         <- solved_params[4]
  psi_tilde   <- solved_params[5]
  omega       <- solved_params[6]
  theta       <- solved_params[7]
  var_c_error <- solved_params[8] 
  if (taste) {
    varcsi    <- solved_params[9] }
  else {
    varcsi    <- 0.0
  }
  # read standard errors
  var_perm_se    <- standard_errors[1] 
  var_tran_se    <- standard_errors[2]
  phi_se         <- standard_errors[3] 
  psi_se         <- standard_errors[4]
  psi_tilde_se   <- standard_errors[5]
  omega_se       <- standard_errors[6]
  theta_se       <- standard_errors[7]
  var_c_error_se <- standard_errors[8] 
  if (taste) {
    varcsi_se    <- standard_errors[9] }
  else {
    varcsi_se    <- 0.0
  }

  output = list("var_perm"=var_perm,
                "var_perm_se"=var_perm_se,
                "var_tran"=var_tran, 
                "var_tran_se"=var_tran_se,
                "phi"=phi, 
                "phi_se"=phi_se, 
                "psi"=psi, 
                "psi_se"=psi_se, 
                "psi_tilde"=psi_tilde, 
                "psi_tilde_se"=psi_tilde_se, 
                "omega"=omega, 
                "omega_se"=omega_se, 
                "theta"=theta, 
                "theta_se"=theta_se, 
                "var_c_error"=var_c_error,
                "var_c_error_se"=var_c_error_se,
                " varcsi"=varcsi, 
                "varcsi_se"=varcsi_se)
  return (output)
}
  
  