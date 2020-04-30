# Estimates minimum distance for splurge model

require(numDeriv)
require(MCMCpack)
require(ks)

#This is the covariance of two stochastic processes, driven by the same underlying shocks, 
#but which decay at different exponential rates omega and theta
cov_omega_theta <-function(omega, theta){
  omegax = 1.0/(1.0-exp(-omega))
  thetax = 1.0/(1.0-exp(-theta))
  omth = omega + theta
  omthx = 1.0/(1.0-exp(-omth))

  var = 2.0*thetax*omegax +
         thetax*omegax*( 
                 ((2.0-exp(-theta))*(2.0-exp(-omega))+1.0)/(omth*omthx) 
                 - (3.0-exp(-theta))/(theta*thetax) - (3.0-exp(-omega))/(omega*omegax)  
                  ) +
         1.0/(omth*omegax*thetax)
  # Covariance for moving the theta process up to T+1
  cov_1 = thetax*omegax*((2.0-exp(-theta))/(theta*thetax) 
                       + 1.0/(omega*omegax) - (2.0-exp(-theta))/(omth*omthx) 
                       -1.0 ) -
           omegax/thetax*((2.0-exp(-omega))/(omth*omthx) - 1.0/(theta*thetax) ) +
           exp(-theta)/(omegax*thetax*omth)
  # Covariance for moving the theta process up to T+2
  cov_2 = -omegax/thetax * (1.0/(theta*thetax) - 1.0/(omth*omthx)) -
         omegax/thetax * (exp(-theta)*(2.0-exp(-omega))/(omth*omthx) - exp(-theta)/(theta*thetax) ) +
         exp(-2.0*theta)/(omegax*thetax*omth)
#cov_M = exp(-(M-2.0)*theta)*cov2
  # Covariance for moving the omega process up to T+1
  cov_m1 = thetax*omegax*((2.0-exp(-omega))/(omega*omegax) +
                       1.0/(theta*thetax) - (2.0-exp(-omega))/(omth*omthx)
                       -1.0 ) -
          thetax/omegax*((2.0-exp(-theta))/(omth*omthx) - 1.0/(omega*omegax) ) +
          exp(-omega)/(omegax*thetax*omth)
  # Covariance for moving the omega process up to T+2
  cov_m2 = -thetax/omegax * (1.0/(omega*omegax) - 1.0/(omth*omthx)) -
                thetax/omegax * (exp(-omega)*(2.0-exp(-theta))/(omth*omthx) - exp(-omega)/(omega*omegax) ) +
                exp(-2.0*omega)/(omegax*thetax*omth)
#cov_mM = exp((2.0+M)*omega)*cov2
  cov =  c(cov_m2, cov_m1, var, cov_1, cov_2)
  return (cov)
}

########################################################################################################################################
# Implied covariance structure under the original BPP model (no time aggregation)
implied_cov_splurge <-function(params, T) {
  
  # read in the parameters
  var_perm    <- params[1] 
  var_tran    <- params[2]
  phi         <- params[3] 
  phi_tilde   <- params[4] 
  psi         <- params[5]
  psi_tilde   <- params[6]
  omega       <- params[7]
  theta       <- params[8]
  var_c_error <- params[9] 
  varcsi      <- params[10] 
  bonus       <- params[11] 

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
  cov_om_th_0  = cov_om_th[3]
  cov_om_th_1  = cov_om_th[4]
  cov_om_th_2  = cov_om_th[5]


  #/* This is the covariance of Income */
  for (j in 1:T){
    cov_y[j,j] = 2.0/3.0*var_perm +
                   var_tran*((1-bonus)*var_omega + 2.0*bonus)
  }
  for (j in 2:T){
    cov_y[j-1,j] <- 1.0/6.0*var_perm +
                     var_tran*((1-bonus)*cov_omega_1 - bonus)
    cov_y[j,j-1] <- cov_y[j-1,j]
  }
  for (M in 2:(T-1)){
    for (j in (M+1):T){
      cov_y[j-M,j]<- var_tran*(1-bonus)*cov_omega_2*exp(-(M-2)*omega)
      cov_y[j,j-M] <- cov_y[j-M,j]
    }
  }
  #/* This is the covariance of consumption */
  for (j in 1:T){
    cov_c[j,j] <- (2.0/3.0*phi^2 + 2*phi_tilde^2)*var_perm + 
                                           var_tran*(  psi_tilde^2       *var_omega 
                                                     + 2.0*psi*psi_tilde *cov_om_th_0 
                                                     + psi^2             *var_theta) + varcsi
  }
  for (j in 2:T){
    cov_c[j-1,j] <- (1.0/6.0*phi^2 - phi_tilde^2)*var_perm + 
                                             var_tran*(  psi_tilde^2   *cov_omega_1 
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
                                   + psi_tilde^2   *cov_omega_2) *exp(-(M-2)*omega) )
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
    cov_y_c[j,j] = (2.0/3.0*phi)*var_perm + 
                                (1-bonus)*var_tran * (  psi_tilde *var_omega 
                                                      + psi       *cov_om_th_0 ) + 
                                    bonus*var_tran * ( psi_tilde  *2.0
                                                      + psi       *(2.0/(1-exp(-theta)) - (3.0-exp(-theta))/theta))
  }
  for (j in 2:T){
    cov_y_c[j-1,j] <- (1.0/6.0*phi - 0.5*phi_tilde)*var_perm +
                             (1-bonus)*var_tran * (  psi_tilde *cov_omega_1
                                                   + psi       *cov_om_th_m1 ) + 
                                bonus *var_tran * (  psi_tilde   * (-1)
                                                   + psi       *   (1/theta - 1/(1-exp(-theta))))
    cov_y_c[j,j-1] <- (1.0/6.0*phi - 0.5*phi_tilde)*var_perm +
                             (1-bonus)*var_tran * (  psi_tilde *cov_omega_1
                                                   + psi       *cov_om_th_1 ) +
                                bonus *var_tran * (  psi_tilde   * (-1)
                                                   + psi       *   ((2.0-exp(-theta))/theta -1.0/(1-exp(-theta))  +   + (1.0-exp(-theta))^2/theta) )
  }
  for (M in 2:(T-1)){
    for (j in (M+1):T){
      cov_y_c[j-M,j] <- (1-bonus)*var_tran * (  psi_tilde *cov_omega_2
                                              + psi       *cov_om_th_m2 ) *exp(-(M-2)*omega) + 
                            bonus*var_tran * (  psi_tilde *0.0
                                              + psi       *0.0 ) *exp(-(M-2)*omega)
      cov_y_c[j,j-M] <- (1-bonus)*var_tran * (  psi_tilde *cov_omega_2   *exp(-(M-2)*omega)
                                              + psi       *cov_om_th_2   *exp(-(M-2)*theta) ) +
                            bonus*var_tran * (  psi_tilde *0.0 
                                              + psi  * (-(1-exp(-theta))^3/theta) )    *exp(-(M-2)*theta)
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
splurge_parameter_estimation <- function(c_vector, Omega, T, init_params=NULL, fixed_index=NULL,moments_for_estimation=NULL){
  
  #set initial parameters if not given
  if (is.null(init_params)){
    init_params <- matrix(0,nrow=9,ncol=1)
    init_params[1] <- 0.003  # var_perm
    init_params[2] <- 0.003  # var_tran
    init_params[3] <- 0.9    # phi
    init_params[4] <- 0.9    # phi_tilde
    init_params[5] <- 0.3    # psi
    init_params[6] <- 0.2    # psi_tilde
    init_params[7] <- 5.0    # omega - exponential decay of transitory shock
    init_params[8] <- 0.4    # theta - exponential decay of consumption response to transitory shock
    init_params[9] <- 0.06   # var_c_error 
    init_params[10] <- 0.01 #variance of taste shocks
    init_params[11] <- 0.00 #bonus
  }

  #fix certain parameters if required
  if (!is.null(fixed_index)) {
      estimate_params = init_params[!fixed_index]
      fixed_values = init_params[fixed_index]
  } else {
    estimate_params = init_params
    fixed_index = matrix(FALSE,nrow=length(init_params),ncol=0)
    fixed_values = matrix(0,nrow=0,ncol=0)
  }
  #lower and upper bounds for parameters
  lower = init_params*0.0 +0.001
  lower[5] = -0.5 #allow psi to go negative
  upper = init_params*0.0 + 20.0
  lower = lower[!fixed_index]
  upper = upper[!fixed_index]

  #Select moments used in the estimation
  if (is.null(moments_for_estimation)){
    moments_for_estimation = matrix(TRUE,nrow=length(c_vector),ncol=1)
  }
  c_vector_used = c_vector[moments_for_estimation]
  Omega_used    = Omega[moments_for_estimation,moments_for_estimation]

  implied_cov_splurge_varying_params_only <-function(estimate_params, T,fixed_index,fixed_values,moments_for_estimation) {
    params = matrix(0,nrow=length(fixed_index),ncol=1)
    params[!fixed_index] = estimate_params
    params[fixed_index] = fixed_values
    model_cov <- implied_cov_splurge(params, T)
    model_cov_used = model_cov[moments_for_estimation]
    return (model_cov_used)
  }
  objectiveFun <-function(estimate_params, T, empirical_cov, weight_matrix,fixed_index,fixed_values,moments_for_estimation){
    model_cov <- implied_cov_splurge_varying_params_only(estimate_params, T,fixed_index,fixed_values,moments_for_estimation)
    distance <- (model_cov-empirical_cov) %*% weight_matrix %*% (model_cov-empirical_cov)
    return (distance)
  }
  
  # Define the weight matrix as Equal Weight Minimum Distance
  weight_matrix <- diag(diag(Omega_used)^(-1))
  #weight_matrix <- diag(length(c_vector_used))
  
  ret <- objectiveFun(estimate_params, T, c_vector_used, weight_matrix,fixed_index,fixed_values,moments_for_estimation)
  
  #solved_objective <- nlm(objectiveFun, estimate_params, T, c_vector_used, weight_matrix,fixed_index,fixed_values,moments_for_estimation, iterlim = 1000, print.level=2)
  #solved_params <- solved_objective$estimate
  solved_objective <- optim(estimate_params, objectiveFun, gr=NULL, T, c_vector_used, weight_matrix,fixed_index,fixed_values,moments_for_estimation, method="L-BFGS-B",lower=lower,upper=upper)
  solved_params <- solved_objective$par
  jacob <- jacobian(implied_cov_splurge_varying_params_only, solved_params,T=T,fixed_index=fixed_index,fixed_values=fixed_values,moments_for_estimation=moments_for_estimation)
  
  Sandwich1 <- solve(t(jacob) %*% weight_matrix %*% jacob)
  Sandwich2 <- t(jacob) %*% weight_matrix %*% Omega_used %*% weight_matrix %*% jacob
  cov_params <- Sandwich1 %*% Sandwich2 %*% Sandwich1
  standard_errors <- diag(cov_params)^0.5
  
  output_params = matrix(0,nrow=length(fixed_index),ncol=1)
  output_params[!fixed_index] = solved_params
  output_params[fixed_index] = fixed_values
  
  output_se = matrix(0,nrow=length(fixed_index),ncol=1)
  output_se[!fixed_index] = standard_errors
  output_se[fixed_index] = 0.0
  
  # read solution
  var_perm    <- output_params[1] 
  var_tran    <- output_params[2]
  phi         <- output_params[3] 
  phi_tilde   <- output_params[4] 
  psi         <- output_params[5]
  psi_tilde   <- output_params[6]
  omega       <- output_params[7]
  theta       <- output_params[8]
  var_c_error <- output_params[9] 
  varcsi    <- output_params[10] 
  bonus    <- output_params[11] 

  # read standard errors
  var_perm_se    <- output_se[1] 
  var_tran_se    <- output_se[2]
  phi_se         <- output_se[3] 
  phi_tilde_se   <- output_se[4] 
  psi_se         <- output_se[5]
  psi_tilde_se   <- output_se[6]
  omega_se       <- output_se[7]
  theta_se       <- output_se[8]
  var_c_error_se <- output_se[9] 
  varcsi_se      <- output_se[10] 
  bonus_se      <- output_se[11]

  output = list("var_perm"        =var_perm,
                 "var_tran"       =var_tran, 
                "phi"             =phi, 
                "phi_tilde"       =phi_tilde, 
                "psi"             =psi, 
                "psi_tilde"       =psi_tilde, 
                "omega"           =omega, 
                "theta"           =theta, 
                "var_c_error"     =var_c_error,
                "varcsi"          =varcsi, 
                "bonus"          =bonus, 
                "var_perm_se"     =var_perm_se,
                "var_tran_se"     =var_tran_se, 
                "phi_se"          =phi_se, 
                "phi_tilde_se"    =phi_tilde_se, 
                "psi_se"          =psi_se, 
                "psi_tilde_se"    =psi_tilde_se, 
                "omega_se"        =omega_se, 
                "theta_se"        =theta_se, 
                "var_c_error_se"  =var_c_error_se,
                "varcsi_se"       =varcsi_se,
                "bonus_se"        =bonus_se)
  return (output) 
  }
  
splurge_parameter_step_by_step <- function(c_vector, Omega, T, init_params, fixed_index=NULL){

  fixed_index1 = c(FALSE,      # var_perm
                  FALSE,      # var_tran
                  TRUE,      # phi
                  TRUE,      # phi_tilde
                  TRUE,      # psi
                  TRUE,      # psi_tilde
                  FALSE,      # omega
                  TRUE,      # theta
                  TRUE,      # var_c_error
                  TRUE,     # taste shocks
                  FALSE)     # bonus
  if (!is.null(fixed_index)){
    fixed_index1 = (fixed_index | fixed_index1)
  }
  
  # select which moments to use for estimation
  use_c_cov = FALSE
  use_y_cov = TRUE
  use_yc_cov = FALSE
  moments_for_estimation_matrix = matrix(TRUE,nrow=2*T,ncol=2*T)
  moments_for_estimation_matrix[1:T,1:T]                 <-   matrix(use_c_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[(T+1):(2*T),1:T]         <-   matrix(use_yc_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[1:T,(T+1):(2*T)]         <-   matrix(use_yc_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[(T+1):(2*T),(T+1):(2*T)] <-   matrix(use_y_cov,nrow=T,ncol=T)
  moments_for_estimation = vech(moments_for_estimation_matrix)
  
  #Next replicate BPP
  BPP_output_step1      = splurge_parameter_estimation(c_vector  , Omega, T, init_params=init_params,fixed_index=fixed_index1,moments_for_estimation=moments_for_estimation) 
  
  init_params <- matrix(0,nrow=9,ncol=1)
  init_params[1] <- BPP_output_step1$var_perm    # var_perm
  init_params[2] <- BPP_output_step1$var_tran    # var_tran
  init_params[3] <- BPP_output_step1$phi         # phi
  init_params[4] <- BPP_output_step1$phi_tilde   # phi_tilde
  init_params[5] <- BPP_output_step1$psi         # psi
  init_params[6] <- BPP_output_step1$psi_tilde   # psi_tilde
  init_params[7] <- BPP_output_step1$omega       # omega - exponential decay of transitory shock
  init_params[8] <- BPP_output_step1$theta       # theta - exponential decay of consumption response to transitory shock
  init_params[9] <- BPP_output_step1$var_c_error # var_c_error  
  init_params[10] <- BPP_output_step1$varcsi    #variance of taste shocks
  init_params[11] <- BPP_output_step1$bonus    #bonus
  
  
  fixed_index2 = c(TRUE,      # var_perm
                  TRUE,      # var_tran
                  FALSE,      # phi
                  FALSE,      # phi_tilde
                  FALSE,      # psi
                  FALSE,      # psi_tilde
                  TRUE,      # omega
                  FALSE,      # theta
                  TRUE,      # var_c_error
                  TRUE,     # taste shocks
                  TRUE)     # bonus
  if (!is.null(fixed_index)){
    fixed_index2 = (fixed_index | fixed_index2)
  }
  # select which moments to use for estimation
  use_c_cov = FALSE
  use_y_cov = FALSE
  use_yc_cov = TRUE
  moments_for_estimation_matrix = matrix(TRUE,nrow=2*T,ncol=2*T)
  moments_for_estimation_matrix[1:T,1:T]                 <-   matrix(use_c_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[(T+1):(2*T),1:T]         <-   matrix(use_yc_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[1:T,(T+1):(2*T)]         <-   matrix(use_yc_cov,nrow=T,ncol=T)
  moments_for_estimation_matrix[(T+1):(2*T),(T+1):(2*T)] <-   matrix(use_y_cov,nrow=T,ncol=T)
  moments_for_estimation = vech(moments_for_estimation_matrix)
  
  #Next replicate BPP
  BPP_output_step2      = splurge_parameter_estimation(c_vector  , Omega, T, init_params=init_params,fixed_index=fixed_index2,moments_for_estimation=moments_for_estimation) 
  return (BPP_output_step2)
}
  
