

R_code_folder =  "C:/Users/edmun/OneDrive/Documents/Research/Denmark/Splurge/Code"
moments_BPP_dir = "C:/Users/edmun/OneDrive/Documents/Research/Denmark/Splurge/Data/BPP_moments"

source(paste(R_code_folder,"/Estimate_splurge.r", sep=""))

###############################################################################
# Full sample estimation
#First load the moments
c_vector = as.vector(t(read.csv(file=paste(moments_BPP_dir,"/moments_all_c_vector.txt", sep=""), header=FALSE, sep=",")))
Omega =    as.matrix(read.csv(file=paste(moments_BPP_dir,"/moments_all_omega.txt",    sep=""), header=FALSE, sep=","))
T=12

init_params <- matrix(0,nrow=9,ncol=1)
init_params[1] <- 0.005  # var_perm
init_params[2] <- 0.003  # var_tran
init_params[3] <- 1.0    # phi
init_params[4] <- 0.0    # phi_tilde
init_params[5] <- 0.0    # psi
init_params[6] <- 0.0    # psi_tilde
init_params[7] <- 0.5    # omega - exponential decay of transitory shock
init_params[8] <- 1.0    # theta - exponential decay of consumption response to transitory shock
init_params[9] <- 0.06   # var_c_error  
init_params[10] <- 0.00 #variance of taste shocks
init_params[11] <- 0.5 #"bonus" parameter

fixed_index = c(FALSE,      # var_perm
                FALSE,      # var_tran
                FALSE,      # phi
                TRUE,      # phi_tilde
                FALSE,      # psi
                TRUE,      # psi_tilde
                FALSE,      # omega
                FALSE,      # theta
                TRUE,      # var_c_error
                TRUE,      # varcsi
                FALSE)     # "bonus parameter"

#Next replicate BPP
BPP_output      = splurge_parameter_step_by_step(c_vector  , Omega, T, init_params=init_params,fixed_index=fixed_index) 

 source(paste(R_code_folder,"/scratchpad_calcmean_mat.r", sep=""))
 BPP_output_mean = splurge_parameter_step_by_step(c_vec_mean, Omega, T, init_params=init_params,fixed_index=fixed_index) 


########################################################
###############################################################################
# Function to estimate parameters for each category for which we have moments
estimation_splurge_by_category<- function(moments_BPP_dir,moments_stub,category_set, T=12, init_params,fixed_index) {
  category_params = array(0, dim=c(length(category_set),8))
  category_se = array(0, dim=c(length(category_set),8))
  for (i in 1:length(category_set)){
    this_category = as.character(category_set[i])
    this_c_vector = as.vector(t(read.csv(file=paste(moments_BPP_dir,"/",moments_stub,i,"c_vector.txt", sep=""), header=FALSE, sep=",")))
    this_omega = as.matrix(read.csv(file=paste(moments_BPP_dir,"/",moments_stub,i,"_omega.txt", sep=""), header=FALSE, sep=","))
    this_output = splurge_parameter_step_by_step(this_c_vector, this_omega,T, init_params=init_params,fixed_index=fixed_index)
    category_params[i,1] = this_output$var_perm
    category_params[i,2] = this_output$var_tran
    category_params[i,3] = this_output$phi
    category_params[i,4] = this_output$phi_tilde
    category_params[i,5] = this_output$psi
    category_params[i,6] = this_output$psi_tilde
    category_params[i,7] = this_output$omega
    category_params[i,8] = this_output$theta
    category_se[i,1]     = this_output$var_perm_se
    category_se[i,2]     = this_output$var_tran_se
    category_se[i,3]     = this_output$phi_se
    category_se[i,4]     = this_output$phi_tilde_se
    category_se[i,5]     = this_output$psi_se
    category_se[i,6]    = this_output$psi_tilde_se
    category_se[i,7]   = this_output$omega_se
    category_se[i,8]    = this_output$theta_se
  }
  output = list("category_params"=category_params,"category_se"=category_se)
  return (output)
}

###############################################################################
# load liquid weath quintile data and create graph
moments_stub = "moments_by_liquid_wealth_quantile"
num_quantiles = 5
round_digits = -3
wealth_quantile_set = as.character(1:num_quantiles)
output =estimation_splurge_by_category(moments_BPP_dir,moments_stub, make.names(wealth_quantile_set), T=12,init_params=init_params,fixed_index=fixed_index)
wealth_quantile_output=output
wealth_quantile_params = output$category_params
wealth_quantile_se = output$category_se
