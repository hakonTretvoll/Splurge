
c_vector = as.vector(t(read.csv(file=paste(moments_BPP_dir,"/moments_all_c_vector.txt", sep=""), header=FALSE, sep=",")))
Omega =    as.matrix(read.csv(file=paste(moments_BPP_dir,"/moments_all_omega.txt",    sep=""), header=FALSE, sep=","))

# c_vector = as.vector(t(read.csv(file=paste(moments_BPP_dir,"/moments_by_liquid_wealth_quantile1c_vector.txt", sep=""), header=FALSE, sep=",")))
# Omega =    as.matrix(read.csv(file=paste(moments_BPP_dir,"/moments_by_liquid_wealth_quantile1_omega.txt",    sep=""), header=FALSE, sep=","))


source(paste(R_code_folder,"/scratchpad_calcmean_mat.r", sep=""))
init_params[5] <- -0.200   # psi
init_params[6] <- 0.3  # psi_tilde
# params[7] <- 1.12    # omega
init_params[8] <- 0.1      # theta
BPP_output_mean = splurge_parameter_step_by_step(c_vec_mean, Omega, T, init_params=init_params,fixed_index=fixed_index)


params = init_params
params[1] <- BPP_output_mean$var_perm    # var_perm
params[2] <- BPP_output_mean$var_tran    # var_tran
params[3] <- BPP_output_mean$phi         # phi
params[4] <- BPP_output_mean$phi_tilde   # phi_tilde
params[5] <- BPP_output_mean$psi         # psi
params[6] <- BPP_output_mean$psi_tilde   # psi_tilde
params[7] <- BPP_output_mean$omega       # omega - exponential decay of transitory shock
params[8] <- BPP_output_mean$theta       # theta - exponential decay of consumption response to transitory shock
params[9] <- BPP_output_mean$var_c_error # var_c_error  
params[10] <- BPP_output_mean$varcsi    #variance of taste shocks
params[11] <- BPP_output_mean$bonus    #bonus

# # params[1] <- 0.0    # var_perm
  # params[5] <- -0.15   # psi
#  params[6] <- 0.3  # psi_tilde
# # params[7] <- 1.12    # omega
#  params[8] <- 1.0      # theta
# # params[11] <- 0.0    #bonus

imp_cov=invvech(implied_cov_splurge(params,T))

#plot income
plot(c_mat_mean[T+1,(T+1):(2*T)],ylim = c(-0.0015,0.0015))
lines(c_mat_mean[T+1,(T+1):(2*T)],col="red")
lines(imp_cov[T+1,(T+1):(2*T)],lty=2,col="red")
lines(c_mat_mean[1,1:T]*0.0)

#plot cov yc
plot(c_mat_mean[(T+1),1:T],col="red",ylim = c(-0.0015,0.0015))
lines(c_mat_mean[(T+1),1:T],col="red")
lines(c_mat_mean[(T+1):(2*T),1],col="blue")
lines(imp_cov[(T+1),1:T],lty=2,col="red")
lines(imp_cov[(T+1):(2*T),1],lty=2,col="blue")
lines(c_mat_mean[1,1:T]*0.0)

# plot responses
num_periods = 10
period_points = 20
time = (1:(num_periods*period_points))/period_points
perm_inc_shock = params[1]^0.5*(time*0.0+1.0)
perm_con        =params[3]*params[1]^0.5*(time*0.0+1.0)
tran_inc = params[2]^0.5*(exp(-params[7]*time))*(1-params[11])
tran_con = params[6]*params[2]^0.5*(exp(-params[7]*time))*(1-params[11]) + params[5]*params[2]^0.5*(exp(-params[8]*time))

plot(time,perm_inc_shock,lty=1,ylim=c(-perm_inc_shock[1]*1.5,perm_inc_shock[1]*1.5))
lines(time, perm_con,col="green")
lines(time,time*0.0)
lines(time, tran_inc,col="red")
lines(time, tran_con,col="orange")

#cumulative income/expenditure
tran_inc_cum = params[2]^0.5*params[11] + cumsum(tran_inc)/period_points
tran_con_inc = params[6]*tran_inc_cum + cumsum(params[5]*params[2]^0.5*(exp(-params[8]*time)))/period_points
plot(time,tran_inc_cum,ylim=c(-0.05,max(tran_inc_cum)))
lines(time,tran_con_inc, col="red")
lines(time,time*0.0)




