"""

"""
import numpy as np
from pathlib import Path
from min_distance import parameter_estimation, parameter_estimation_by_subgroup
from tools import vech_indices

moments_BPP_dir = Path("../../Data/BPP_moments/") 

###############################################################################
#First load the moments
empirical_moments_all = np.genfromtxt(Path(moments_BPP_dir,"moments_all_c_vector.txt"), delimiter=',')
Omega_all =    np.genfromtxt(Path(moments_BPP_dir,"moments_all_omega.txt"), delimiter=',')
T=12
#Just doing income for now - remove other moments
income_moments = np.array([[False]*2*T]*2*T, dtype=bool)
income_moments[T:,T:] = True
vech_indices2T = vech_indices(2*T)
income_moments = income_moments[vech_indices2T]
empirical_moments_inc = empirical_moments_all[income_moments]
Omega_inc = Omega_all[income_moments,:][:,income_moments]

# set up initial guess and bounds
init_params = np.array([0.005,  #permanent variance
                        0.003,  #transitory variance
                        0.5,    #decay parameter of slightly persistant transitory shock
                        0.5,   #fraction of transitory variance that has no persistence
                        0.0])  # decay parameter of perm shock
optimize_index = np.array([0,  #permanent variance
                           1,  #transitory variance
                           2,    #decay parameter of slightly persistant transitory shock
                           3,   #fraction of transitory variance that has no persistence
                          -1])  # decay parameter of perm shock
bounds     = [(0.000001,0.1),
              (0.000001,0.1),
              (0.01,5.0),
              (0.0,0.9999),     #currently the L-BFGS-B solving mechanism calculates numerical derivatives by adding epsilon at the bound, so we bound below one. This has been fixed in scipy but not yet released (as of scipy 1.4.1)
              (0.0,0.1)]

#Do estimation
estimates_cont, estimate_se_cont = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_continuous")
estimates_monthly, estimate_se_monthly = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_monthly")
estimates_monthly_allJan, estimate_se_monthly_allJan = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_monthly",var_monthly_weights=np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))

# Fix bonus at zero for MA1 calculations, as difficult (or impossible) to distinguish from theta
init_params[3] = 0.0
optimize_index[3] = -1
estimates_annual, estimate_se_annual = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_annual")
estimates_monthly_MA1, estimate_se_monthly_MA1 = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_MA1_monthly")
estimates_monthly_MA1_allJan, estimate_se_monthly_MA1_allJan = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index, model="PermTranBonus_MA1_monthly",var_monthly_weights=np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))

#block_length = T+1
#init_params =    np.array(np.concatenate(([0.005]*block_length,  #permanent variance
#                                          [0.003]*block_length,  #transitory variance
#                                          [0.5  ]*block_length,    #decay parameter of slightly persistant transitory shock
#                                          [0.5  ]*block_length,   #fraction of transitory variance that has no persistence
#                                          [0.0  ]*block_length )))  # decay parameter of perm shock
#optimize_index = np.array(np.concatenate((np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 10]),  #permanent variance
#                                          np.array([0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10])+block_length,  #transitory variance
#                                          np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0])+2*block_length,    #decay parameter of slightly persistant transitory shock
#                                          np.array([0, 0, 0, 0, 4, 4, 4, 4, 4, 9,  9,  9,  9])+3*block_length,   #fraction of transitory variance that has no persistence
#                                                   [-1 ]*block_length )))  # decay parameter of perm shock
#bounds     = [(0.000001,0.1)]*block_length + \
#             [(0.000001,0.1)]*block_length + \
#             [(0.01,5.0)     ]*block_length + \
#             [(0.0,0.9999)  ]*block_length + \
#             [(0.0,0.1)     ]*block_length
#
#estimates, estimate_se = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds, optimize_index=optimize_index)
#




#subgroup_stub = "moments_by_liquid_wealth_quantile"
#num_quantiles = 5
#subgroup_names = []
#for i in range(num_quantiles):
#    subgroup_names += ["X"+str(i+1)]
#
#
#liquid_wealth_estimates, liquid_wealth_se = parameter_estimation_by_subgroup(moments_BPP_dir,subgroup_stub,subgroup_names, T, init_params, optimize_index=optimize_index, bounds=bounds)
#

