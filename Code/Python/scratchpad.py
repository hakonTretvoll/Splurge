import numpy as np


    


var_perm = 0.005
var_tran = 0.003
bonus = 0.3
omega = 0.8
T=4
init_params = np.array([var_perm,  #permanent variance
                        var_tran,  #transitory variance
                        omega,    #decay parameter of slightly persistant transitory shock
                        bonus])   #fraction of transitory variance that has no persistence
composite = implied_inc_cov_composite(init_params,T)
orig = implied_inc_cov_continuous(init_params,T)
    
sub_periods=50
init_params[0]=0.0
impact_bonus = impact_matrix_bonus(var_tran,T,sub_periods)
implied_cov_bonus_discrete = implied_inc_cov_discrete(impact_bonus, T, sub_periods)    
implied_cov_bonus_cont = implied_inc_cov_continuous(init_params, T)


init_params[3]=0.0
impact_tran = impact_matrix_tran(var_tran,T,sub_periods)
implied_cov_tran_discrete = implied_inc_cov_discrete(impact_tran, T, sub_periods)
implied_cov_tran_cont = implied_inc_cov_continuous(init_params, T)







