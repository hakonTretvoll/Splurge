import numpy as np

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

def impact_matrix_tran(var_tran, half_life, T, sub_periods=12, pre_periods=10):
    omega = np.log(2)/half_life
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
    perm_var = params[0]
    tran_var = params[1]
    half_life = params[2]
    bonus = params[3]
    perm_inc_cov = implied_inc_cov_continuous([perm_var,0.0,half_life,0.0],T)
    tran_inc_cov = implied_inc_cov_continuous([0.0,tran_var*(1-bonus),half_life,0.0],T)
    bonus_inc_cov = implied_inc_cov_continuous([0.0,tran_var*bonus,half_life,1.0],T)
    implied_inc_cov_composite = perm_inc_cov + tran_inc_cov + bonus_inc_cov
    return implied_inc_cov_composite
    


var_perm = 0.005
var_tran = 0.003
bonus = 0.3
half_life = 0.8
T=4
init_params = np.array([var_perm,  #permanent variance
                        var_tran,  #transitory variance
                        half_life,    #half life of slightly persistant transitory shock
                        bonus])   #fraction of transitory variance that has no persistence
composite = implied_inc_cov_composite(init_params,T)
orig = implied_inc_cov_continuous(init_params,T)
    
sub_periods=50
init_params[0]=0.0
impact_bonus = impact_matrix_bonus(var_tran,T,sub_periods)
implied_cov_bonus_discrete = implied_inc_cov_discrete(impact_bonus, T, sub_periods)    
implied_cov_bonus_cont = implied_inc_cov_continuous(init_params, T)


init_params[3]=0.0
impact_tran = impact_matrix_tran(var_tran,half_life,T,sub_periods)
implied_cov_tran_discrete = implied_inc_cov_discrete(impact_tran, T, sub_periods)
implied_cov_tran_cont = implied_inc_cov_continuous(init_params, T)







