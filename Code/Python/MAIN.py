"""

"""
import numpy as np
from pathlib import Path
from min_distance import parameter_estimation, vech_indices

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
                        0.5,    #half life of slightly persistant transitory shock
                        0.5])   #fraction of transitory variance that has no persistence
bounds     = [(0.000001,0.1),
              (0.000001,0.1),
              (0.1,5.0),
              (0.0,1.0)]

# Do estimation
estimates, estimate_se = parameter_estimation(empirical_moments_inc, Omega_inc, T, init_params, bounds=bounds)