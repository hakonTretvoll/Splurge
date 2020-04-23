

R_code_folder =  "C:/Users/edmun/OneDrive/Documents/Research/Denmark/Splurge/Code"
moments_BPP_dir = "C:/Users/edmun/OneDrive/Documents/Research/Denmark/Splurge/Data/BPP_moments"

source(paste(R_code_folder,"/Estimate_splurge.r", sep=""))

###############################################################################
# Full sample estimation
#First load the moments
c_vector = as.vector(t(read.csv(file=paste(moments_BPP_dir,"/moments_all_c_vector.txt", sep=""), header=FALSE, sep=",")))
Omega =    as.matrix(read.csv(file=paste(moments_BPP_dir,"/moments_all_omega.txt",    sep=""), header=FALSE, sep=","))
T=12

#Next replicate BPP
BPP_output = splurge_parameter_estimation(c_vector, Omega, T, taste=1) 