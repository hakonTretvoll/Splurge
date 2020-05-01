# Caculate a matrix with mean elements
c_mat_mean = invvech(c_vector)*0.0
for (t in 1:(T-1)){
  diag(c_mat_mean[t:T,1:(T+1-t)])=mean(diag(invvech(c_vector)[t:T,1:(T+1-t)]))
  diag(c_mat_mean[1:(T+1-t),t:T])=mean(diag(invvech(c_vector)[1:(T+1-t),t:T]))
  
  diag(c_mat_mean[(T+t):(2*T),1:(T+1-t)])=mean(diag(invvech(c_vector)[(T+t):(2*T),1:(T+1-t)]))
  diag(c_mat_mean[(T+1):(2*T+1-t),t:T])=mean(diag(invvech(c_vector)[(T+1):(2*T+1-t),t:T]))
  
  diag(c_mat_mean[t:T,(T+1):(2*T+1-t)])=mean(diag(invvech(c_vector)[t:T,(T+1):(2*T+1-t)]))
  diag(c_mat_mean[1:(T+1-t),(T+t):(2*T)])=mean(diag(invvech(c_vector)[1:(T+1-t),(T+t):(2*T)]))
  
  diag(c_mat_mean[(T+t):(2*T),(T+1):(2*T+1-t)])=mean(diag(invvech(c_vector)[(T+t):(2*T),(T+1):(2*T+1-t)]))
  diag(c_mat_mean[(T+1):(2*T+1-t),(T+t):(2*T)])=mean(diag(invvech(c_vector)[(T+1):(2*T+1-t),(T+t):(2*T)]))
}
c_mat_mean[T,1]=invvech(c_vector)[T,1]
c_mat_mean[1,T]=invvech(c_vector)[1,T]

c_mat_mean[(2*T),1]=invvech(c_vector)[(2*T),1]
c_mat_mean[(T+1),T]=invvech(c_vector)[(T+1),T]

c_mat_mean[T,(T+1)]=invvech(c_vector)[T,(T+1)]
c_mat_mean[1,(2*T)]=invvech(c_vector)[1,(2*T)]

c_mat_mean[(T+1),(2*T)]=invvech(c_vector)[(T+1),(2*T)]
c_mat_mean[(2*T),(T+1)]=invvech(c_vector)[(2*T),(T+1)]
c_vec_mean = vech(c_mat_mean)





