%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% compute log probability, log pi %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%m = current log parameter vector
%z = current vector of measurements
%z_hat = vector of "exact" measurements
%sig = standard deviation parameter in likelihood
%sig_pr = standard deviation parameter in prior
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%grad_log_pi = gradient of logarithm of posterior probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function grad_log_pi = grad_log_probability(m,z,dz,z_hat,sig,sig_pr)

%compute log likelihood: here dz is the gradient of z wrt m = log(theta)
grad_log_L = -dz'*(z-z_hat)/(sig^2);

%compute log prior
grad_log_pi_pr = -(m-sig_pr^2)/(sig_pr^2);

%compute log posterior
grad_log_pi = grad_log_L + grad_log_pi_pr;
