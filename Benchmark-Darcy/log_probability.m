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
%log_pi = logarithm of posterior probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function log_pi = log_probability(m,z,z_hat,sig,sig_pr)

%compute log likelihood
log_L = -sum((z-z_hat).^2)/(2*sig^2);

%compute log prior
log_pi_pr = -sum((m-sig_pr^2).^2)/(2*sig_pr^2);

%compute log posterior
log_pi = log_L+log_pi_pr;
