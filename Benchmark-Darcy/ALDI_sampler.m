%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run ALDI sampler to estimate posterior distribution %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define number of chains, chain length, and lag time
N = input('number of particles: ');
N_L = input('length of each Markov chain: ');
lag = input('lag time for measurements: ');
dt = input('time step: ');
num_workers = input('number of parallel workers: ');
L = N_L/lag;

%start parallel workers
parpool(num_workers)

%load precomputations
load precomputations.mat

%define data matrix of mean theta values
data = zeros(64,L);

%define initial m = log(theta) values
ms = normrnd(0,0.1,[64 N]);

%initialize running mean values of theta
theta_mean = zeros(64,1);

tic

for k=1:L
    
    for l=1:lag

        %compute particle mean vector and covariance matrix
        ms_mean = mean(ms,2);
        sqrt_particle_covar = (ms-ms_mean)/sqrt(N);
        particle_covar = sqrt_particle_covar*sqrt_particle_covar';

        %update particle positions
        parfor n=1:N

            %compute gradients for each particle
            m = ms(:,n);
            [z,dz] = forward_solver_with_gradient_(exp(m));
            gradient = grad_log_probability_(m,z,dz);
    
            %update particle positions
            ms(:,n) = m + ...
                         particle_covar*gradient*dt + ... 
                         ((d+1)/N)*(m-ms_mean)*dt + ...
                         sqrt(2*dt)*sqrt_particle_covar*normrnd(0,1,[N 1]);
        end

        %update running mean values of theta
        theta_mean = theta_mean + mean(exp(ms),2);

    end
    
    %update data matrix
    data(:,k) = mean(exp(ms),2);

end

%normalize running mean values of theta
theta_mean = theta_mean/N_L;

toc

%shut down parallel pool
poolobj = gcp('nocreate');
delete(poolobj);

%save data to Matlab workspace, labeled by N and N_L
save (['ALDI_data_N_' num2str(N) '_N_L_' num2str(N_L) '_dt_' num2str(dt) '.mat'])
