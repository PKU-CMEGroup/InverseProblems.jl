%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run Langevin sampler to estimate posterior distribution %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define number of chains, chain length, and lag time
N = input('number of independent Markov chains: ');
N_L = input('length of each Markov chain: ');
lag = input('lag time for measurements: ');
dt = input('time step: ');
workers = input('number of parallel workers: ');
L = N_L/lag;

%open Matlab parallel pool
%parpool(workers)

%load precomputations
load precomputations.mat

%define lag time and data matrix
data = zeros(64,L,N);        %data matrix of samples at lag times
theta_means = zeros(64,N);   %overall mean of theta

tic

parfor n=1:N

    %set initial m = log(theta) and theta mean
    m = zeros(64,1) + normrnd(0,0.1,[64 1]);
    theta_mean = zeros(64,1);

    for k=1:L

        for l=1:lag

            %compute gradient
            [z,dz] = forward_solver_with_gradient_(exp(m));
            gradient = grad_log_probability_(m,z,dz);

            %compute new m value
            m = m + gradient*dt + sqrt(2*dt)*normrnd(0,1,[64 1]);
            
            %update mean of theta
            theta_mean = theta_mean + exp(m);
        
        end
    
        %update data matrix
        data(:,k,n) = exp(m);

    end

    %update theta means
    theta_means(:,n) = theta_mean/N_L;
    
end

toc

%shut down parallel pool
poolobj = gcp('nocreate');
delete(poolobj);

%compute statistics on data set
[theta_mean,covars,autocovar] = get_statistics(data,theta_means);

%save data to Matlab workspace, labeled by N and N_L
save (['Langevin_data_N_' num2str(N) '_N_L_' num2str(N_L) '.mat'])
