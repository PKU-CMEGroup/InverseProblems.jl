%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run SVGD sampler to estimate posterior distribution %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define number of chains, chain length, and lag time
N = input('number of particles: ');
N_L = input('number of steps of gradient descent: ');
lag = input('lag time for measurements: ');
dt = input('step size: ');
alph = input('length scale parameter: ');
num_workers = input('number of parallel workers: ');
useC = input('use covariance matrix? ''y'' for yes, ''n'' for no: ');
L = N_L/lag;

%load covariance matrix data
if useC == 'y'
    covariance_matrix   %load covariance matrix Cm from pilot runs
else
    Cm = eye(64);   %use uninformed identity covariance matrix
end

%start parallel workers
parpool(num_workers)

%load precomputations
load precomputations.mat

%define data matrix of mean theta values
data = zeros(64,L);

%initialize running mean values of theta
theta_mean = zeros(64,1);

%define initial m = log(theta) values
ms = normrnd(0,0.2,[64 N]);

%initialize gradients
gradients = zeros(64,N);

tic

for k=1:L        
    for l=1:lag
        %compute gradients
        parfor i=1:N
            m = ms(:,i);
            [z,dz] = forward_solver_with_gradient_(exp(m));
            gradients(:,i) = grad_log_probability_(m,z,dz);
        end 
        %update particle positions
        dms = zeros(64,N);
        for i=1:N  
            %get particle i and initialize update vector
            mi = ms(:,i); dmi = 0; 
            for j=1:N    
                %get particle j and its gradient
                mj = ms(:,j); gradient = gradients(:,j);        
                %compute distance kernel
                K = exp(-norm(mj-mi)^2/(2*alph));
                %add to update vector
                dmi = dmi + K*(gradient-(mj-mi)/alph);   
            end       
            %store update
            dms(:,i) = dmi;    
        end            
        %update particle positions
        ms = ms + (dt/N)*Cm*dms;
        %update running mean values of theta
        theta_mean = theta_mean + mean(exp(ms),2);
    end      
    %update data matrix
    data(:,k) = mean(exp(ms,2)); 
    %periodically save data to Matlab workspace
    if k*l/1000 == floor(k*l/1000)
        save (['SVGD_data_N_' num2str(N) '_N_L_' num2str(N_L) '_dt_' ...
            num2str(dt) '_alph_' num2str(alph) '_useC_' useC '.mat'])
    end
end

toc

%normalize running mean values of theta
theta_mean = theta_mean/N_L;

%shut down parallel pool
poolobj = gcp('nocreate');
delete(poolobj);

%save data to workspace
save (['SVGD_data_N_' num2str(N) '_N_L_' num2str(N_L) '_dt_' ...
         num2str(dt) '_alph_' num2str(alph) '_useC_' useC '.mat'])

