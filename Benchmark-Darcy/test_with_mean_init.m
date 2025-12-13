%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Run MCMC sampler to estimate posterior distribution %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Fixed parameters
N = 5;                % number of independent Markov chains
N_L = 1e6;            % length of each Markov chain
lag = 10;             % lag time for measurements
workers = 5;  % still ask for workers
L = N_L / lag;

% Open MATLAB parallel pool
parpool(workers);

% Load precomputations
load precomputations.mat

% Load mean_ref from file (64x1 vector of positive floats)
mean_ref = load('mean_ref.txt');  % assumes one column, 64 rows
if size(mean_ref,1) ~= 64 || any(mean_ref <= 0)
    error('mean_ref.txt must contain 64 positive floating-point numbers.');
end

% Initialize data structures
data = zeros(64, L, N);           % data matrix of samples at lag times
theta_means = zeros(64, N);       % overall mean of theta

tic

parfor n = 1:N

    % Set initial m to log(mean_ref), and initialize accumulators
    m = log(mean_ref);            % 64 x 1 vector
    theta_mean = zeros(64, 1);
    z = forward_solver_(exp(m));  % initial forward solve

    total_steps = 0;

    for k = 1:L

        for l = 1:lag

            % Define proposal: theta_tilde in log-space
            xi = normrnd(0, sig_prop, [64, 1]);
            m_tilde = m + xi;

            % Compute new z values
            z_tilde = forward_solver_(exp(m_tilde));

            % Compute log posterior probabilities
            log_pi_tilde = log_probability_(m_tilde, z_tilde);
            log_pi = log_probability_(m, z);

            % Metropolis acceptance step
            accept = exp(log_pi_tilde - log_pi);
            if rand < accept
                m = m_tilde;
                z = z_tilde;
            end

            % Accumulate theta = exp(m) for mean estimation
            theta_mean = theta_mean + exp(m);

            total_steps = total_steps + 1;

            % print process per 1e4 steps, stressing the chain number
            if mod(total_steps, 1e4) == 0
                fprintf('Chain %d: completed %d iterations\n', n, total_steps);
            end

        end

        % Store sample at lag interval
        data(:, k, n) = exp(m);

    end

    % Compute mean over entire chain
    theta_means(:, n) = theta_mean / N_L;

end

toc

% Shut down parallel pool
poolobj = gcp('nocreate');
if ~isempty(poolobj)
    delete(poolobj);
end

% Compute statistics on dataset
[theta_mean, covars, autocovar] = get_statistics(data, theta_means);

% Save results with fixed N and N_L in filename
save(['MH_data_N_' num2str(N) '_N_L_' num2str(N_L) '.mat'], ...
     'data', 'theta_means', 'theta_mean', 'covars', 'autocovar');