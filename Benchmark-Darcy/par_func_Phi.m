function B = par_func_Phi(A)
    persistent pool_initialized
    load precomputations.mat forward_solver_ log_probability_
    
    [m, n, ~] = size(A);
    B = zeros(m, n);
    
    if isempty(pool_initialized)
        if isempty(gcp('nocreate'))
            parpool('Processes', 'IdleTimeout', Inf);
        end
        pool_initialized = true;
    end
    
    parfor i = 1:m
        temp_row = zeros(1, n);
        for j = 1:n
            x = squeeze(A(i, j, :));
            theta = exp(x);
            z = forward_solver_(theta);
            temp_row(j) = -log_probability_(x, z);
        end
        B(i, :) = temp_row;
    end
end