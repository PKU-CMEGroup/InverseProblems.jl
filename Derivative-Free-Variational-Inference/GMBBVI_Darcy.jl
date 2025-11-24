using MATLAB, MAT, Random, LinearAlgebra, Distributed
include("../Inversion/GMBBVI.jl")
include("../Inversion/AnnealingInitialize.jl")
include("../Benchmark-Darcy/mean_ref.jl")


script_dir = pwd()
mat"cd($script_dir)"
mat"addpath('../Benchmark-Darcy')"
mat"load('../Benchmark-Darcy/precomputations.mat')"

@info "MATLAB files loaded successfully"

function func_Phi(x)
    theta = exp.(x)
    z = mat"forward_solver_($theta)"
    log_prob = mat"log_probability_($x, $z)"
    return -log_prob
end

function compute_exp_expectation(x_w, x_mean, xx_cov)

    N_modes, N_x = size(x_mean)
    x_w ./= sum(x_w)
    exp_mean = zeros(N_x)

    Threads.@threads for i in 1:N_x
        exp_mean[i] = sum(x_w[im]*exp( x_mean[im,i]+xx_cov[im,i,i]/2) for im = 1:N_modes)
    end

    return exp_mean

end


Random.seed!(123);

N_x = 64 # dimension of Darcy problem (fixed)
N_modes = 1  # number of modes in Gaussian mixture
N_ens = 8*N_x 
N_iter = 1000  
dt = 0.5


σ_0 = 2.0
x0_w  = ones(N_modes)/N_modes
μ0, Σ0 = σ_0*σ_0*ones(N_x), Matrix(σ_0*σ_0*I(N_x)) 
x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
for im = 1:N_modes
    x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
    xx0_cov[im, :, :] .= Σ0
end

@info "Running: Darcy problem"
@info "N_modes = ", N_modes, "N_iter = ", N_iter, "N_ens = ", N_ens

# annealing process
# x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = initialize_with_annealing(func_Phi, x0_w, x0_mean, xx0_cov, N_iter = 1)
x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = x0_w, x0_mean, xx0_cov

# run GMBBVI
@time  obj = Gaussian_mixture_GMBBVI(func_Phi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal; N_iter = N_iter, dt = dt, N_ens = N_ens)

open("data.txt", "w") do io
    for iter = 0:N_iter
        x_w  = exp.(obj.logx_w[iter+1])
        xx_cov  = obj.xx_cov[iter+1]
        x_mean  = obj.x_mean[iter+1]
        error_vec = mean_ref - compute_exp_expectation(x_w, x_mean, xx_cov)
        rel_error = norm(error_vec) / norm(mean_ref)
        pre_error = norm(error_vec./mean_ref)
        line = string(iter, " ", rel_error, " ", pre_error)
        println(line)      # print to terminal
        println(io, line)  # print to data.txt
        
        if iter == N_iter
            println("error vector:")
            println(error_vec)
        end
    end
end

mat"quit"
exit(0)
