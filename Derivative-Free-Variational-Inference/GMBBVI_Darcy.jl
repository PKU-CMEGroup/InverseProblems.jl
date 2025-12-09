using MATLAB, MAT, Random, LinearAlgebra, Distributed, JLD2
include("../Inversion/GMBBVI.jl")
include("../Inversion/Plot.jl")
include("../Benchmark-Darcy/mean_ref.jl")

script_dir = pwd()
mat"cd($script_dir)"
mat"addpath('../Benchmark-Darcy')"
@info "MATLAB files loaded successfully"

function func_Phi_par(x_ens)
    return mxcall(:par_func_Phi, 1, x_ens)
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
N_ens = 15*N_x
N_iter = 200
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

obj = load("gmbbvi_Darcy.jld2")["obj"]
x0_w, x0_mean, xx0_cov = exp.(obj.logx_w[end]), obj.x_mean[end], obj.xx_cov[end]
x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = x0_w, x0_mean, xx0_cov

# run GMBBVI
@time  obj = Gaussian_mixture_GMBBVI_par(func_Phi_par, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal; N_iter = N_iter, dt = dt, N_ens = N_ens)

open("Darcy_data.txt", "w") do io
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

@save "gmbbvi_Darcy.jld2" obj

#######################

function calculate_errors(x_w, x_mean, xx_cov, N_ens=0)
    N_modes, N_x= size(x_mean)
    error = zeros(3,N_modes)
    V = zeros(1,N_modes,N_x)
    V[1,:,:] .= x_mean
    error[1,:] .= func_Phi_par(V)[1,:]
    for im=1:N_modes
        error[2,im] = norm(xx_cov[im,:,:])
        error[3,im] = x_w[im]/sum(x_w)
    end
    
    return error

end


# V = zeros(1,2,64)
# V[1,1,:] = log.(mean_ref)
# V[1,1,:] = log.(mean_ref-error)

# @show func_Phi_par(V)

obj = load("gmbbvi_Darcy.jld2")["obj"]
N_iter = length(obj.x_mean)


N_iter = length(obj.logx_w)-1
N_modes = obj.N_modes
errors = zeros(3, N_modes, N_iter+1)

for iter =1:N_iter+1
    if (iter-1)%1==0  @info "iter", iter ,"/", N_iter   end
    errors[:, :, iter] = calculate_errors(exp.(obj.logx_w[iter]), obj.x_mean[iter], obj.xx_cov[iter])
end

fig, ax = PyPlot.subplots(ncols=4, nrows=1, figsize=(19,6))
linestyles = ["o"; "x"; "s"; "v"; "p"; "h"; "1"; "2"; "3"; "4"]
markevery = 200


for im = 1: N_modes
    ax[1].semilogy(Array(0:N_iter), errors[1, im, :], marker=linestyles[im], color = "C"*string(im), fillstyle="none", markevery=markevery, label= "mode "*string(im))
    ax[2].semilogy(Array(0:N_iter), errors[2, im, :], marker=linestyles[im], color = "C"*string(im), fillstyle="none", markevery=markevery, label= "mode "*string(im))
    ax[3].plot(Array(0:N_iter), errors[3, im, :], marker=linestyles[im], color = "C"*string(im), fillstyle="none", markevery=markevery, label= "mode "*string(im))
end
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel(L"\Phi_R")
ax[1].legend()

ax[2].set_xlabel("Iterations")
ax[2].set_ylabel("Frobenius norm of covariance")
ax[2].legend()

ax[3].set_xlabel("Iterations")
ax[3].set_ylabel(L"Weights")
ax[3].legend()

fig.savefig("Darcy-2D-convergence.pdf")

@save "gmbbvi_Darcy_error.jld2" errors

exit(0)
