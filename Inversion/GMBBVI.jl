using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions
include("QuadratureRule.jl")
include("GaussianMixture.jl")
include("Scheduler.jl")

mutable struct GMBBVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} 
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} 
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Vector{Array{FT, 3}}
    "a vector of lower triangular square root matrix for current time step of size (N_parameters x N_parameters)"
    sqrt_xx_cov::Vector{LowerTriangular{FT, Matrix{FT}}}
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "number of sampling points (to compute expectation using MC)"
    N_ens::IT
    "weight clipping"
    w_min::FT
end


function GMBBVIObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Array{FT, 3};
                # setup for Gaussian mixture part
                N_ens::IT = 10,
                w_min::FT = 1.0e-8) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)

    sqrt_xx_cov = [cholesky(xx0_cov[im,:,:]).L for im = 1:size(xx0_cov,1)]

    name = "GMBBVI"

    iter = 0
    GMBBVIObj(name,
            logx_w, x_mean, xx_cov, sqrt_xx_cov, N_modes, N_x,
            iter, N_ens, w_min)
end


function ensemble_GMBBVI(x_ens, forward)
    N_modes, N_ens, N_x = size(x_ens)
    F = zeros(N_modes, N_ens)   
    
    Threads.@threads for i = 1:N_ens
        for im = 1:N_modes
            F[im, i] = forward(x_ens[im, i, :])
        end
    end
    
    return F
end


function update_ensemble!(gmgd::GMBBVIObj{FT, IT}, ensemble_func::Function, dt_max::FT, iter::IT, N_iter::IT, scheduler_type::String) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]
    x_w = exp.(logx_w)
    x_w /= sum(x_w)
    sqrt_xx_cov = gmgd.sqrt_xx_cov
   
    # compute square root matrix
    inv_sqrt_xx_cov = [inv(sqrt_xx_cov[im]) for im =1:N_modes]

 
    N_ens = gmgd.N_ens
    ############ Generate sigma points
    x_p_normal = zeros(N_modes, N_ens, N_x)
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p_normal[im,:,:] = construct_ensemble(zeros(N_x), I(N_x); c_weights = nothing, N_ens = N_ens)
        x_p[im,:,:] = x_p_normal[im,:,:]*sqrt_xx_cov[im]' .+ x_mean[im,:]'
    end


    ########### function evaluation, Φᵣ, N_modes by N_ens
    Phi_R = ensemble_func(x_p)

    ########### log rho_a, N_modes by N_ens,  without 1/(2π^N_x/2) in rho_a
    log_rhoa = log.(Gaussian_mixture_density(x_w, x_mean, inv_sqrt_xx_cov, reshape(x_p, N_modes*N_ens, N_x))) 
    log_rhoa = reshape(log_rhoa, N_modes, N_ens)    
    
    ########### log_rhoa + Phi_R - E[log_rhoa + Phi_R], N_modes by N_ens
    log_ratio = log_rhoa + Phi_R
    log_ratio_mean = mean(log_ratio, dims=2)
    log_ratio_demeaned = log_ratio .- log_ratio_mean
    
    ########### compute residuals for covariances, means, weights
    d_logx_w, log_ratio_x_mean, log_ratio_xx_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes 
        # E[x(logρ+Phi(m+Lx) - E(logρ+Phi))]
        log_ratio_x_mean[im,:] = log_ratio_demeaned[im, :]' * x_p_normal[im,:,:] / N_ens   
        # E[xx'(logρ+Phi(m+Lx) - E(logρ+Phi))] 
        log_ratio_xx_mean[im,:,:] = x_p_normal[im,:,:]' * (x_p_normal[im,:,:] .* log_ratio_demeaned[im,:]) / N_ens
        d_logx_w[im] = -log_ratio_mean[im]
    end

    # Compute eigen decomposion of E[xx'(logρ+Phi(m+Lx) - E(logρ+Phi))] 
    eigens = [eigen(Symmetric(log_ratio_xx_mean[im,:,:])) for im = 1:N_modes]
    matrix_norm = [maximum(abs.(eigens[im].values)) for im = 1:N_modes]
    
    dts = scheduler(iter, N_iter, scheduler_type = scheduler_type) * min.(dt_max,   dt_max./ (matrix_norm)) 
    
    dts .= minimum(dts)
    
    
    ########### update covariances, means, weights with different time steps
    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    # update x_mean_n 
    for im =1:N_modes
        x_mean_n[im,:] += -dts[im] * sqrt_xx_cov[im] * log_ratio_x_mean[im,:]
    end

    # update xx_cov_n, sqrt_xx_cov
    for im =1:N_modes
        sqrt_xx_cov_n    = sqrt_xx_cov[im] * (eigens[im].vectors .*  (exp.(-dts[im]*0.5*eigens[im].values))')
        xx_cov_n[im,:,:] = sqrt_xx_cov_n  * sqrt_xx_cov_n'
        sqrt_xx_cov[im] .= cholesky(xx_cov_n[im,:,:]).L
        if !isposdef(Hermitian(xx_cov_n[im, :, :]))
            @show gmgd.iter
            @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :]), xx_cov_n[im, :, :]
            @assert(isposdef(xx_cov_n[im, :, :]))
        end
    end 

    
    # update logx_w_n
    logx_w_n += dts .* d_logx_w

    # Normalization
    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    logx_w_n .= log.(x_w_n)
    
    ######### Save results
    push!(gmgd.x_mean, x_mean_n)
    push!(gmgd.xx_cov, xx_cov_n)
    push!(gmgd.logx_w, logx_w_n) 

end


""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi )"""
##########
function Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov;
         N_iter = 100, dt = 5.0e-1, N_ens = -1, scheduler_type = "stable_cos_decay", w_min = 1.0e-8)

    _, N_x = size(x0_mean) 
    if N_ens == -1 
        N_ens = 2*N_x+1  
    end

    gmgdobj=GMBBVIObj(
        x0_w, x0_mean, xx0_cov;
        N_ens = N_ens,
        w_min = w_min)

    func(x_ens) = ensemble_GMBBVI(x_ens, func_Phi) 

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func, dt,  i,  N_iter, scheduler_type) 
    end
    
    return gmgdobj
end
