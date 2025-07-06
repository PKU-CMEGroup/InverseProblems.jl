using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions
include("QuadratureRule.jl")
include("GaussianMixture.jl")

mutable struct BBVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} 
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} 
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "weather to keep covariance matrix diagonal"
    diagonal_covariance::Bool
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "number of sampling points (to compute expectation using MC)"
    N_ens::IT
    "weight clipping"
    w_min::FT
    
    # Adam optimizer parameters
    "Adam first moment estimate for weights"
    S_w::Vector{Array{FT, 1}}
    "Adam second moment estimate for weights"
    M_w::Vector{Array{FT, 1}}
    "Adam first moment estimate for means"
    S_mean::Vector{Array{FT, 2}}
    "Adam second moment estimate for means"
    M_mean::Vector{Array{FT, 2}}
    "Adam first moment estimate for covariances"
    S_cov::Vector{Array{FT, 3}}
    "Adam second moment estimate for covariances"
    M_cov::Vector{Array{FT, 3}}
    "Adam parameters"
    ρ1::FT
    ρ2::FT
    α::FT
    ϵ::FT
end


function BBVIObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                diagonal_covariance::Bool = false,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                N_ens::IT = 10,
                w_min::FT = 1.0e-8,
                # Adam parameters
                ρ1::FT = 0.9,
                ρ2::FT = 0.9,
                α::FT = 0.5,
                ϵ::FT = 1e-8) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    # Initialize Adam moments
    S_w = [zeros(size(x0_w))]
    M_w = [zeros(size(x0_w))]
    S_mean = [zeros(size(x0_mean))]
    M_mean = [zeros(size(x0_mean))]
    S_cov = [zeros(size(xx0_cov))]
    M_cov = [zeros(size(xx0_cov))]
    
    name = "BBVI"

    iter = 0

    BBVIObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance, diagonal_covariance, 
            sqrt_matrix_type, N_ens, w_min,
            S_w, M_w, S_mean, M_mean, S_cov, M_cov,
            ρ1, ρ2, α, ϵ)
end


function ensemble_BBVI(x_ens, forward)
    N_modes, N_ens, N_x = size(x_ens)
    F = zeros(N_modes, N_ens)   
    
    Threads.@threads for i = 1:N_ens
        for im = 1:N_modes
            F[im, i] = forward(x_ens[im, i, :])
        end
    end
    
    return F
end


function update_ensemble!(gmgd::BBVIObj{FT, IT}, ensemble_func::Function, dt_max::FT, iter::IT, N_iter::IT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance

    gmgd.iter += 1
    k = gmgd.iter
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    x_w = exp.(logx_w)
    xx_cov  = gmgd.xx_cov[end]
    x_w = exp.(logx_w)
    x_w /= sum(x_w)

    # compute square root matrix
    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=sqrt_matrix_type) 
        push!(sqrt_xx_cov, sqrt_cov)
        push!(inv_sqrt_xx_cov, inv_sqrt_cov) 
    end

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
    
    matrix_norm = zeros(N_modes)
    for im = 1 : N_modes
        matrix_norm[im] = opnorm(log_ratio_xx_mean[im,:,:], 2)
    end
    # set an upper bound dt_max, with cos annealing
    # dt = min(dt_max,  (0.01 + (1.0 - 0.01)*cos(pi/2 * iter/N_iter)) / (maximum(matrix_norm))) # keep the matrix postive definite, avoid too large cov update.
#     dts = min.(dt_max,  (1.0) ./ (matrix_norm)) # avoid too large cov update.
#     dts = min.(gmgd.α,  (1.0) ./ (matrix_norm)) # avoid too large cov update.
#     dts = min.((0.01 + (1.0 - 0.01)*cos(pi/2 * iter/N_iter)) * gmgd.α,  1.0 ./ (matrix_norm)) # avoid too large cov update.
#     dts .= minimum(dts)
    dts = zeros(N_modes)
    dts .= (0.01 + (1.0 - 0.01)*cos(pi/2 * iter/N_iter)) * gmgd.α

    ########### update covariances, means, weights with different time steps
    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    # Adam update for means
    new_S_mean = gmgd.ρ1 * gmgd.S_mean[end] + (1 - gmgd.ρ1) * log_ratio_x_mean
    new_M_mean = gmgd.ρ2 * gmgd.M_mean[end] + (1 - gmgd.ρ2) * (log_ratio_x_mean .^ 2)
    
    # Bias correction
    S_mean_hat = new_S_mean / (1 - gmgd.ρ1^k)
    M_mean_hat = new_M_mean / (1 - gmgd.ρ2^k)
        
#     x_mean_n += -dts .* S_mean_hat ./ (sqrt.(M_mean_hat) .+ gmgd.ϵ)
    
    if update_covariance
        # Adam update for covariances
        new_S_cov = gmgd.ρ1 * gmgd.S_cov[end] + (1 - gmgd.ρ1) * log_ratio_xx_mean
        new_M_cov = gmgd.ρ2 * gmgd.M_cov[end] + (1 - gmgd.ρ2) * (log_ratio_xx_mean .^ 2)
        
        # Bias correction
        S_cov_hat = new_S_cov / (1 - gmgd.ρ1^k)
        M_cov_hat = new_M_cov / (1 - gmgd.ρ2^k)
        
        for im =1:N_modes
            # Compute the update direction
            update_dir = S_cov_hat[im,:,:] ./ (sqrt.(M_cov_hat[im,:,:]) .+ gmgd.ϵ)
            update_dir = clamp.(update_dir, -10.0, 10.0)
            
            # Apply update to square root of covariance
            sqrt_xx_cov_n = sqrt_xx_cov[im] * exp(-0.5 * dts[im] * update_dir)
            
            xx_cov_n[im,:,:] = sqrt_xx_cov_n * sqrt_xx_cov_n'
            x_mean_n[im,:] += -dts[im] * (sqrt_xx_cov_n * S_mean_hat[im,:]) ./ (sqrt.(M_mean_hat[im,:]) .+ gmgd.ϵ)

            if diagonal_covariance
                xx_cov_n[im, :, :] = diagm(diag(xx_cov_n[im, :, :]))
            end
            if !isposdef(Hermitian(xx_cov_n[im, :, :]))
                @show gmgd.iter
                @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :]), xx_cov_n[im, :, :]
                @assert(isposdef(xx_cov_n[im, :, :]))
            end
        end
    end
    
    # Adam update for weights
    new_S_w = gmgd.ρ1 * gmgd.S_w[end] + (1 - gmgd.ρ1) * d_logx_w
    new_M_w = gmgd.ρ2 * gmgd.M_w[end] + (1 - gmgd.ρ2) * (d_logx_w .^ 2)
    
    # Bias correction
    S_w_hat = new_S_w / (1 - gmgd.ρ1^k)
    M_w_hat = new_M_w / (1 - gmgd.ρ2^k)
    
    # Update weights
    logx_w_n += dts .* S_w_hat ./ (sqrt.(M_w_hat) .+ gmgd.ϵ)
#     logx_w_n += dts .* d_logx_w

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
        
    # Store Adam moments for next iteration
    push!(gmgd.S_w, new_S_w)
    push!(gmgd.M_w, new_M_w)
    push!(gmgd.S_mean, new_S_mean)
    push!(gmgd.M_mean, new_M_mean)
    push!(gmgd.S_cov, new_S_cov)
    push!(gmgd.M_cov, new_M_cov)

end


""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi )"""
##########
function Gaussian_mixture_BBVI(func_Phi, x0_w, x0_mean, xx0_cov;
        diagonal_covariance::Bool = false, N_iter = 100, dt = 5.0e-1, N_ens = -1, 
        w_min = 1.0e-8, sqrt_matrix_type = "Cholesky",
        # Adam parameters
        ρ1 = 0.9, ρ2 = 0.9, α = 0.5, ϵ = 1e-8)

    _, N_x = size(x0_mean) 
    if N_ens == -1 
        N_ens = 2*N_x+1  
    end

    gmgdobj = BBVIObj(
        x0_w, x0_mean, xx0_cov;
        update_covariance = true,
        diagonal_covariance = diagonal_covariance,
        sqrt_matrix_type = "Cholesky",
        N_ens = N_ens,
        w_min = w_min,
        ρ1 = ρ1, ρ2 = ρ2, α = α, ϵ = ϵ)

    func(x_ens) = ensemble_BBVI(x_ens, func_Phi) 

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func, dt,  i,  N_iter) 
    end
    
    return gmgdobj
end

