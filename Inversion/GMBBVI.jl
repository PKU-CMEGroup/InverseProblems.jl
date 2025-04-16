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
                w_min::FT = 1.0e-8) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    name = "BBVI"

    iter = 0

    BBVIObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance, diagonal_covariance, 
            sqrt_matrix_type, N_ens, w_min)
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

function construct_Gaussian_ensemble(x_mean, sqrt_cov; N_ens = 100)

    N_x = size(x_mean,1)
    # generate random weights on the fly
    c_weights = rand(Normal(0, 1), N_x, N_ens)

    xs = ones(N_ens)*x_mean' + (sqrt_cov * c_weights)'       

    return xs
end

function update_ensemble!(gmgd::BBVIObj{FT, IT}, ensemble_func::Function, dt_max::FT, iter::IT, N_iter::IT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance

    gmgd.iter += 1
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
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p[im,:,:] = construct_Gaussian_ensemble(x_mean[im,:], sqrt_xx_cov[im]; N_ens = N_ens)
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
    d_logx_w, d_x_mean, d_xx_cov = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes 
        # E[(x-m)(logρ+Phi - E(logρ+Phi))]
        d_x_mean[im,:] = - log_ratio[im, :]' * (x_p[im,:,:] .- x_mean[im,:]') / N_ens 

        # E[(x-m)(x-m)'(logρ+Phi - E(logρ+Phi))] 
        d_xx_cov[im,:,:] = -(x_p[im,:,:]' .- x_mean[im,:]) * ((x_p[im,:,:] .- x_mean[im,:]') .* log_ratio_demeaned[im,:]) / N_ens
        
        d_logx_w[im] = -log_ratio_mean[im]

    end
    
    matrix_norm = []
    for im = 1 : N_modes
        push!(matrix_norm, opnorm( d_xx_cov[im,:,:]*inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im], 2))
    end
    # set an upper bound dt_max to keep the matrix postive definite
    dt = min(dt_max,  0.99 / (maximum(matrix_norm))) 
    
    ########### update covariances, means, weights
    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    if update_covariance
        
        for im =1:N_modes
            
            xx_cov_n[im,:,:] += dt * d_xx_cov[im,:,:]
            xx_cov_n[im,:,:] = Hermitian(xx_cov_n[im,:,:])
            if diagonal_covariance
                xx_cov_n[im, :, :] = diagm(diag(xx_cov_n[im, :, :]))
            end
            if !isposdef(Hermitian(xx_cov_n[im, :, :]))
                @show gmgd.iter
                @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :])
                @assert(isposdef(xx_cov_n[im, :, :]))
            end
        end
    end 
    x_mean_n += dt * d_x_mean 
    logx_w_n += dt * d_logx_w

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
function Gaussian_mixture_BBVI(func_Phi, x0_w, x0_mean, xx0_cov;
        diagonal_covariance::Bool = false, N_iter = 100, dt = 5.0e-1, N_ens = -1, w_min = 1.0e-8)

    _, N_x = size(x0_mean) 
    if N_ens == -1 
        N_ens = 2*N_x+1  
    end

    gmgdobj=BBVIObj(
        x0_w, x0_mean, xx0_cov;
        update_covariance = true,
        diagonal_covariance = diagonal_covariance,
        sqrt_matrix_type = "Cholesky",
        N_ens = N_ens,
        w_min = w_min)

    func(x_ens) = ensemble_BBVI(x_ens, func_Phi) 

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func, dt,  i,  N_iter) 
    end
    
    return gmgdobj
end
