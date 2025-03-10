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
    "single_Gaussian, Gaussian_mixture"
    random_quadrature_type
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
                random_quadrature_type::String = "Gaussian_mixture",
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
            sqrt_matrix_type, random_quadrature_type, N_ens, w_min)
end

function Gaussian_mixture_sampler(logx_w, x_mean, sqrt_xx_cov, N_sample)
    x_w = exp.(logx_w)
    x_w /= sum(x_w)
    modes_dist = Categorical(x_w)
    modes_sample = rand(modes_dist,N_sample)
    N_modes, N_x = size(x_mean)
    xs = zeros(N_sample,N_x)
    for i = 1:N_sample
        im = modes_sample[i]
        xs[i,:] = (sqrt_xx_cov[im]*rand(Normal(0,1), N_x)+x_mean[im,:])'
    end
    return xs
end
   
""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi)"""
function update_ensemble!(gmgd::BBVIObj{FT, IT}, func_Phi::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance
    random_quadrature_type = gmgd.random_quadrature_type

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]
    x_w = exp.(logx_w)
    x_w ./= sum(x_w)

    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=sqrt_matrix_type) 
        push!(sqrt_xx_cov, sqrt_cov)
        push!(inv_sqrt_xx_cov, inv_sqrt_cov) 
    end

    N_ens = gmgd.N_ens
    d_logx_w, d_x_mean, d_xx_cov = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    if random_quadrature_type == "single_Gaussian"
        for im = 1:N_modes 

            # generate sampling points subject to Normal(x_mean [im,:], xx_cov[im]), size=(N_ens, N_x)
            x_p =  construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = nothing, N_ens = N_ens)

            # x_p = zeros(N_x,2*N_x+1)
            # x_p[:,1:N_x] = sqrt_xx_cov/(N_x+0.5)
            # x_p[:,N_x+1:2*N_x] = -sqrt_xx_cov/(N_x+0.5)
            # x_p += x_mean[im,:]*ones(2*N_x+1)'
            # x_p = x_p'

            # log_ratio[i] = logρ[x_p[i,:]] + log func_Phi[x_p[i,:]]
            log_ratio = zeros(N_ens) 
            for i = 1:N_ens
                for imm = 1:N_modes
                    log_ratio[i] += x_w[imm]*Gaussian_density_helper(x_mean[imm,:], inv_sqrt_xx_cov[imm], x_p[i,:])
                end
                log_ratio[i] = log(log_ratio[i])+func_Phi(x_p[i,:])
            end

            # E[logρ+Phi]
            log_ratio_mean = mean(log_ratio)

            # E[(x-m)(logρ+Phi)]
            log_ratio_m1 = mean( (x_p[i,:]-x_mean[im,:])*(log_ratio[i] - log_ratio_mean) for i=1:N_ens)   

            # E[(x-m)(x-m)'(logρ+Phi)] - E[(x-m)(x-m)'] E(logρ+Phi)
            # E[(x-m)(x-m)'(logρ+Phi - E(logρ+Phi))] 
            log_ratio_m2 = mean(( x_p[i,:]-x_mean[im,:])*((x_p[i,:]-x_mean[im,:])'*(log_ratio[i] - log_ratio_mean)) for i=1:N_ens)  
            
            d_x_mean[im,:] = -log_ratio_m1
            d_xx_cov[im,:,:] = -log_ratio_m2
            d_logx_w[im] = -log_ratio_mean

        end
    
    elseif random_quadrature_type == "Gaussian_mixture"

        N_single = div(N_ens, N_modes)

        xs = vcat([construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = nothing, N_ens = N_single) for im = 1:N_modes]...  )  # size=(N_ens,N_x)
        
        xw = vcat([x_w[im]*ones(N_single) for im = 1:N_modes]...)

        xs_G = zeros(N_ens,N_modes)
        for im = 1:N_modes, i = 1:N_ens
            # @show size(xs)
            xs_G[i,im] = Gaussian_density_helper(x_mean[im,:], inv_sqrt_xx_cov[im], xs[i,:])
        end

        xs_GM = xs_G*x_w # xs_GM[i]: Gaussian mixture density at xs[i,:]
        xs_Phi = [func_Phi(xs[i,:]) for i = 1:N_ens]

        xs_log_ratio = log.(xs_GM) + xs_Phi
        xs_log_ratio .-= mean(xs_log_ratio)

        N = zeros(N_ens,N_modes)
        for im = 1:N_modes, i = 1:N_ens
            N[i,im] = xs_log_ratio[i]*xs_G[i,im]/xs_GM[i]
        end
        
        for im = 1:N_modes
            d_logx_w[im] = -sum(xw[i]*N[i,im] for i = 1:N_ens)/N_single
            d_x_mean[im,:] = -sum(xw[i]*(xs[i,:]-x_mean[im,:])*N[i,im] for i = 1:N_ens)/N_single
            d_xx_cov[im,:,:] = -sum(xw[i]*(xs[i,:]-x_mean[im,:])*(xs[i,:]-x_mean[im,:])'*N[i,im] for i = 1:N_ens)/N_single-d_logx_w[im]*xx_cov[im,:,:]
        end
    else 
        @error "UNDEFINED random_quadrature_type!"
    end
    
    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    matrix_norm = []
    for im = 1 : N_modes
        push!(matrix_norm, opnorm( inv_sqrt_xx_cov[im]*d_xx_cov[im,:,:]*inv_sqrt_xx_cov[im]', 2))
    end
    dt = min(dt_max,  0.5 / (maximum(matrix_norm))) # keep the matrix postive definite.
    # if gmgd.iter%50==0  @show gmgd.iter,dt  end

    # dt = dt_max


    if update_covariance
        for im =1:N_modes
            temp = inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im]*d_xx_cov[im,:,:]
            xx_cov_n[im,:,:] = xx_cov[im,:,:]*inv(I-dt*temp)
            # xx_cov_n[im,:,:] += dt*d_xx_cov[im,:,:]
            xx_cov_n[im, :, :] = Hermitian(xx_cov_n[im, :, :])
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


##########
function Gaussian_mixture_BBVI(func_Phi, x0_w, x0_mean, xx0_cov;
     diagonal_covariance::Bool = false, random_quadrature_type::String = "Gaussian_mixture", N_iter = 100, dt = 5.0e-1, N_ens = -1)

    N_modes , N_x = size(x0_mean)
    if random_quadrature_type == "Gaussian_mixture"
        if N_ens == -1  N_ens = N_modes * N_x  end
    elseif random_quadrature_type == "single_Gaussian"
        if N_ens == -1  N_ens = 2* N_x  end
    else 
        @error "UNDEFINED random_quadrature_type in BBVI"
    end

    gmgdobj=BBVIObj(
        x0_w, x0_mean, xx0_cov;
        update_covariance = true,
        diagonal_covariance = diagonal_covariance,
        sqrt_matrix_type = "Cholesky",
        random_quadrature_type = random_quadrature_type, 
        N_ens = N_ens,
        w_min = 1.0e-8)

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func_Phi, dt) 
    end
    
    return gmgdobj
end
