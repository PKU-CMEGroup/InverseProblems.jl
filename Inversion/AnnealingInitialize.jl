using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions

include("./EstimateTstart.jl")
include("./GMBBVI.jl")

"""
    AnnealingSchedule{FT<:AbstractFloat}

一个用于存储退火计划参数和状态的小型结构体。
"""
mutable struct AnnealingSchedule{FT<:AbstractFloat}
    "是否启用退火"
    enable::Bool
    "初始温度"
    T_start::FT
    "最终温度"
    T_end::FT
    "当前温度"
    current_T::FT
end

function AnnealingSchedule(; 
        enable::Bool=true, 
        T_start::FT=10.0, 
        T_end::FT=1.0, 
        ) where {FT<:AbstractFloat}
    
    initial_T = enable ? T_start : T_end
    return AnnealingSchedule{FT}(enable, T_start, T_end, initial_T)
end

mutable struct GMBBVIAnnealingObj{FT<:AbstractFloat, IT<:Int}
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
    "Annealing schedule object"
    annealing::AnnealingSchedule{FT}
end

function GMBBVIAnnealingObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Array{FT, 3};
                # setup for Gaussian mixture part
                N_ens::IT = 10,
                w_min::FT = 1.0e-8,
                T_start::FT = 10.0,
                T_end::FT = 1.0,
                ) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]
    push!(logx_w, log.(x0_w))
    x_mean = Array{FT,2}[]
    push!(x_mean, x0_mean)
    xx_cov = Array{FT,3}[]
    push!(xx_cov, xx0_cov)

    sqrt_xx_cov = [cholesky(xx0_cov[im,:,:]).L for im = 1:size(xx0_cov,1)]
    name = "GMBBVI_Annealing"
    iter = 0
    
    annealing_schedule = AnnealingSchedule(
        enable=true, 
        T_start=T_start, 
        T_end=T_end, 
    )

    GMBBVIAnnealingObj(name,
            logx_w, x_mean, xx_cov, sqrt_xx_cov, N_modes, N_x,
            iter, N_ens, w_min, annealing_schedule)
end


"""
    update_ensemble_annealing!

"""
function update_ensemble_annealing!(gmgd::GMBBVIAnnealingObj{FT, IT}, ensemble_func::Function, dt_max::FT, iter::IT, N_iter::IT, scheduler_type::String = "stable_cos_decay") where {FT<:AbstractFloat, IT<:Int}
    annealing = gmgd.annealing
    annealing.current_T = scheduler(iter, N_iter; η_min = annealing.T_end, η_max = annealing.T_start, scheduler_type = scheduler_type)
    current_T = annealing.current_T

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes
    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]
    x_w = exp.(logx_w)
    x_w /= sum(x_w)
    sqrt_xx_cov = gmgd.sqrt_xx_cov
    inv_sqrt_xx_cov = [inv(sqrt_xx_cov[im]) for im =1:N_modes]
    
    N_ens = gmgd.N_ens
    x_p_normal = zeros(N_modes, N_ens, N_x)
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p_normal[im,:,:] = construct_ensemble(zeros(N_x), I(N_x); c_weights = nothing, N_ens = N_ens)
        x_p[im,:,:] = x_p_normal[im,:,:]*sqrt_xx_cov[im]' .+ x_mean[im,:]'
    end

    Phi_R = ensemble_func(x_p)

    log_rhoa = log.(Gaussian_mixture_density(x_w, x_mean, inv_sqrt_xx_cov, reshape(x_p, N_modes*N_ens, N_x))) 
    log_rhoa = reshape(log_rhoa, N_modes, N_ens)    
    
    log_ratio = log_rhoa + Phi_R / current_T
    
    log_ratio_mean = mean(log_ratio, dims=2)
    log_ratio_demeaned = log_ratio .- log_ratio_mean
    
    d_logx_w, log_ratio_x_mean, log_ratio_xx_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes 
        log_ratio_x_mean[im,:] = log_ratio_demeaned[im, :]' * x_p_normal[im,:,:] / N_ens   
        log_ratio_xx_mean[im,:,:] = x_p_normal[im,:,:]' * (x_p_normal[im,:,:] .* log_ratio_demeaned[im,:]) / N_ens
        d_logx_w[im] = -log_ratio_mean[im]
    end

    eigens = [eigen(Symmetric(log_ratio_xx_mean[im,:,:])) for im = 1:N_modes]
    matrix_norm = [maximum(abs.(eigens[im].values)) for im = 1:N_modes]

    dts = min.(dt_max, dt_max ./ (matrix_norm))
    # dts = min.(scheduler(iter, N_iter, scheduler_type = scheduler_type) * dt_max,   dt_max./ (matrix_norm)) 
    # dts .= minimum(dts)


    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)
    sqrt_xx_cov_n_list = Vector{LowerTriangular{FT, Matrix{FT}}}(undef, N_modes)
    
    for im = 1:N_modes
        sqrt_xx_cov_n    = sqrt_xx_cov[im] * (eigens[im].vectors .*  (exp.(-dts[im]*0.5*eigens[im].values))')
        xx_cov_n[im,:,:] = sqrt_xx_cov_n  * sqrt_xx_cov_n'
        L = cholesky(Hermitian(xx_cov_n[im,:,:])).L
        sqrt_xx_cov_n_list[im] = L
    end 

    for im = 1:N_modes
        x_mean_n[im,:] += -dts[im] * sqrt_xx_cov_n_list[im] * log_ratio_x_mean[im,:]
    end

    logx_w_n += dts .* d_logx_w

    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    if sum(clip_ind) < length(x_w_n)
        x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    end
    logx_w_n .= log.(x_w_n)
    
    return (logx_w_n, x_mean_n, xx_cov_n, sqrt_xx_cov_n_list)
end

function initialize_with_annealing(
    func_Phi, 
    x0_w, 
    x0_mean, 
    xx0_cov;
    N_iter::Int = 100,
    dt::Float64 = 0.5,
    N_ens::Int = -1, 
    w_min::Float64 = 1.0e-8,
    alpha::Float64 = 0.1,
    scheduler_type = "stable_cos_decay"
)

    phi_max_val = 1.0e30
    function func_Phi_stable(θ::AbstractVector{T}) where {T<:AbstractFloat}
        val = func_Phi(θ)
        if isfinite(val)
            return min(val, T(phi_max_val))
        else
            return T(phi_max_val)
        end
    end

    _, N_x = size(x0_mean) 

    if N_ens == -1 
        N_ens = 2*N_x+1  
    end
    
    T_start =  estimate_T_start_from_gradients(
        func_Phi, 
        x0_w, 
        x0_mean, 
        xx0_cov;
        N_ens=N_ens,
        alpha=alpha,
        construct_ensemble_func = construct_ensemble,
        ensemble_GMBBVI_func = ensemble_GMBBVI,
        gaussian_mixture_density_func = Gaussian_mixture_density
    )

    T_end = 1.0
    @info "initialize_with_annealing: T_start = ", T_start, "T_end = ", T_end, ", scheduler_type = ", scheduler_type
    
    

    # 使用我们为退火专门创建的 GMBBVIAnnealingObj
    gmgd_anneal_obj = GMBBVIAnnealingObj(
        x0_w, x0_mean, xx0_cov;
        N_ens = N_ens,
        w_min = w_min,
        T_start = T_start,
        T_end = T_end,
    )

    func = (x_ens) -> ensemble_GMBBVI(x_ens, func_Phi) 

    i = 1
    while i <= N_iter
        logx_w_n, x_mean_n, xx_cov_n, sqrt_xx_cov_n = update_ensemble_annealing!(gmgd_anneal_obj, func, dt, i, N_iter, scheduler_type)
        push!(gmgd_anneal_obj.logx_w, logx_w_n)
        push!(gmgd_anneal_obj.x_mean, x_mean_n)
        push!(gmgd_anneal_obj.xx_cov, xx_cov_n)
        gmgd_anneal_obj.sqrt_xx_cov .= sqrt_xx_cov_n
        gmgd_anneal_obj.iter = i 
        i += 1
    end
    
    w_final = exp.(gmgd_anneal_obj.logx_w[end])
    mean_final = gmgd_anneal_obj.x_mean[end]
    cov_final = gmgd_anneal_obj.xx_cov[end]
    
    return w_final, mean_final, cov_final
end