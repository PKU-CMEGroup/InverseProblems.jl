using LinearAlgebra
using Statistics
using Distributions 

function _compute_gradient_components(
    x_p_normal, log_ratio, sqrt_xx_cov,
)
    N_modes, N_ens, N_x = size(x_p_normal)
    # 中心化 log_ratio 
    log_ratio_mean = mean(log_ratio, dims=2)
    log_ratio_demeaned = log_ratio .- log_ratio_mean

    # 计算梯度分量
    d_logx_w = -vec(log_ratio_mean) # vec() 将 (N,1) 矩阵变为向量
    log_ratio_x_mean = zeros(N_modes, N_x)
    
    for im = 1:N_modes
        log_ratio_x_mean[im,:] = log_ratio_demeaned[im, :]' * x_p_normal[im,:,:] / N_ens
        log_ratio_x_mean[im,:] = sqrt_xx_cov[im]' * log_ratio_x_mean[im,:]
        # log_ratio_xx_mean[im,:,:] = x_p_normal[im,:,:]' * (x_p_normal[im,:,:] .* log_ratio_demeaned[im,:]) / N_ens
    end

    # 将所有梯度分量“压平”到一个长向量中
    flat_grad_w = d_logx_w
    flat_grad_mean = vec(log_ratio_x_mean)
    
    # 将所有梯度向量拼接在一起
    # total_flat_gradient = vcat(flat_grad_w, flat_grad_mean)
    total_flat_gradient = flat_grad_mean

    return norm(total_flat_gradient)
end


"""
    estimate_T_start_from_gradients(
        func_Phi, x0_w, x0_mean, x0_cov;
        N_ens=100, alpha=0.1,
        # 传入GMBBVI.jl中的辅助函数
        construct_ensemble_func,
        ensemble_GMBBVI_func,
        gaussian_mixture_density_func
    )

根据能量项和熵项的梯度范数比来估算一个合理的初始温度 T_start。

# Arguments
- `func_Phi`: 你的势能函数 Φ(θ)。
- `x0_w`, `x0_mean`, `x0_cov`: 用于定义初始近似分布 q(θ) 的GMM参数。
- `N_ens`: 用于蒙特卡洛估计的样本数量。
- `alpha`: 目标比例，即 ||能量梯度|| / ||熵梯度||。
- `..._func`: 从外部传入 GMBBVI 代码中定义的辅助函数，以保持模块独立。

# Returns
- `T_start_est`: 建议的初始温度 T_start。
"""
function estimate_T_start_from_gradients(
    func_Phi, 
    x0_w::Vector{FT}, 
    x0_mean::Matrix{FT}, 
    x0_cov::Array{FT, 3};
    N_ens::IT=100, 
    alpha::FT=0.1,
    construct_ensemble_func::Function,
    ensemble_GMBBVI_func::Function,
    gaussian_mixture_density_func::Function
) where {FT<:AbstractFloat, IT<:Int}

    # STEP 1: 设置初始的 GMM 状态
    N_modes, N_x = size(x0_mean)
    x_w = x0_w / sum(x0_w)
    sqrt_xx_cov = [cholesky(Hermitian(x0_cov[im,:,:])).L for im = 1:N_modes]
    inv_sqrt_xx_cov = [inv(L) for L in sqrt_xx_cov]

    # STEP 2: 执行一次蒙特卡洛采样
    x_p_normal = zeros(N_modes, N_ens, N_x)
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p_normal[im,:,:] = construct_ensemble_func(zeros(N_x), Matrix(1.0I, N_x, N_x); c_weights=nothing, N_ens=N_ens)
        x_p[im,:,:] = x_p_normal[im,:,:] * sqrt_xx_cov[im]' .+ x0_mean[im,:]'
    end

    # 预先计算 Phi_R 和 log_rhoa
    Phi_R = ensemble_GMBBVI_func(x_p, func_Phi)
    log_rhoa_flat = log.(gaussian_mixture_density_func(x_w, x0_mean, inv_sqrt_xx_cov, reshape(x_p, N_modes*N_ens, N_x)))
    log_rhoa = reshape(log_rhoa_flat, N_modes, N_ens)

    # STEP 3: 计算能量项梯度的范数 ||∇E[-Φ]||
    norm_energy_grad = _compute_gradient_components(x_p_normal, -Phi_R, sqrt_xx_cov)

    # STEP 4: 计算熵项梯度的范数 ||∇H(q)||
    norm_entropy_grad = _compute_gradient_components(x_p_normal, -log_rhoa, sqrt_xx_cov)

    # STEP 5: 计算 T_start
    if norm_entropy_grad < 1e-9
        @warn "Entropy gradient norm is close to zero. Cannot reliably estimate T_start. Returning default 10.0"
        return 10.0
    end

    T_start_est = norm_energy_grad / (alpha * norm_entropy_grad)
    
    return T_start_est
end



"""
    estimate_T_start_from_phi(
        func_Phi, x0_w, x0_mean, x0_cov;
        N_ens=100, alpha=0.1,
        # 传入GMBBVI.jl中的辅助函数
        construct_ensemble_func,
        ensemble_GMBBVI_func,
        gaussian_mixture_density_func
    )

根据能量项和熵项的梯度范数比来估算一个合理的初始温度 T_start。

# Arguments
- `func_Phi`: 你的势能函数 Φ(θ)。
- `x0_w`, `x0_mean`, `x0_cov`: 用于定义初始近似分布 q(θ) 的GMM参数。
- `N_ens`: 用于蒙特卡洛估计的样本数量。
- `alpha`: 目标比例，即 ||能量梯度|| / ||熵梯度||。
- `..._func`: 从外部传入 GMBBVI 代码中定义的辅助函数，以保持模块独立。

# Returns
- `T_start_est`: 建议的初始温度 T_start。
"""
function estimate_T_start_from_phi(
    func_Phi, 
    x0_w::Vector{FT}, 
    x0_mean::Matrix{FT}, 
    x0_cov::Array{FT, 3};
    N_ens::IT=100,
    construct_ensemble_func = construct_ensemble,
    ensemble_GMBBVI_func::Function
) where {FT<:AbstractFloat, IT<:Int}

    # STEP 1: 设置初始的 GMM 状态
    N_modes, N_x = size(x0_mean)
    x_w = x0_w / sum(x0_w)
    sqrt_xx_cov = [cholesky(Hermitian(x0_cov[im,:,:])).L for im = 1:N_modes]
    inv_sqrt_xx_cov = [inv(L) for L in sqrt_xx_cov]

    # STEP 2: 执行一次蒙特卡洛采样
    x_p_normal = zeros(N_modes, N_ens, N_x)
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p_normal[im,:,:] = construct_ensemble_func(zeros(N_x), Matrix(1.0I, N_x, N_x); c_weights=nothing, N_ens=N_ens)
        x_p[im,:,:] = x_p_normal[im,:,:] * sqrt_xx_cov[im]' .+ x0_mean[im,:]'
    end

    # 预先计算 Phi_R 
    Phi_R = ensemble_GMBBVI_func(x_p, func_Phi)
    
    T_start_est = (maximum(Phi_R) - minimum(Phi_R))
    
    return T_start_est
end