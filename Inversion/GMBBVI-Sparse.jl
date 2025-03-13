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
    # "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    # xx_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "a vector of arrays of size (N_modes) containing the modal term ϵ, used for computing covariances of the parameters"
    ϵ::Vector{Array{FT, 1}}
    "a vector of arrays of size (N_modes x N_parameters x N_rank) containing the modal term Q, used for computing covariances of the parameters"
    Q::Union{Vector{Array{FT, 3}}, Nothing}
    "updating rule: direct or EM"
    updating_rule::String
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
                # xx0_cov::Union{Array{FT, 3}, Nothing};
                ϵ0::Array{FT, 1},
                Q0::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                diagonal_covariance::Bool = false,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                N_ens::IT = 10,
                w_min::FT = 1.0e-8,
                updating_rule::String = "EM") where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    # xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    # push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    ϵ = Array{FT,1}[]           # array of Array{FT, 1}'s
    push!(ϵ, ϵ0)                # insert parameters at end of array (in this case just 1st entry)
    Q = Array{FT,3}[]           # array of Array{FT, 3}'s
    push!(Q, Q0)                # insert parameters at end of array (in this case just 1st entry)
    
    name = "BBVI-Sparse"

    iter = 0

    BBVIObj(name,
            logx_w, x_mean, ϵ, Q, updating_rule, N_modes, N_x,
            iter, update_covariance, diagonal_covariance,
            sqrt_matrix_type, N_ens, w_min)
end

   
function randomized_structured_svd(
    ε::Float64, 
    Q::Matrix{Float64}, 
    δt::Float64, 
    R::Matrix{Float64}, 
    D::Diagonal{Float64, Vector{Float64}}, 
    k::Int; 
    p::Int=5, 
    q::Int=2, 
    power_iter_stable::Bool=true
)
    # 获取矩阵维度
    m = size(Q, 1)
    r = size(R, 2)
    l = k + p  # 总采样数

    # 1. 生成随机高斯矩阵 Ω (m × l)
    Ω = randn(m, l)

    # 2. 计算初始投影 Y = (εI + Q Q' + δt R D R') * Ω
    Y = ε .* Ω + Q * (Q' * Ω) + δt .* R * (D * (R' * Ω))

    # 3. 幂迭代增强数值稳定性
    for _ in 1:q
        # 计算 A * Y (A = εI + Q Q' + δt R D R')
        AY = ε .* Y + Q * (Q' * Y) + δt .* R * (D * (R' * Y))
        # 计算 A * (A * Y)
        Y = ε .* AY + Q * (Q' * AY) + δt .* R * (D * (R' * AY))
        # 可选的正交化步骤
        if power_iter_stable
            Y, _ = qr(Y)
        end
    end

    # 4. 对 Y 进行QR分解得到正交基 Q_ortho
    Q_ortho = Matrix(qr(Y).Q)

    # 5. 构造低维矩阵 B = Q_ortho' * A
    term1 = ε * Q_ortho'  # 来自 εI
    term2 = (Q_ortho' * Q) * Q' # 来自 Q Q'
    term3 = δt * (Q_ortho' * R) * D * R'  # 来自 δt R D R'
    B = term1 + term2 + term3

    # 6. 对 B 进行精确SVD分解
    Ũ, Σ, V = svd(B)

    # 7. 重构左奇异向量矩阵 U
    U = Q_ortho * Ũ

    # 8. 截断到前 k 个主成分
    return U[:, 1:k], Σ[1:k], V[:, 1:k]
end

function new_rank_scalar_approximation(eps0, Q0, R0, D0, N_r; rank_plus::Int = 3, A = nothing, eps_min = 0.01, method = "RandSVD")
    """Approximate A = eps0*I + Q0*Q0' + R0*D0*R0'  with  A = eps*I +QQ',
    based on PPCA method and eigenvalue decomposition(by RandSVD / Nystrom). """
    if method == "RandSVD"
        if A == nothing
            N_x = size(Q0,1)
            Omega = randn(N_x, N_r+rank_plus)
            Y = eps0*Omega + Q0*(Q0'*Omega) + (R0.*D0')*(R0'*Omega)
            Yqr = qr(Y)
            Q = Matrix(Yqr.Q)  
            B = eps0*Q' + Q'*Q0*Q0' + Q'*(R0.*D0')*R0'
            trA = eps0*N_x + sum(Q0.*Q0) + sum(R0.*(R0.*D0'))
        else
            trA_inv = tr(inv(A))
            N_x = size(A,1)
            Omega = randn(N_x, N_r+rank_plus)
            Y = A*Omega
            Yqr = qr(Y)
            Q = Matrix(Yqr.Q)  
            B = Q'*A
        end 
        U0, D, _ = svd(B)
        U = (Q*U0)[:,1:N_r]
        D = D[1:N_r]
        eps = (N_x-N_r)/(trA_inv-sum(1.0./D))
        eps = max(eps, eps_min)
        newQ = hcat( [sqrt(max(D[i]-eps,1.0e-6))*U[:,i] for i=1:N_r]... )
    else
        @error "Undefined rank_scalar_approximation method"
    end
    return eps, newQ
end

function low_rank_kl(ϵ, Q, ϵ0, Q0, R, D; A = nothing)
    N_x, N_r = size(Q)
    QQI = Q' * Q + ϵ * I
    det_term = log(det(QQI)) + (N_x - N_r) * log(ϵ)
    inv_QQI = inv(QQI)
    trace_term = (tr(A) - tr(Q * inv_QQI * (Q' * A))) / ϵ
    return trace_term + det_term
end

function update(ϵ, Q, ϵ0, Q0, R0, D0; A = nothing)
    D, K = size(Q)
    I_K = Matrix(I, K, K)
    I_D = Matrix(I, D, D)
    
    α = Q' / ϵ
    β = α * (I_D - Q * ((I_K + α * Q) \ α))
    AβT = A * β'
    Q_update = (AβT) / (β * AβT + I_K - β * Q)
    trA = ϵ0 * D + sum(Q0.*Q0) + sum(R0 .* (R0 .* D0'))
    second_term = tr(Q_update * AβT')
    ϵ_update = trA - second_term
    ϵ_update /= D
    return ϵ_update, Q_update
end

function EM_patch(ϵ0, Q0, R, D, N_r; maxiter = 10, tol = 1e-1, η = 1.0, A = nothing, miniter = 3, verbose = false)
    @assert 1.0 <= η < 2.0
    counter = 0
    ϵ = copy(ϵ0)
    Q = copy(Q0)

    current_kl = low_rank_kl(ϵ, Q, ϵ0, Q0, R, D; A = A)
    kld = Float64[]
    rkld = Float64[]

    for counter in 1:maxiter
        ϵ_update, Q_update = update(ϵ, Q, ϵ0, Q0, R, D; A = A)
        
        if any(isnan.(ϵ_update)) || any(isnan.(Q_update))
            return NaN, Q, counter
        else
            if tol == 0.0
                ϵ = (1 - η) * ϵ + η * ϵ_update
                Q = (1 - η) .* Q .+ η .* Q_update
            else
                old_kl = current_kl
                current_kl = low_rank_kl(ϵ_update, Q_update, ϵ0, Q0, R, D; A = A)
                if isnan(current_kl) || isinf(current_kl)
                    println("$(counter), NaN/Inf in KL")
                    return NaN, Q .* NaN, counter
                end
                
                ϵ = (1 - η) * ϵ + η * ϵ_update
                Q = (1 - η) .* Q .+ η .* Q_update
                push!(rkld, abs(current_kl / old_kl - 1))
                push!(kld, current_kl - old_kl)
                
                if counter > miniter
                    second_der = (kld[end] + kld[end-2] - 2*kld[end-1]) / kld[end-1]
                    if old_kl != 0.0 && all(rkld[end] .< tol)
                        verbose && println("$(counter) iterations to reach convergence")
                        return ϵ, Q, counter
                    end
                end
            end
        end
    end
    return ϵ, Q, counter
end

""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi)"""
function update_ensemble!(gmgd::BBVIObj{FT, IT}, func_Phi::Function, dt_max::FT, η::FT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    ϵ  = gmgd.ϵ[end]
    Q  = gmgd.Q[end]
    _, _, N_r = size(Q)


    N_ens = gmgd.N_ens
    d_logx_w, d_x_mean, d_xx_cov_R, d_xx_cov_D = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_r), zeros(N_modes, N_r)

    for im = 1:N_modes 

        # generate sampling points subject to Normal(x_mean [im,:], xx_cov[im]), size=(N_ens, N_x)
        # θ_k = m_k + Q_k N(0, I_Nᵣ) + √ϵ N(0, I_Nθ)
        x_p = ones(N_ens)*x_mean[im,:]' + (Q[im, :, :] * rand(Normal(0, 1), N_r, N_ens) + sqrt(ϵ[im]) * rand(Normal(0, 1), N_x, N_ens))'
        # log_ratio[i] = logρ[x_p[i,:]] + log func_Phi[x_p[i,:]]
        
        # if im==1 && gmgd.iter==1  @show sum((x_p[i,:]-x_mean[im,:])*(x_p[i,:]-x_mean[im,:])'-xx_cov[im,:,:] for i=1:N_ens)/N_ens  end

        log_ratio = zeros(2 * N_ens) 
        for i = 1:N_ens
            for imm = 1:N_modes
                log_ratio[i] += exp(logx_w[imm]) * Gaussian_density_helper_sparse(x_mean[imm,:], ϵ[imm], Q[imm, :, :], x_p[i,:])
                log_ratio[N_ens + i] += exp(logx_w[imm]) * Gaussian_density_helper_sparse(x_mean[imm,:], ϵ[imm], Q[imm, :, :], 2 * x_mean[imm,:] - x_p[i,:])
            end
            log_ratio[i] = log(log_ratio[i])+func_Phi(x_p[i,:])
            log_ratio[N_ens + i] = log(log_ratio[N_ens + i]) + func_Phi(2 * x_mean[im,:] - x_p[i,:])
        end

        # E[logρ+Phi]
        log_ratio_mean = mean(log_ratio)
        log_ratio .-= mean(log_ratio)

        # E[(x-m)(logρ+Phi)]
        log_ratio_m1 = mean( (x_p[i,:]-x_mean[im,:])*(log_ratio[i]-log_ratio[i+N_ens]) for i=1:N_ens)   

        ############################################ where N_r and N_ens might need to change ######################################
        # E[(x-m)(x-m)'(logρ+Phi)] - E[(x-m)(x-m)'] E(logρ+Phi)
        # E[(x-m)(x-m)'(logρ+Phi - E(logρ+Phi))] 
        # ̇c[im, :, :] = d_xx_cov_R[im, :, :] * diagm(d_xx_cov_D) * d_xx_cov_R[im, :, :]'
        for i = 1 : N_r
            d_xx_cov_R[im, :, i] = x_p[i,:] - x_mean[im,:]
            d_xx_cov_D[im, i] = (-log_ratio[i] - log_ratio[N_ens + i]) / (2 * N_r)
        end
        ############################################ where N_r and N_ens might need to change ######################################
        
        d_x_mean[im,:] = -log_ratio_m1
        d_logx_w[im] = -log_ratio_mean

    end
    x_mean_n = copy(x_mean) 
    logx_w_n = copy(logx_w)

    matrix_norm = []
    d⁻_xx_cov_D = copy(d_xx_cov_D)
    d⁻_xx_cov_D[d⁻_xx_cov_D .> 0] .= 0
    for im = 1 : N_modes
        RD = d_xx_cov_R[im, :, :] * Diagonal(sqrt.(- d⁻_xx_cov_D[im,:]))
        # push!(matrix_norm, opnorm( d_xx_cov[im,:,:]*inv(xx_cov[im,:,:]) , 2))
        if isnan(norm(RD' * RD - RD' * Q[im, :, :] * inv(Q[im, :, :]' * Q[im, :, :] + ϵ[im] * Matrix(I, N_r, N_r)) * Q[im, :, :]' * RD))
            println(d_xx_cov_R[im, :, :], d_xx_cov_D)
        end
        push!(matrix_norm, opnorm(RD' * RD - RD' * Q[im, :, :] * inv(Q[im, :, :]' * Q[im, :, :] + ϵ[im] * Matrix(I, N_r, N_r)) * Q[im, :, :]' * RD , 2)/ϵ[im] ) 
    end
    # dt = dt_max
    dt = min(dt_max,  0.99 / (maximum(matrix_norm))) # keep the matrix postive definite.
    # if gmgd.iter%10==0  @show gmgd.iter,dt  end

    if update_covariance
        for im =1:N_modes
            # Incomplete failure method
            # U, Σ, V = randomized_structured_svd(ϵ[im], Q[im, :, :], dt, d_xx_cov_R[im, :, :], Diagonal(d_xx_cov_D[im, :]), N_r; q=2)
            # indices = partialsortperm(Σ, 1:N_r, rev=true)
            # remaining_indices = setdiff(1:length(Σ), indices)
            # remaining_values = @view Σ[remaining_indices]
            # ϵ[im] = (N_x - N_r) / sum(1 ./ remaining_values)
            # for i = 1 : N_r
            #     Q[im, :, i] = sqrt(Σ[indices[i]] - ϵ[im]) * U[:, indices[i]]
            # end

            A = ϵ[im]*I + Q[im,:,:]*Q[im,:,:]' + d_xx_cov_R[im, :, :].*(d_xx_cov_D[im, :]')*d_xx_cov_R[im, :, :]'*dt
            if gmgd.updating_rule == "EM"
                # EM algorithm
                ϵ[im], Q[im, :, :], _ = EM_patch(ϵ[im], Q[im, :, :], d_xx_cov_R[im, :, :], dt*d_xx_cov_D[im, :], N_r; maxiter = 10, tol = 1e-5, A = A, η = η)
            elseif gmgd.updating_rule == "direct"
                # Che's LowRankBBVI 1st method
                new_ϵ, new_Q = new_rank_scalar_approximation(ϵ[im], Q[im, :, :], d_xx_cov_R[im, :, :], dt*d_xx_cov_D[im, :], N_r; A=A)
                ϵ[im] = η * new_ϵ + (1 - η) * ϵ[im]
                Q[im, :, :] = η * new_Q + + (1 - η) * Q[im, :, :]
            else
                @info "error! updating rule not found!"
            end
            
            
            # xx_cov_n[im, :, :] = Hermitian(xx_cov_n[im, :, :])
            # if diagonal_covariance
            #     xx_cov_n[im, :, :] = diagm(diag(xx_cov_n[im, :, :]))
            # end
            # if !isposdef(Hermitian(xx_cov_n[im, :, :]))
            #     @show gmgd.iter
            #     @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :])
            #     @assert(isposdef(xx_cov_n[im, :, :]))
            # end
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
    push!(gmgd.ϵ, ϵ)
    push!(gmgd.Q, Q)
    push!(gmgd.logx_w, logx_w_n) 
end


##########
function Gaussian_mixture_BBVI_Sparse(func_Phi, x0_w, x0_mean, ϵ0, Q0;
     diagonal_covariance::Bool = false, N_iter = 100, dt = 5.0e-1, N_ens = -1, η = 1.0, updating_rule = "EM")

    _, N_x = size(x0_mean) 
    if N_ens == -1 
        N_ens = 5*N_x
    end

    gmgdobj=BBVIObj(
        x0_w, x0_mean, ϵ0, Q0;
        update_covariance = true,
        diagonal_covariance = diagonal_covariance,
        sqrt_matrix_type = "Cholesky",
        N_ens = N_ens,
        w_min = 1.0e-8,
        updating_rule = updating_rule)

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func_Phi, dt, η) 
    end
    
    return gmgdobj
end
