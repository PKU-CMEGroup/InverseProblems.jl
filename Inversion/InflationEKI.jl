using LinearAlgebra, Random, Statistics, Distributions

mutable struct EKIObj{FT<:AbstractFloat, IT<:Int}
    # Type of ensemble Kalman inversion: "EKI", "ETKI", "EAKI", "DEKI"
    filter_type::String
    # Ensemble of parameters at each iteration: θ ∈ R^(N_θ × N_ens)
    θ::Vector{Array{FT,2}}
    # Predicted observations for each ensemble
    y_pred::Vector{Array{FT,2}}
    # Observed data vector
    y::Array{FT,1}
    # Lower triangular sqrt of observation covariance
    Σ_y_sqrt::LowerTriangular{FT}
    # Ensemble size
    N_ens::IT
    # Parameter dimension
    N_θ::IT
    # Observation dimension
    N_y::IT
    # Artificial time step
    Δτ::FT
    # Dropout rate for dropout optimization EAKI/ETKI
    dropout_rate::FT
    # Whether to use the inflated mean-field prediction step
    inflation::Bool
    # Reference deviation step size h in Algorithm 2.1 of dropout EKI
    dropout_deviation_Δτ::FT
    # Reference mean step size h_tilde in Algorithm 2.1 of dropout EKI
    dropout_mean_Δτ::FT
    # Singular-value truncation bound M_G in Algorithm 2.1 of dropout EKI
    dropout_linearization_bound::FT
    # Projected dropout mean update: "sequential" or "joint"
    dropout_correction_mode::String
    # Relative weight of the complementary block in the joint reduced model
    joint_dropout_weight::FT
end

# Constructor
function EKIObj(filter_type::String, θ0::Array{FT,2}, y_pred_0::Array{FT,2}, y0::Array{FT,1}, 
                        Σ_y::Array{FT,2}, Δτ::FT, dropout_rate::FT=FT(0.5), inflation::Bool=true,
                        dropout_deviation_Δτ::FT=Δτ, dropout_mean_Δτ::FT=Δτ,
                        dropout_linearization_bound::FT=FT(Inf),
                        dropout_correction_mode::String="sequential",
                        joint_dropout_weight::FT=one(FT)) where FT<:AbstractFloat
    θ = [θ0]
    y_pred = [y_pred_0]

    N_θ, N_ens = size(θ0)
    N_y = size(y0, 1)

    # Robust observation-noise square root: try the standard Cholesky first,
    # then add a small diagonal regularization if the covariance is not
    # numerically positive definite.
    local Σ_y_sqrt
    try
        Σ_y_sqrt = LowerTriangular(cholesky(Symmetric(Matrix(Σ_y))).L)
    catch
        # If Cholesky fails, project the covariance onto the positive-semidefinite
        # cone via its eigenvalue decomposition before computing the sqrt.
        eigF = eigen(Symmetric(Matrix(Σ_y)))
        λmax = maximum(abs, eigF.values)
        λfloor = λmax == 0 ? one(FT) : max(eps(FT) * λmax, FT(1e-12) * λmax)
        λ = max.(eigF.values, λfloor)
        Σ_y_reg = eigF.vectors * Diagonal(λ) * eigF.vectors'
        Σ_y_sqrt = LowerTriangular(cholesky(Symmetric(Σ_y_reg)).L)
    end

    dropout_correction_mode in ("sequential", "joint") ||
        error("dropout_correction_mode must be sequential or joint")
    joint_dropout_weight > 0 || error("joint_dropout_weight must be positive")

    obj = EKIObj(filter_type, θ, y_pred, y0, Σ_y_sqrt, N_ens, N_θ, N_y, Δτ, dropout_rate, inflation,
                 dropout_deviation_Δτ, dropout_mean_Δτ, dropout_linearization_bound,
                 dropout_correction_mode, joint_dropout_weight)
    return obj
end

function active_svd_rank(s::AbstractVector{FT}) where FT<:AbstractFloat
    isempty(s) && return 0
    max_s = maximum(s)
    max_s == zero(FT) && return 0
    tol = max(eps(FT) * length(s) * max_s, FT(1e-12) * max_s)
    r = findlast(s .> tol)
    return isnothing(r) ? 0 : r
end

# ---------------------------------------------------------------------------
# Numerical-stability helpers
# ---------------------------------------------------------------------------

# Robust pseudo-inverse: some LAPACK versions fail in pinv's divide-and-conquer
# SVD on rank-deficient matrices, so fall back to an explicit thin SVD.
function _safe_pinv(A::AbstractMatrix{FT}) where FT<:AbstractFloat
    try
        return pinv(A)
    catch
        F = svd(A; full=false, alg=LinearAlgebra.QRIteration())
        tol = max(eps(FT) * maximum(size(A)) * maximum(F.S), FT(1e-12) * maximum(F.S))
        r = count(>(tol), F.S)
        if r == 0
            return zeros(FT, size(A, 2), size(A, 1))
        end
        return F.V[:, 1:r] * Diagonal(1 ./ F.S[1:r]) * F.U[:, 1:r]'
    end
end

# Solve A*x = b using a SVD-based pseudo-inverse.  This avoids direct
# factorization of near-singular sample covariance matrices and gracefully
# handles rank-deficient systems.
function stable_solve(A::AbstractMatrix, b::Union{AbstractVector,AbstractMatrix})
    return _safe_pinv(A) * b
end

# Inverse of a diagonal represented by a vector, with small singular values
# clamped to zero to avoid enormous amplification.
function safe_diag_inv(d::AbstractVector{FT}) where FT<:AbstractFloat
    isempty(d) && return similar(d, FT)
    m = maximum(abs, d)
    m == zero(FT) && return zeros(FT, length(d))
    tol = max(eps(FT) * length(d) * m, FT(1e-12) * m)
    return [abs(x) > tol ? inv(x) : zero(FT) for x in d]
end

function projected_dropout_components(eki::EKIObj, forward::Function,
                                      center::Array{FT,2}, Zb::Array{FT,2},
                                      Yb::Array{FT,2}, Σ_y::Array{FT,2}) where FT<:AbstractFloat
    ρ = dropout_mask(eki)
    Z_tilde = ρ .* Zb

    θ_tilde = center .+ Z_tilde * sqrt(eki.N_ens - 1)
    x_tilde = forward(θ_tilde)
    x_tilde_mean = mean(x_tilde, dims=2)
    Y_tilde = (x_tilde .- x_tilde_mean) ./ sqrt(eki.N_ens - 1)

    L = _safe_pinv(Yb' * stable_solve(Σ_y, Yb)) * (Yb' * stable_solve(Σ_y, Y_tilde))
    Z_tilde_⊥ = Z_tilde - Zb * L
    Y_tilde_⊥ = Y_tilde - Yb * L

    return Z_tilde_⊥, Y_tilde_⊥, x_tilde_mean
end

function dropout_optimization_mean(eki::EKIObj, forward::Function, m_hat::Array{FT,2},
                                   Zb::Array{FT,2}, Yb::Array{FT,2},
                                   Σ_y::Array{FT,2}) where FT<:AbstractFloat
    Z_tilde_⊥, Y_tilde_⊥, _ =
        projected_dropout_components(eki, forward, m_hat, Zb, Yb, Σ_y)
    x_tilde_m_n = forward(m_hat)

    Δm = reshape((Z_tilde_⊥ * Y_tilde_⊥') *
                 stable_solve(Y_tilde_⊥ * Y_tilde_⊥' + Σ_y, eki.y .- x_tilde_m_n), :, 1)
    return m_hat .+ Δm
end

function joint_projected_dropout_mean(eki::EKIObj, forward::Function,
                                      mn::Array{FT,2}, x_mn::AbstractArray{FT},
                                      Zb::Array{FT,2}, Yb::Array{FT,2},
                                      Σ_y::Array{FT,2}) where FT<:AbstractFloat
    Z_tilde_⊥, Y_tilde_⊥, _ =
        projected_dropout_components(eki, forward, mn, Zb, Yb, Σ_y)
    weight = sqrt(eki.joint_dropout_weight)
    Z_aug = hcat(Zb, weight .* Z_tilde_⊥)
    Y_aug = hcat(Yb, weight .* Y_tilde_⊥)
    Cθx_aug = Z_aug * Y_aug'
    Cxx_aug = Y_aug * Y_aug'
    return mn .+ reshape(Cθx_aug *
        stable_solve(Cxx_aug + Σ_y, eki.y .- vec(x_mn)), :, 1)
end

function dropout_mask(eki::EKIObj{FT}) where FT<:AbstractFloat
    dropout_λ = 1 - eki.dropout_rate
    0 < dropout_λ <= 1 || error("dropout_rate must satisfy 0 <= dropout_rate < 1")

    ρ = rand(Bernoulli(dropout_λ), eki.N_θ)
    while sum(ρ) == 0
        ρ = rand(Bernoulli(dropout_λ), eki.N_θ)
    end
    return reshape(FT.(ρ), :, 1)
end

function deki_linearized_observation_deviations(T::Array{FT,2}, Y::Array{FT,2}, M_G::FT) where FT<:AbstractFloat
    M_G > 0 || error("dropout_linearization_bound must be positive")

    svd_T = svd(T; full=false)
    r_T = active_svd_rank(svd_T.S)
    if r_T == 0
        return zeros(FT, size(Y, 1), size(T, 2))
    end

    S_T = svd_T.S[1:r_T]
    V_T = svd_T.V[:,1:r_T]

    svd_Y = svd(Y; full=false)
    r_Y = active_svd_rank(svd_Y.S)
    if r_Y == 0
        return zeros(FT, size(Y, 1), size(T, 2))
    end

    W = svd_Y.U[:,1:r_Y]
    R = Diagonal(svd_Y.S[1:r_Y]) * svd_Y.V[:,1:r_Y]'
    A = R * V_T * Diagonal(safe_diag_inv(S_T))

    if isfinite(M_G)
        svd_A = svd(A; full=false)
        A = svd_A.U * Diagonal(min.(svd_A.S, M_G)) * svd_A.V'
    end

    Q = Diagonal(S_T) * V_T'
    return W * A * Q
end

# Ensemble forward evaluation (parallel)
function ensemble_forward(forward::Function, θ::Array{FT,2}, N_y::Int) where FT<:AbstractFloat
    y_pred = zeros(FT, N_y, size(θ,2))
    Threads.@threads for j in 1:size(θ,2)
        y_pred[:,j] = forward(θ[:,j])
    end
    return y_pred
end

# Unified ensemble update function
function update_ensemble!(eki::EKIObj{FT}, forward::Function) where FT<:AbstractFloat

    # --- Prediction step: inflated mean-field dynamics ---
    filter_type = eki.filter_type
    θ_prev = eki.θ[end]
    mn = mean(θ_prev, dims=2)
    if eki.inflation
        0 < eki.Δτ < 1 || error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
        θb = mn .+ sqrt(1 / (1 - eki.Δτ)) .* (θ_prev .- mn)
        Σ_y_sqrt_n = sqrt(1 / eki.Δτ) * eki.Σ_y_sqrt
    else
        θb = θ_prev
        Σ_y_sqrt_n = eki.Σ_y_sqrt
    end

    # --- Predicted observations ---
    xb = forward(θb)
    xbar = mean(xb, dims=2)
    x_mean_bar = forward(mn)
    y_obs = reshape(eki.y, :, 1)

    # --- Compute deviations ---
    Zb = (θb .- mn) ./ sqrt(eki.N_ens - 1)
    Yb = (xb .- xbar) ./ sqrt(eki.N_ens - 1)

    Σ_y_n = Σ_y_sqrt_n * Σ_y_sqrt_n'
    Σ_y = eki.Σ_y_sqrt * eki.Σ_y_sqrt'

    Cθx = Zb * Yb'
    Cxx = Yb * Yb' + Σ_y_n

    # Kalman gain (SVD-based pseudo-inverse for robustness on rank-deficient
    # or ill-conditioned sample covariances)
    K = Cθx * _safe_pinv(Cxx)
    # --- Analysis step based on filter_type ---
    if filter_type == "EKI" || filter_type == "NF-EKI"
    
        θ_new = similar(θb)
        if filter_type == "EKI" 
            ν = Σ_y_sqrt_n * rand(MvNormal(zeros(eki.N_y), I(eki.N_y)))
            θ_new = θb + K * (y_obs .- xb .- reshape(ν, :, 1))
        else
            θ_new = θb + K * (y_obs .- xb)
        end
        
    elseif filter_type == "dropout-EKI"
        # Dropout optimization based on the noise-free EKI covariance update
        θ_hat = θb + K * (y_obs .- xb)
        m_hat = mean(θ_hat, dims=2)
        Z_hat = (θ_hat .- m_hat) ./ sqrt(eki.N_ens - 1)
        m_new = dropout_optimization_mean(eki, forward, m_hat, Zb, Yb, Σ_y_n)
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)

    elseif filter_type == "DEKI"
        # Dropout EKI, Algorithm 2.1: mean/deviation separation with parameter dropout
        T = θb .- mn
        Z = T ./ sqrt(eki.N_ens - 1)
        Cuu_norm = maximum(svdvals(Z))^2

        if Cuu_norm == 0
            θ_new = copy(θb)
        else
            reg_norm = sqrt(eps(FT)) * max(Cuu_norm, one(FT))
            h_n = eki.dropout_deviation_Δτ / (Cuu_norm + reg_norm)
            h_tilde_n = eki.dropout_mean_Δτ / (Cuu_norm + reg_norm)
            ρ = dropout_mask(eki)

            θ_tilde = mn .+ ρ .* T
            x_tilde = forward(θ_tilde)

            Z_tilde = (θ_tilde .- mn) ./ sqrt(eki.N_ens - 1)
            # Y_tilde = (x_tilde .- x_mn) ./ sqrt(eki.N_ens - 1)
            Y_tilde = (x_tilde .- mean(x_tilde, dims=2)) ./ sqrt(eki.N_ens - 1)
            Cθx_tilde = Z_tilde * Y_tilde'
            Cxx_tilde = Y_tilde * Y_tilde'

            mn1 = mn .+ reshape(h_tilde_n * Cθx_tilde *
                                 stable_solve(Σ_y + h_tilde_n * Cxx_tilde, eki.y .- vec(x_mean_bar)), :, 1)
            # mn1 = mn .+ reshape(h_tilde_n * Cθx_tilde *
            #                      ((Σ_y_n + h_tilde_n * Cxx_tilde) \ (eki.y .- mean(x_tilde, dims=2))), :, 1)

            # Y_linear = xb .- x_mn
            Y_linear = xb .- mean(xb, dims=2)
            G_T = deki_linearized_observation_deviations(T, Y_linear, eki.dropout_linearization_bound)
            Cθx_linear = (T * G_T') ./ (eki.N_ens - 1)
            Cxx_linear = (G_T * G_T') ./ (eki.N_ens - 1)
            T_new = T - h_n * Cθx_linear * stable_solve(Σ_y + h_n * Cxx_linear, G_T)
            θ_new = mn1 .+ T_new
        end

    elseif filter_type == "dropout-EAKI"
        # Dropout optimization based on the EAKI covariance update
        P, Db_sqrt, V = svd(Zb; full=false)
        m_hat = mn .+ reshape(K * (eki.y .- vec(x_mean_bar)), :, 1)
        r = active_svd_rank(Db_sqrt)
        if r == 0
            θ_new = m_hat .+ zeros(FT, eki.N_θ, size(θb, 2))
        else
            P_r = P[:,1:r]
            V_r = V[:,1:r]
            temp = Σ_y_sqrt_n \ Yb
            S = Symmetric(V_r' * ((I + temp' * temp) \ V_r))
            eig = eigen(S)
            U, D = eig.vectors, eig.values
            D .= max.(D, eps(Float64))
            A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
            d_tol = max(eps(FT) * length(Db_sqrt) * maximum(Db_sqrt), FT(1e-12) * maximum(Db_sqrt))
            d_safe = max.(Db_sqrt[1:r], d_tol)
            A2 = Diagonal(sqrt.(max.(D, zero(FT)))) * Diagonal(1 ./ d_safe) * P_r'
            Z_hat = A1 * (A2 * Zb)
            if eki.dropout_correction_mode == "joint"
                m_new = joint_projected_dropout_mean(
                    eki, forward, mn, x_mean_bar, Zb, Yb, Σ_y_n)
            else
                m_new = dropout_optimization_mean(eki, forward, m_hat, Zb, Yb, Σ_y_n)
            end
            θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
        end

    elseif filter_type == "dropout-ETKI"
        # Dropout optimization based on the ETKI covariance update
        temp = Σ_y_sqrt_n \ Yb
        eig = eigen(Symmetric(temp' * temp))
        P, D = eig.vectors, eig.values
        T = P * Diagonal(1 ./ sqrt.(max.(1 .+ D, eps(FT)))) * P'
        Z_hat = Zb * T
        m_hat = mn .+ reshape(K * (eki.y .- vec(x_mean_bar)), :, 1)
        eki.dropout_correction_mode == "joint" &&
            error("joint dropout correction is currently implemented only for dropout-EAKI")
        m_new = dropout_optimization_mean(eki, forward, m_hat, Zb, Yb, Σ_y_n)
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)

    
    elseif filter_type == "EAKI"
        # Ensemble Adjustment Kalman Inversion
        # Compact SVD of Zb
        P, Db_sqrt, V = svd(Zb; full=false)
        mn1 = mn .+ reshape(K * (eki.y .- vec(x_mean_bar)), :, 1)
        r = active_svd_rank(Db_sqrt)
        if r == 0
            θ_new = mn1 .+ zeros(FT, eki.N_θ, size(θb, 2))
        else
            P_r = P[:,1:r]
            V_r = V[:,1:r]
            temp = Σ_y_sqrt_n \ Yb
            S = Symmetric(V_r' * ((I + temp' * temp) \ V_r))   # r × r
            eig = eigen(S)
            U, D = eig.vectors, eig.values
            A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
            d_tol = max(eps(FT) * length(Db_sqrt) * maximum(Db_sqrt), FT(1e-12) * maximum(Db_sqrt))
            d_safe = max.(Db_sqrt[1:r], d_tol)
            A2 = Diagonal(sqrt.(max.(D, zero(FT)))) * Diagonal(1 ./ d_safe) * P_r'
            θ_new = A1 * (A2 * (θb .- mn)) .+ mn1
        end

    elseif filter_type == "ETKI"
        # Ensemble Transform Kalman Inversion
        temp = Σ_y_sqrt_n \ Yb
        eig = eigen(Symmetric(temp' * temp))
        P, D = eig.vectors, eig.values
        T = P * Diagonal(1 ./ sqrt.(max.(1 .+ D, eps(FT)))) * P'
        mn1 = mn .+ reshape(K * (eki.y .- vec(x_mean_bar)), :, 1)
        Z_new = Zb * T
        θ_new = mn1 .+ Z_new * sqrt(eki.N_ens - 1)
    else
        error("Unknown filter_type: $(filter_type)")
    end

    if any(x -> !isfinite(x), θ_new)
        @warn "Non-finite ensemble update detected; reverting to previous ensemble" filter_type=filter_type
        θ_new = copy(θ_prev)
    end

    size(θ_new) == size(θ_prev) || error("θ_new has size $(size(θ_new)), expected $(size(θ_prev))")
    push!(eki.θ, θ_new)
    push!(eki.y_pred, xb)
    return eki
end

# Main EKI run
function EKI_Run(forward::Function, θ0::Array{FT,2}, Σ_y::Array{FT,2}, y::Array{FT,1};
                 filter_type::String="EKI", Δτ::FT=0.5, N_iter::Int=50, forward_parallel::Bool=false,
                 dropout_rate::FT=FT(0.5), inflation::Bool=true,
                 dropout_deviation_Δτ::Union{Nothing,FT}=nothing,
                 dropout_mean_Δτ::Union{Nothing,FT}=nothing,
                 dropout_linearization_bound::FT=FT(Inf),
                 dropout_correction_mode::String="sequential",
                 joint_dropout_weight::FT=one(FT),
                 iteration_callback::Union{Nothing,Function}=nothing) where FT<:AbstractFloat
    N_y = size(y, 1)
    size(θ0, 2) > 1 || error("Need at least 2 ensemble members")
    if inflation && !(0 < Δτ < 1)
        error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
    end
    dropout_deviation_Δτ_actual = isnothing(dropout_deviation_Δτ) ? Δτ : dropout_deviation_Δτ
    dropout_mean_Δτ_actual = isnothing(dropout_mean_Δτ) ? Δτ : dropout_mean_Δτ

    # Obtain forward function for parallel evaluation
    func(x) = forward_parallel ? forward(x) : ensemble_forward(forward, x, N_y)

    y_pred_0 = func(θ0)
    # EKI obj initialization
    obj = EKIObj(filter_type, θ0, y_pred_0, y, Σ_y, Δτ, dropout_rate, inflation,
                 dropout_deviation_Δτ_actual, dropout_mean_Δτ_actual, dropout_linearization_bound,
                 dropout_correction_mode, joint_dropout_weight)
    @info "Running ", filter_type, " with ensemble size ", size(θ0,2)
    iteration_callback !== nothing && iteration_callback(obj, 0)
    for n in 1:N_iter
        if n % max(1, div(N_iter,10)) == 0
            @info ("Iteration ", n, "/", N_iter)
        end
        update_ensemble!(obj, func)
        iteration_callback !== nothing && iteration_callback(obj, n)
    end
    return obj
end

# Optional: compute errors vs true observations
function opt_errors(eki::EKIObj)
    N_iter = length(eki.θ) - 1
    errors = zeros(Float64, N_iter+1)
    for i in 0:N_iter
        res = eki.y_pred[i+1] .- eki.y
        errors[i+1] = 0.5 * norm(eki.Σ_y_sqrt \ res)^2 / eki.N_ens
    end
    return errors
end
