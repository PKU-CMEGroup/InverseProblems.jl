using LinearAlgebra, Random, Statistics, Distributions

struct JointWeightConfig{FT<:AbstractFloat}
    w_min::FT
    w_max::FT
    smoothing::FT
end

function JointWeightConfig(::Type{FT}=Float64;
        w_min::Real=0.1, w_max::Real=10.0,
        smoothing::Real=0.25) where FT<:AbstractFloat
    values = FT.((w_min, w_max, smoothing))
    all(isfinite, values) ||
        throw(ArgumentError("joint weight configuration must be finite"))
    0 < values[1] <= values[2] ||
        throw(ArgumentError("require 0 < w_min <= w_max"))
    0 < values[3] <= 1 ||
        throw(ArgumentError("smoothing must lie in (0, 1]"))
    return JointWeightConfig{FT}(values...)
end

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
    # Optional trust-ratio adaptation of the joint complementary-block weight
    joint_weight_mode::String
    joint_weight_config::JointWeightConfig{FT}
    joint_weight_history::Vector{FT}
    joint_trust_ratio_history::Vector{FT}
    joint_predicted_reduction_history::Vector{FT}
    joint_actual_reduction_history::Vector{FT}
    joint_previous_objective::FT
    joint_previous_predicted_reduction::FT
    # Optional globalization of only the dropout-EAKI mean translation
    mean_line_search::Bool
    # Dimensionless Armijo backtracking constants
    mean_line_search_contraction::FT
    mean_line_search_armijo_c::FT
    # Explicit mean/anomaly state avoids cancellation when unobserved anomalies grow
    mean_state::Array{FT,2}
    anomaly_state::Array{FT,2}
    # Accepted mean prediction, reused by the next iteration
    mean_prediction_cache::Union{Nothing,Array{FT,2}}
    # Diagnostics for the optional mean globalization
    mean_step_γ::Vector{FT}
    mean_step_source::Vector{String}
    mean_step_trials::Vector{IT}
    mean_step_backtracks::Vector{IT}
    mean_step_reduction_ratio::Vector{FT}
end

# Constructor
function EKIObj(filter_type::String, θ0::Array{FT,2}, y_pred_0::Array{FT,2}, y0::Array{FT,1}, 
                        Σ_y::Array{FT,2}, Δτ::FT, dropout_rate::FT=FT(0.5), inflation::Bool=true,
                        dropout_deviation_Δτ::FT=Δτ, dropout_mean_Δτ::FT=Δτ,
                        dropout_linearization_bound::FT=FT(Inf),
                        dropout_correction_mode::String="sequential",
                        joint_dropout_weight::FT=one(FT),
                        joint_weight_mode::String="fixed",
                        joint_weight_config::JointWeightConfig{FT}=JointWeightConfig(FT),
                        mean_line_search::Bool=false,
                        mean_line_search_contraction::FT=FT(0.5),
                        mean_line_search_armijo_c::FT=FT(1e-4)) where FT<:AbstractFloat
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
    joint_weight_mode in ("fixed", "adaptive") ||
        throw(ArgumentError("joint_weight_mode must be fixed or adaptive"))
    joint_weight_mode == "adaptive" && dropout_correction_mode != "joint" &&
        throw(ArgumentError("adaptive joint weights require dropout_correction_mode=joint"))
    joint_weight_mode == "adaptive" &&
        !(joint_weight_config.w_min <= joint_dropout_weight <= joint_weight_config.w_max) &&
        throw(ArgumentError(
            "adaptive joint_dropout_weight must lie within its configured bounds"))
    0 < mean_line_search_contraction < 1 ||
        error("mean_line_search_contraction must lie in (0, 1)")
    0 < mean_line_search_armijo_c < 1 ||
        error("mean_line_search_armijo_c must lie in (0, 1)")

    mean_state = mean(θ0, dims=2)
    anomaly_state = (θ0 .- mean_state) ./ sqrt(N_ens - 1)

    obj = EKIObj(filter_type, θ, y_pred, y0, Σ_y_sqrt, N_ens, N_θ, N_y, Δτ, dropout_rate, inflation,
                 dropout_deviation_Δτ, dropout_mean_Δτ, dropout_linearization_bound,
                 dropout_correction_mode, joint_dropout_weight,
                 joint_weight_mode, joint_weight_config,
                 FT[], FT[], FT[], FT[], FT(NaN), FT(NaN),
                 mean_line_search, mean_line_search_contraction,
                 mean_line_search_armijo_c, mean_state, anomaly_state, nothing,
                 FT[], String[], Int[], Int[], FT[])
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

function adaptive_joint_weight(current::FT, actual_reduction,
                               predicted_reduction,
                               cfg::JointWeightConfig{FT}) where FT<:AbstractFloat
    if !(isfinite(actual_reduction) && isfinite(predicted_reduction))
        return clamp(current, cfg.w_min, cfg.w_max)
    end
    ratio = predicted_reduction <= eps(FT) ? zero(FT) :
        clamp(actual_reduction / predicted_reduction, zero(FT), one(FT))
    target = cfg.w_min + (cfg.w_max - cfg.w_min) * ratio
    return clamp((one(FT) - cfg.smoothing) * current +
                 cfg.smoothing * target, cfg.w_min, cfg.w_max)
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

function joint_projected_dropout_proposal(eki::EKIObj, forward::Function,
        mn::Array{FT,2}, x_mn::AbstractArray{FT},
        Zb::Array{FT,2}, Yb::Array{FT,2},
        Σ_y::Array{FT,2}) where FT<:AbstractFloat
    residual = eki.y .- vec(x_mn)
    objective(v) = FT(0.5) * dot(v, stable_solve(Σ_y, v))
    current_objective = objective(residual)
    actual_reduction = isfinite(eki.joint_previous_objective) ?
        eki.joint_previous_objective - current_objective : FT(NaN)
    trust_ratio = isfinite(actual_reduction) &&
            eki.joint_previous_predicted_reduction > eps(FT) ?
        actual_reduction / eki.joint_previous_predicted_reduction : FT(NaN)

    if eki.joint_weight_mode == "adaptive" && isfinite(actual_reduction)
        eki.joint_dropout_weight = adaptive_joint_weight(
            eki.joint_dropout_weight, actual_reduction,
            eki.joint_previous_predicted_reduction,
            eki.joint_weight_config)
    end
    effective_weight = eki.joint_dropout_weight

    Z_tilde_⊥, Y_tilde_⊥, _ =
        projected_dropout_components(eki, forward, mn, Zb, Yb, Σ_y)
    block_scale = sqrt(effective_weight)
    Z_aug = hcat(Zb, block_scale .* Z_tilde_⊥)
    Y_aug = hcat(Yb, block_scale .* Y_tilde_⊥)
    Cθx_aug = Z_aug * Y_aug'
    Cxx_signal = Y_aug * Y_aug'
    coefficient = stable_solve(Cxx_signal + Σ_y, residual)
    direction = vec(Cθx_aug * coefficient)
    observation_change = vec(Cxx_signal * coefficient)
    predicted_residual = residual - observation_change
    predicted_reduction = current_objective - objective(predicted_residual)

    push!(eki.joint_weight_history, effective_weight)
    push!(eki.joint_trust_ratio_history, trust_ratio)
    push!(eki.joint_predicted_reduction_history, predicted_reduction)
    push!(eki.joint_actual_reduction_history, actual_reduction)
    eki.joint_previous_objective = current_objective
    eki.joint_previous_predicted_reduction = predicted_reduction

    return (;
        direction,
        observation_change,
        weight=effective_weight,
        trust_ratio,
        predicted_reduction,
        actual_reduction,
    )
end

function joint_projected_dropout_mean(eki::EKIObj, forward::Function,
                                      mn::Array{FT,2}, x_mn::AbstractArray{FT},
                                      Zb::Array{FT,2}, Yb::Array{FT,2},
                                      Σ_y::Array{FT,2}) where FT<:AbstractFloat
    proposal = joint_projected_dropout_proposal(
        eki, forward, mn, x_mn, Zb, Yb, Σ_y)
    return mn .+ reshape(proposal.direction, :, 1)
end

# ---------------------------------------------------------------------------
# Optional globalization of the dropout-EAKI mean translation
# ---------------------------------------------------------------------------

_prediction_matrix(x::AbstractVector) = reshape(x, :, 1)
_prediction_matrix(x::AbstractMatrix) = x

function _prediction_objective(eki::EKIObj, prediction::AbstractArray)
    residual = vec(prediction) - eki.y
    whitened_residual = eki.Σ_y_sqrt \ residual
    return dot(whitened_residual, whitened_residual) / 2
end

function _mean_direction_statistics(eki::EKIObj{FT}, x_mean::AbstractArray,
                                    q::AbstractVector) where FT<:AbstractFloat
    residual = eki.Σ_y_sqrt \ (vec(x_mean) - eki.y)
    whitened_q = eki.Σ_y_sqrt \ q
    slope = dot(residual, whitened_q)
    curvature = dot(whitened_q, whitened_q)
    if !(isfinite(slope) && isfinite(curvature)) || slope >= 0 ||
            curvature <= eps(FT) * max(dot(residual, residual), one(FT))
        return nothing
    end
    γ = -slope / curvature
    return isfinite(γ) && γ > 0 ? (; slope, curvature, γ) : nothing
end

function _try_mean_armijo(eki::EKIObj{FT}, forward::Function,
        mn::Array{FT,2}, x_mean::AbstractArray,
        direction::AbstractVector, q::AbstractVector;
        known_unit_prediction=nothing) where FT<:AbstractFloat
    statistics = _mean_direction_statistics(eki, x_mean, q)
    statistics === nothing && return (; accepted=false, trials=0, backtracks=0)

    ϕ0 = _prediction_objective(eki, x_mean)
    γ = statistics.γ
    min_γ = sqrt(eps(FT)) * (one(FT) + norm(mn)) /
            max(norm(direction), eps(FT))
    trials = 0
    backtracks = 0

    while γ >= min_γ
        candidate = vec(mn) + γ .* direction
        if any(x -> !isfinite(x), candidate)
            γ *= eki.mean_line_search_contraction
            backtracks += 1
            continue
        end

        if known_unit_prediction !== nothing &&
                isapprox(γ, one(FT); rtol=8eps(FT), atol=zero(FT))
            prediction = _prediction_matrix(known_unit_prediction)
        else
            prediction = _prediction_matrix(forward(reshape(candidate, :, 1)))
            trials += 1
        end
        ϕ_trial = _prediction_objective(eki, prediction)
        armijo_bound = ϕ0 + eki.mean_line_search_armijo_c * γ * statistics.slope
        if isfinite(ϕ_trial) && ϕ_trial <= armijo_bound
            predicted_reduction = -γ * statistics.slope -
                                  γ^2 * statistics.curvature / 2
            ratio = predicted_reduction > 0 ?
                (ϕ0 - ϕ_trial) / predicted_reduction : FT(NaN)
            return (;
                accepted=true,
                mean=reshape(candidate, :, 1),
                prediction=Matrix{FT}(prediction),
                γ,
                trials,
                backtracks,
                reduction_ratio=FT(ratio),
            )
        end
        γ *= eki.mean_line_search_contraction
        backtracks += 1
    end
    return (; accepted=false, trials, backtracks)
end

function _globalize_dropout_mean(eki::EKIObj{FT}, forward::Function,
        mn::Array{FT,2}, x_mean::AbstractArray,
        full_direction::AbstractVector, full_q::AbstractVector,
        subspace_direction::AbstractVector, subspace_q::AbstractVector;
        known_subspace_prediction=nothing) where FT<:AbstractFloat
    sources = (
        ("dropout", full_direction, full_q, nothing),
        ("EAKI-fallback", subspace_direction, subspace_q,
            known_subspace_prediction),
    )
    total_trials = 0
    total_backtracks = 0
    for (source, direction, q, known_prediction) in sources
        norm(direction) > 0 || continue
        result = _try_mean_armijo(
            eki, forward, mn, x_mean, direction, q;
            known_unit_prediction=known_prediction)
        total_trials += result.trials
        total_backtracks += result.backtracks
        result.accepted || continue
        return merge(result, (;
            source,
            trials=total_trials,
            backtracks=total_backtracks,
        ))
    end

    return (;
        accepted=true,
        mean=copy(mn),
        prediction=Matrix{FT}(_prediction_matrix(x_mean)),
        γ=zero(FT),
        trials=total_trials,
        backtracks=total_backtracks,
        reduction_ratio=FT(NaN),
        source="covariance-only",
    )
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
    use_explicit_mean_state = eki.mean_line_search && filter_type == "dropout-EAKI"
    if use_explicit_mean_state
        mn = copy(eki.mean_state)
        Z_prev = copy(eki.anomaly_state)
    else
        mn = mean(θ_prev, dims=2)
        Z_prev = (θ_prev .- mn) ./ sqrt(eki.N_ens - 1)
    end
    if eki.inflation
        0 < eki.Δτ < 1 || error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
        if use_explicit_mean_state
            Zb = Z_prev ./ sqrt(1 - eki.Δτ)
            θb = mn .+ sqrt(eki.N_ens - 1) .* Zb
        else
            θb = mn .+ sqrt(1 / (1 - eki.Δτ)) .* (θ_prev .- mn)
        end
        Σ_y_sqrt_n = sqrt(1 / eki.Δτ) * eki.Σ_y_sqrt
    else
        if use_explicit_mean_state
            Zb = Z_prev
            θb = mn .+ sqrt(eki.N_ens - 1) .* Zb
        else
            θb = θ_prev
        end
        Σ_y_sqrt_n = eki.Σ_y_sqrt
    end

    # --- Predicted observations ---
    xb = forward(θb)
    xbar = mean(xb, dims=2)
    x_mean_bar = use_explicit_mean_state && eki.mean_prediction_cache !== nothing ?
        eki.mean_prediction_cache : _prediction_matrix(forward(mn))
    y_obs = reshape(eki.y, :, 1)

    # --- Compute deviations ---
    use_explicit_mean_state ||
        (Zb = (θb .- mn) ./ sqrt(eki.N_ens - 1))
    Yb = (xb .- xbar) ./ sqrt(eki.N_ens - 1)

    Σ_y_n = Σ_y_sqrt_n * Σ_y_sqrt_n'
    Σ_y = eki.Σ_y_sqrt * eki.Σ_y_sqrt'

    Cθx = Zb * Yb'
    Cxx = Yb * Yb' + Σ_y_n

    # Kalman gain (SVD-based pseudo-inverse for robustness on rank-deficient
    # or ill-conditioned sample covariances)
    K = Cθx * _safe_pinv(Cxx)
    mean_step_result = nothing
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
            Z_hat = zeros(FT, eki.N_θ, size(θb, 2))
            if use_explicit_mean_state
                mean_step_result = (;
                    accepted=true,
                    mean=copy(mn),
                    prediction=Matrix{FT}(x_mean_bar),
                    γ=zero(FT),
                    trials=0,
                    backtracks=0,
                    reduction_ratio=FT(NaN),
                    source="covariance-only",
                )
                θ_new = mean_step_result.mean .+
                        Z_hat * sqrt(eki.N_ens - 1)
            else
                θ_new = m_hat .+ Z_hat * sqrt(eki.N_ens - 1)
            end
        else
            P_r = P[:,1:r]
            V_r = V[:,1:r]
            temp = Σ_y_sqrt_n \ Yb
            if use_explicit_mean_state
                gram = Symmetric(Matrix(I, size(temp, 2), size(temp, 2)) + temp' * temp)
                transformed_V = try
                    cholesky(gram) \ V_r
                catch
                    stable_solve(Matrix(gram), V_r)
                end
                S = Symmetric(V_r' * transformed_V)
            else
                S = Symmetric(V_r' * ((I + temp' * temp) \ V_r))
            end
            eig = eigen(S)
            U, D = eig.vectors, eig.values
            D .= max.(D, eps(Float64))
            A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
            d_tol = max(eps(FT) * length(Db_sqrt) * maximum(Db_sqrt), FT(1e-12) * maximum(Db_sqrt))
            d_safe = max.(Db_sqrt[1:r], d_tol)
            A2 = Diagonal(sqrt.(max.(D, zero(FT)))) * Diagonal(1 ./ d_safe) * P_r'
            Z_hat = A1 * (A2 * Zb)
            if use_explicit_mean_state
                Z_hat .-= mean(Z_hat, dims=2)
                residual = eki.y .- vec(x_mean_bar)
                subspace_coefficient = stable_solve(Cxx, residual)
                subspace_direction = vec(m_hat - mn)
                subspace_q_model = vec(Yb * (Yb' * subspace_coefficient))

                if eki.dropout_correction_mode == "joint"
                    joint_proposal = joint_projected_dropout_proposal(
                        eki, forward, mn, x_mean_bar, Zb, Yb, Σ_y_n)
                    mean_step_result = _globalize_dropout_mean(
                        eki, forward, mn, x_mean_bar,
                        joint_proposal.direction,
                        joint_proposal.observation_change,
                        subspace_direction, subspace_q_model)
                else
                    Z_tilde_⊥, Y_tilde_⊥, _ = projected_dropout_components(
                        eki, forward, m_hat, Zb, Yb, Σ_y_n)
                    x_m_hat = _prediction_matrix(forward(m_hat))
                    dropout_coefficient = stable_solve(
                        Y_tilde_⊥ * Y_tilde_⊥' + Σ_y_n,
                        eki.y .- vec(x_m_hat))
                    dropout_direction = vec(
                        Z_tilde_⊥ * (Y_tilde_⊥' * dropout_coefficient))
                    dropout_q = vec(
                        Y_tilde_⊥ * (Y_tilde_⊥' * dropout_coefficient))
                    subspace_q = vec(x_m_hat) - vec(x_mean_bar)
                    mean_step_result = _globalize_dropout_mean(
                        eki, forward, mn, x_mean_bar,
                        subspace_direction + dropout_direction,
                        subspace_q + dropout_q,
                        subspace_direction, subspace_q;
                        known_subspace_prediction=x_m_hat)
                end
                θ_new = mean_step_result.mean .+
                        Z_hat * sqrt(eki.N_ens - 1)
            elseif eki.dropout_correction_mode == "joint"
                m_new = joint_projected_dropout_mean(
                    eki, forward, mn, x_mean_bar, Zb, Yb, Σ_y_n)
                θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
            else
                m_new = dropout_optimization_mean(eki, forward, m_hat, Zb, Yb, Σ_y_n)
                θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
            end
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

    reverted_nonfinite = any(x -> !isfinite(x), θ_new)
    if reverted_nonfinite
        @warn "Non-finite ensemble update detected; reverting to previous ensemble" filter_type=filter_type
        θ_new = copy(θ_prev)
    end

    if use_explicit_mean_state
        if reverted_nonfinite
            eki.mean_state = copy(mn)
            eki.anomaly_state = copy(Z_prev)
            eki.mean_prediction_cache = Matrix{FT}(x_mean_bar)
            mean_step_result = (;
                accepted=true,
                mean=copy(mn),
                prediction=Matrix{FT}(x_mean_bar),
                γ=zero(FT),
                trials=mean_step_result === nothing ? 0 : mean_step_result.trials,
                backtracks=mean_step_result === nothing ? 0 : mean_step_result.backtracks,
                reduction_ratio=FT(NaN),
                source="numerical-revert",
            )
        else
            eki.mean_state = copy(mean_step_result.mean)
            eki.anomaly_state = copy(Z_hat)
            eki.mean_prediction_cache = copy(mean_step_result.prediction)
        end
        push!(eki.mean_step_γ, mean_step_result.γ)
        push!(eki.mean_step_source, mean_step_result.source)
        push!(eki.mean_step_trials, mean_step_result.trials)
        push!(eki.mean_step_backtracks, mean_step_result.backtracks)
        push!(eki.mean_step_reduction_ratio, mean_step_result.reduction_ratio)
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
                 joint_weight_mode::String="fixed",
                 joint_weight_min::FT=FT(0.1),
                 joint_weight_max::FT=FT(10.0),
                 joint_weight_smoothing::FT=FT(0.25),
                 mean_line_search::Bool=false,
                 mean_line_search_contraction::FT=FT(0.5),
                 mean_line_search_armijo_c::FT=FT(1e-4),
                 iteration_callback::Union{Nothing,Function}=nothing) where FT<:AbstractFloat
    N_y = size(y, 1)
    size(θ0, 2) > 1 || error("Need at least 2 ensemble members")
    if inflation && !(0 < Δτ < 1)
        error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
    end
    mean_line_search && filter_type != "dropout-EAKI" &&
        error("mean_line_search is implemented only for dropout-EAKI")
    mean_line_search && !inflation &&
        error("mean_line_search requires inflation=true")
    joint_weight_mode in ("fixed", "adaptive") ||
        throw(ArgumentError("joint_weight_mode must be fixed or adaptive"))
    joint_weight_mode == "adaptive" &&
        !(filter_type == "dropout-EAKI" && dropout_correction_mode == "joint") &&
        throw(ArgumentError("adaptive joint weights require joint dropout-EAKI"))
    dropout_deviation_Δτ_actual = isnothing(dropout_deviation_Δτ) ? Δτ : dropout_deviation_Δτ
    dropout_mean_Δτ_actual = isnothing(dropout_mean_Δτ) ? Δτ : dropout_mean_Δτ

    # Obtain forward function for parallel evaluation
    func(x) = forward_parallel ? forward(x) : ensemble_forward(forward, x, N_y)

    y_pred_0 = func(θ0)
    joint_weight_config = JointWeightConfig(FT;
        w_min=joint_weight_min,
        w_max=joint_weight_max,
        smoothing=joint_weight_smoothing)
    # EKI obj initialization
    obj = EKIObj(filter_type, θ0, y_pred_0, y, Σ_y, Δτ, dropout_rate, inflation,
                 dropout_deviation_Δτ_actual, dropout_mean_Δτ_actual, dropout_linearization_bound,
                 dropout_correction_mode, joint_dropout_weight,
                 joint_weight_mode, joint_weight_config,
                 mean_line_search, mean_line_search_contraction,
                 mean_line_search_armijo_c)
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
