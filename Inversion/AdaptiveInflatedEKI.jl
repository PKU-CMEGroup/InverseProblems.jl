module AdaptiveInflatedEKI

using LinearAlgebra, Random, Statistics, SparseArrays

export AdaptiveInflatedEKI, EKIObj, EKI_Run,
       update_ensemble!, ensemble_forward, opt_errors,
       LowRankPrior, prior_coordinates, prior_coordinate_dimension,
       prior_penalty, low_rank_augmented_inner, EKI_Run_Low_Rank_Prior,
       low_rank_opt_errors

abstract type AbstractCovarianceMetric{FT<:AbstractFloat} end

struct DenseCovarianceMetric{FT<:AbstractFloat,F} <: AbstractCovarianceMetric{FT}
    factor::F
end

struct DiagonalCovarianceMetric{FT<:AbstractFloat,V<:AbstractVector{FT}} <:
       AbstractCovarianceMetric{FT}
    standard_deviation::V
end

struct SparseCovarianceMetric{FT<:AbstractFloat,IT<:Integer,L} <:
       AbstractCovarianceMetric{FT}
    sqrt_permuted::L
    permutation::Vector{IT}
end

function covariance_metric(covariance::Matrix{FT}) where FT<:AbstractFloat
    covariance_symmetric = Symmetric(copy(covariance), :L)
    factor = try
        cholesky(covariance_symmetric)
    catch
        eig = eigen(covariance_symmetric)
        lambda_max = maximum(abs, eig.values)
        lambda_floor = lambda_max == zero(FT) ? one(FT) :
            max(eps(FT) * lambda_max, FT(1e-12) * lambda_max)
        regularized = eig.vectors * Diagonal(max.(eig.values, lambda_floor)) * eig.vectors'
        cholesky(Symmetric(regularized, :L))
    end
    return DenseCovarianceMetric{FT,typeof(factor)}(factor)
end

function covariance_metric(covariance::Diagonal{FT}) where FT<:AbstractFloat
    diagonal = covariance.diag
    all(isfinite, diagonal) || throw(ArgumentError("covariance must be finite"))
    all(>(zero(FT)), diagonal) || throw(ArgumentError("diagonal covariance must be positive"))
    standard_deviation = sqrt.(diagonal)
    return DiagonalCovarianceMetric{FT,typeof(standard_deviation)}(standard_deviation)
end

function covariance_metric(covariance::SparseMatrixCSC{FT,IT}) where {FT<:AbstractFloat,IT<:Integer}
    size(covariance, 1) == size(covariance, 2) ||
        throw(DimensionMismatch("covariance must be square"))
    factor = cholesky(Symmetric(covariance))
    permutation = collect(factor.p)
    sqrt_permuted = LowerTriangular(sparse(factor.L))
    return SparseCovarianceMetric{FT,eltype(permutation),typeof(sqrt_permuted)}(
        sqrt_permuted, permutation,
    )
end

whiten(metric::DenseCovarianceMetric, values) = metric.factor.L \ values
whiten(metric::DiagonalCovarianceMetric, values::AbstractVector) =
    values ./ metric.standard_deviation
whiten(metric::DiagonalCovarianceMetric, values::AbstractMatrix) =
    values ./ reshape(metric.standard_deviation, :, 1)
whiten(metric::SparseCovarianceMetric, values::AbstractVector) =
    metric.sqrt_permuted \ values[metric.permutation]
whiten(metric::SparseCovarianceMetric, values::AbstractMatrix) =
    metric.sqrt_permuted \ values[metric.permutation, :]

weighted_objective(metric::AbstractCovarianceMetric, residual) =
    sum(abs2, whiten(metric, residual)) / 2
Base.:(\)(metric::AbstractCovarianceMetric, values) = whiten(metric, values)

function sample_noise(metric::DenseCovarianceMetric{FT}, count::Int) where FT
    return metric.factor.L * randn(FT, size(metric.factor.L, 1), count)
end
function sample_noise(metric::DiagonalCovarianceMetric{FT}, count::Int) where FT
    return metric.standard_deviation .* randn(FT, length(metric.standard_deviation), count)
end
function sample_noise(metric::SparseCovarianceMetric{FT}, count::Int) where FT
    permuted = metric.sqrt_permuted * randn(FT, size(metric.sqrt_permuted, 1), count)
    result = similar(permuted)
    result[metric.permutation, :] = permuted
    return result
end

struct LowRankPrior{FT<:AbstractFloat,F}
    mean::Vector{FT}
    coordinate_dimension::Int
    coordinate_map::F
end

function LowRankPrior(mean::AbstractVector{FT}, U::AbstractMatrix) where FT<:AbstractFloat
    size(U, 1) == length(mean) || throw(DimensionMismatch("mean and U disagree"))
    F = svd(Matrix{FT}(U); full=false, alg=LinearAlgebra.QRIteration())
    active_svd_rank(F.S) == size(U, 2) ||
        throw(ArgumentError("U must have full column rank"))
    coordinates(values::AbstractVector) = F.V * ((F.U' * values) ./ F.S)
    coordinates(values::AbstractMatrix) = F.V * ((F.U' * values) ./ F.S)
    return LowRankPrior{FT,typeof(coordinates)}(collect(mean), size(U, 2), coordinates)
end

function LowRankPrior(mean::AbstractVector{FT}, coordinate_dimension::Integer,
        coordinate_map::Function) where FT<:AbstractFloat
    coordinate_dimension > 0 || throw(ArgumentError("coordinate_dimension must be positive"))
    return LowRankPrior{FT,typeof(coordinate_map)}(
        collect(mean), Int(coordinate_dimension), coordinate_map)
end

LowRankPrior(mean::Vector{FT}, dimension::Int, map::F) where
        {FT<:AbstractFloat,F<:Function} = LowRankPrior{FT,F}(mean, dimension, map)

function prior_coordinates(prior::LowRankPrior, values::AbstractVector)
    coordinates = prior.coordinate_map(values)
    length(coordinates) == prior.coordinate_dimension ||
        throw(DimensionMismatch("prior coordinate map returned the wrong length"))
    return coordinates
end

function prior_coordinates(prior::LowRankPrior, values::AbstractMatrix)
    coordinates = prior.coordinate_map(values)
    size(coordinates) == (prior.coordinate_dimension, size(values, 2)) ||
        throw(DimensionMismatch("prior coordinate map returned the wrong size"))
    return coordinates
end

prior_coordinate_dimension(prior::LowRankPrior) = prior.coordinate_dimension
prior_penalty(prior::LowRankPrior, θ::AbstractVector) =
    sum(abs2, prior_coordinates(prior, θ - prior.mean)) / 2

function low_rank_augmented_inner(Σ_y_sqrt, prior, Ya, Za, Yb, Zb; scale=1)
    return scale^2 .* ((Σ_y_sqrt \ Ya)' * (Σ_y_sqrt \ Yb) +
        prior_coordinates(prior, Za)' * prior_coordinates(prior, Zb))
end

mutable struct EKIObj{FT<:AbstractFloat,IT<:Int,MT<:AbstractCovarianceMetric{FT}}
    filter_type::String
    θ::Vector{Array{FT,2}}
    y_pred::Vector{Array{FT,2}}
    y::Array{FT,1}
    Σ_y_sqrt::MT
    N_ens::IT
    N_θ::IT
    N_y::IT
    Δτ::FT
    dropout_rate::FT
    inflation::Bool
    dropout_correction_mode::String
    joint_dropout_weight::FT
    joint_weight_mode::String
    joint_previous_objective::FT
    joint_previous_predicted_reduction::FT
    mean_line_search::Bool
end

function EKIObj(filter_type::String, θ0::Array{FT,2}, y_pred_0::Array{FT,2},
        y::Array{FT,1}, Σ_y::AbstractMatrix{FT}, Δτ::FT,
        dropout_rate::FT=FT(0.5), inflation::Bool=true,
        dropout_correction_mode::String="joint",
        joint_dropout_weight::FT=one(FT),
        joint_weight_mode::String="fixed",
        mean_line_search::Bool=false) where FT<:AbstractFloat
    N_θ, N_ens = size(θ0)
    N_y = length(y)
    size(y_pred_0) == (N_y, 1) || throw(DimensionMismatch("y_pred_0 must have size (N_y, 1)"))
    dropout_correction_mode in ("sequential", "joint") ||
        throw(ArgumentError("dropout_correction_mode must be sequential or joint"))
    joint_dropout_weight > 0 || throw(ArgumentError("joint_dropout_weight must be positive"))
    joint_weight_mode in ("fixed", "adaptive") ||
        throw(ArgumentError("joint_weight_mode must be fixed or adaptive"))
    metric = covariance_metric(Σ_y)
    return EKIObj(
        filter_type, [θ0], [y_pred_0], y, metric, N_ens, N_θ, N_y, Δτ,
        dropout_rate, inflation, dropout_correction_mode,
        joint_dropout_weight, joint_weight_mode,
        FT(NaN), FT(NaN), mean_line_search,
    )
end

function active_svd_rank(s::AbstractVector{FT}) where FT<:AbstractFloat
    isempty(s) && return 0
    maximum(s) == zero(FT) && return 0
    tolerance = max(eps(FT) * length(s) * maximum(s), FT(1e-12) * maximum(s))
    rank = findlast(s .> tolerance)
    return isnothing(rank) ? 0 : rank
end

function _safe_pinv_apply(A::AbstractMatrix{FT}, B) where FT<:AbstractFloat
    factorization = svd(A; full=false, alg=LinearAlgebra.QRIteration())
    rank = active_svd_rank(factorization.S)
    rank == 0 && return zeros(
        FT, size(A, 2), size(B, 2),
    )
    return factorization.V[:, 1:rank] * (
        (factorization.U[:, 1:rank]' * B) ./
        reshape(factorization.S[1:rank], :, 1)
    )
end

function _ensemble_factor(A)
    all(isfinite, A) || throw(ArgumentError("non-finite ensemble anomalies"))
    return cholesky(Symmetric(I + A' * A))
end

function safe_diag_inv(values::AbstractVector{FT}) where FT<:AbstractFloat
    isempty(values) && return similar(values)
    scale = maximum(abs, values)
    scale == zero(FT) && return zeros(FT, length(values))
    tolerance = max(eps(FT) * length(values) * scale, FT(1e-12) * scale)
    return [abs(value) > tolerance ? inv(value) : zero(FT) for value in values]
end

function ensemble_forward(forward::Function, θ::Array{FT,2}, N_y::Int) where FT<:AbstractFloat
    predictions = zeros(FT, N_y, size(θ, 2))
    Threads.@threads for j in axes(θ, 2)
        predictions[:, j] = forward(θ[:, j])
    end
    return predictions
end

function dropout_mask(eki::EKIObj{FT}) where FT<:AbstractFloat
    keep_probability = one(FT) - eki.dropout_rate
    0 < keep_probability <= 1 ||
        throw(ArgumentError("dropout_rate must satisfy 0 <= dropout_rate < 1"))
    mask = rand(eki.N_θ) .< keep_probability
    while !any(mask)
        mask .= rand(eki.N_θ) .< keep_probability
    end
    return reshape(FT.(mask), :, 1)
end

_scaled_whiten(eki::EKIObj, values, scale) = scale .* whiten(eki.Σ_y_sqrt, values)

_augmented_anomalies(eki::EKIObj, Y, Z, scale, ::Nothing) =
    _scaled_whiten(eki, Y, scale)
_augmented_anomalies(eki::EKIObj, Y, Z, scale, prior::LowRankPrior) =
    vcat(_scaled_whiten(eki, Y, scale), scale .* prior_coordinates(prior, Z))

_augmented_residual(eki::EKIObj, y_residual, θ_residual, scale, ::Nothing) =
    _scaled_whiten(eki, y_residual, scale)
_augmented_residual(eki::EKIObj, y_residual, θ_residual, scale,
        prior::LowRankPrior) = vcat(
    _scaled_whiten(eki, y_residual, scale),
    scale .* prior_coordinates(prior, θ_residual),
)

_objective(eki::EKIObj, y_residual, θ, ::Nothing) =
    weighted_objective(eki.Σ_y_sqrt, y_residual)
_objective(eki::EKIObj, y_residual, θ, prior::LowRankPrior) =
    weighted_objective(eki.Σ_y_sqrt, y_residual) + prior_penalty(prior, θ)

function _ensemble_gain_mul(Z, A, factor, augmented_residual)
    coefficient = factor \ (A' * augmented_residual)
    return Z * coefficient, coefficient
end

function _projected_dropout_components(eki::EKIObj{FT}, forward::Function,
        center, Zb, A, scale, prior=nothing) where FT<:AbstractFloat
    Z_tilde = dropout_mask(eki) .* Zb
    θ_tilde = center .+ Z_tilde * sqrt(eki.N_ens - 1)
    x_tilde = forward(θ_tilde)
    Y_tilde = (x_tilde .- mean(x_tilde, dims=2)) ./ sqrt(eki.N_ens - 1)
    A_tilde = _augmented_anomalies(eki, Y_tilde, Z_tilde, scale, prior)
    projection = _safe_pinv_apply(A, A_tilde)
    return Z_tilde - Zb * projection, Y_tilde,
           A_tilde - A * projection, projection
end

function _dropout_optimization_mean(eki::EKIObj, forward::Function,
        m_hat, Zb, A, scale, prior=nothing)
    Z_perp, _, A_perp, _ = _projected_dropout_components(
        eki, forward, m_hat, Zb, A, scale, prior,
    )
    prediction = forward(m_hat)
    θ_residual = isnothing(prior) ? nothing : prior.mean .- vec(m_hat)
    residual = _augmented_residual(
        eki, eki.y .- vec(prediction), θ_residual, scale, prior,
    )
    factor = _ensemble_factor(A_perp)
    coefficient = factor \ (A_perp' * residual)
    return m_hat .+ reshape(Z_perp * coefficient, :, 1)
end

function adaptive_joint_weight(current::FT, actual_reduction, predicted_reduction) where FT
    lower, upper, smoothing = FT(0.1), FT(10), FT(0.25)
    if !(isfinite(actual_reduction) && isfinite(predicted_reduction))
        return clamp(current, lower, upper)
    end
    ratio = predicted_reduction <= eps(FT) ? zero(FT) :
        clamp(actual_reduction / predicted_reduction, zero(FT), one(FT))
    target = lower + (upper - lower) * ratio
    return clamp((one(FT) - smoothing) * current + smoothing * target, lower, upper)
end

function _joint_projected_dropout_proposal(eki::EKIObj{FT}, forward::Function,
        mn, x_mn, Zb, Yb, A, scale, prior=nothing) where FT<:AbstractFloat
    residual = eki.y .- vec(x_mn)
    current_objective = scale^2 * _objective(eki, residual, vec(mn), prior)
    actual_reduction = isfinite(eki.joint_previous_objective) ?
        eki.joint_previous_objective - current_objective : FT(NaN)
    weight_value = eki.joint_dropout_weight
    if eki.joint_weight_mode == "adaptive" && isfinite(actual_reduction)
        weight_value = adaptive_joint_weight(
            eki.joint_dropout_weight, actual_reduction,
            eki.joint_previous_predicted_reduction,
        )
    end
    Z_perp, Y_tilde, A_perp, projection =
        _projected_dropout_components(
            eki, forward, mn, Zb, A, scale, prior,
        )
    Y_perp = Y_tilde - Yb * projection
    weight = sqrt(weight_value)
    Z_augmented = hcat(Zb, weight .* Z_perp)
    Y_augmented = hcat(Yb, weight .* Y_perp)
    A_augmented = hcat(A, weight .* A_perp)
    θ_residual = isnothing(prior) ? nothing : prior.mean .- vec(mn)
    augmented_residual = _augmented_residual(
        eki, residual, θ_residual, scale, prior,
    )
    factor = _ensemble_factor(A_augmented)
    coefficient = factor \ (A_augmented' * augmented_residual)
    direction = vec(Z_augmented * coefficient)
    observation_change = vec(Y_augmented * coefficient)
    predicted_reduction = current_objective - scale^2 * _objective(
        eki, residual - observation_change, vec(mn) + direction, prior,
    )
    return (; direction, observation_change, weight=weight_value,
            predicted_reduction, current_objective)
end

function _try_mean_armijo(eki::EKIObj{FT}, forward::Function, mn, x_mean,
        direction, q, prior=nothing) where FT
    all(isfinite, direction) && all(isfinite, q) || return nothing
    residual = _augmented_residual(
        eki, vec(x_mean) - eki.y,
        isnothing(prior) ? nothing : vec(mn) - prior.mean, one(FT), prior)
    change = _augmented_residual(
        eki, q, isnothing(prior) ? nothing : direction, one(FT), prior)
    slope, curvature = dot(residual, change), dot(change, change)
    slope < 0 && curvature > eps(FT) * max(dot(residual, residual), one(FT)) ||
        return nothing
    objective0 = _objective(eki, vec(x_mean) - eki.y, vec(mn), prior)
    gamma = -slope / curvature
    minimum_gamma = sqrt(eps(FT)) * (one(FT) + norm(mn)) /
                    max(norm(direction), eps(FT))
    while gamma >= minimum_gamma
        candidate = vec(mn) + gamma .* direction
        if !all(isfinite, candidate)
            gamma *= FT(0.5)
            continue
        end
        prediction = reshape(forward(reshape(candidate, :, 1)), :, 1)
        objective_trial = _objective(
            eki, vec(prediction) - eki.y, candidate, prior,
        )
        armijo_bound = objective0 + FT(1e-4) * gamma * slope
        if isfinite(objective_trial) && objective_trial <= armijo_bound
            predicted_reduction = -gamma * slope - gamma^2 * curvature / 2
            return (; accepted=true, mean=reshape(candidate, :, 1),
                    prediction=Matrix{FT}(prediction),
                    predicted_reduction=FT(predicted_reduction))
        end
        gamma *= FT(0.5)
    end
    return nothing
end

function _globalize_dropout_mean(eki::EKIObj{FT}, forward::Function, mn, x_mean,
        full_direction, full_q, fallback_direction, fallback_q, prior=nothing) where FT
    for (direction, q) in ((full_direction, full_q), (fallback_direction, fallback_q))
        norm(direction) > 0 || continue
        result = _try_mean_armijo(eki, forward, mn, x_mean, direction, q, prior)
        result === nothing || return result
    end
    return (; accepted=true, mean=copy(mn), prediction=Matrix{FT}(x_mean),
            predicted_reduction=zero(FT))
end

function deki_linearized_observation_deviations(T::Array{FT,2}, Y::Array{FT,2},
        bound::FT) where FT<:AbstractFloat
    bound > 0 || throw(ArgumentError("dropout_linearization_bound must be positive"))
    svd_T = svd(T; full=false)
    rank_T = active_svd_rank(svd_T.S)
    rank_T == 0 && return zeros(FT, size(Y, 1), size(T, 2))
    svd_Y = svd(Y; full=false)
    rank_Y = active_svd_rank(svd_Y.S)
    rank_Y == 0 && return zeros(FT, size(Y, 1), size(T, 2))
    singular_T = svd_T.S[1:rank_T]
    vectors_T = svd_T.V[:, 1:rank_T]
    W = svd_Y.U[:, 1:rank_Y]
    R = Diagonal(svd_Y.S[1:rank_Y]) * svd_Y.V[:, 1:rank_Y]'
    derivative = R * vectors_T * Diagonal(safe_diag_inv(singular_T))
    if isfinite(bound)
        derivative_svd = svd(derivative; full=false)
        derivative = derivative_svd.U * Diagonal(min.(derivative_svd.S, bound)) * derivative_svd.V'
    end
    return W * derivative * Diagonal(singular_T) * vectors_T'
end

function _eaki_anomalies(Zb, factor)
    P, singular_values, V = svd(Zb; full=false)
    rank = active_svd_rank(singular_values)
    rank == 0 && return zeros(eltype(Zb), size(Zb))
    P_rank = P[:, 1:rank]
    V_rank = V[:, 1:rank]
    inverse_gram_V = factor \ V_rank
    eig = eigen(Symmetric(V_rank' * inverse_gram_V))
    first = P_rank * Diagonal(singular_values[1:rank]) * eig.vectors
    second = Diagonal(sqrt.(max.(eig.values, zero(eltype(Zb))))) *
             Diagonal(safe_diag_inv(singular_values[1:rank])) * P_rank'
    return first * (second * Zb)
end

function _etki_transform(A)
    eig = eigen(Symmetric(A' * A))
    return eig.vectors * Diagonal(1 ./ sqrt.(max.(1 .+ eig.values, eps(eltype(A))))) *
           eig.vectors'
end

function update_ensemble!(eki::EKIObj{FT}, forward::Function,
        prior::Union{Nothing,LowRankPrior}=nothing) where FT<:AbstractFloat
    θ_prev = eki.θ[end]
    use_line_search = eki.mean_line_search && eki.filter_type == "dropout-EAKI"
    mn = mean(θ_prev, dims=2)
    Z_prev = (θ_prev .- mn) ./ sqrt(eki.N_ens - 1)
    if eki.inflation
        0 < eki.Δτ < 1 || throw(ArgumentError("Δτ must lie in (0, 1) with inflation"))
        Zb = Z_prev ./ sqrt(1 - eki.Δτ)
        θb = mn .+ Zb * sqrt(eki.N_ens - 1)
        scale = sqrt(eki.Δτ)
    else
        Zb = Z_prev
        θb = θ_prev
        scale = one(FT)
    end
    xb = forward(θb)
    x_mean = eki.y_pred[end]
    Yb = (xb .- mean(xb, dims=2)) ./ sqrt(eki.N_ens - 1)
    prior_residual = isnothing(prior) ? nothing : prior.mean .- vec(mn)
    filter_type = eki.filter_type
    mean_result = nothing
    joint_proposal = nothing

    if filter_type == "DEKI"
        T = θb .- mn
        covariance_norm = maximum(svdvals(Zb))^2
        if covariance_norm == 0
            θ_new = copy(θb)
        else
            regularizer = sqrt(eps(FT)) * max(covariance_norm, one(FT))
            h = eki.Δτ / (covariance_norm + regularizer)
            h_mean = h
            θ_tilde = mn .+ dropout_mask(eki) .* T
            x_tilde = forward(θ_tilde)
            Z_tilde = (θ_tilde .- mn) ./ sqrt(eki.N_ens - 1)
            Y_tilde = (x_tilde .- mean(x_tilde, dims=2)) ./ sqrt(eki.N_ens - 1)
            mean_scale = scale * sqrt(h_mean)
            mean_A = _augmented_anomalies(eki, Y_tilde, Z_tilde, mean_scale, nothing)
            mean_factor = _ensemble_factor(mean_A)
            mean_residual = _augmented_residual(
                eki, eki.y .- vec(x_mean), nothing, mean_scale, nothing,
            )
            mean_update, _ = _ensemble_gain_mul(
                Z_tilde, mean_A, mean_factor, mean_residual,
            )

            linearized = deki_linearized_observation_deviations(
                T, xb .- mean(xb, dims=2), FT(Inf),
            )
            Y_linearized = linearized ./ sqrt(eki.N_ens - 1)
            deviation_scale = scale * sqrt(h)
            deviation_A = _augmented_anomalies(
                eki, Y_linearized, Zb, deviation_scale, nothing,
            )
            deviation_factor = _ensemble_factor(deviation_A)
            deviation_residual = _augmented_residual(
                eki, linearized, nothing, deviation_scale, nothing,
            )
            deviation_update, _ = _ensemble_gain_mul(
                Zb, deviation_A, deviation_factor, deviation_residual,
            )
            θ_new = mn .+ reshape(mean_update, :, 1) .+ T .- deviation_update
        end
    else
        A = _augmented_anomalies(eki, Yb, Zb, scale, prior)
        factor = _ensemble_factor(A)
        base_residual = _augmented_residual(
            eki, eki.y .- vec(x_mean), prior_residual, scale, prior,
        )
        mean_increment, base_coefficient = _ensemble_gain_mul(
            Zb, A, factor, base_residual,
        )
        m_hat = mn .+ reshape(mean_increment, :, 1)

    if filter_type in ("EKI", "NF-EKI")
        residual = reshape(eki.y, :, 1) .- xb
        if filter_type == "EKI"
            noise_scale = eki.inflation ? inv(sqrt(eki.Δτ)) : one(FT)
            residual .-= noise_scale .* sample_noise(eki.Σ_y_sqrt, eki.N_ens)
        end
        augmented_residual = _augmented_residual(
            eki, residual, nothing, scale, nothing,
        )
        update, _ = _ensemble_gain_mul(
            Zb, A, factor, augmented_residual,
        )
        θ_new = θb + update
    elseif filter_type == "EAKI"
        θ_new = m_hat .+ _eaki_anomalies(Zb, factor) * sqrt(eki.N_ens - 1)
    elseif filter_type == "ETKI"
        transform = _etki_transform(A)
        θ_new = m_hat .+ Zb * transform * sqrt(eki.N_ens - 1)
    elseif filter_type == "dropout-EKI"
        ensemble_residual = _augmented_residual(
            eki, reshape(eki.y, :, 1) .- xb, nothing, scale, nothing,
        )
        ensemble_update, _ = _ensemble_gain_mul(
            Zb, A, factor, ensemble_residual,
        )
        θ_hat = θb + ensemble_update
        θ_hat .+= m_hat .- mean(θ_hat, dims=2)
        Z_hat = (θ_hat .- m_hat) ./ sqrt(eki.N_ens - 1)
        m_new = _dropout_optimization_mean(
            eki, forward, m_hat, Zb, A, scale, prior,
        )
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
    elseif filter_type == "dropout-EAKI"
        Z_hat = _eaki_anomalies(Zb, factor)
        Z_hat .-= mean(Z_hat, dims=2)
        if eki.dropout_correction_mode == "joint"
            proposal = _joint_projected_dropout_proposal(
                eki, forward, mn, x_mean, Zb, Yb, A, scale, prior)
            if use_line_search
                mean_result = _globalize_dropout_mean(
                    eki, forward, mn, x_mean, proposal.direction,
                    proposal.observation_change, vec(mean_increment),
                    vec(Yb * base_coefficient), prior)
                θ_new = mean_result.mean .+ Z_hat * sqrt(eki.N_ens - 1)
                accepted_prediction = scale^2 * mean_result.predicted_reduction
            else
                m_new = mn .+ reshape(proposal.direction, :, 1)
                θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
                accepted_prediction = proposal.predicted_reduction
            end
            joint_proposal = merge(proposal,
                (; accepted_predicted_reduction=accepted_prediction))
        else
            m_new = _dropout_optimization_mean(
                eki, forward, m_hat, Zb, A, scale, prior)
            θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
        end
    elseif filter_type == "dropout-ETKI"
        eki.dropout_correction_mode == "joint" && throw(ArgumentError(
            "joint correction is implemented only for dropout-EAKI"))
        transform = _etki_transform(A)
        Z_hat = Zb * transform
        m_new = _dropout_optimization_mean(
            eki, forward, m_hat, Zb, A, scale, prior,
        )
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
    else
        throw(ArgumentError("Unknown filter_type: $(filter_type)"))
    end
    end
    size(θ_new) == size(θ_prev) || throw(DimensionMismatch("ensemble update has wrong size"))
    all(isfinite, θ_new) || throw(ArgumentError("non-finite ensemble update"))
    mean_prediction = reshape(
        use_line_search ? mean_result.prediction : forward(mean(θ_new, dims=2)), :, 1)
    size(mean_prediction) == (eki.N_y, 1) ||
        throw(DimensionMismatch("mean prediction has wrong size"))
    all(isfinite, mean_prediction) || throw(ArgumentError("non-finite mean prediction"))
    mean_prediction = Matrix{FT}(mean_prediction)
    push!(eki.θ, θ_new)
    push!(eki.y_pred, mean_prediction)
    if joint_proposal !== nothing
        eki.joint_dropout_weight = joint_proposal.weight
        eki.joint_previous_objective = joint_proposal.current_objective
        eki.joint_previous_predicted_reduction =
            joint_proposal.accepted_predicted_reduction
    end
    return eki
end

function EKI_Run(forward::Function, θ0::Array{FT,2}, Σ_y::AbstractMatrix{FT}, y::Array{FT,1};
        filter_type::String="EKI", Δτ::FT=FT(0.5), N_iter::Int=50,
        forward_parallel::Bool=false, dropout_rate::FT=FT(0.5), inflation::Bool=true,
        dropout_correction_mode::String="joint",
        joint_dropout_weight::FT=one(FT), joint_weight_mode::String="fixed",
        mean_line_search::Bool=false,
        prior::Union{Nothing,LowRankPrior}=nothing) where FT<:AbstractFloat
    size(θ0, 2) > 1 || throw(ArgumentError("Need at least 2 ensemble members"))
    mean_line_search && !(filter_type == "dropout-EAKI" &&
                          dropout_correction_mode == "joint" && inflation) &&
        throw(ArgumentError("mean_line_search requires joint inflated dropout-EAKI"))
    joint_weight_mode == "adaptive" &&
        !(filter_type == "dropout-EAKI" && dropout_correction_mode == "joint") &&
        throw(ArgumentError("adaptive joint weights require joint dropout-EAKI"))
    N_y = length(y)
    func(values) = forward_parallel ? forward(values) : ensemble_forward(forward, values, N_y)
    mean_forward(values) = forward_parallel ? forward(values) :
        reshape(forward(vec(values)), :, 1)
    wrapped_forward(values) = size(values, 2) == 1 ? mean_forward(values) : func(values)
    y_pred_0 = mean_forward(mean(θ0, dims=2))
    object = EKIObj(
        filter_type, θ0, y_pred_0, y, Σ_y, Δτ, dropout_rate, inflation,
        dropout_correction_mode, joint_dropout_weight, joint_weight_mode,
        mean_line_search,
    )
    @info "Running ", filter_type, " with ensemble size ", size(θ0, 2)
    for iteration in 1:N_iter
        iteration % max(1, div(N_iter, 10)) == 0 &&
            @info ("Iteration ", iteration, "/", N_iter)
        update_ensemble!(object, wrapped_forward, prior)
    end
    return object
end

function opt_errors(eki::EKIObj)
    return [weighted_objective(eki.Σ_y_sqrt, vec(prediction) - eki.y)
            for prediction in eki.y_pred]
end

function EKI_Run_Low_Rank_Prior(forward::Function, θ0::Array{FT,2},
        Σ_y::AbstractMatrix{FT}, y::Array{FT,1}, prior::LowRankPrior{FT};
        filter_type::String="dropout-EAKI", kwargs...) where FT<:AbstractFloat
    filter_type == "dropout-EAKI" || throw(ArgumentError(
        "low-rank prior augmentation supports dropout-EAKI only"))
    size(θ0, 1) == length(prior.mean) ||
        throw(DimensionMismatch("ensemble and prior disagree"))
    return EKI_Run(forward, θ0, Σ_y, y; filter_type, prior, kwargs...)
end

function low_rank_opt_errors(eki::EKIObj, prior::LowRankPrior)
    errors = opt_errors(eki)
    for i in eachindex(errors)
        errors[i] += prior_penalty(prior, vec(mean(eki.θ[i], dims=2)))
    end
    return errors
end

end

using .AdaptiveInflatedEKI
