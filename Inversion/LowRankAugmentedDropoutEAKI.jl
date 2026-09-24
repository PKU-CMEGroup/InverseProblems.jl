module LowRankAugmentedDropoutEAKI

include("InflatedEKI.jl")

export LowRankPrior, prior_coordinates, prior_coordinate_dimension,
       prior_penalty, sample_prior,
       low_rank_augmented_inner, EKI_Run_Low_Rank_Prior,
       low_rank_opt_errors, EKI_Run, opt_errors

"""Low-rank Gaussian prior, represented by a factor or a coordinate map."""
struct LowRankPrior{FT<:AbstractFloat}
    mean::Vector{FT}
    left_vectors::Matrix{FT}
    singular_values::Vector{FT}
    right_vectors::Matrix{FT}
    coordinate_map::Union{Nothing,Function}
    coordinate_dimension::Int
end

function LowRankPrior(mean::AbstractVector{FT}, U::AbstractMatrix) where FT<:AbstractFloat
    size(U, 1) == length(mean) || throw(DimensionMismatch("mean and U disagree"))
    size(U, 2) > 0 || throw(ArgumentError("U must have at least one column"))
    U_mat = Matrix{FT}(U)
    all(isfinite, U_mat) || throw(ArgumentError("U must contain only finite values"))
    factorization = svd(U_mat; full=false, alg=LinearAlgebra.QRIteration())
    tolerance = max(
        eps(FT) * maximum(size(U_mat)) * maximum(factorization.S),
        FT(1e-12) * maximum(factorization.S),
    )
    count(>(tolerance), factorization.S) == size(U, 2) ||
        throw(ArgumentError("U must have full column rank"))
    return LowRankPrior(
        collect(mean), Matrix(factorization.U), collect(factorization.S),
        Matrix(factorization.V), nothing, size(U, 2),
    )
end

function LowRankPrior(
    mean::AbstractVector{FT},
    coordinate_dimension::Integer,
    coordinate_map::Function,
) where FT<:AbstractFloat
    coordinate_dimension > 0 || throw(ArgumentError("coordinate_dimension must be positive"))
    all(isfinite, mean) || throw(ArgumentError("mean must contain only finite values"))
    return LowRankPrior(
        collect(mean), zeros(FT, length(mean), 0), FT[], zeros(FT, 0, 0),
        coordinate_map, Int(coordinate_dimension),
    )
end

# Moore-Penrose coordinates U^+ z from a thin SVD, without forming U^+ or U*U'.
function prior_coordinates(prior::LowRankPrior, z::AbstractVector)
    coordinates = isnothing(prior.coordinate_map) ?
        prior.right_vectors * ((prior.left_vectors' * z) ./ prior.singular_values) :
        prior.coordinate_map(z)
    length(coordinates) == prior.coordinate_dimension ||
        throw(DimensionMismatch("prior coordinate map returned the wrong length"))
    all(isfinite, coordinates) || throw(ArgumentError("prior coordinates must be finite"))
    return coordinates
end

function prior_coordinates(prior::LowRankPrior, Z::AbstractMatrix)
    coordinates = isnothing(prior.coordinate_map) ?
        prior.right_vectors * (
            (prior.left_vectors' * Z) ./ reshape(prior.singular_values, :, 1)
        ) : prior.coordinate_map(Z)
    size(coordinates) == (prior.coordinate_dimension, size(Z, 2)) ||
        throw(DimensionMismatch("prior coordinate map returned the wrong size"))
    all(isfinite, coordinates) || throw(ArgumentError("prior coordinates must be finite"))
    return coordinates
end

prior_coordinate_dimension(prior::LowRankPrior) = prior.coordinate_dimension

prior_penalty(prior::LowRankPrior, theta::AbstractVector) =
    sum(abs2, prior_coordinates(prior, theta - prior.mean)) / 2

function sample_prior(rng::AbstractRNG, prior::LowRankPrior, ensemble_size::Int)
    ensemble_size > 1 || throw(ArgumentError("ensemble_size must exceed one"))
    isnothing(prior.coordinate_map) ||
        throw(ArgumentError("sampling requires an explicit prior factor"))
    xi = randn(rng, eltype(prior.mean), length(prior.singular_values), ensemble_size)
    return prior.mean .+ prior.left_vectors * (
        prior.singular_values .* (prior.right_vectors' * xi)
    )
end

function low_rank_augmented_inner(
    Sigma_y_sqrt,
    prior::LowRankPrior,
    Y_a,
    Z_a,
    Y_b,
    Z_b;
    scale=one(eltype(prior.mean)),
)
    Y_a_scaled = scale .* (Sigma_y_sqrt \ Y_a)
    Y_b_scaled = scale .* (Sigma_y_sqrt \ Y_b)
    Z_a_scaled = scale .* prior_coordinates(prior, Z_a)
    Z_b_scaled = scale .* prior_coordinates(prior, Z_b)
    return Y_a_scaled' * Y_b_scaled + Z_a_scaled' * Z_b_scaled
end

function _augmented_anomalies(Sigma_y_sqrt, prior, Y, Z, scale)
    return vcat(
        scale .* (Sigma_y_sqrt \ Y),
        scale .* prior_coordinates(prior, Z),
    )
end

function _augmented_residual(
    Sigma_y_sqrt,
    prior,
    y_residual,
    theta_residual,
    scale,
)
    return vcat(
        scale .* (Sigma_y_sqrt \ y_residual),
        scale .* prior_coordinates(prior, theta_residual),
    )
end

struct EnsembleSpaceSVD{FT<:AbstractFloat}
    left_vectors::Matrix{FT}
    singular_values::Vector{FT}
    right_vectors::Matrix{FT}
end

function _require_finite(values, stage::AbstractString)
    all(isfinite, values) ||
        throw(ArgumentError("non-finite values in $(stage)"))
    return values
end

function _ensemble_space_svd(A::AbstractMatrix{FT}, stage::AbstractString) where FT<:AbstractFloat
    _require_finite(A, stage)
    factorization = svd(Matrix(A); full=false, alg=LinearAlgebra.QRIteration())
    _require_finite(factorization.S, "$(stage) singular values")
    return EnsembleSpaceSVD(
        Matrix(factorization.U), collect(factorization.S), Matrix(factorization.V),
    )
end

# Apply (I + A'A)^(-1) without forming A'A.  The ratio s/hypot(1,s)
# remains bounded even when a finite singular value is too large to square.
function _solve_identity_plus_gram(
    factorization::EnsembleSpaceSVD,
    B,
    stage::AbstractString,
)
    _require_finite(B, "$(stage) right-hand side")
    V = factorization.right_vectors
    ratios = [
        (s / hypot(one(s), s))^2 for s in factorization.singular_values
    ]
    result = B - V * (ratios .* (V' * B))
    return _require_finite(result, stage)
end

# Apply (I + A'A)^(-1)A'r directly from A = Q*diag(s)*V'.
function _solve_identity_plus_gram_adjoint(
    factorization::EnsembleSpaceSVD,
    residual,
    stage::AbstractString,
)
    _require_finite(residual, "$(stage) residual")
    weights = [
        (s / hypot(one(s), s)) / hypot(one(s), s)
        for s in factorization.singular_values
    ]
    result = factorization.right_vectors * (
        weights .* (factorization.left_vectors' * residual)
    )
    return _require_finite(result, stage)
end

function _svd_pinv_apply(A::AbstractMatrix, B, stage::AbstractString)
    factorization = _ensemble_space_svd(A, stage)
    _require_finite(B, "$(stage) right-hand side")
    r = active_svd_rank(factorization.singular_values)
    r == 0 && return zeros(eltype(A), size(A, 2), size(B, 2))
    result = factorization.right_vectors[:, 1:r] * (
        (factorization.left_vectors[:, 1:r]' * B) ./
        reshape(factorization.singular_values[1:r], :, 1)
    )
    return _require_finite(result, stage)
end

function _projected_dropout_components(
    eki::EKIObj{FT},
    forward::Function,
    center,
    Zb,
    A,
    Sigma_y_sqrt,
    prior::LowRankPrior,
    scale,
) where FT<:AbstractFloat
    rho = dropout_mask(eki)
    Z_tilde = rho .* Zb

    theta_tilde = center .+ Z_tilde * sqrt(eki.N_ens - 1)
    x_tilde = forward(theta_tilde)
    Y_tilde = (x_tilde .- mean(x_tilde, dims=2)) ./ sqrt(eki.N_ens - 1)
    A_tilde = _augmented_anomalies(Sigma_y_sqrt, prior, Y_tilde, Z_tilde, scale)

    L = _svd_pinv_apply(A, A_tilde, "dropout complement projection")
    return Z_tilde - Zb * L, A_tilde - A * L
end

function _dropout_optimization_mean(
    eki::EKIObj{FT},
    forward::Function,
    m_hat,
    Zb,
    A,
    Sigma_y_sqrt,
    prior::LowRankPrior,
    scale,
) where FT<:AbstractFloat
    Z_perp, A_perp = _projected_dropout_components(
        eki, forward, m_hat, Zb, A, Sigma_y_sqrt, prior, scale,
    )
    x_m_hat = forward(m_hat)
    residual = _augmented_residual(
        Sigma_y_sqrt,
        prior,
        eki.y .- vec(x_m_hat),
        prior.mean .- vec(m_hat),
        scale,
    )
    factorization = _ensemble_space_svd(A_perp, "sequential dropout anomalies")
    coefficient = _solve_identity_plus_gram_adjoint(
        factorization, residual, "sequential dropout mean solve",
    )
    return m_hat .+ reshape(Z_perp * coefficient, :, 1)
end

function _joint_projected_dropout_mean(
    eki::EKIObj{FT},
    forward::Function,
    mn,
    x_mn,
    Zb,
    A,
    Sigma_y_sqrt,
    prior::LowRankPrior,
    scale,
) where FT<:AbstractFloat
    Z_perp, A_perp = _projected_dropout_components(
        eki, forward, mn, Zb, A, Sigma_y_sqrt, prior, scale,
    )
    weight = sqrt(eki.joint_dropout_weight)
    Z_augmented = hcat(Zb, weight .* Z_perp)
    A_augmented = hcat(A, weight .* A_perp)
    residual = _augmented_residual(
        Sigma_y_sqrt,
        prior,
        eki.y .- vec(x_mn),
        prior.mean .- vec(mn),
        scale,
    )
    factorization = _ensemble_space_svd(A_augmented, "joint dropout anomalies")
    coefficient = _solve_identity_plus_gram_adjoint(
        factorization, residual, "joint dropout mean solve",
    )
    return mn .+ reshape(Z_augmented * coefficient, :, 1)
end

# Low-rank augmented dropout-EAKI update. The formal augmented map is
# [G(theta); theta], but U^+ coordinates keep every solve in ensemble space.
# Deliberately no projection onto range(U) is applied after grid dropout.
function update_ensemble_low_rank_prior!(
    eki::EKIObj{FT},
    forward::Function,
    prior::LowRankPrior{FT},
) where FT<:AbstractFloat
    eki.filter_type == "dropout-EAKI" ||
        error("low-rank prior augmentation currently supports dropout-EAKI only")

    theta_prev = eki.θ[end]
    mn = mean(theta_prev, dims=2)
    if eki.inflation
        0 < eki.Δτ < 1 || error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
        theta_b = mn .+ sqrt(1 / (1 - eki.Δτ)) .* (theta_prev .- mn)
        scale = sqrt(eki.Δτ)
    else
        theta_b = theta_prev
        scale = one(FT)
    end

    xb = forward(theta_b)
    xbar = mean(xb, dims=2)
    x_mean_bar = eki.y_pred[end]
    Zb = (theta_b .- mn) ./ sqrt(eki.N_ens - 1)
    Yb = (xb .- xbar) ./ sqrt(eki.N_ens - 1)

    A = _augmented_anomalies(eki.Σ_y_sqrt, prior, Yb, Zb, scale)
    factorization = _ensemble_space_svd(A, "base augmented anomalies")
    residual = _augmented_residual(
        eki.Σ_y_sqrt,
        prior,
        eki.y .- vec(x_mean_bar),
        prior.mean .- vec(mn),
        scale,
    )
    base_coefficient = _solve_identity_plus_gram_adjoint(
        factorization, residual, "base augmented mean solve",
    )
    m_hat = mn .+ reshape(Zb * base_coefficient, :, 1)

    P, Db_sqrt, V = svd(Zb; full=false)
    r = active_svd_rank(Db_sqrt)
    if r == 0
        theta_new = m_hat .+ zeros(FT, eki.N_θ, size(theta_b, 2))
    else
        P_r = P[:, 1:r]
        V_r = V[:, 1:r]
        transformed_V = _solve_identity_plus_gram(
            factorization, V_r, "EAKI anomaly solve",
        )
        eig = eigen(Symmetric(V_r' * transformed_V))
        U_eig, D = eig.vectors, max.(eig.values, zero(FT))
        A1 = P_r * Diagonal(Db_sqrt[1:r]) * U_eig
        A2 = Diagonal(sqrt.(D)) *
             Diagonal(safe_diag_inv(Db_sqrt[1:r])) * P_r'
        Z_hat = A1 * (A2 * Zb)

        m_new = if eki.dropout_correction_mode == "joint"
            _joint_projected_dropout_mean(
                eki, forward, mn, x_mean_bar, Zb, A,
                eki.Σ_y_sqrt, prior, scale,
            )
        else
            _dropout_optimization_mean(
                eki, forward, m_hat, Zb, A,
                eki.Σ_y_sqrt, prior, scale,
            )
        end
        theta_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
    end

    _require_finite(theta_new, "updated ensemble")
    x_mean_new = forward(mean(theta_new, dims=2))
    _require_finite(x_mean_new, "updated mean prediction")

    size(theta_new) == size(theta_prev) ||
        error("theta_new has size $(size(theta_new)), expected $(size(theta_prev))")
    push!(eki.θ, theta_new)
    push!(eki.y_pred, x_mean_new)
    return eki
end

function EKI_Run_Low_Rank_Prior(
    forward::Function,
    theta0::Array{FT,2},
    Sigma_y::Array{FT,2},
    y::Array{FT,1},
    prior::LowRankPrior{FT};
    filter_type::String="dropout-EAKI",
    Δτ::FT=FT(0.5),
    N_iter::Int=50,
    forward_parallel::Bool=false,
    dropout_rate::FT=FT(0.5),
    inflation::Bool=true,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::FT=one(FT),
) where FT<:AbstractFloat
    size(theta0, 1) == length(prior.mean) ||
        throw(DimensionMismatch("theta0 and prior disagree"))
    size(theta0, 2) > 1 || error("Need at least 2 ensemble members")
    filter_type == "dropout-EAKI" ||
        error("low-rank prior augmentation currently supports dropout-EAKI only")
    if inflation && !(0 < Δτ < 1)
        error("Δτ must satisfy 0 < Δτ < 1 when inflation is enabled")
    end

    N_y = length(y)
    func(theta) = forward_parallel ? forward(theta) : ensemble_forward(forward, theta, N_y)
    y_pred_0 = func(mean(theta0, dims=2))
    obj = EKIObj(
        filter_type, theta0, y_pred_0, y, Sigma_y, Δτ,
        dropout_rate, inflation, Δτ, Δτ, FT(Inf),
        dropout_correction_mode, joint_dropout_weight,
    )

    @info ("Running low-rank prior-augmented ", filter_type,
           " with ensemble size ", size(theta0, 2))
    for n in 1:N_iter
        if n % max(1, div(N_iter, 10)) == 0
            @info ("Iteration ", n, "/", N_iter)
        end
        update_ensemble_low_rank_prior!(obj, func, prior)
    end
    return obj
end

function low_rank_opt_errors(eki::EKIObj, prior::LowRankPrior)
    errors = opt_errors(eki)
    for i in eachindex(errors)
        mean_theta = vec(mean(eki.θ[i], dims=2))
        errors[i] += prior_penalty(prior, mean_theta)
    end
    return errors
end

end
