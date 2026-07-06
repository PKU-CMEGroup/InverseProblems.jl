using LinearAlgebra
using Random
using Statistics
using Distributions
using PyPlot

include("../Inversion/InflationEKI.jl")

normalize_vec(v::AbstractVector) = v / norm(v)

function orthonormal_basis(X::AbstractMatrix)
    U, S, _ = svd(X; full=false)
    isempty(S) && return zeros(eltype(X), size(X, 1), 0)
    maximum(S) == 0 && return zeros(eltype(X), size(X, 1), 0)
    tol = max(eps(eltype(S)) * length(S) * maximum(S), 1.0e-12 * maximum(S))
    r = count(S .> tol)
    return Matrix(U[:, 1:r])
end

function matrix_rank(X::AbstractMatrix)
    S = svdvals(X)
    isempty(S) && return 0
    maximum(S) == 0 && return 0
    tol = max(eps(eltype(S)) * length(S) * maximum(S), 1.0e-12 * maximum(S))
    return count(S .> tol)
end

function centered_full_rank_coefficients(N_sub::Int, N_ens::Int)
    N_sub <= N_ens - 1 || error("Need N_sub <= N_ens - 1 because centered anomalies have rank at most N_ens - 1")

    coeff = randn(N_sub, N_ens)
    coeff .-= mean(coeff, dims=2)
    while matrix_rank(coeff) < N_sub
        coeff .= randn(N_sub, N_ens)
        coeff .-= mean(coeff, dims=2)
    end
    return coeff
end

function setup_seteki(N_y::Int, N_θ::Int, N_ens::Int;
                      N_sub::Union{Nothing,Int}=nothing,
                      missing_direction::String="observable",
                      default_subspace_fraction::Float64=0.5)
    N_θ > N_y || error("This subspace test needs N_θ > N_y so that null(G) contains a missing direction")
    G = rand(N_y, N_θ)

    row_basis = orthonormal_basis(G')
    rankG = size(row_basis, 2)
    rankG > 0 || error("Generated a zero-rank forward operator")

    max_sub_dim = missing_direction == "observable" ? rankG - 1 : rankG
    0 < default_subspace_fraction <= 1 ||
        error("default_subspace_fraction must satisfy 0 < default_subspace_fraction <= 1")
    default_sub_dim = max(1, floor(Int, default_subspace_fraction * max_sub_dim))
    N_sub_actual = isnothing(N_sub) ? min(default_sub_dim, N_ens - 1) : N_sub
    1 <= N_sub_actual <= min(rankG, N_ens - 1) ||
        error("N_sub must satisfy 1 <= N_sub <= min(rank(G), N_ens - 1)")

    Q_sub = row_basis[:, 1:N_sub_actual]
    θ0_mean = Q_sub * randn(N_sub_actual)
    θ0 = θ0_mean .+ Q_sub * centered_full_rank_coefficients(N_sub_actual, N_ens)

    if missing_direction == "observable"
        N_sub_actual < rankG ||
            error("Observable missing direction needs N_sub < rank(G); reduce N_sub")
        q_missing = row_basis[:, N_sub_actual + 1]
    elseif missing_direction == "nullspace"
        null_basis = nullspace(G)
        size(null_basis, 2) > 0 || error("Generated G has no numerical nullspace")
        q_missing = normalize_vec(null_basis[:, 1])
    else
        error("missing_direction must be \"observable\" or \"nullspace\"")
    end

    return G, θ0, Q_sub, q_missing
end

function projection_errors(θ_hist, θ_ref, Q_sub)
    subspace_errors = zeros(Float64, length(θ_hist))
    complement_errors = zeros(Float64, length(θ_hist))
    relative_errors = zeros(Float64, length(θ_hist))

    for i in eachindex(θ_hist)
        θ_mean = dropdims(mean(θ_hist[i], dims=2), dims=2)
        err = θ_mean - θ_ref
        err_sub = Q_sub * (Q_sub' * err)
        err_perp = err - err_sub

        subspace_errors[i] = norm(err_sub)
        complement_errors[i] = norm(err_perp)
        relative_errors[i] = norm(err) / norm(θ_ref)
    end

    return subspace_errors, complement_errors, relative_errors
end

function weighted_min_norm_solution(G::AbstractMatrix, Σ_y::AbstractMatrix, y::AbstractVector)
    Σ_inv_G = Σ_y \ G
    return pinv(G' * Σ_inv_G) * (Σ_inv_G' * y)
end

function affine_subspace_ls_solution(G::AbstractMatrix, Σ_y::AbstractMatrix,
                                     y::AbstractVector, θ0::AbstractMatrix, Q_sub::AbstractMatrix)
    θ0_mean = dropdims(mean(θ0, dims=2), dims=2)
    A = G * Q_sub
    rhs = y - G * θ0_mean
    Σ_inv_A = Σ_y \ A
    coeff = pinv(A' * Σ_inv_A) * (Σ_inv_A' * rhs)
    return θ0_mean + Q_sub * coeff
end

function run_deterministic_eki(G::AbstractMatrix, θ0::AbstractMatrix, Σ_y::AbstractMatrix,
                               y::AbstractVector, N_iter::Int)
    θ_hist = [copy(θ0)]
    y_obs = reshape(y, :, 1)

    for _ in 1:N_iter
        θ = θ_hist[end]
        θ_mean = mean(θ, dims=2)
        Z = (θ .- θ_mean) ./ sqrt(size(θ, 2) - 1)
        Γ = Z * Z'
        S = G * Γ * G' + Σ_y
        K = Γ * G' / S
        push!(θ_hist, θ + K * (y_obs .- G * θ))
    end

    return θ_hist
end

function plot_style(i::Int)
    markers = ["o", "s", "^", "d", "v", "x"]
    linestyles = ["-", "--", "-.", ":"]
    return (
        marker=markers[mod1(i, length(markers))],
        linestyle=linestyles[mod1(i, length(linestyles))],
    )
end

function run_inflation_eki_subspace_test(;
    N_θ::Int=1000,
    N_y::Int=500,
    N_ens::Int=1000,
    N_sub::Union{Nothing,Int}=nothing,
    N_iter::Int=30,
    Δτ::Float64=0.2,
    dropout_rate::Float64=0.3,
    noise_std::Float64=1.0e-3,
    truth_missing_scale::Float64=5.0,
    missing_direction::String="observable",
    default_subspace_fraction::Float64=0.5,
    seed::Int=2026,
    filter_types=["ETKI", "D-ETKI", "DF-ETKI"],
    save_prefix::String=joinpath(@__DIR__, "InflationEKI_Subspace_Test"),
)
    Random.seed!(seed)

    G, θ0, Q_sub, q_missing = setup_seteki(N_y, N_θ, N_ens;
                                           N_sub=N_sub,
                                           missing_direction=missing_direction,
                                           default_subspace_fraction=default_subspace_fraction)
    θ_ref = Q_sub * randn(size(Q_sub, 2)) + truth_missing_scale * q_missing
    y = G * θ_ref + noise_std * randn(N_y)
    Σ_y = Array(Diagonal(fill(noise_std^2, N_y)))
    θ_star = weighted_min_norm_solution(G, Σ_y, y)
    θ_affine_star = affine_subspace_ls_solution(G, Σ_y, y, θ0, Q_sub)

    forward(θ) = G * θ

    objs = Dict{String, EKIObj}()
    subspace_errors = Dict{String, Vector{Float64}}()
    complement_errors = Dict{String, Vector{Float64}}()
    relative_errors = Dict{String, Vector{Float64}}()
    data_misfits = Dict{String, Vector{Float64}}()

    for filter_type in filter_types
        @info "Running setEKI subspace test" filter_type N_θ N_y N_ens N_sub=size(Q_sub, 2) missing_direction N_iter Δτ dropout_rate
        if filter_type == "DET-EKI"
            θ_hist = run_deterministic_eki(G, copy(θ0), Σ_y, y, N_iter)
        else
            obj = EKI_Run(
                forward,
                copy(θ0),
                Σ_y,
                y;
                filter_type=filter_type,
                Δτ=Δτ,
                N_iter=N_iter,
                dropout_rate=dropout_rate,
            )
            θ_hist = obj.θ
            objs[filter_type] = obj
        end

        sub_e, comp_e, rel_e = projection_errors(θ_hist, θ_ref, Q_sub)
        subspace_errors[filter_type] = sub_e
        complement_errors[filter_type] = comp_e
        relative_errors[filter_type] = rel_e
        data_misfits[filter_type] = [
            norm(G * dropdims(mean(θ, dims=2), dims=2) - y) / norm(y)
            for θ in θ_hist
        ]
    end

    fig, ax = PyPlot.subplots(ncols=3, figsize=(15, 4))
    for (i, filter_type) in enumerate(filter_types)
        ites = 0:N_iter
        style = plot_style(i)
        ax[1].semilogy(ites, subspace_errors[filter_type], marker=style.marker, linestyle=style.linestyle, fillstyle="none", markevery=5, label=filter_type)
        ax[2].semilogy(ites, complement_errors[filter_type], marker=style.marker, linestyle=style.linestyle, fillstyle="none", markevery=5, label=filter_type)
        ax[3].semilogy(ites, relative_errors[filter_type], marker=style.marker, linestyle=style.linestyle, fillstyle="none", markevery=5, label=filter_type)
    end

    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Initial subspace error to truth")
    ax[1].grid()

    ax[2].set_xlabel("Iterations")
    ax[2].set_ylabel("Orthogonal complement error to truth")
    ax[2].grid()

    ax[3].set_xlabel("Iterations")
    ax[3].set_ylabel("Relative truth error")
    ax[3].grid()
    ax[3].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_prefix * "_errors.png", dpi=180)
    PyPlot.close(fig)

    fig, ax = PyPlot.subplots(ncols=1, figsize=(6, 4))
    for (i, filter_type) in enumerate(filter_types)
        style = plot_style(i)
        comp0 = complement_errors[filter_type][1]
        comp_ratio = complement_errors[filter_type] ./ comp0
        ax.semilogy(0:N_iter, comp_ratio, marker=style.marker, linestyle=style.linestyle, fillstyle="none", markevery=5, label=filter_type)
    end
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Complement error / initial")
    ax.grid()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_prefix * "_complement_ratio.png", dpi=180)
    PyPlot.close(fig)

    fig, ax = PyPlot.subplots(ncols=1, figsize=(6, 4))
    for (i, filter_type) in enumerate(filter_types)
        style = plot_style(i)
        ax.semilogy(0:N_iter, data_misfits[filter_type], marker=style.marker, linestyle=style.linestyle, fillstyle="none", markevery=5, label=filter_type)
    end
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Relative data misfit")
    ax.grid()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_prefix * "_misfit.png", dpi=180)
    PyPlot.close(fig)

    return (
        G=G,
        θ_ref=θ_ref,
        θ_star=θ_star,
        θ_affine_star=θ_affine_star,
        θ0=θ0,
        Q_sub=Q_sub,
        q_missing=q_missing,
        missing_direction=missing_direction,
        default_subspace_fraction=default_subspace_fraction,
        rank_G=matrix_rank(G),
        rank_GQ_sub=matrix_rank(G * Q_sub),
        objs=objs,
        subspace_errors=subspace_errors,
        complement_errors=complement_errors,
        relative_errors=relative_errors,
        data_misfits=data_misfits,
    )
end

function env_parse(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_parse(name::String, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_parse_optional_int(name::String)
    value = get(ENV, name, "")
    return isempty(strip(value)) ? nothing : parse(Int, value)
end

function env_filter_types(default_filter_types)
    filter_type_string = get(ENV, "FILTER_TYPES", join(default_filter_types, ","))
    return String.(strip.(split(filter_type_string, ",")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    job_id = get(ENV, "SLURM_JOB_ID", "local")
    default_filter_types = ["DET-EKI", "ETKI", "D-ETKI", "DF-ETKI"]

    run_inflation_eki_subspace_test(
        N_θ=env_parse("N_THETA", 1000),
        N_y=env_parse("N_Y", 500),
        N_ens=env_parse("N_ENS", 1000),
        N_sub=env_parse_optional_int("N_SUB"),
        N_iter=env_parse("N_ITER", 30),
        Δτ=env_parse("DELTA_T", 0.2),
        dropout_rate=env_parse("DROPOUT_RATE", 0.3),
        noise_std=env_parse("NOISE_STD", 1.0e-3),
        truth_missing_scale=env_parse("TRUTH_MISSING_SCALE", 5.0),
        missing_direction=get(ENV, "MISSING_DIRECTION", "observable"),
        default_subspace_fraction=env_parse("DEFAULT_SUBSPACE_FRACTION", 0.5),
        seed=env_parse("SEED", 2026),
        filter_types=env_filter_types(default_filter_types),
        save_prefix=get(ENV, "SAVE_PREFIX", joinpath(@__DIR__, "InflationEKI_Subspace_Test_" * job_id)),
    )
end
job_id = get(ENV, "SLURM_JOB_ID", "local")
    default_filter_types = ["DET-EKI", "ETKI", "D-ETKI", "DF-ETKI"]

    run_inflation_eki_subspace_test(
        N_θ=env_parse("N_THETA", 1000),
        N_y=env_parse("N_Y", 500),
        N_ens=env_parse("N_ENS", 1000),
        N_sub=env_parse_optional_int("N_SUB"),
        N_iter=env_parse("N_ITER", 50),
        Δτ=env_parse("DELTA_T", 0.2),
        dropout_rate=env_parse("DROPOUT_RATE", 0.5),
        noise_std=env_parse("NOISE_STD", 1.0e-3),
        truth_missing_scale=env_parse("TRUTH_MISSING_SCALE", 5.0),
        missing_direction=get(ENV, "MISSING_DIRECTION", "observable"),
        default_subspace_fraction=env_parse("DEFAULT_SUBSPACE_FRACTION", 0.5),
        seed=env_parse("SEED", 2026),
        filter_types=env_filter_types(default_filter_types),
        save_prefix=get(ENV, "SAVE_PREFIX", joinpath(@__DIR__, "InflationEKI_Subspace_Test_" * job_id)),
    )