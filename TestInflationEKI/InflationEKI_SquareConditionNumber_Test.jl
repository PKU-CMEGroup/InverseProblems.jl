using LinearAlgebra, Random, Statistics, Serialization

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")
using PyPlot

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "..", "Inversion", "CMAES.jl"))

# ---------------------------------------------------------------- helpers

function matrix_rank(X::AbstractMatrix)
    S = svdvals(X)
    isempty(S) && return 0
    maximum(S) == 0 && return 0
    tol = max(eps(eltype(S)) * length(S) * maximum(S), 1.0e-12 * maximum(S))
    return count(S .> tol)
end

function centered_full_rank_coefficients(N_sub::Int, N_ens::Int)
    N_sub <= N_ens - 1 ||
        error("Need N_sub <= N_ens - 1 because centered anomalies have rank at most N_ens - 1")
    coeff = randn(N_sub, N_ens)
    coeff .-= mean(coeff, dims=2)
    while matrix_rank(coeff) < N_sub
        coeff .= randn(N_sub, N_ens)
        coeff .-= mean(coeff, dims=2)
    end
    return coeff
end

function compute_initial_offset_direction(Q_sub::AbstractMatrix, Q_missing::AbstractMatrix,
                                          N::Int, mode::String)
    if mode == "qmissing" && size(Q_missing, 2) == 0
        error("initial_offset_direction=\"qmissing\" requires N_missing >= 1")
    end
    if mode == "qsub"
        c = randn(size(Q_sub, 2))
        return Q_sub * (c / norm(c))
    elseif mode == "qmissing"
        c = randn(size(Q_missing, 2))
        return Q_missing * (c / norm(c))
    elseif mode == "qsub_qmissing"
        d = Q_sub * randn(size(Q_sub, 2)) + Q_missing * randn(size(Q_missing, 2))
        return d / norm(d)
    elseif mode == "full_random"
        d = randn(N)
        return d / norm(d)
    else
        error("Unknown initial_offset_direction: $(mode)")
    end
end

# ------------------------------------------------- square full-rank problem

# G = U diag(σ) V', with N_y = N_θ = N, so there is no nullspace.
function square_conditioned_operator(N::Int; cond::Float64=1.0)
    U, _ = qr(randn(N, N))
    V, _ = qr(randn(N, N))
    U = Matrix(U)
    V = Matrix(V)
    σ = cond <= 1.0 ? ones(N) : [cond^(-(i - 1) / (N - 1)) for i in 1:N]
    G = U * Diagonal(σ) * V'
    return G, V, σ
end

function setup_square_condition_problem(N::Int, N_ens::Int;
        cond::Float64=1.0,
        N_sub::Int=20,
        N_missing::Int=0,
        truth_signal::Float64=30.0,
        truth_missing_signal::Float64=30.0,
        noise_std::Float64=1.0,
        initial_offset_scale::Float64=0.0,
        initial_offset_direction::String="qsub_qmissing")
    N >= 2 || error("N must be at least 2")
    1 <= N_sub <= N || error("N_sub must satisfy 1 <= N_sub <= N")
    N_sub <= N_ens - 1 ||
        error("Need N_sub <= N_ens - 1 because centered anomalies have rank at most N_ens - 1")
    N_missing >= 0 || error("N_missing must be at least 0")
    N_sub + N_missing <= N || error("Need N_sub + N_missing <= N")

    G, V, σ = square_conditioned_operator(N; cond=cond)
    Q_sub = V[:, 1:N_sub]
    σ_sub = σ[1:N_sub]

    if N_missing > 0
        Q_missing = V[:, N_sub+1:N_sub+N_missing]
        σ_missing = σ[N_sub+1:N_sub+N_missing]
    else
        Q_missing = zeros(N, 0)
        σ_missing = Float64[]
    end

    θ0 = Q_sub * centered_full_rank_coefficients(N_sub, N_ens)
    if initial_offset_scale != 0.0
        d = compute_initial_offset_direction(Q_sub, Q_missing, N, initial_offset_direction)
        θ0 .+= initial_offset_scale .* reshape(d, :, 1)
    end

    θ_ref = Q_sub * (truth_signal ./ σ_sub)
    if N_missing > 0
        θ_ref += Q_missing * (truth_missing_signal ./ σ_missing)
    end

    y = G * θ_ref
    Σ_y = Array(Diagonal(fill(noise_std^2, N)))

    return (N=N, G=G, V=V, σ=σ, Q_sub=Q_sub, Q_missing=Q_missing,
            θ0=θ0, θ_ref=θ_ref, y=y, Σ_y=Σ_y)
end

# ------------------------------------------------- method configurations

struct MethodConf
    label::String
    filter_type::String
    inflation::Bool
end

function default_method_configs()
    return MethodConf[
        MethodConf("DEKI",                  "DEKI",        false),
        MethodConf("Inflation-EAKI",          "EAKI",        true),
        MethodConf("Inflation-dropout-EAKI",  "dropout-EAKI", true),
        MethodConf("CMA-ES",                  "CMAES",       false),
    ]
end

# ------------------------------------------------- runners

function misfit_curve(θ_hist, G::AbstractMatrix, y::AbstractVector)
    ynorm = max(norm(y), eps(Float64))
    return [norm(G * dropdims(mean(θ, dims=2), dims=2) - y) / ynorm for θ in θ_hist]
end

function fixed_length_trajectory(history::Vector, n::Int, final_point)
    if length(history) >= n
        return vcat(history[1:(n-1)], [copy(final_point)])
    else
        out = copy(history)
        while length(out) < n
            push!(out, copy(final_point))
        end
        return out
    end
end

function run_eki_method(problem, mc::MethodConf, seed::Int,
                        Δτ::Float64, N_iter::Int, dropout_rate::Float64)
    Random.seed!(seed)
    obj = EKI_Run(θ -> problem.G * θ, copy(problem.θ0), problem.Σ_y, problem.y;
                  filter_type=mc.filter_type,
                  Δτ=Δτ,
                  N_iter=N_iter,
                  dropout_rate=dropout_rate,
                  inflation=mc.inflation)
    return misfit_curve(obj.θ, problem.G, problem.y)
end

function run_cmaes_method(problem, seed::Int, N_iter::Int,
                          cma_sigma0::Float64, N_ens::Int, noise_std::Float64)
    Random.seed!(seed)
    residual(θ) = (problem.G * θ - problem.y) ./ noise_std
    objective(θ) = sum(abs2, residual(θ))
    x0 = vec(mean(problem.θ0, dims=2))
    out = run_cmaes(objective, x0; sigma0=cma_sigma0, max_iter=N_iter,
                    popsize=N_ens, seed=seed)
    traj = fixed_length_trajectory(collect(out.best_history), N_iter + 1, out.best_x)
    return [norm(problem.G * p - problem.y) / max(norm(problem.y), eps(Float64)) for p in traj]
end

function median_curve(curves::Vector{Vector{Float64}}, n::Int)
    med = fill(NaN, n)
    for i in 1:n
        vals = Float64[c[i] for c in curves if i <= length(c) && isfinite(c[i])]
        med[i] = isempty(vals) ? NaN : median(vals)
    end
    return med
end

function experiment_tag(; N, N_ens, N_sub, missing_dims, initial_offset_scale,
                        initial_offset_direction, seed, n_seeds, N_iter)
    dir = replace(initial_offset_direction, r"[^a-zA-Z0-9]+" => "_")
    nm = join(string.(missing_dims), "_")
    return "N$(N)_Ne$(N_ens)_sub$(N_sub)_nm$(nm)_off$(initial_offset_scale)_dir$(dir)_seed$(seed)_ns$(n_seeds)_iter$(N_iter)"
end


# ------------------------------------------------- driver

function run_square_condition_study(;
        N::Int=100,
        N_ens::Int=21,
        N_sub::Int=20,
        missing_dims::Vector{Int}=[0, 2, 10, 20],
        N_iter::Int=500,
        Δτ::Float64=0.2,
        dropout_rate::Float64=0.5,
        noise_std::Float64=1.0,
        truth_signal::Float64=30.0,
        truth_missing_signal::Float64=30.0,
        cond_numbers::Vector{Float64}=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1.0e3, 3.0e3, 1.0e4],
        method_configs::Vector{MethodConf}=default_method_configs(),
        seed::Int=2026,
        n_seeds::Int=3,
        cma_sigma0::Float64=1.0,
        initial_offset_scale::Float64=0.0,
        initial_offset_direction::String="qsub_qmissing",
        save_prefix::String=joinpath(@__DIR__, "Figs", "SquareConditionNumber", "DEKI_EAKI_CMA_SquareConditionNumber"))

    tag = experiment_tag(; N=N, N_ens=N_ens, N_sub=N_sub, missing_dims=missing_dims,
                          initial_offset_scale=initial_offset_scale,
                          initial_offset_direction=initial_offset_direction,
                          seed=seed, n_seeds=n_seeds, N_iter=N_iter)
    save_prefix = save_prefix * "_" * tag

    Random.seed!(seed)
    results = Dict{String, Dict{Int, Dict{Int, Dict{Int, Vector{Float64}}}}}()
    for mc in method_configs
        results[mc.label] = Dict(
            ri => Dict(ki => Dict{Int, Vector{Float64}}()
                       for ki in eachindex(cond_numbers))
            for ri in eachindex(missing_dims)
        )
    end

    for (ri, nmiss) in enumerate(missing_dims)
        for (ki, cond) in enumerate(cond_numbers)
            for si in 1:n_seeds
                Random.seed!(seed + 10000 * ki + si)
                prob = setup_square_condition_problem(
                    N, N_ens;
                    cond=cond,
                    N_sub=N_sub,
                    N_missing=nmiss,
                    truth_signal=truth_signal,
                    truth_missing_signal=truth_missing_signal,
                    noise_std=noise_std,
                    initial_offset_scale=initial_offset_scale,
                    initial_offset_direction=initial_offset_direction,
                )

                for (mi, mc) in enumerate(method_configs)
                    Random.seed!(seed + 10000 * ki + si + 100000 * mi)
                    @info "Running" N=N cond method=mc.label seed=si N_missing=nmiss iter=N_iter
                    try
                        curve = mc.filter_type == "CMAES" ?
                            run_cmaes_method(prob, seed, N_iter, cma_sigma0, N_ens, noise_std) :
                            run_eki_method(prob, mc, seed, Δτ, N_iter, dropout_rate)
                        results[mc.label][ri][ki][si] = curve
                    catch err
                        @warn "Run diverged (numerical error)" cond method=mc.label seed=si err=sprint(showerror, err)[1:min(120, end)]
                        results[mc.label][ri][ki][si] = fill(NaN, N_iter + 1)
                    end
                end
            end
        end
    end


    # ------------------------------------------------------------- plotting
    ites = 0:N_iter
    log_conds = log10.(cond_numbers)
    cmap = PyPlot.matplotlib.cm.get_cmap("viridis")
    cond_colors = [cmap((lc - minimum(log_conds)) /
                        (maximum(log_conds) - minimum(log_conds) + 1e-12))
                   for lc in log_conds]

    fig, ax = PyPlot.subplots(nrows=length(missing_dims), ncols=length(method_configs),
                              figsize=(4.2 * length(method_configs), 3.6 * length(missing_dims)))
    row_vals = [Float64[] for _ in missing_dims]

    for (ri, nmiss) in enumerate(missing_dims)
        for (mi, mc) in enumerate(method_configs)
            panel = length(missing_dims) == 1 ? (length(method_configs) == 1 ? ax : ax[mi]) : ax[ri, mi]
            panel.yaxis.set_minor_formatter(PyPlot.matplotlib.ticker.NullFormatter())
            panel.yaxis.set_minor_locator(PyPlot.matplotlib.ticker.NullLocator())
            for (ki, cond) in enumerate(cond_numbers)
                curves = [results[mc.label][ri][ki][si] for si in 1:n_seeds]
                med = median_curve(curves, N_iter + 1)
                append!(row_vals[ri], med[isfinite.(med)])
                panel.semilogy(ites, med, color=cond_colors[ki], linewidth=1.8, label="κ=$(cond)")
            end
            panel.set_xlabel("Iterations")
            if mi == 1
                panel.set_ylabel("N_missing = $nmiss\nRelative data misfit")
            else
                panel.set_ylabel("Relative data misfit")
            end
            panel.grid(true, which="both", alpha=0.3)
            if ri == 1
                panel.set_title(mc.label, fontsize=11)
            end
            if ri == length(missing_dims)
                panel.legend(loc="best", fontsize=7, ncol=2)
            end
        end
    end

    # Unified y limits across methods in the same row.
    for (ri, nmiss) in enumerate(missing_dims)
        vals = row_vals[ri]
        isempty(vals) && continue
        ymin = minimum(vals)
        ymax = maximum(vals)
        if !isfinite(ymin) || ymin <= 0
            ymin = ymax * 1e-6
        end
        if !isfinite(ymax) || ymax <= 0
            continue
        end
        ylo = ymin / 3.0
        yhi = ymax * 3.0
        for (mi, mc) in enumerate(method_configs)
            panel = length(missing_dims) == 1 ? (length(method_configs) == 1 ? ax : ax[mi]) : ax[ri, mi]
            panel.set_ylim(ylo, yhi)
        end
    end

    fig.suptitle("Square full-rank G: convergence of relative data misfit", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figdir = dirname(save_prefix)
    isempty(figdir) || mkpath(figdir)
    fig.savefig(save_prefix * "_convergence.png", dpi=160)
    PyPlot.close(fig)

    serialize(save_prefix * "_results.jls", (; N, N_ens, N_sub, missing_dims, N_iter,
                                             Δτ, dropout_rate, noise_std, truth_signal,
                                             truth_missing_signal, cond_numbers,
                                             method_configs, seed, n_seeds, cma_sigma0,
                                             initial_offset_scale, initial_offset_direction,
                                             results))
    @info "Saved square full-rank study" save_prefix
    return results
end

# ------------------------------------------------- env parsing + main

function env_parse(name::String, default::Int)
    return parse(Int, get(ENV, name, string(default)))
end

function env_parse(name::String, default::Float64)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_cond_numbers(default::Vector{Float64})
    s = get(ENV, "COND_NUMBERS", join(string.(default), ","))
    return parse.(Float64, strip.(split(s, ",")))
end

function env_int_list(name::String, default::Vector{Int})
    s = get(ENV, name, join(string.(default), ","))
    return parse.(Int, strip.(split(s, ",")))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_square_condition_study(
        N=env_parse("N", 100),
        N_ens=env_parse("N_ENS", 21),
        N_sub=env_parse("N_SUB", 20),
        missing_dims=env_int_list("N_MISSING_VALUES", [0, 2, 10, 20]),
        N_iter=env_parse("N_ITER", 500),
        Δτ=env_parse("DELTA_T", 0.2),
        dropout_rate=env_parse("DROPOUT_RATE", 0.5),
        noise_std=env_parse("NOISE_STD", 1.0),
        truth_signal=env_parse("TRUTH_SIGNAL", 30.0),
        truth_missing_signal=env_parse("TRUTH_MISSING_SIGNAL", 30.0),
        cond_numbers=env_cond_numbers([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1.0e3, 3.0e3, 1.0e4]),
        seed=env_parse("SEED", 2026),
        n_seeds=env_parse("N_SEEDS", 1),
        cma_sigma0=env_parse("CMA_SIGMA0", 1.0),
        initial_offset_scale=env_parse("INITIAL_OFFSET_SCALE", 0.0),
        initial_offset_direction=get(ENV, "INITIAL_OFFSET_DIRECTION", "qsub_qmissing"),
        save_prefix=get(ENV, "SAVE_PREFIX",
                        joinpath(@__DIR__, "Figs", "SquareConditionNumber", "DEKI_EAKI_CMA_SquareConditionNumber")),
    )
end

