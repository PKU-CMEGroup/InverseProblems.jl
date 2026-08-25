using LinearAlgebra
using Random
using Statistics
using Distributions
using Serialization
using PyPlot

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "..", "Inversion", "CMAES.jl"))
include(joinpath(@__DIR__, "..", "Inversion", "DFOLS.jl"))

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

function plot_style(i::Int)
    markers = ["o", "s", "^", "d", "v", "x"]
    linestyles = ["-", "--", "-.", ":"]
    return (
        color="C$(mod(i - 1, 10))",
        marker=markers[mod1(i, length(markers))],
        linestyle=linestyles[mod1(i, length(linestyles))],
    )
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

# ------------------------------------------------- conditioned problem

# G = U diag(σ) V' with σ log-spaced in [1, 1/cond], so cond(G) = cond.
# Returns G (N_y × N_θ), V (N_θ × N_y, columns = right singular vectors = row-space frame), σ.
function conditioned_forward_operator(N_y::Int, N_θ::Int; cond::Float64=1.0)
    U, _ = qr(randn(N_y, N_y))
    Vfull, _ = qr(randn(N_θ, N_θ))
    V = Matrix(Vfull)[:, 1:N_y]
    σ = cond <= 1.0 ? ones(N_y) : [cond^(-(i - 1) / (N_y - 1)) for i in 1:N_y]
    G = Matrix(U) * Diagonal(σ) * V'
    return G, V, σ
end

# Conditioned linear inverse problem with balanced spectral excitation and a
# controlled component in the complement of the initial ensemble subspace.
#
# The initial ensemble lives in Q_sub = V[:, 1:N_sub] and has mean zero.  The truth
# is
#   θ_ref = Q_sub * (truth_signal ./ σ[1:N_sub])
#         + q_missing * (truth_missing_signal / σ[N_sub+1]),
#   q_missing = V[:, N_sub+1].
# Thus every retained observation direction and the first missing observation
# direction each carry a prescribed signal amplitude.  The missing direction is
# observable (it lies in the row space of G) but is NOT in the initial ensemble
# span, which is exactly the component that dropout-EAKI's projected orthogonal
# correction should recover, while plain EAKI cannot.
function setup_conditioned_linear_problem(N_y::Int, N_θ::Int, N_ens::Int;
                                           cond::Float64=1.0,
                                           N_sub::Union{Nothing,Int}=nothing,
                                           default_subspace_fraction::Float64=0.5,
                                           truth_signal::Float64=30.0,
                                           truth_missing_signal::Float64=30.0,
                                           noise_std::Float64=1.0,
                                           full_space_init::Bool=false)
    N_θ > N_y || error("This study uses a rectangular forward map with N_θ > N_y")
    G, V, σ = conditioned_forward_operator(N_y, N_θ; cond=cond)

    N_sub_actual = isnothing(N_sub) ? max(1, floor(Int, default_subspace_fraction * N_y)) : N_sub
    1 <= N_sub_actual < N_y || error("N_sub must satisfy 1 <= N_sub < N_y")
    N_sub_actual <= N_ens - 1 ||
        error("Need N_sub <= N_ens - 1 because the ensemble has at most N_ens-1 independent anomalies")

    Q_sub = V[:, 1:N_sub_actual]
    σ_sub = σ[1:N_sub_actual]
    q_missing = V[:, N_sub_actual + 1]
    σ_missing = σ[N_sub_actual + 1]

    # initial ensemble: default is mean zero and supported only on Q_sub; with
    # full_space_init=true use a centered full-space Gaussian ensemble, which is
    # the regime where the DEKI paper's min_s P(s,s)>0 condition holds.
    if full_space_init
        θ0 = randn(N_θ, N_ens)
        θ0 .-= mean(θ0, dims=2)
    else
        θ0 = Q_sub * centered_full_rank_coefficients(N_sub_actual, N_ens)
    end

    # truth: balanced signal in every retained singular direction, plus one
    # controlled component in the complement of the initial ensemble subspace
    θ_ref = Q_sub * (truth_signal ./ σ_sub) +
            q_missing * (truth_missing_signal / σ_missing)
    y = G * θ_ref + noise_std * randn(N_y)
    Σ_y = Array(Diagonal(fill(noise_std^2, N_y)))

    θ_star = weighted_min_norm_solution(G, Σ_y, y)
    θ_affine_star = affine_subspace_ls_solution(G, Σ_y, y, θ0, Q_sub)

    κ_sub = σ_sub[1] / σ_sub[end]               # condition number of G restricted to Q_sub
    noise_floor_rel = sqrt(N_y) * noise_std / max(norm(y), eps(Float64))
    affine_floor_rel = norm(G * θ_affine_star - y) / max(norm(y), eps(Float64))

    return (G=G, V=V, σ=σ, Q_sub=Q_sub, q_missing=q_missing, θ0=θ0, θ_ref=θ_ref,
            y=y, Σ_y=Σ_y, θ_star=θ_star, θ_affine_star=θ_affine_star,
            κ_sub=κ_sub, σ_missing=σ_missing, noise_floor_rel=noise_floor_rel,
            affine_floor_rel=affine_floor_rel)
end

# ------------------------------------------------- prior augmentation

# Turn forward(θ) = G θ into an augmented full-column-rank forward model with a
# weak Gaussian prior θ ~ N(0, prior_std² I):
#     forward(θ) = [G θ; θ],   y_aug = [y; 0],   Σ_aug = blkdiag(Σ_y, prior_std² I).
# The default prior_std is large (1000) so it stabilizes the nullspace of G without
# dominating the retained spectrum used in the condition-number study.
function prior_augmented_system(G::AbstractMatrix, y::AbstractVector, Σ_y::AbstractMatrix,
                                N_θ::Int, prior_std::Float64)
    N_y = size(G, 1)
    G_aug = vcat(G, Matrix{Float64}(I, N_θ, N_θ))
    y_aug = vcat(y, zeros(N_θ))
    Σ_y_aug = zeros(N_y + N_θ, N_y + N_θ)
    Σ_y_aug[1:N_y, 1:N_y] = Σ_y
    Σ_y_aug[N_y+1:end, N_y+1:end] = Matrix(prior_std^2 * I, N_θ, N_θ)
    forward(θ) = G_aug * θ
    return forward, y_aug, Σ_y_aug
end

# ------------------------------------------------- metrics

function compute_metrics(θ_hist, G::AbstractMatrix, V::AbstractMatrix, Q_sub::AbstractMatrix,
                         q_missing::AbstractVector, θ_ref::AbstractVector, y::AbstractVector,
                         θ_affine_star::AbstractVector)
    N = length(θ_hist)
    misfit = zeros(N)
    rel_err = zeros(N)
    obs_rel_err = zeros(N)
    comp_err = zeros(N)
    rel_affine = zeros(N)
    missing_rel = zeros(N)
    P_row = V * V'          # projection onto the row space of G
    truth_missing = q_missing' * θ_ref
    for i in 1:N
        m = dropdims(mean(θ_hist[i], dims=2), dims=2)
        misfit[i] = norm(G * m - y) / max(norm(y), eps(Float64))
        e = m - θ_ref
        rel_err[i] = norm(e) / max(norm(θ_ref), eps(Float64))
        obs_rel_err[i] = norm(P_row * e) / max(norm(θ_ref), eps(Float64))
        comp_err[i] = norm(e - Q_sub * (Q_sub' * e))
        rel_affine[i] = norm(m - θ_affine_star) / max(norm(θ_affine_star), eps(Float64))
        missing_rel[i] = abs(q_missing' * e) / max(abs(truth_missing), eps(Float64))
    end
    return (misfit=misfit, rel_err=rel_err, obs_rel_err=obs_rel_err,
            comp_err=comp_err, rel_affine=rel_affine, missing_rel=missing_rel)
end

# ------------------------------------------------- small statistics / fits

function seed_summary(seed_values::AbstractVector)
    n_seeds = length(seed_values)
    n_iter = length(first(seed_values)) - 1
    med = Vector{Float64}(undef, n_iter + 1)
    lo = Vector{Float64}(undef, n_iter + 1)
    hi = Vector{Float64}(undef, n_iter + 1)
    for i in 1:(n_iter + 1)
        f = Float64[s[i] for s in seed_values if isfinite(s[i])]
        if isempty(f)
            med[i] = NaN; lo[i] = NaN; hi[i] = NaN
        elseif length(f) == 1
            med[i] = f[1]; lo[i] = f[1]; hi[i] = f[1]
        else
            med[i] = median(f)
            lo[i] = quantile(f, 0.25)
            hi[i] = quantile(f, 0.75)
        end
    end
    n_finite = count(isfinite, Float64[s[end] for s in seed_values])
    return med, lo, hi, n_finite
end

# Iterations needed to reduce the excess misfit (distance above the noise floor)
# by a factor 1-τ.  τ=0.5 means the misfit has covered half of the distance from
# its initial value to the noise floor.  Returns Inf when not reached.
function excess_iters(misfit::Vector{Float64}, floor_rel::Float64, τ::Float64)
    e0 = misfit[1] - floor_rel
    e0 <= 0 && return 0.0
    target = floor_rel + τ * e0
    j = findfirst(x -> x <= target, misfit)
    return isnothing(j) ? Inf : Float64(j - 1)
end

# Empirical linear convergence rate β for one trajectory:
#   y_n ≈ floor + (y_0 - floor) * exp(-β n).
# The slope is fitted by OLS in log space over all iterations whose excess is still
# above cutoff_frac of the initial excess.  The caller must pass the appropriate
# floor: noise floor for a method that can leave the initial subspace, affine
# subspace-LS floor for a subspace-confined method such as EAKI.
function estimate_rate(y::Vector{Float64}, floor_rel::Float64; burnin::Int=5,
                       cutoff_frac::Float64=0.02)
    e0 = y[1] - floor_rel
    e0 > 0 || return NaN
    xs = Float64[]
    ys = Float64[]
    for k in (burnin + 1):length(y)
        ex = y[k] - floor_rel
        if isfinite(ex) && ex > cutoff_frac * e0
            push!(xs, Float64(k - 1))
            push!(ys, log(ex))
        end
    end
    length(xs) >= 10 || return NaN
    mx = mean(xs)
    my = mean(ys)
    denom = sum((xs .- mx) .^ 2)
    denom > 0 || return NaN
    slope = sum((xs .- mx) .* (ys .- my)) / denom
    return -slope
end

# Rate of the missing-direction error itself, target floor 0:
#   missing_rel_n ≈ missing_rel_0 * exp(-β_missing n).
# This is the relevant quantity for dropout-EAKI's projected orthogonal correction.
function estimate_missing_rate(missing_rel::Vector{Float64}; burnin::Int=5,
                               cutoff_frac::Float64=0.02)
    return estimate_rate(missing_rel, 0.0; burnin=burnin, cutoff_frac=cutoff_frac)
end

# Floor used for the misfit rate of a given method.
# EAKI is confined to the initial affine subspace, so its floor is the affine LS
# residual.  The other two methods aim at the full noise floor.
function method_floor(mc, prob)
    return mc.filter_type == "EAKI" ? prob.affine_floor_rel : prob.noise_floor_rel
end

function loglog_fit(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    mask = isfinite.(x) .& isfinite.(y) .& (x .> 0) .& (y .> 0)
    lx = log10.(x[mask])
    ly = log10.(y[mask])
    n = length(lx)
    if n < 3
        return (slope=NaN, intercept=NaN, C=NaN, r2=NaN, n=n)
    end
    mx = mean(lx)
    my = mean(ly)
    denom = sum((lx .- mx) .^ 2)
    denom > 0 || return (slope=NaN, intercept=NaN, C=NaN, r2=NaN, n=n)
    slope = sum((lx .- mx) .* (ly .- my)) / denom
    intercept = my - slope * mx
    yhat = intercept .+ slope .* lx
    ss_res = sum((ly .- yhat) .^ 2)
    ss_tot = sum((ly .- my) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
    return (slope=slope, intercept=intercept, C=10.0^intercept, r2=r2, n=n)
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
        MethodConf("DFO-LS",                  "DFOLS",       false),
    ]
end

# ------------------------------------------------- optimizer runners

# Weighted least-squares residual for the same prior-augmented system used by EKI.
function _optimizer_weighted_residual(forward, y_run, Σ_y_run)
    L = cholesky(Symmetric(Σ_y_run)).L
    return θ -> L \ (forward(θ) - y_run)
end

# Turn a trajectory of point estimates into the metric history used by this study.
function _point_trajectory_metrics(point_history::AbstractVector, prob)
    θ_hist = [reshape(collect(Float64, p), :, 1) for p in point_history]
    return compute_metrics(θ_hist, prob.G, prob.V, prob.Q_sub, prob.q_missing,
                           prob.θ_ref, prob.y, prob.θ_affine_star)
end

# Force a fixed length (N_iter+1) for plotting.  If the optimizer produced more
# points, keep the early trajectory and attach the true final best point at the end.
function _fixed_length_trajectory(point_history::Vector, n::Int, final_point)
    if length(point_history) >= n
        return vcat(point_history[1:(n-1)], [copy(final_point)])
    else
        out = copy(point_history)
        while length(out) < n
            push!(out, copy(final_point))
        end
        return out
    end
end

function run_cmaes_method_condition(forward, y_run, Σ_y_run, prob,
                                    x0::Vector{Float64}, N_iter::Int, seed::Int;
                                    sigma0::Float64=1.0,
                                    popsize::Union{Int,Nothing}=nothing)
    residual = _optimizer_weighted_residual(forward, y_run, Σ_y_run)
    objective(θ) = sum(abs2, residual(θ))
    cma_result = run_cmaes(objective, x0; sigma0=sigma0, max_iter=N_iter,
                           popsize=popsize, seed=seed)
    traj = _fixed_length_trajectory(collect(cma_result.best_history), N_iter + 1,
                                    cma_result.best_x)
    metrics = _point_trajectory_metrics(traj, prob)
    return (metrics=metrics, raw=cma_result)
end

function run_dfols_method_condition(forward, y_run, Σ_y_run, prob,
                                    x0::Vector{Float64}, N_iter::Int, seed::Int;
                                    maxfun::Union{Int,Nothing}=nothing,
                                    rhobeg::Union{Real,Nothing}=nothing,
                                    rhoend::Real=1e-8,
                                    objfun_has_noise::Bool=false,
                                    kwargs...)
    residual = _optimizer_weighted_residual(forward, y_run, Σ_y_run)

    # Use enough evaluations that DFO-LS can produce roughly N_iter internal steps;
    # 1000 is DFO-LS's default cap for high-dimensional problems.
    if maxfun === nothing
        maxfun = max(1000, 3 * N_iter)
    end

    result = run_dfols(residual, x0;
                       maxfun=maxfun,
                       rhobeg=rhobeg,
                       rhoend=rhoend,
                       objfun_has_noise=objfun_has_noise,
                       save_history=true,
                       kwargs...)

    if result.history === nothing || isempty(result.history.x)
        traj = [copy(x0); [copy(result.x) for _ in 1:N_iter]]
    else
        # DFO-LS diagnostic history starts after the initial evaluation; prepend x0.
        traj = vcat([copy(x0)], collect(result.history.x))
        traj = _fixed_length_trajectory(traj, N_iter + 1, result.x)
    end
    metrics = _point_trajectory_metrics(traj, prob)
    return (metrics=metrics, raw=result)
end

# ------------------------------------------------- driver

function run_condition_number_study(;
    N_θ::Int=200,
    N_y::Int=80,
    N_ens::Int=100,
    N_sub::Union{Nothing,Int}=nothing,
    N_iter::Int=500,
    Δτ::Float64=0.2,
    dropout_rate::Float64=0.5,
    noise_std::Float64=1.0,
    truth_signal::Float64=30.0,
    truth_missing_signal::Float64=30.0,
    prior_std::Union{Nothing,Float64}=1000.0,
    full_space_init::Bool=false,
    default_subspace_fraction::Float64=0.5,
    cond_numbers::Vector{Float64}=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1.0e3, 3.0e3, 1.0e4],
    method_configs::Vector{MethodConf}=default_method_configs(),
    seed::Int=2026,
    n_seeds::Int=3,
    cma_sigma0::Float64=1.0,
    dfols_maxfun::Union{Int,Nothing}=nothing,
    save_prefix::String=joinpath(@__DIR__, "Figs", "DEKI_ConditionNumber", "DEKI_EAKI_CMA_DFO_ConditionNumber"),
)
    Random.seed!(seed)

    results = Dict{String, Dict{Int, Dict{Int, Any}}}()
    rates   = Dict{String, Dict{Int, Vector{Float64}}}()
    missing_rates = Dict{String, Dict{Int, Vector{Float64}}}()
    problems = Dict{Int, Dict{Int, Any}}()

    for mc in method_configs
        results[mc.label] = Dict(ki => Dict{Int,Any}() for ki in eachindex(cond_numbers))
        rates[mc.label]   = Dict(ki => Float64[] for ki in eachindex(cond_numbers))
        missing_rates[mc.label] = Dict(ki => Float64[] for ki in eachindex(cond_numbers))
    end

    for ki in eachindex(cond_numbers)
        problems[ki] = Dict{Int,Any}()
    end

    for (ki, cond) in enumerate(cond_numbers)
        for si in 1:n_seeds
            # same problem realization for the two methods and the same (cond, seed)
            Random.seed!(seed + 10000 * ki + si)
            prob = setup_conditioned_linear_problem(N_y, N_θ, N_ens;
                                                    cond=cond,
                                                    N_sub=N_sub,
                                                    default_subspace_fraction=default_subspace_fraction,
                                                    truth_signal=truth_signal,
                                                    truth_missing_signal=truth_missing_signal,
                                                    noise_std=noise_std,
                                                    full_space_init=full_space_init)
            problems[ki][si] = prob
            G = prob.G
            if isnothing(prior_std)
                forward(θ) = G * θ
                y_run = prob.y
                Σ_y_run = prob.Σ_y
            else
                forward, y_run, Σ_y_run = prior_augmented_system(G, prob.y, prob.Σ_y,
                                                                  N_θ, prior_std)
            end

            for (mi, mc) in enumerate(method_configs)
                Random.seed!(seed + 10000 * ki + si + 100000 * mi)
                @info "Running" cond method=mc.label seed=si iter=N_iter
                diverged = false
                metrics = nothing
                try
                    if mc.filter_type == "CMAES"
                        x0_opt = vec(mean(prob.θ0, dims=2))
                        out = run_cmaes_method_condition(forward, y_run, Σ_y_run, prob,
                                                         x0_opt, N_iter, seed;
                                                         sigma0=cma_sigma0)
                        metrics = out.metrics
                    elseif mc.filter_type == "DFOLS"
                        x0_opt = vec(mean(prob.θ0, dims=2))
                        out = run_dfols_method_condition(forward, y_run, Σ_y_run, prob,
                                                         x0_opt, N_iter, seed;
                                                         maxfun=dfols_maxfun)
                        metrics = out.metrics
                    else
                        obj = EKI_Run(
                            forward,
                            copy(prob.θ0),
                            Σ_y_run,
                            y_run;
                            filter_type=mc.filter_type,
                            Δτ=Δτ,
                            N_iter=N_iter,
                            dropout_rate=dropout_rate,
                            inflation=mc.inflation,
                        )
                        metrics = compute_metrics(obj.θ, G, prob.V, prob.Q_sub, prob.q_missing,
                                                  prob.θ_ref, prob.y, prob.θ_affine_star)
                    end
                    if metrics !== nothing && any(!isfinite, metrics.misfit)
                        diverged = true
                        metrics = nothing
                    end
                catch err
                    @warn "Run diverged (numerical error)" cond method=mc.label seed=si err=sprint(showerror, err)[1:min(120, end)]
                    diverged = true
                end

                if diverged
                    NaNvec = fill(NaN, N_iter + 1)
                    metrics = (misfit=copy(NaNvec), obs_rel_err=copy(NaNvec),
                               rel_err=copy(NaNvec), comp_err=copy(NaNvec),
                               rel_affine=copy(NaNvec), missing_rel=copy(NaNvec))
                    push!(rates[mc.label][ki], NaN)
                    push!(missing_rates[mc.label][ki], NaN)
                else
                    push!(rates[mc.label][ki],
                          estimate_rate(metrics.misfit, method_floor(mc, prob)))
                    push!(missing_rates[mc.label][ki],
                          estimate_missing_rate(metrics.missing_rel))
                end
                results[mc.label][ki][si] = (metrics=metrics, diverged=diverged)
            end
        end
    end

    # save raw results before plotting
    figdir = dirname(save_prefix)
    isempty(figdir) || mkpath(figdir)
    serialize(save_prefix * "_results.jls", (; cond_numbers, method_configs, n_seeds, N_iter,
                                             results, rates, missing_rates, problems))
    @info "Raw results saved (before plotting)" save_prefix * "_results.jls"

    # ------------------------------------------------------------- plotting
    ites = 0:N_iter
    metric_keys = [:misfit, :obs_rel_err, :rel_err, :comp_err, :rel_affine, :missing_rel]
    metric_labels = [
        "relative data misfit ‖Gθ̄-y‖/‖y‖",
        "observable rel. error ‖VV'(θ̄-θref)‖/‖θref‖",
        "relative error ‖θ̄-θref‖/‖θref‖",
        "complement error ‖(I-QQ')(θ̄-θref)‖",
        "rel. error to affine LS target θ_affine_star",
        "missing-direction rel. error |q⊥'(θ̄-θref)|/|q⊥'θref|",
    ]
    log_conds = log10.(cond_numbers)
    cmap = PyPlot.matplotlib.cm.get_cmap("viridis")
    cond_colors = [cmap((lc - minimum(log_conds)) / (maximum(log_conds) - minimum(log_conds) + 1e-12))
                   for lc in log_conds]

    try
        # Figure 1: convergence curves
        fig, ax = PyPlot.subplots(nrows=length(method_configs), ncols=length(metric_keys),
                                  figsize=(4.2 * length(metric_keys), 3.6 * length(method_configs)))
        for (mi, mc) in enumerate(method_configs)
            for (mj, key) in enumerate(metric_keys)
                panel = length(method_configs) == 1 ? ax[mj] : ax[mi, mj]
                for (ki, cond) in enumerate(cond_numbers)
                    seed_values = [results[mc.label][ki][si][:metrics][key] for si in 1:n_seeds]
                    med, _, _, _ = seed_summary(seed_values)
                    label = "κ=$(cond)"
                    n_div = count(results[mc.label][ki][si][:diverged] for si in 1:n_seeds)
                    n_div > 0 && (label *= " ($n_div/$n_seeds div)")
                    panel.semilogy(ites, med, color=cond_colors[ki], linewidth=1.8, label=label)
                end
                panel.set_xlabel("Iterations")
                panel.set_ylabel(metric_labels[mj])
                panel.grid(true, which="both", alpha=0.3)
                panel.set_title(mc.label, fontsize=11)
                if mj == length(metric_keys)
                    panel.legend(loc="best", fontsize=7, ncol=2)
                end
            end
        end
        fig.suptitle("Condition-number sensitivity: EKI variants vs CMA-ES vs DFO-LS",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(save_prefix * "_convergence.png", dpi=160)
        PyPlot.close(fig)

        # Figure 2: final value vs condition number
        fig, ax = PyPlot.subplots(ncols=length(metric_keys), figsize=(4.2 * length(metric_keys), 3.6))
        for (mj, key) in enumerate(metric_keys)
            panel = ax[mj]
            for (mi, mc) in enumerate(method_configs)
                style = plot_style(mi)
                finals = Float64[]
                for ki in eachindex(cond_numbers)
                    vals = Float64[results[mc.label][ki][si][:metrics][key][end] for si in 1:n_seeds]
                    f = filter(isfinite, vals)
                    push!(finals, isempty(f) ? NaN : median(f))
                end
                panel.loglog(cond_numbers, finals, marker=style.marker, linestyle=style.linestyle,
                             fillstyle="none", markevery=1, label=mc.label)
            end
            panel.set_xlabel("condition number κ(G)")
            panel.set_ylabel(metric_labels[mj])
            panel.grid(true, which="both", alpha=0.3)
            panel.legend(loc="best", fontsize=8)
            panel.set_title("final value at iter $N_iter", fontsize=10)
        end
        fig.tight_layout()
        fig.savefig(save_prefix * "_final_vs_cond.png", dpi=160)
        PyPlot.close(fig)

        # Figure 3: iterations to reduce excess misfit by fraction τ
        τ_levels = [0.9, 0.7, 0.5, 0.3]
        fig, ax = PyPlot.subplots(ncols=length(τ_levels), figsize=(4.2 * length(τ_levels), 3.6))
        for (tj, τ) in enumerate(τ_levels)
            panel = ax[tj]
            for (mi, mc) in enumerate(method_configs)
                style = plot_style(mi)
                its = Float64[]
                for ki in eachindex(cond_numbers)
                    vals = Float64[]
                    for si in 1:n_seeds
                        if !results[mc.label][ki][si][:diverged]
                            push!(vals, excess_iters(results[mc.label][ki][si][:metrics].misfit,
                                                     method_floor(mc, problems[ki][si]), τ))
                        end
                    end
                    f = filter(isfinite, vals)
                    push!(its, isempty(f) ? NaN : median(f))
                end
                panel.semilogx(cond_numbers, its, marker=style.marker, linestyle=style.linestyle,
                               fillstyle="none", markevery=1, label=mc.label)
            end
            panel.set_xlabel("condition number κ(G)")
            panel.set_ylabel("iterations for τ=$τ excess reduction")
            panel.set_ylim(bottom=0)
            panel.grid(true, which="both", alpha=0.3)
            panel.legend(loc="best", fontsize=8)
        end
        fig.tight_layout()
        fig.savefig(save_prefix * "_iters_to_threshold.png", dpi=160)
        PyPlot.close(fig)

        # Figure 4: empirical linear convergence rate vs condition number
        fig, ax = PyPlot.subplots(figsize=(6.4, 4.6))
        for (mi, mc) in enumerate(method_configs)
            style = plot_style(mi)
            rmed = Float64[]
            for ki in eachindex(cond_numbers)
                r = filter(isfinite, rates[mc.label][ki])
                push!(rmed, isempty(r) ? NaN : median(r))
            end
            mask = isfinite.(rmed) .& (rmed .> 0)
            if count(mask) >= 3
                fit = loglog_fit(cond_numbers[mask], rmed[mask])
                kk = 10.0 .^ range(log10(minimum(cond_numbers[mask])), log10(maximum(cond_numbers[mask])), length=50)
                ax.loglog(cond_numbers[mask], rmed[mask], marker=style.marker,
                          linestyle="none", fillstyle="none", color=style.color, label=mc.label)
                ax.loglog(kk, fit.C .* kk .^ fit.slope, linestyle=style.linestyle,
                          color=style.color,
                          label="$(mc.label) fit: β=$(round(fit.C, sigdigits=3))·κ^$(round(fit.slope, digits=3))")
            end
        end
        ax.set_xlabel("condition number κ(G)")
        ax.set_ylabel("empirical rate β (per iteration)")
        ax.grid(true, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Convergence rate vs condition number", fontsize=11)
        fig.tight_layout()
        fig.savefig(save_prefix * "_rate_vs_cond.png", dpi=160)
        PyPlot.close(fig)

        # Figure 5: missing-direction recovery rate vs condition number
        fig, ax = PyPlot.subplots(figsize=(6.4, 4.6))
        for (mi, mc) in enumerate(method_configs)
            style = plot_style(mi)
            rmed = Float64[]
            for ki in eachindex(cond_numbers)
                r = filter(isfinite, missing_rates[mc.label][ki])
                push!(rmed, isempty(r) ? NaN : median(r))
            end
            mask = isfinite.(rmed) .& (rmed .> 1.0e-8)
            if count(mask) >= 3
                fit = loglog_fit(cond_numbers[mask], rmed[mask])
                kk = 10.0 .^ range(log10(minimum(cond_numbers[mask])), log10(maximum(cond_numbers[mask])), length=50)
                ax.loglog(cond_numbers[mask], rmed[mask], marker=style.marker,
                          linestyle="none", fillstyle="none", color=style.color, label=mc.label)
                ax.loglog(kk, fit.C .* kk .^ fit.slope, linestyle=style.linestyle,
                          color=style.color,
                          label="$(mc.label) fit: β_miss=$(round(fit.C, sigdigits=3))·κ^$(round(fit.slope, digits=3))")
            else
                ax.loglog(cond_numbers[mask], rmed[mask], marker=style.marker,
                          linestyle="none", fillstyle="none", color=style.color, label=mc.label)
            end
        end
        ax.set_xlabel("condition number κ(G)")
        ax.set_ylabel("missing-direction recovery rate β_miss (per iteration)")
        ax.grid(true, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Missing-direction recovery rate vs condition number", fontsize=11)
        fig.tight_layout()
        fig.savefig(save_prefix * "_missing_rate_vs_cond.png", dpi=160)
        PyPlot.close(fig)
    catch err
        @warn "Plotting failed; raw results already saved" err=sprint(showerror, err)[1:min(200, end)]
    end

    # ------------------------------------------------------------- summary
    println("\n==============================")
    println("Condition-number study summary (median over $n_seeds seeds, iter $N_iter)")
    println("==============================")
    for mc in method_configs
        println("\n[", mc.label, "]")
        println(rpad("cond", 10), rpad("misfit", 12), rpad("obs_rel", 12),
                rpad("rel_err", 12), rpad("comp_err", 12), rpad("miss_rel", 12),
                rpad("β", 10), rpad("β_miss", 10), rpad("iters τ=0.5", 12), "div")
        for ki in eachindex(cond_numbers)
            med = Dict{Any,Any}()
            for k in metric_keys
                vals = Float64[results[mc.label][ki][si][:metrics][k][end] for si in 1:n_seeds]
                f = filter(isfinite, vals)
                med[k] = isempty(f) ? NaN : median(f)
            end
            rmed = Float64[]
            mrate = Float64[]
            τ05 = Float64[]
            for si in 1:n_seeds
                if !results[mc.label][ki][si][:diverged]
                    r = rates[mc.label][ki][si]
                    isfinite(r) && push!(rmed, r)
                    rm = missing_rates[mc.label][ki][si]
                    isfinite(rm) && push!(mrate, rm)
                    push!(τ05, excess_iters(results[mc.label][ki][si][:metrics].misfit,
                                             method_floor(mc, problems[ki][si]), 0.5))
                end
            end
            τ_med = median(τ05)
            n_div = count(results[mc.label][ki][si][:diverged] for si in 1:n_seeds)
            println(rpad(cond_numbers[ki], 10),
                    rpad(isnan(med[:misfit]) ? "div" : round(med[:misfit], sigdigits=4), 12),
                    rpad(isnan(med[:obs_rel_err]) ? "div" : round(med[:obs_rel_err], sigdigits=4), 12),
                    rpad(isnan(med[:rel_err]) ? "div" : round(med[:rel_err], sigdigits=4), 12),
                    rpad(isnan(med[:comp_err]) ? "div" : round(med[:comp_err], sigdigits=4), 12),
                    rpad(isnan(med[:missing_rel]) ? "div" : round(med[:missing_rel], sigdigits=4), 12),
                    rpad(isempty(rmed) ? "div" : round(median(rmed), sigdigits=4), 10),
                    rpad(isempty(mrate) ? "div" : round(median(mrate), sigdigits=4), 10),
                    rpad(isinf(τ_med) ? ">$N_iter" : round(τ_med, sigdigits=4), 12),
                    "$n_div/$n_seeds")
        end
    end

    # ---------------------------------------------------- quantitative fits
    println("\n------------------------------")
    println("Quantitative condition-number fits (log-log OLS)")
    println("------------------------------")

    println("\n1. Empirical rate β(κ) = C·κ^p  [β fitted from log-excess misfit slope]")
    for mc in method_configs
        rmed = Float64[]
        for ki in eachindex(cond_numbers)
            r = filter(isfinite, rates[mc.label][ki])
            push!(rmed, isempty(r) ? NaN : median(r))
        end
        mask = isfinite.(rmed) .& (rmed .> 0)
        fit = loglog_fit(cond_numbers[mask], rmed[mask])
        println("   ", rpad(mc.label, 20),
                " β ≈ ", round(fit.C, sigdigits=4), " · κ^(", round(fit.slope, digits=3), ")",
                ",  R²=", round(fit.r2, digits=3), ",  n=", fit.n)
    end

    println("\n2. Iterations N_τ(κ) = A·κ^p  [τ = fraction of initial excess misfit removed]")
    for τ in (0.9, 0.7, 0.5)
        println("   τ=$τ:")
        for mc in method_configs
            its = Float64[]
            for ki in eachindex(cond_numbers)
                vals = Float64[]
                for si in 1:n_seeds
                    if !results[mc.label][ki][si][:diverged]
                        push!(vals, excess_iters(results[mc.label][ki][si][:metrics].misfit,
                                                 method_floor(mc, problems[ki][si]), τ))
                    end
                end
                f = filter(isfinite, vals)
                push!(its, isempty(f) ? NaN : median(f))
            end
            mask = isfinite.(its)
            fit = loglog_fit(cond_numbers[mask], its[mask])
            if fit.n >= 4
                println("      ", rpad(mc.label, 20),
                        " N_$τ ≈ ", round(fit.C, sigdigits=4), " · κ^(", round(fit.slope, digits=3), ")",
                        ",  R²=", round(fit.r2, digits=3), ",  n=", fit.n)
            else
                println("      ", rpad(mc.label, 20), " not enough finite thresholds for fit (n=", fit.n, ")")
            end
        end
    end

    println("\n3. Final observable error E_final(κ) = A·κ^p")
    for mc in method_configs
        vals = Float64[]
        for ki in eachindex(cond_numbers)
            f = filter(isfinite, Float64[results[mc.label][ki][si][:metrics].obs_rel_err[end]
                                          for si in 1:n_seeds])
            push!(vals, isempty(f) ? NaN : median(f))
        end
        mask = isfinite.(vals) .& (vals .> 0)
        fit = loglog_fit(cond_numbers[mask], vals[mask])
        println("   ", rpad(mc.label, 20),
                " E_final ≈ ", round(fit.C, sigdigits=4), " · κ^(", round(fit.slope, digits=3), ")",
                ",  R²=", round(fit.r2, digits=3), ",  n=", fit.n)
    end

    println("\n4. Final missing-direction relative error M_final(κ) = A·κ^p")
    for mc in method_configs
        vals = Float64[]
        for ki in eachindex(cond_numbers)
            f = filter(isfinite, Float64[results[mc.label][ki][si][:metrics].missing_rel[end]
                                          for si in 1:n_seeds])
            push!(vals, isempty(f) ? NaN : median(f))
        end
        mask = isfinite.(vals) .& (vals .> 0)
        fit = loglog_fit(cond_numbers[mask], vals[mask])
        println("   ", rpad(mc.label, 20),
                " M_final ≈ ", round(fit.C, sigdigits=4), " · κ^(", round(fit.slope, digits=3), ")",
                ",  R²=", round(fit.r2, digits=3), ",  n=", fit.n)
    end

    println("\n5. Missing-direction recovery rate β_miss(κ) = C·κ^p  [fit on log missing_rel]")
    for mc in method_configs
        vals = Float64[]
        for ki in eachindex(cond_numbers)
            f = filter(isfinite, missing_rates[mc.label][ki])
            push!(vals, isempty(f) ? NaN : median(f))
        end
        mask = isfinite.(vals) .& (vals .> 1.0e-8)
        fit = loglog_fit(cond_numbers[mask], vals[mask])
        if fit.n >= 3
            println("   ", rpad(mc.label, 20),
                    " β_miss ≈ ", round(fit.C, sigdigits=4), " · κ^(", round(fit.slope, digits=3), ")",
                    ",  R²=", round(fit.r2, digits=3), ",  n=", fit.n)
        else
            println("   ", rpad(mc.label, 20), " not enough positive β_miss points for fit (n=", fit.n, ")")
        end
    end

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

if abspath(PROGRAM_FILE) == @__FILE__
    run_condition_number_study(
        N_θ=env_parse("N_THETA", 200),
        N_y=env_parse("N_Y", 80),
        N_ens=env_parse("N_ENS", 100),
        N_sub=isempty(get(ENV, "N_SUB", "")) ? nothing : parse(Int, ENV["N_SUB"]),
        N_iter=env_parse("N_ITER", 500),
        Δτ=env_parse("DELTA_T", 0.2),
        dropout_rate=env_parse("DROPOUT_RATE", 0.5),
        noise_std=env_parse("NOISE_STD", 1.0),
        truth_signal=env_parse("TRUTH_SIGNAL", 30.0),
        truth_missing_signal=env_parse("TRUTH_MISSING_SIGNAL", 30.0),
        prior_std=(haskey(ENV, "PRIOR_STD") && isempty(strip(ENV["PRIOR_STD"]))) ? nothing : env_parse("PRIOR_STD", 1000.0),
        full_space_init=get(ENV, "FULL_SPACE_INIT", "0") in ("1", "true", "TRUE"),
        default_subspace_fraction=env_parse("DEFAULT_SUBSPACE_FRACTION", 0.5),
        cond_numbers=env_cond_numbers([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1.0e3, 3.0e3, 1.0e4]),
        seed=env_parse("SEED", 2026),
        n_seeds=env_parse("N_SEEDS", 3),
        cma_sigma0=env_parse("CMA_SIGMA0", 1.0),
        dfols_maxfun=isempty(get(ENV, "DFOLS_MAXFUN", "")) ? nothing : env_parse("DFOLS_MAXFUN", 1000),
        save_prefix=get(ENV, "SAVE_PREFIX",
                        joinpath(@__DIR__, "Figs", "DEKI_ConditionNumber", "DEKI_EAKI_CMA_DFO_ConditionNumber")),
    )
end
