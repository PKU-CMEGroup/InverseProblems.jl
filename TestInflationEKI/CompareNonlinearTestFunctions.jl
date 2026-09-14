using LinearAlgebra
using Random
using Serialization
using Statistics

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using PyPlot
using Optimization
using OptimizationCMAEvolutionStrategy

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "NonlinearFunctions.jl"))

# Main knobs. Change these before running the script.
const TEST_FUNCTION_NAMES = ["rosenbrock", "rastrigin"] # ["weakly_nonlinear", "rastrigin", "rosenbrock"]
const THETA_DIM = 100
const N_ENS = 50
const MAX_EVAL = 100 * (2 * N_ENS + 2)
const N_GRID = 200
const INFLATION_DT = 0.5
const DROPOUT_RATE = 0.5
const INITIAL_STD = 3.0
const NEAR_ZERO_TOL = 0.2
const SEED = 1234
const OUTPUT_DIR = joinpath(@__DIR__, "Figs")

const EKI_DT_KEY = Symbol(Char(0x0394), Char(0x03c4))
const EKI_THETA_FIELDS = (Symbol("theta"), Symbol(Char(0x03b8)), Symbol(Char(0x80c3)))
const CMAES = OptimizationCMAEvolutionStrategy.CMAEvolutionStrategy

Base.@kwdef struct NonlinearComparisonConfig
    function_names::Vector{String} = copy(TEST_FUNCTION_NAMES)
    theta_dim::Int = THETA_DIM
    n_ens::Int = N_ENS
    max_eval::Int = MAX_EVAL
    n_grid::Int = N_GRID
    inflation_dt::Float64 = INFLATION_DT
    dropout_rate::Float64 = DROPOUT_RATE
    initial_std::Float64 = INITIAL_STD
    near_zero_tol::Float64 = NEAR_ZERO_TOL
    seed::Int = SEED
    output_dir::String = OUTPUT_DIR
    save_plots::Bool = true
end

struct NonlinearComparisonProblem
    name::String
    theta_dim::Int
    args::NamedTuple
    forward::Function
    y::Vector{Float64}
    sigma_y::Matrix{Float64}
    theta0::Matrix{Float64}
    reference_mean::Vector{Float64}
    x_grid::Matrix{Float64}
    y_grid::Matrix{Float64}
    heatmap::Matrix{Float64}
    xs::Vector{Float64}
    ys::Vector{Float64}
end

function ensure_parent_dir(path::String)
    dir = dirname(path)
    !isempty(dir) && mkpath(dir)
end

positive_for_log(x) = max.(x, eps(Float64))
slug(name::String) = replace(lowercase(name), r"[^a-z0-9]+" => "-")

function plot_window(args::NamedTuple)
    if args.name == "rastrigin"
        return ((-5.12, 5.12), (-5.12, 5.12))
    elseif args.name == "rosenbrock"
        return ((-3.0, 3.0), (-2.0, 10.0))
    elseif args.name == "weakly_nonlinear"
        radius = 3.0
        return (
            (args.ref_theta[1] - radius, args.ref_theta[1] + radius),
            (args.ref_theta[2] - radius, args.ref_theta[2] + radius),
        )
    end
    error("Unknown nonlinear test function: $(args.name)")
end

function normalized_density_heatmap(args, theta_dim::Int, n_grid::Int)
    xlim, ylim = plot_window(args)
    xs = collect(range(xlim[1], xlim[2], length=n_grid))
    ys = collect(range(ylim[1], ylim[2], length=n_grid))
    x_grid = repeat(xs, 1, n_grid)
    y_grid = repeat(ys', n_grid, 1)

    potential = similar(x_grid, Float64)
    theta = copy(args.ref_theta)
    for i in axes(x_grid, 1), j in axes(x_grid, 2)
        theta[1] = x_grid[i, j]
        theta[2] = y_grid[i, j]
        potential[i, j] = func_phi(theta, args)
    end

    shifted = potential .- minimum(potential)
    weights = exp.(-shifted)
    heatmap = weights ./ sum(weights)
    return (; xs, ys, x_grid, y_grid, heatmap)
end

function initial_ensemble(args::NamedTuple, cfg::NonlinearComparisonConfig, rng::AbstractRNG)
    if args.name == "rastrigin"
        xlim, _ = plot_window(args)
        return xlim[1] .+ (xlim[2] - xlim[1]) .* rand(rng, cfg.theta_dim, cfg.n_ens)
    end
    return cfg.initial_std .* randn(rng, cfg.theta_dim, cfg.n_ens)
end

function make_nonlinear_problem(name::String, cfg::NonlinearComparisonConfig, seed::Int)
    args = make_test_args(name; dim=cfg.theta_dim, seed=seed)
    forward(theta) = func_F(theta, args)
    y = zeros(Float64, length(forward(zeros(cfg.theta_dim))))
    sigma_y = Matrix{Float64}(I, length(y), length(y))

    rng = MersenneTwister(seed + 17)
    theta0 = initial_ensemble(args, cfg, rng)
    grid = normalized_density_heatmap(args, cfg.theta_dim, cfg.n_grid)

    return NonlinearComparisonProblem(
        args.name,
        cfg.theta_dim,
        args,
        forward,
        y,
        sigma_y,
        theta0,
        copy(args.ref_theta),
        grid.x_grid,
        grid.y_grid,
        grid.heatmap,
        grid.xs,
        grid.ys,
    )
end

function ensemble_covariance(theta_ens::AbstractMatrix)
    size(theta_ens, 2) <= 1 && return zeros(size(theta_ens, 1), size(theta_ens, 1))
    theta_mean = mean(theta_ens, dims=2)
    anomalies = (theta_ens .- theta_mean) ./ sqrt(size(theta_ens, 2) - 1)
    return anomalies * anomalies'
end

function objective_context(problem::NonlinearComparisonProblem)
    residual(theta) = problem.forward(theta)
    phi(theta) = 0.5 * sum(abs2, problem.forward(theta))
    return (; residual, phi)
end

function evals_per_iteration(method::String, n_ens::Int)
    if method == "dropout-EAKI" || method == "dropout-ETKI"
        return 2 * n_ens + 2
    elseif method == "DEKI"
        return n_ens + 1
    elseif method == "CMA-ES"
        return n_ens
    end
    error("Unknown method: $(method)")
end

function iterations_for_eval_budget(method::String, cfg::NonlinearComparisonConfig)
    return fld(cfg.max_eval, evals_per_iteration(method, cfg.n_ens))
end

function evaluation_axis_for_history(method::String, n_ens::Int, n_hist::Int)
    step = evals_per_iteration(method, n_ens)
    return collect(0:step:step * (n_hist - 1))
end

function ensemble_history(source, theta_dim::Int)
    theta = nothing
    for field in EKI_THETA_FIELDS
        if hasproperty(source, field)
            theta = getfield(source, field)
            break
        end
    end
    theta === nothing && error("Cannot find ensemble history on $(typeof(source))")

    if size(theta[1], 1) == theta_dim
        return theta
    elseif size(theta[1], 2) == theta_dim
        return [permutedims(theta_i) for theta_i in theta]
    end
    error("Cannot determine ensemble orientation for theta size $(size(theta[1]))")
end

function objective_curve_from_ensembles(history, phi::Function)
    representative_points = [vec(mean(theta, dims=2)) for theta in history]
    objective_values = [phi(theta_mean) for theta_mean in representative_points]
    covariance_norm = [norm(ensemble_covariance(theta)) for theta in history]
    return (; objective_values, covariance_norm, representative_points)
end

count_near_zero_dimensions(theta::AbstractVector, tol::Real) = count(abs.(theta) .<= tol)

function run_eki_method(problem::NonlinearComparisonProblem, ctx, cfg::NonlinearComparisonConfig,
        filter_type::String, label::String, seed::Int)
    Random.seed!(seed)

    inflation = (filter_type != "DEKI")
    n_iter = iterations_for_eval_budget(filter_type, cfg)

    ekiobj = EKI_Run(
        problem.forward,
        copy(problem.theta0),
        problem.sigma_y,
        problem.y;
        filter_type=filter_type,
        Δτ = cfg.inflation_dt,
        N_iter=n_iter,
        dropout_rate=cfg.dropout_rate,
        inflation=inflation,
    )

    history = ensemble_history(ekiobj, problem.theta_dim)
    curves = objective_curve_from_ensembles(history, ctx.phi)
    evaluation_axis = evaluation_axis_for_history(filter_type, cfg.n_ens, length(history))
    final_mean_point = curves.representative_points[end]
    return (;
        label,
        method=filter_type,
        n_iter,
        max_eval=cfg.max_eval,
        evals_per_iteration=evals_per_iteration(filter_type, cfg.n_ens),
        final_mean_point,
        near_zero_count=count_near_zero_dimensions(final_mean_point, cfg.near_zero_tol),
        near_zero_tol=cfg.near_zero_tol,
        object=ekiobj,
        final_points=history[end],
        representative_points=curves.representative_points,
        optimization_error=curves.objective_values,
        covariance_norm=curves.covariance_norm,
        x_axis=evaluation_axis,
    )
end

function covariance_norm_from_columns(points::AbstractMatrix)
    size(points, 2) <= 1 && return 0.0
    return norm(ensemble_covariance(points))
end

function run_cma_es(problem::NonlinearComparisonProblem, ctx, cfg::NonlinearComparisonConfig, seed::Int)
    x0 = vec(mean(problem.theta0, dims=2))
    n_iter = iterations_for_eval_budget("CMA-ES", cfg)
    evaluation_count = Ref(0)
    function objective(x)
        value = ctx.phi(Vector{Float64}(x))
        evaluation_count[] += 1
        return value
    end

    best_values = Float64[ctx.phi(x0)]
    best_points = Vector{Float64}[copy(x0)]
    covariance_norm = Float64[sqrt(problem.theta_dim) * cfg.initial_std^2]
    evaluation_axis = Int[0]
    final_population = copy(problem.theta0)

    function callback(opt, y, fvals, perm)
        push!(best_points, Vector{Float64}(CMAES.xbest(opt)))
        push!(best_values, CMAES.fbest(opt))

        population = Matrix{Float64}(CMAES.compute_input(opt.p, y))
        final_population = population
        push!(covariance_norm, covariance_norm_from_columns(population))
        push!(evaluation_axis, evaluation_count[])
        return nothing
    end

    opt = CMAES.minimize(
        objective,
        x0,
        cfg.initial_std;
        popsize=cfg.n_ens,
        maxiter=n_iter,
        seed=UInt(seed),
        verbosity=0,
        callback=callback,
    )

    if evaluation_count[] > evaluation_axis[end]
        push!(best_points, Vector{Float64}(CMAES.xbest(opt)))
        push!(best_values, CMAES.fbest(opt))
        push!(covariance_norm, covariance_norm[end])
        push!(evaluation_axis, evaluation_count[])
    elseif best_values[end] != CMAES.fbest(opt)
        best_points[end] = Vector{Float64}(CMAES.xbest(opt))
        best_values[end] = CMAES.fbest(opt)
    end

    final_mean_point = vec(mean(final_population, dims=2))
    return (;
        label="CMA-ES",
        method="CMA-ES",
        n_iter,
        max_eval=cfg.max_eval,
        evals_per_iteration=evals_per_iteration("CMA-ES", cfg.n_ens),
        final_mean_point,
        near_zero_count=count_near_zero_dimensions(final_mean_point, cfg.near_zero_tol),
        near_zero_tol=cfg.near_zero_tol,
        object=opt,
        final_points=final_population,
        representative_points=best_points,
        optimization_error=best_values,
        covariance_norm,
        x_axis=evaluation_axis,
    )
end

function run_problem_comparison(problem::NonlinearComparisonProblem, cfg::NonlinearComparisonConfig, seed::Int)
    ctx = objective_context(problem)
    runs = NamedTuple[]
    push!(runs, run_eki_method(problem, ctx, cfg, "dropout-EAKI", "dropout EAKI", seed + 1))
    push!(runs, run_eki_method(problem, ctx, cfg, "dropout-ETKI", "dropout ETKI", seed + 2))
    push!(runs, run_eki_method(problem, ctx, cfg, "DEKI", "DEKI", seed + 3))
    push!(runs, run_cma_es(problem, ctx, cfg, seed + 4))
    return runs
end

function near_zero_dimension_summary(problem::NonlinearComparisonProblem, runs)
    return [
        (;
            method=run.label,
            near_zero_count=run.near_zero_count,
            theta_dim=problem.theta_dim,
            tol=run.near_zero_tol,
        )
        for run in runs
    ]
end

function report_near_zero_dimensions(problem::NonlinearComparisonProblem, runs)
    problem.name == "rastrigin" || return nothing
    for item in near_zero_dimension_summary(problem, runs)
        @info "Final mean dimensions near zero" function_name=problem.name method=item.method count=item.near_zero_count dim=item.theta_dim tol=item.tol
    end
    return nothing
end
 
function plot_style(index::Int)
    markers = ["o", "s", "^", "d", "v", "x"]
    linestyles = ["-", "--", "-.", ":", "-"]
    return (
        color="C$(mod(index - 1, 10))",
        marker=markers[mod1(index, length(markers))],
        linestyle=linestyles[mod1(index, length(linestyles))],
    )
end

function plot_heatmap_and_final_points!(ax, problem::NonlinearComparisonProblem, runs)
    ax.pcolormesh(problem.x_grid, problem.y_grid, problem.heatmap, cmap="viridis")
    ax.contour(problem.x_grid, problem.y_grid, problem.heatmap, 12; colors="white",
        linewidths=0.6, alpha=0.55)
    ax.scatter(problem.reference_mean[1], problem.reference_mean[2];
        marker="*", s=120, color="black", label="reference minimizer", zorder=5)
    for (index, run) in enumerate(runs)
        style = plot_style(index)
        points = run.final_points
        ax.scatter(points[1, :], points[2, :]; s=18, alpha=0.55,
            color=style.color, marker=style.marker, linewidths=0.4,
            label=run.label, zorder=3)
    end

    ax.set_title("$(problem.name): density slice")
    ax.set_xlabel("theta_1")
    ax.set_ylabel("theta_2")
    ax.set_xlim(extrema(problem.xs))
    ax.set_ylim(extrema(problem.ys))
    ax.legend(fontsize=7, loc="best", frameon=false)
end

function plot_curve_panel!(ax, runs, field::Symbol, title::String, ylabel::String)
    for (index, run) in enumerate(runs)
        style = plot_style(index)
        curve = positive_for_log(getproperty(run, field))
        markevery = max(1, div(length(curve), 8))
        ax.semilogy(run.x_axis, curve;
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            fillstyle="none",
            markevery=markevery,
            linewidth=1.5,
            label=run.label,
        )
    end
    ax.set_title(title)
    ax.set_xlabel("forward model evaluations")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, maximum(getproperty.(runs, :max_eval)))
    ax.grid(true, alpha=0.25)
    ax.legend(fontsize=7, loc="best", frameon=false)
end

function default_output_file(cfg::NonlinearComparisonConfig)
    function_part = join(slug.(cfg.function_names), "_")
    return joinpath(cfg.output_dir,
        "nonlinear_functions_$(function_part)_dim$(cfg.theta_dim)_ens$(cfg.n_ens)_eval$(cfg.max_eval).png")
end

function plot_comparison(problems, results::Dict{String, Vector{NamedTuple}}, output_file::String)
    fig, axs = PyPlot.subplots(length(problems), 3;
        figsize=(18, 4.8 * length(problems)), squeeze=false)

    for (row, problem) in enumerate(problems)
        runs = results[problem.name]
        plot_heatmap_and_final_points!(axs[row, 1], problem, runs)
        plot_curve_panel!(axs[row, 2], runs, :optimization_error,
            "optimization error", "Phi(theta)")
        plot_curve_panel!(axs[row, 3], runs, :covariance_norm,
            "covariance norm", "||Cov(points)||_F")
    end

    fig.tight_layout()
    ensure_parent_dir(output_file)
    fig.savefig(output_file, dpi=180)
    PyPlot.close(fig)
    return output_file
end

function run_nonlinear_test_function_comparison(
        cfg::NonlinearComparisonConfig=NonlinearComparisonConfig())
    problems = NonlinearComparisonProblem[]
    results = Dict{String, Vector{NamedTuple}}()
    near_zero_reports = Dict{String, Vector{NamedTuple}}()

    for (index, name) in enumerate(cfg.function_names)
        problem = make_nonlinear_problem(name, cfg, cfg.seed + 1000 * index)
        @info "Running nonlinear test-function comparison" function_name=problem.name dim=problem.theta_dim max_eval=cfg.max_eval n_ens=cfg.n_ens
        push!(problems, problem)
        runs = run_problem_comparison(problem, cfg, cfg.seed + 1000 * index)
        results[problem.name] = runs
        near_zero_reports[problem.name] = near_zero_dimension_summary(problem, runs)
        report_near_zero_dimensions(problem, runs)
    end

    plot_file = cfg.save_plots ? plot_comparison(problems, results, default_output_file(cfg)) : nothing
    result_file = joinpath(cfg.output_dir,
        "nonlinear_functions_$(join(slug.(cfg.function_names), "_"))_dim$(cfg.theta_dim)_ens$(cfg.n_ens)_eval$(cfg.max_eval).jls")
    ensure_parent_dir(result_file)
    serialize(result_file, (; config=cfg, problems, results, near_zero_reports, plot_file))

    return (; config=cfg, problems, results, near_zero_reports, plot_file, result_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_nonlinear_test_function_comparison()
    @info "Saved nonlinear test-function comparison" result.plot_file result.result_file
end
