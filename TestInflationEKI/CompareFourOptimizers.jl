using LinearAlgebra
using Random
using Serialization
using Statistics

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using PyPlot
using Optimization
using OptimizationCMAEvolutionStrategy
using DFOLS

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "..", "Derivative-Free-Variational-Inference", "MultiModal.jl"))

# Main knobs. Change these before running the script.
const PROBLEM_NAMES = ["Circle", "Rosenbrock", "Double_banana"]
const THETA_DIM = 100
const N_ITER = 500
const N_ENS = 50
const N_GRID = 250
const INFLATION_DT = 0.2
const DROPOUT_RATE = 0.5
const INITIAL_STD = 1.0
const SEED = 20260824
const OUTPUT_DIR = joinpath(@__DIR__, "Figs")

const EKI_DT_KEY = Symbol(Char(0x0394), Char(0x03c4))
const EKI_THETA_FIELDS = (Symbol("theta"), Symbol(Char(0x03b8)), Symbol(Char(0x80c3)))
const CMAES = OptimizationCMAEvolutionStrategy.CMAEvolutionStrategy

Base.@kwdef struct ComparisonConfig
    problem_names::Vector{String} = copy(PROBLEM_NAMES)
    theta_dim::Int = THETA_DIM
    n_iter::Int = N_ITER
    n_ens::Int = N_ENS
    n_grid::Int = N_GRID
    inflation_dt::Float64 = INFLATION_DT
    dropout_rate::Float64 = DROPOUT_RATE
    initial_std::Float64 = INITIAL_STD
    seed::Int = SEED
    output_dir::String = OUTPUT_DIR
    save_plots::Bool = true
end

struct ProblemSpec
    label::String
    gtype::String
    y::Vector{Float64}
    sigma_eta::Vector{Float64}
    arg
    marginal_dim::Int
    xlim::Tuple{Float64,Float64}
    ylim::Tuple{Float64,Float64}
end

struct ComparisonProblem
    name::String
    theta_dim::Int
    spec::ProblemSpec
    args::Tuple
    marginal_args::Tuple
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

function problem_spec(name::String, theta_dim::Int)
    theta_dim >= 2 || error("theta_dim must be at least 2")
    n_tail = theta_dim - 2
    key = lowercase(replace(name, "-" => "_", " " => "_"))

    if key in ("rosenbrock", "banana")
        lambda = 10.0
        return ProblemSpec(
            "Rosenbrock",
            "Banana",
            vcat([0.0, 1.0], zeros(n_tail)),
            vcat([sqrt(10.0), sqrt(10.0)], ones(n_tail)),
            lambda,
            2,
            (-4.0, 4.0),
            (-2.0, 15.0),
        )
    elseif key in ("double_banana", "doublebanana")
        lambda = 100.0
        return ProblemSpec(
            "Double_banana",
            "Double_banana",
            vcat([log(lambda + 1.0), 0.0, 1.0], zeros(n_tail)),
            vcat([0.3, 1.0, 1.0], ones(n_tail)),
            lambda,
            3,
            (-3.0, 3.0),
            (-3.0, 3.0),
        )
    elseif key == "circle"
        return ProblemSpec(
            "Circle",
            "Circle",
            vcat([1.0], zeros(n_tail)),
            vcat([0.3], ones(n_tail)),
            Matrix{Float64}(I, 2, 2),
            1,
            (-3.0, 3.0),
            (-3.0, 3.0),
        )
    end
    error("Unknown problem name: $(name)")
end

full_args(spec::ProblemSpec) = (spec.y, spec.sigma_eta, spec.arg, spec.gtype)

function marginal_args(spec::ProblemSpec)
    return (
        spec.y[1:spec.marginal_dim],
        spec.sigma_eta[1:spec.marginal_dim],
        spec.arg,
        spec.gtype,
    )
end

function grid_heatmap(spec::ProblemSpec, theta_dim::Int, n_grid::Int)
    xs = collect(range(spec.xlim[1], spec.xlim[2], length=n_grid))
    ys = collect(range(spec.ylim[1], spec.ylim[2], length=n_grid))
    x_grid = repeat(xs, 1, n_grid)
    y_grid = repeat(ys', n_grid, 1)

    args_2d = marginal_args(spec)
    potential = similar(x_grid, Float64)
    weights = similar(x_grid, Float64)
    for i in axes(x_grid, 1), j in axes(x_grid, 2)
        theta = [x_grid[i, j], y_grid[i, j]]
        potential[i, j] = Phi(theta, args_2d)
    end

    shifted = potential .- minimum(potential)
    heatmap = log10.(positive_for_log(shifted .+ eps(Float64)))
    weights .= exp.(-shifted)

    weight_sum = sum(weights)
    mean_pair = [
        sum(x_grid .* weights) / weight_sum,
        sum(y_grid .* weights) / weight_sum,
    ]
    reference_mean = fill(sum(mean_pair), theta_dim)
    reference_mean[1:2] .= mean_pair
    return (; xs, ys, x_grid, y_grid, heatmap, reference_mean)
end

function make_problem(name::String, cfg::ComparisonConfig, seed::Int)
    spec = problem_spec(name, cfg.theta_dim)
    args = full_args(spec)
    grid = grid_heatmap(spec, cfg.theta_dim, cfg.n_grid)
    forward(theta) = G(theta, spec.arg, spec.gtype)
    sigma_y = Array(Diagonal(spec.sigma_eta .^ 2))

    length(spec.y) == length(forward(zeros(cfg.theta_dim))) ||
        error("incompatible spec $(spec.label) and theta_dim=$(cfg.theta_dim)")

    rng = MersenneTwister(seed)
    theta0 = cfg.initial_std .* randn(rng, cfg.theta_dim, cfg.n_ens)

    return ComparisonProblem(
        spec.label,
        cfg.theta_dim,
        spec,
        args,
        marginal_args(spec),
        forward,
        spec.y,
        sigma_y,
        theta0,
        grid.reference_mean,
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

function objective_context(problem::ComparisonProblem)
    residual(theta) = F(theta, problem.args)
    phi(theta) = Phi(theta, problem.args)
    return (; residual, phi)
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
    means = [vec(mean(theta, dims=2)) for theta in history]
    objective_values = [phi(theta_mean) for theta_mean in means]
    covariance_norm = [norm(ensemble_covariance(theta)) for theta in history]
    return (; means, objective_values, covariance_norm)
end

function run_eki_method(problem::ComparisonProblem, ctx, cfg::ComparisonConfig,
        filter_type::String, label::String, seed::Int)
    Random.seed!(seed)
    ekiobj = EKI_Run(
        problem.forward,
        copy(problem.theta0),
        problem.sigma_y,
        problem.y;
        filter_type=filter_type,
        EKI_DT_KEY => cfg.inflation_dt,
        N_iter=cfg.n_iter,
        dropout_rate=cfg.dropout_rate,
        inflation=true,
    )
    history = ensemble_history(ekiobj, problem.theta_dim)
    curves = objective_curve_from_ensembles(history, ctx.phi)
    return (;
        label,
        method=filter_type,
        object=ekiobj,
        final_points=history[end],
        representative_points=curves.means,
        optimization_error=curves.objective_values,
        covariance_norm=curves.covariance_norm,
        x_axis=0:(length(curves.objective_values) - 1),
    )
end

function covariance_norm_from_columns(points::AbstractMatrix)
    size(points, 2) <= 1 && return 0.0
    return norm(ensemble_covariance(points))
end

function best_point_cloud(points::Vector{Vector{Float64}}, values::Vector{Float64}, n_points::Int)
    isempty(points) && error("Cannot build point cloud from an empty history")
    order = sortperm(values)
    keep = order[1:min(n_points, length(order))]
    return hcat(points[keep]...)
end

function run_cma_es(problem::ComparisonProblem, ctx, cfg::ComparisonConfig, seed::Int)
    x0 = vec(mean(problem.theta0, dims=2))
    objective(x) = ctx.phi(Vector{Float64}(x))

    best_values = Float64[objective(x0)]
    best_points = Vector{Float64}[copy(x0)]
    covariance_norm = Float64[sqrt(problem.theta_dim) * cfg.initial_std^2]
    final_population = copy(problem.theta0)

    function callback(opt, y, fvals, perm)
        best_point = Vector{Float64}(CMAES.xbest(opt))
        push!(best_points, best_point)
        push!(best_values, CMAES.fbest(opt))

        population = Matrix{Float64}(CMAES.compute_input(opt.p, y))
        final_population = population
        push!(covariance_norm, covariance_norm_from_columns(population))
        return nothing
    end

    opt = CMAES.minimize(
        objective,
        x0,
        cfg.initial_std;
        popsize=cfg.n_ens,
        maxiter=cfg.n_iter,
        seed=UInt(seed),
        verbosity=0,
        callback=callback,
    )

    if isempty(best_points) || best_values[end] != CMAES.fbest(opt)
        push!(best_points, Vector{Float64}(CMAES.xbest(opt)))
        push!(best_values, CMAES.fbest(opt))
        push!(covariance_norm, covariance_norm[end])
    end

    return (;
        label="CMA-ES",
        method="CMA-ES",
        object=opt,
        final_points=final_population,
        representative_points=best_points,
        optimization_error=best_values,
        covariance_norm,
        x_axis=0:(length(best_values) - 1),
    )
end

function dfols_solve_with_budget(objfun::Function, x0::Vector{Float64}, maxfun::Int)
    return DFOLS.dfols[:solve](
        objfun,
        x0;
        maxfun=maxfun,
        rhobeg=0.5,
        rhoend=1e-8,
    )
end

function run_dfols(problem::ComparisonProblem, ctx, cfg::ComparisonConfig)
    x0 = vec(mean(problem.theta0, dims=2))
    points = Vector{Float64}[copy(x0)]
    best_points = Vector{Float64}[copy(x0)]
    best_values = Float64[ctx.phi(x0)]
    covariance_norm = Float64[0.0]

    function objfun(x)
        theta = Vector{Float64}(x)
        value = ctx.phi(theta)
        push!(points, copy(theta))
        if value < best_values[end]
            push!(best_values, value)
            push!(best_points, copy(theta))
        else
            push!(best_values, best_values[end])
            push!(best_points, copy(best_points[end]))
        end
        cloud = best_point_cloud(points, [ctx.phi(p) for p in points], cfg.n_ens)
        push!(covariance_norm, covariance_norm_from_columns(cloud))
        return ctx.residual(theta)
    end

    maxfun = max(cfg.n_iter + 1, 2 * problem.theta_dim + 1)
    solution = try
        dfols_solve_with_budget(objfun, x0, maxfun)
    catch err
        @warn "DFO-LS stopped before returning a solution" problem=problem.name exception=(err, catch_backtrace())
        nothing
    end

    if solution !== nothing
        theta = Vector{Float64}(solution[:x])
        value = ctx.phi(theta)
        push!(points, copy(theta))
        if value < best_values[end]
            push!(best_values, value)
            push!(best_points, copy(theta))
        else
            push!(best_values, best_values[end])
            push!(best_points, copy(best_points[end]))
        end
        cloud = best_point_cloud(points, [ctx.phi(p) for p in points], cfg.n_ens)
        push!(covariance_norm, covariance_norm_from_columns(cloud))
    end

    final_values = [ctx.phi(p) for p in points]
    final_cloud = best_point_cloud(points, final_values, cfg.n_ens)

    return (;
        label="DFO-LS",
        method="DFO-LS",
        object=solution,
        final_points=final_cloud,
        representative_points=best_points,
        optimization_error=best_values,
        covariance_norm,
        x_axis=0:(length(best_values) - 1),
    )
end

function run_problem_comparison(problem::ComparisonProblem, cfg::ComparisonConfig, seed::Int)
    ctx = objective_context(problem)
    runs = NamedTuple[]
    push!(runs, run_eki_method(problem, ctx, cfg, "EAKI", "Inflation EAKI", seed + 1))
    push!(runs, run_eki_method(problem, ctx, cfg, "DEKI", "DEKI", seed + 2))
    push!(runs, run_cma_es(problem, ctx, cfg, seed + 3))
    push!(runs, run_dfols(problem, ctx, cfg))
    return runs
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

function plot_heatmap_and_final_points!(ax, problem::ComparisonProblem, runs)
    ax.pcolormesh(problem.x_grid, problem.y_grid, problem.heatmap, cmap="viridis")
    ax.contour(problem.x_grid, problem.y_grid, problem.heatmap, 12; colors="white",
        linewidths=0.6, alpha=0.55)
    ax.scatter(problem.reference_mean[1], problem.reference_mean[2];
        marker="*", s=120, color="black", label="reference mean", zorder=5)

    for (index, run) in enumerate(runs)
        style = plot_style(index)
        points = run.final_points
        ax.scatter(points[1, :], points[2, :]; s=18, alpha=0.55,
            color=style.color, marker=style.marker, linewidths=0.4,
            label=run.label, zorder=3)
    end

    ax.set_title("$(problem.name): log objective slice")
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
    ax.set_xlabel("iteration / evaluation")
    ax.set_ylabel(ylabel)
    ax.grid(true, alpha=0.25)
    ax.legend(fontsize=7, loc="best", frameon=false)
end

function default_output_file(cfg::ComparisonConfig)
    problem_part = join(slug.(cfg.problem_names), "_")
    return joinpath(cfg.output_dir,
        "four_optimizers_$(problem_part)_dim$(cfg.theta_dim)_ens$(cfg.n_ens)_iter$(cfg.n_iter).png")
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

function run_four_optimizer_comparison(cfg::ComparisonConfig=ComparisonConfig())
    problems = ComparisonProblem[]
    results = Dict{String, Vector{NamedTuple}}()

    for (index, name) in enumerate(cfg.problem_names)
        problem = make_problem(name, cfg, cfg.seed + 1000 * index)
        @info "Running four-optimizer comparison" problem=problem.name dim=problem.theta_dim n_iter=cfg.n_iter n_ens=cfg.n_ens
        push!(problems, problem)
        results[problem.name] = run_problem_comparison(problem, cfg, cfg.seed + 1000 * index)
    end

    plot_file = cfg.save_plots ? plot_comparison(problems, results, default_output_file(cfg)) : nothing
    result_file = joinpath(cfg.output_dir,
        "four_optimizers_$(join(slug.(cfg.problem_names), "_"))_dim$(cfg.theta_dim)_ens$(cfg.n_ens)_iter$(cfg.n_iter).jls")
    ensure_parent_dir(result_file)
    serialize(result_file, (; config=cfg, problems, results, plot_file))

    return (; config=cfg, problems, results, plot_file, result_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_four_optimizer_comparison()
    @info "Saved four-optimizer comparison" result.plot_file result.result_file
end
