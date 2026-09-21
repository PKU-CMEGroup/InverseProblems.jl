using LinearAlgebra
using Random
using Serialization
using Statistics

ENV["MPLBACKEND"] = get(ENV, "MPLBACKEND", "Agg")

using PyPlot

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "..", "Inversion", "CMAES.jl"))
include(joinpath(@__DIR__, "NonlinearFunctions.jl"))

# Main knobs. Change these before running the script.
const TEST_FUNCTION_NAMES = [
    "monotone_cubic",
    "paired_rosenbrock",
    "rosenbrock",
    "rastrigin",
]
const THETA_DIM = 100
const N_ENS = 50
const MAX_EVAL = 100 * (2 * N_ENS + 2)
const N_GRID = 200
const INFLATION_DT = 0.5
const DROPOUT_RATE = 0.5
const MEAN_LINE_SEARCH = true
const MEAN_LINE_SEARCH_CONTRACTION = 0.5
const MEAN_LINE_SEARCH_ARMIJO_C = 1e-4
const JOINT_WEIGHT_MODE = "adaptive"
const JOINT_WEIGHT_MIN = 0.1
const JOINT_WEIGHT_MAX = 10.0
const JOINT_WEIGHT_SMOOTHING = 0.25
const INITIAL_STD = 1.0
const NEAR_ZERO_TOL = 0.2
const SEED = 1234
const OUTPUT_DIR = joinpath(@__DIR__, "Figs")

const EKI_DT_KEY = Symbol(Char(0x0394), Char(0x03c4))
const EKI_THETA_FIELDS = (Symbol("theta"), Symbol(Char(0x03b8)), Symbol(Char(0x80c3)))
Base.@kwdef struct NonlinearComparisonConfig
    function_names::Vector{String} = copy(TEST_FUNCTION_NAMES)
    theta_dim::Int = THETA_DIM
    n_ens::Int = N_ENS
    max_eval::Int = MAX_EVAL
    n_grid::Int = N_GRID
    inflation_dt::Float64 = INFLATION_DT
    dropout_rate::Float64 = DROPOUT_RATE
    mean_line_search::Bool = MEAN_LINE_SEARCH
    mean_line_search_contraction::Float64 = MEAN_LINE_SEARCH_CONTRACTION
    mean_line_search_armijo_c::Float64 = MEAN_LINE_SEARCH_ARMIJO_C
    initial_std::Float64 = INITIAL_STD
    rastrigin_initial_center::Float64 = 1.0
    rastrigin_initial_half_width::Float64 = 4.0
    paired_rosenbrock_initial_std::Float64 = 0.3
    rosenbrock_initial_std::Float64 = 0.3
    joint_dropout_weight::Float64 = 1.0
    joint_weight_mode::String = JOINT_WEIGHT_MODE
    joint_weight_min::Float64 = JOINT_WEIGHT_MIN
    joint_weight_max::Float64 = JOINT_WEIGHT_MAX
    joint_weight_smoothing::Float64 = JOINT_WEIGHT_SMOOTHING
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

const PROBLEM_SEED_OFFSETS = Dict(
    "monotone_cubic" => 1000,
    "paired_rosenbrock" => 2000,
    "rastrigin" => 3000,
    "rotated_rastrigin" => 4000,
    "rosenbrock" => 5000,
    "weakly_nonlinear" => 6000,
)

problem_seed(base_seed::Int, name::String) =
    base_seed + get(PROBLEM_SEED_OFFSETS, name) do
        error("No stable seed offset registered for $(name)")
    end

function plot_window(args::NamedTuple)
    if args.name == "rastrigin"
        return ((-5.12, 5.12), (-5.12, 5.12))
    elseif args.name == "rotated_rastrigin"
        return ((-5.12, 5.12), (-5.12, 5.12))
    elseif args.name in ("rosenbrock", "paired_rosenbrock")
        return ((-2.0, 2.0), (-1.0, 3.0))
    elseif args.name in ("weakly_nonlinear", "monotone_cubic")
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
    if args.name in ("rastrigin", "rotated_rastrigin")
        lower = cfg.rastrigin_initial_center - cfg.rastrigin_initial_half_width
        width = 2 * cfg.rastrigin_initial_half_width
        return lower .+ width .* rand(rng, cfg.theta_dim, cfg.n_ens)
    elseif args.name == "paired_rosenbrock"
        theta_center = repeat([-1.2, 1.0], div(cfg.theta_dim, 2))
        return theta_center .+
               cfg.paired_rosenbrock_initial_std .* randn(rng, cfg.theta_dim, cfg.n_ens)
    elseif args.name == "rosenbrock"
        # The same classical alternating start is now coupled through every
        # adjacent pair by the chained Rosenbrock forward map.
        theta_center = [isodd(i) ? -1.2 : 1.0 for i in 1:cfg.theta_dim]
        return theta_center .+
               cfg.rosenbrock_initial_std .* randn(rng, cfg.theta_dim, cfg.n_ens)
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

function minimum_evals_per_iteration(method::String, n_ens::Int)
    if method == "EAKI"
        return n_ens + 1
    elseif method == "dropout-EAKI-sequential"
        return 2 * n_ens + 2
    elseif method == "dropout-EAKI-joint"
        return 2 * n_ens + 1
    elseif method == "DEKI"
        return 2 * n_ens + 1
    elseif method == "CMA-ES"
        return n_ens
    end
    error("Unknown method: $(method)")
end

function iterations_for_eval_budget(method::String, cfg::NonlinearComparisonConfig)
    remaining = max(cfg.max_eval - cfg.n_ens, 0)
    return fld(remaining, minimum_evals_per_iteration(method, cfg.n_ens))
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

mutable struct ForwardTracker
    evaluations::Int
    best_value::Float64
    best_point::Vector{Float64}
end

ForwardTracker(theta_dim::Int) = ForwardTracker(0, Inf, zeros(theta_dim))

function observation_objective(problem::NonlinearComparisonProblem, prediction::AbstractVector)
    residual = prediction - problem.y
    return 0.5 * dot(residual, problem.sigma_y \ residual)
end

function tracked_forward(problem::NonlinearComparisonProblem, tracker::ForwardTracker,
                         theta::AbstractArray)
    theta_matrix = ndims(theta) == 1 ? reshape(theta, :, 1) : theta
    predictions = Matrix{Float64}(undef, length(problem.y), size(theta_matrix, 2))
    for j in axes(theta_matrix, 2)
        point = Vector{Float64}(view(theta_matrix, :, j))
        prediction = problem.forward(point)
        predictions[:, j] = prediction
        value = observation_objective(problem, prediction)
        tracker.evaluations += 1
        if value < tracker.best_value
            tracker.best_value = value
            tracker.best_point = point
        end
    end
    return predictions
end

count_near_zero_dimensions(theta::AbstractVector, tol::Real) = count(abs.(theta) .<= tol)

function run_eki_method(problem::NonlinearComparisonProblem, ctx, cfg::NonlinearComparisonConfig,
        filter_type::String, label::String, seed::Int;
        correction_mode::String="sequential")
    Random.seed!(seed)

    inflation = (filter_type != "DEKI")
    budget_name = filter_type == "dropout-EAKI" ?
        "dropout-EAKI-$(correction_mode)" : filter_type
    n_iter = iterations_for_eval_budget(budget_name, cfg)
    tracker = ForwardTracker(problem.theta_dim)
    counted_forward(theta) = tracked_forward(problem, tracker, theta)
    use_mean_line_search = filter_type == "dropout-EAKI" && cfg.mean_line_search
    joint_weight_mode = filter_type == "dropout-EAKI" && correction_mode == "joint" ?
        cfg.joint_weight_mode : "fixed"

    evaluation_history = Int[]
    best_value_history = Float64[]
    best_point_history = Vector{Float64}[]
    mean_point_history = Vector{Float64}[]
    mean_value_history = Float64[]
    covariance_history = Float64[]
    ensemble_history_at_callback = Matrix{Float64}[]

    function record_iteration(obj, iteration)
        ensemble = copy(obj.θ[end])
        if obj.mean_line_search
            mean_point = vec(obj.mean_state)
            mean_value = obj.mean_prediction_cache === nothing ?
                ctx.phi(mean_point) :
                observation_objective(problem, vec(obj.mean_prediction_cache))
            covariance_value = norm(obj.anomaly_state * obj.anomaly_state')
        else
            mean_point = vec(mean(ensemble, dims=2))
            mean_value = ctx.phi(mean_point)
            covariance_value = norm(ensemble_covariance(ensemble))
        end
        push!(evaluation_history, tracker.evaluations)
        push!(best_value_history, tracker.best_value)
        push!(best_point_history, copy(tracker.best_point))
        push!(mean_point_history, mean_point)
        push!(mean_value_history, mean_value)
        push!(covariance_history, covariance_value)
        push!(ensemble_history_at_callback, ensemble)
        return nothing
    end

    ekiobj = EKI_Run(
        counted_forward,
        copy(problem.theta0),
        problem.sigma_y,
        problem.y;
        filter_type=filter_type,
        Δτ = cfg.inflation_dt,
        N_iter=n_iter,
        dropout_rate=cfg.dropout_rate,
        inflation=inflation,
        forward_parallel=true,
        dropout_correction_mode=correction_mode,
        joint_dropout_weight=cfg.joint_dropout_weight,
        joint_weight_mode=joint_weight_mode,
        joint_weight_min=cfg.joint_weight_min,
        joint_weight_max=cfg.joint_weight_max,
        joint_weight_smoothing=cfg.joint_weight_smoothing,
        mean_line_search=use_mean_line_search,
        mean_line_search_contraction=cfg.mean_line_search_contraction,
        mean_line_search_armijo_c=cfg.mean_line_search_armijo_c,
        iteration_callback=record_iteration,
    )

    valid = findall(<=(cfg.max_eval), evaluation_history)
    isempty(valid) && error("Forward budget is smaller than the initial ensemble cost")
    last_valid = valid[end]
    evaluation_axis = evaluation_history[valid]
    best_values = best_value_history[valid]
    best_points = best_point_history[valid]
    mean_points = mean_point_history[valid]
    mean_values = mean_value_history[valid]
    covariance_norm = covariance_history[valid]
    final_points = ensemble_history_at_callback[last_valid]
    final_mean_point = mean_points[end]
    joint_history_length = min(last_valid - 1, length(ekiobj.joint_weight_history))
    return (;
        label,
        method=budget_name,
        n_iter=last_valid - 1,
        max_eval=cfg.max_eval,
        evals_per_iteration=minimum_evals_per_iteration(budget_name, cfg.n_ens),
        final_mean_point,
        near_zero_count=count_near_zero_dimensions(final_mean_point, cfg.near_zero_tol),
        near_zero_tol=cfg.near_zero_tol,
        object=ekiobj,
        final_points,
        representative_points=best_points,
        optimization_error=best_values,
        mean_optimization_error=mean_values,
        covariance_norm,
        x_axis=evaluation_axis,
        mean_step_gamma=use_mean_line_search ?
            copy(ekiobj.mean_step_γ[1:last_valid-1]) : Float64[],
        mean_step_source=use_mean_line_search ?
            copy(ekiobj.mean_step_source[1:last_valid-1]) : String[],
        mean_step_trials=use_mean_line_search ?
            copy(ekiobj.mean_step_trials[1:last_valid-1]) : Int[],
        mean_step_backtracks=use_mean_line_search ?
            copy(ekiobj.mean_step_backtracks[1:last_valid-1]) : Int[],
        mean_step_reduction_ratio=use_mean_line_search ?
            copy(ekiobj.mean_step_reduction_ratio[1:last_valid-1]) : Float64[],
        joint_weight=correction_mode == "joint" ?
            copy(ekiobj.joint_weight_history[1:joint_history_length]) : Float64[],
        joint_trust_ratio=correction_mode == "joint" ?
            copy(ekiobj.joint_trust_ratio_history[1:joint_history_length]) : Float64[],
        joint_predicted_reduction=correction_mode == "joint" ?
            copy(ekiobj.joint_predicted_reduction_history[1:joint_history_length]) : Float64[],
        joint_actual_reduction=correction_mode == "joint" ?
            copy(ekiobj.joint_actual_reduction_history[1:joint_history_length]) : Float64[],
    )
end

function covariance_norm_from_columns(points::AbstractMatrix)
    size(points, 2) <= 1 && return 0.0
    return norm(ensemble_covariance(points))
end

function run_cma_es(problem::NonlinearComparisonProblem, ctx, cfg::NonlinearComparisonConfig, seed::Int)
    x0 = vec(mean(problem.theta0, dims=2))
    n_iter = max(fld(cfg.max_eval - 1, cfg.n_ens), 0)
    tracker = ForwardTracker(problem.theta_dim)
    function objective(x)
        prediction = tracked_forward(problem, tracker, Vector{Float64}(x))
        return observation_objective(problem, vec(prediction))
    end

    coordinate_variance = vec(var(problem.theta0, dims=2, corrected=true))
    sigma0 = sqrt(mean(coordinate_variance))
    result = run_cmaes(
        objective, x0;
        sigma0=sigma0,
        popsize=cfg.n_ens,
        max_iter=n_iter,
        seed=seed,
    )

    evaluation_axis = [1 + (i - 1) * cfg.n_ens for i in eachindex(result.best_f_history)]
    valid = findall(<=(cfg.max_eval), evaluation_axis)
    best_values = result.best_f_history[valid]
    best_points = result.best_history[valid]
    covariance_norm = result.covariance_history[valid]
    mean_points = result.means[valid]
    mean_values = [ctx.phi(point) for point in mean_points]
    final_mean_point = mean_points[end]
    return (;
        label="CMA-ES",
        method="CMA-ES",
        n_iter=length(valid) - 1,
        max_eval=cfg.max_eval,
        evals_per_iteration=cfg.n_ens,
        final_mean_point,
        near_zero_count=count_near_zero_dimensions(final_mean_point, cfg.near_zero_tol),
        near_zero_tol=cfg.near_zero_tol,
        object=result,
        final_points=reshape(copy(result.best_x), :, 1),
        representative_points=best_points,
        optimization_error=best_values,
        mean_optimization_error=mean_values,
        covariance_norm,
        x_axis=evaluation_axis[valid],
    )
end

function run_problem_comparison(problem::NonlinearComparisonProblem, cfg::NonlinearComparisonConfig, seed::Int)
    ctx = objective_context(problem)
    runs = NamedTuple[]
    push!(runs, run_eki_method(problem, ctx, cfg, "EAKI", "Inflation EAKI", seed))
    line_search_suffix = cfg.mean_line_search ? " + mean LS" : ""
    adaptive_weight_suffix = cfg.joint_weight_mode == "adaptive" ?
        " + adaptive w" : ""
    push!(runs, run_eki_method(problem, ctx, cfg, "dropout-EAKI",
        "Sequential projected dropout EAKI$(line_search_suffix)", seed + 1;
        correction_mode="sequential"))
    push!(runs, run_eki_method(problem, ctx, cfg, "dropout-EAKI",
        "Joint projected EAKI$(line_search_suffix)$(adaptive_weight_suffix)", seed + 1;
        correction_mode="joint"))
    push!(runs, run_eki_method(problem, ctx, cfg, "DEKI", "DEKI", seed + 1))
    push!(runs, run_cma_es(problem, ctx, cfg, seed + 1))
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
    fig, axs = PyPlot.subplots(length(problems), 4;
        figsize=(23, 4.8 * length(problems)), squeeze=false)

    for (row, problem) in enumerate(problems)
        runs = results[problem.name]
        plot_heatmap_and_final_points!(axs[row, 1], problem, runs)
        plot_curve_panel!(axs[row, 2], runs, :optimization_error,
            "best evaluated objective", "best Phi(theta)")
        plot_curve_panel!(axs[row, 3], runs, :mean_optimization_error,
            "ensemble/distribution mean objective", "Phi(mean)")
        plot_curve_panel!(axs[row, 4], runs, :covariance_norm,
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

    for name in cfg.function_names
        seed = problem_seed(cfg.seed, name)
        problem = make_nonlinear_problem(name, cfg, seed)
        @info "Running nonlinear test-function comparison" function_name=problem.name dim=problem.theta_dim max_eval=cfg.max_eval n_ens=cfg.n_ens
        push!(problems, problem)
        runs = run_problem_comparison(problem, cfg, seed)
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
