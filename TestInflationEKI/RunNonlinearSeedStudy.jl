using Logging
using Serialization
using Statistics

include(joinpath(@__DIR__, "CompareNonlinearTestFunctions.jl"))

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes")
env_int_list(name, default) = parse.(Int, strip.(split(
    get(ENV, name, join(default, ',')), ',')))

const DEFAULT_SEED_STUDY_FUNCTIONS = ["monotone_cubic", "rosenbrock", "rastrigin"]
const DEFAULT_SEED_STUDY_ENSEMBLE_SIZES = [20, 50]
const SEED_STUDY_GRID_SIZE = 100
const SEED_STUDY_METHOD_LABELS = ["IPD-EAKI", "DEKI", "CMA-ES"]

function seed_study_eaki_config(cfg::NonlinearComparisonConfig)
    names = fieldnames(NonlinearComparisonConfig)
    values = NamedTuple{names}(Tuple(getfield(cfg, name) for name in names))
    updated = merge(values, (
        joint_weight_mode="fixed",
        joint_dropout_weight=1.0,
        mean_line_search=false,
    ))
    return NonlinearComparisonConfig(; updated...)
end

function run_seed_study_methods(problem::NonlinearComparisonProblem,
                                cfg::NonlinearComparisonConfig, seed::Int)
    ctx = objective_context(problem)
    eaki_cfg = seed_study_eaki_config(cfg)
    return [
        run_eki_method(problem, ctx, eaki_cfg, "dropout-EAKI",
            SEED_STUDY_METHOD_LABELS[1], seed; correction_mode="joint"),
        run_eki_method(problem, ctx, cfg, "DEKI",
            SEED_STUDY_METHOD_LABELS[2], seed),
        run_cma_es(problem, ctx, cfg, seed),
    ]
end

function carry_forward_curve(x_axis, values, grid)
    return [values[max(searchsortedlast(x_axis, point), 1)] for point in grid]
end

function make_seed_trajectory_figure(function_names, trajectories, snapshots,
                                     ensemble_sizes, max_eval)
    n_heatmaps = length(ensemble_sizes)
    n_rows = length(function_names)
    fig_height = 4.4 * n_rows
    fig, axs = PyPlot.subplots(n_rows, n_heatmaps + 2;
        figsize=(4.8 * (n_heatmaps + 2), fig_height), squeeze=false)
    ensemble_linestyles = ("-", "--", "-.", ":")

    for (row_index, function_name) in enumerate(function_names)
        for (block_index, n_ens) in enumerate(ensemble_sizes)
            grid = collect(range(n_ens, max_eval, length=160))
            function_runs = [item for item in trajectories if
                item.function_name == function_name && item.n_ens == n_ens]
            snapshot = snapshots[(function_name, n_ens)]
            heatmap_ax = axs[row_index, block_index]
            plot_heatmap_and_final_points!(heatmap_ax, snapshot.problem,
                snapshot.runs; point_size=40, point_alpha=0.9,
                point_edgecolor="black", point_linewidth=0.5,
                rasterize_heatmap=true)
            heatmap_ax.set_title("Objective slice (J=$(n_ens))"; fontsize=14)
            heatmap_ax.set_xlabel("")
            heatmap_ax.set_ylabel("")
            heatmap_ax.tick_params(axis="both", labelsize=11)
            legend = heatmap_ax.get_legend()
            legend !== nothing && legend.remove()

            for (method_index, method) in enumerate(SEED_STUDY_METHOD_LABELS)
                selected = [item for item in function_runs if item.method == method]
                style = plot_style(method_index)
                for (column_index, field) in enumerate((:best_objective, :covariance_norm))
                    curves = hcat([carry_forward_curve(item.x_axis,
                        getproperty(item, field), grid) for item in selected]...)'
                    center = [median(view(curves, :, j)) for j in axes(curves, 2)]
                    lower = [quantile(view(curves, :, j), 0.25) for j in axes(curves, 2)]
                    upper = [quantile(view(curves, :, j), 0.75) for j in axes(curves, 2)]
                    ax = axs[row_index, n_heatmaps + column_index]
                    ax.semilogy(grid, positive_for_log(center);
                        color=style.color,
                        linestyle=ensemble_linestyles[mod1(block_index,
                            length(ensemble_linestyles))],
                        linewidth=1.7, label="$(method) (J=$(n_ens))")
                    ax.fill_between(grid, positive_for_log(lower), positive_for_log(upper);
                        color=style.color, alpha=0.1, linewidth=0)
                end
            end
        end
        axs[row_index, n_heatmaps + 1].set_title("Best evaluated objective"; fontsize=14)
        axs[row_index, n_heatmaps + 2].set_title("Covariance norm"; fontsize=14)
        for column in (n_heatmaps + 1):(n_heatmaps + 2)
            axs[row_index, column].set_xlabel("forward model evaluations"; fontsize=12)
            axs[row_index, column].set_ylabel("")
            axs[row_index, column].tick_params(axis="both", labelsize=11)
            axs[row_index, column].grid(true, alpha=0.25)
        end
    end
    reference_handles, _ = axs[1, 1].get_legend_handles_labels()
    handles, labels = axs[1, n_heatmaps + 1].get_legend_handles_labels()
    pushfirst!(handles, reference_handles[1])
    pushfirst!(labels, "reference minimizer")
    fig.legend(handles, labels; loc="upper center", ncol=length(labels),
        fontsize=13, frameon=false, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig_height), w_pad=1.5)
    return fig, axs
end

function plot_seed_trajectory_summary(function_names, trajectories, snapshots,
                                      ensemble_sizes, max_eval, output_file)
    fig, _ = make_seed_trajectory_figure(function_names, trajectories, snapshots,
        ensemble_sizes, max_eval)
    try
        fig.savefig(output_file, dpi=180)
    finally
        PyPlot.close(fig)
    end
    return output_file
end

function run_nonlinear_seed_study(;
        function_names::Vector{String}=copy(DEFAULT_SEED_STUDY_FUNCTIONS),
        theta_dim::Int=100,
        ensemble_sizes::Vector{Int}=copy(DEFAULT_SEED_STUDY_ENSEMBLE_SIZES),
        max_eval::Int=10_000,
        n_seeds::Int=10,
        seed::Int=2026091500,
        inflation_dt::Float64=0.5,
        dropout_rate::Float64=0.5,
        output_dir::String=joinpath(@__DIR__, "Results"),
        save_plots::Bool=false,
        save_summary_plot::Bool=true,
        verbose::Bool=false)
    n_seeds >= 1 || error("n_seeds must be positive")
    !isempty(ensemble_sizes) || error("ensemble_sizes must not be empty")
    all(>(1), ensemble_sizes) || error("all ensemble sizes must exceed one")
    mkpath(output_dir)
    verbose || disable_logging(Logging.Info)

    rows = NamedTuple[]
    trajectories = NamedTuple[]
    snapshots = Dict{Tuple{String, Int}, NamedTuple}()
    for trial in 1:n_seeds
        trial_seed = seed + trial - 1
        for function_name in function_names
            for n_ens in ensemble_sizes
                cfg = NonlinearComparisonConfig(
                    function_names=copy(function_names),
                    theta_dim=theta_dim,
                    n_ens=n_ens,
                    max_eval=max_eval,
                    n_grid=SEED_STUDY_GRID_SIZE,
                    inflation_dt=inflation_dt,
                    dropout_rate=dropout_rate,
                    rastrigin_initial_center=1.0,
                    rastrigin_initial_half_width=4.0,
                    mean_line_search=false,
                    joint_dropout_weight=1.0,
                    joint_weight_mode="fixed",
                    seed=trial_seed,
                    save_plots=save_plots,
                    output_dir=output_dir,
                )
                current_problem_seed = problem_seed(trial_seed, function_name)
                problem = make_nonlinear_problem(function_name, cfg, current_problem_seed)
                runs = run_seed_study_methods(problem, cfg, current_problem_seed)
                trial == 1 && (snapshots[(function_name, n_ens)] = (; problem, runs))
                for run in runs
                    best_point = run.representative_points[end]
                    push!(rows, (;
                        function_name,
                        n_ens,
                        seed=trial_seed,
                        method=run.label,
                        evaluations=run.x_axis[end],
                        best_objective=run.optimization_error[end],
                        mean_objective=run.mean_optimization_error[end],
                        best_parameter_error=norm(best_point - problem.reference_mean),
                        mean_parameter_error=norm(run.final_mean_point - problem.reference_mean),
                    ))
                    push!(trajectories, (;
                        function_name,
                        n_ens,
                        seed=trial_seed,
                        method=run.label,
                        x_axis=copy(run.x_axis),
                        best_objective=copy(run.optimization_error),
                        mean_objective=copy(run.mean_optimization_error),
                        covariance_norm=copy(run.covariance_norm),
                    ))
                end
            end
        end
    end

    summaries = NamedTuple[]
    for function_name in function_names
        for n_ens in ensemble_sizes
            for method in SEED_STUDY_METHOD_LABELS
                selected = [row for row in rows if row.function_name == function_name &&
                            row.n_ens == n_ens && row.method == method]
                values = getproperty.(selected, :best_objective)
                push!(summaries, (;
                    function_name,
                    n_ens,
                    method,
                    median=median(values),
                    q25=quantile(values, 0.25),
                    q75=quantile(values, 0.75),
                    minimum=minimum(values),
                    maximum=maximum(values),
                ))
            end
        end
    end

    function_tag = join(slug.(function_names), "_")
    ensemble_tag = join(ensemble_sizes, "-")
    tag = "nonlinear_$(function_tag)_dim$(theta_dim)_ens$(ensemble_tag)_eval$(max_eval)_seeds$(n_seeds)"
    csv_file = joinpath(output_dir, tag * ".csv")
    summary_file = joinpath(output_dir, tag * "_summary.csv")
    result_file = joinpath(output_dir, tag * ".jls")
    plot_file = joinpath(output_dir, tag * "_curves.pdf")

    open(csv_file, "w") do io
        println(io, "function,n_ens,seed,method,evaluations,best_objective,mean_objective,best_parameter_error,mean_parameter_error")
        for row in rows
            println(io, join((row.function_name, row.n_ens, row.seed, row.method, row.evaluations,
                row.best_objective, row.mean_objective, row.best_parameter_error,
                row.mean_parameter_error), ','))
        end
    end

    open(summary_file, "w") do io
        println(io, "function,n_ens,method,median,q25,q75,min,max")
        for item in summaries
            println(io, join((item.function_name, item.n_ens, item.method, item.median, item.q25,
                item.q75, item.minimum, item.maximum), ','))
        end
    end

    save_summary_plot && plot_seed_trajectory_summary(
        function_names, trajectories, snapshots, ensemble_sizes, max_eval, plot_file)
    save_summary_plot || (plot_file = nothing)

    serialize(result_file, (; function_names, theta_dim, ensemble_sizes, max_eval,
        n_seeds, seed, inflation_dt, dropout_rate, rows, summaries,
        trajectories, plot_file))

    println("function,n_ens,method,median,q25,q75,min,max")
    for item in summaries
        println(join((item.function_name, item.n_ens, item.method, item.median, item.q25,
            item.q75, item.minimum, item.maximum), ','))
    end
    println("CSV: ", csv_file)
    println("Summary: ", summary_file)
    println("JLS: ", result_file)
    println("Plot: ", plot_file)
    return (; rows, summaries, trajectories, csv_file, summary_file, result_file, plot_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    names = split(get(ENV, "FUNCTION_NAMES", join(DEFAULT_SEED_STUDY_FUNCTIONS, ',')), ',')
    run_nonlinear_seed_study(
        function_names=String.(strip.(names)),
        theta_dim=env_int("THETA_DIM", 100),
        ensemble_sizes=env_int_list("N_ENS_VALUES",
            haskey(ENV, "N_ENS") ? [env_int("N_ENS", 50)] : DEFAULT_SEED_STUDY_ENSEMBLE_SIZES),
        max_eval=env_int("MAX_EVAL", 10_000),
        n_seeds=env_int("N_SEEDS", 10),
        seed=env_int("SEED", 2026091500),
        inflation_dt=env_float("INFLATION_DT", 0.5),
        dropout_rate=env_float("DROPOUT_RATE", 0.5),
        output_dir=get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "Results")),
        save_plots=env_bool("SAVE_PLOTS", false),
        save_summary_plot=env_bool("SAVE_SUMMARY_PLOT", true),
        verbose=env_bool("VERBOSE", false),
    )
end
