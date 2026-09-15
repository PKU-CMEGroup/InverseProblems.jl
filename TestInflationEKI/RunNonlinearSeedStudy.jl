using Logging
using Serialization
using Statistics

include(joinpath(@__DIR__, "CompareNonlinearTestFunctions.jl"))

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes")

function carry_forward_curve(x_axis, values, grid)
    return [values[max(searchsortedlast(x_axis, point), 1)] for point in grid]
end

function plot_seed_trajectory_summary(function_names, trajectories, n_ens, max_eval, output_file)
    grid = collect(range(n_ens, max_eval, length=160))
    fig, axs = PyPlot.subplots(length(function_names), 2;
        figsize=(13, 4.2 * length(function_names)), squeeze=false)

    for (row_index, function_name) in enumerate(function_names)
        function_runs = [item for item in trajectories if item.function_name == function_name]
        methods = unique(getproperty.(function_runs, :method))
        for (method_index, method) in enumerate(methods)
            selected = [item for item in function_runs if item.method == method]
            style = plot_style(method_index)
            for (column_index, field) in enumerate((:best_objective, :mean_objective))
                curves = hcat([carry_forward_curve(item.x_axis,
                    getproperty(item, field), grid) for item in selected]...)'
                center = [median(view(curves, :, j)) for j in axes(curves, 2)]
                lower = [quantile(view(curves, :, j), 0.25) for j in axes(curves, 2)]
                upper = [quantile(view(curves, :, j), 0.75) for j in axes(curves, 2)]
                ax = axs[row_index, column_index]
                ax.semilogy(grid, positive_for_log(center);
                    color=style.color, linestyle=style.linestyle,
                    linewidth=1.7, label=method)
                ax.fill_between(grid, positive_for_log(lower), positive_for_log(upper);
                    color=style.color, alpha=0.14, linewidth=0)
            end
        end
        axs[row_index, 1].set_title("$(function_name): best evaluated objective")
        axs[row_index, 2].set_title("$(function_name): mean objective")
        for column_index in 1:2
            axs[row_index, column_index].set_xlabel("forward model evaluations")
            axs[row_index, column_index].set_ylabel("objective")
            axs[row_index, column_index].grid(true, alpha=0.25)
            axs[row_index, column_index].legend(fontsize=7, frameon=false)
        end
    end
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    PyPlot.close(fig)
    return output_file
end

function run_nonlinear_seed_study(;
        function_names::Vector{String}=["monotone_cubic", "paired_rosenbrock", "rastrigin"],
        theta_dim::Int=100,
        n_ens::Int=50,
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
    mkpath(output_dir)
    verbose || disable_logging(Logging.Info)

    rows = NamedTuple[]
    trajectories = NamedTuple[]
    for trial in 1:n_seeds
        trial_seed = seed + trial - 1
        cfg = NonlinearComparisonConfig(
            function_names=copy(function_names),
            theta_dim=theta_dim,
            n_ens=n_ens,
            max_eval=max_eval,
            n_grid=40,
            inflation_dt=inflation_dt,
            dropout_rate=dropout_rate,
            rastrigin_initial_center=1.0,
            rastrigin_initial_half_width=4.0,
            seed=trial_seed,
            save_plots=save_plots,
            output_dir=output_dir,
        )

        for function_name in function_names
            current_problem_seed = problem_seed(trial_seed, function_name)
            problem = make_nonlinear_problem(function_name, cfg, current_problem_seed)
            runs = run_problem_comparison(problem, cfg, current_problem_seed)
            for run in runs
                best_point = run.representative_points[end]
                push!(rows, (;
                    function_name,
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
                    seed=trial_seed,
                    method=run.label,
                    x_axis=copy(run.x_axis),
                    best_objective=copy(run.optimization_error),
                    mean_objective=copy(run.mean_optimization_error),
                ))
            end
        end
    end

    summaries = NamedTuple[]
    for function_name in function_names
        methods = unique(row.method for row in rows if row.function_name == function_name)
        for method in methods
            selected = [row for row in rows if
                        row.function_name == function_name && row.method == method]
            values = getproperty.(selected, :best_objective)
            push!(summaries, (;
                function_name,
                method,
                median=median(values),
                q25=quantile(values, 0.25),
                q75=quantile(values, 0.75),
                minimum=minimum(values),
                maximum=maximum(values),
            ))
        end
    end

    function_tag = join(slug.(function_names), "_")
    tag = "nonlinear_$(function_tag)_dim$(theta_dim)_ens$(n_ens)_eval$(max_eval)_seeds$(n_seeds)"
    csv_file = joinpath(output_dir, tag * ".csv")
    summary_file = joinpath(output_dir, tag * "_summary.csv")
    result_file = joinpath(output_dir, tag * ".jls")
    plot_file = joinpath(output_dir, tag * "_curves.png")

    open(csv_file, "w") do io
        println(io, "function,seed,method,evaluations,best_objective,mean_objective,best_parameter_error,mean_parameter_error")
        for row in rows
            println(io, join((row.function_name, row.seed, row.method, row.evaluations,
                row.best_objective, row.mean_objective, row.best_parameter_error,
                row.mean_parameter_error), ','))
        end
    end

    open(summary_file, "w") do io
        println(io, "function,method,median,q25,q75,min,max")
        for item in summaries
            println(io, join((item.function_name, item.method, item.median, item.q25,
                item.q75, item.minimum, item.maximum), ','))
        end
    end

    save_summary_plot && plot_seed_trajectory_summary(
        function_names, trajectories, n_ens, max_eval, plot_file)
    save_summary_plot || (plot_file = nothing)

    serialize(result_file, (; function_names, theta_dim, n_ens, max_eval,
        n_seeds, seed, inflation_dt, dropout_rate, rows, summaries,
        trajectories, plot_file))

    println("function,method,median,q25,q75,min,max")
    for item in summaries
        println(join((item.function_name, item.method, item.median, item.q25,
            item.q75, item.minimum, item.maximum), ','))
    end
    println("CSV: ", csv_file)
    println("Summary: ", summary_file)
    println("JLS: ", result_file)
    println("Plot: ", plot_file)
    return (; rows, summaries, trajectories, csv_file, summary_file, result_file, plot_file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    names = split(get(ENV, "FUNCTION_NAMES", "monotone_cubic,paired_rosenbrock,rastrigin"), ',')
    run_nonlinear_seed_study(
        function_names=String.(strip.(names)),
        theta_dim=env_int("THETA_DIM", 100),
        n_ens=env_int("N_ENS", 50),
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
