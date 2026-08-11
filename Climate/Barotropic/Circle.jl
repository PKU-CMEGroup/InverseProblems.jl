using LinearAlgebra, Random, Serialization, Statistics
using PyPlot

include("InflationEKI_Benchmark_Helpers.jl")

const THETA_FIELD = Symbol("theta")

function theta_history(ekiobj_or_wrapper, theta_dim=nothing)
    if hasproperty(ekiobj_or_wrapper, THETA_FIELD)
        theta = getfield(ekiobj_or_wrapper, THETA_FIELD)
        if theta_dim !== nothing && size(theta[1], 2) == theta_dim
            return [permutedims(theta_i) for theta_i in theta]
        end
        return theta
    elseif hasproperty(ekiobj_or_wrapper, :θ)
        theta = getfield(ekiobj_or_wrapper, :θ)
        if theta_dim !== nothing && size(theta[1], 2) == theta_dim
            return [permutedims(theta_i) for theta_i in theta]
        end
        return theta
    else
        error("Cannot access theta history")
    end
end

# ----- Circle problem (2D) -----
# G(θ) = θ₁² + θ₂², y = 1, ση = 0.3, 增广观测 θ → N(0, prior_std²) 先验

circle_observation() = [1.0]
circle_forward(theta::AbstractVector) = [theta[1]^2 + theta[2]^2]
circle_augmented_forward(theta::AbstractVector) = vcat(circle_forward(theta), theta)

function phi_circle(theta::AbstractVector, c2::Float64)
    residual = circle_forward(theta) .- circle_observation()
    return 0.5 * dot(residual, residual) / c2
end

function phi_posterior_circle(theta::AbstractVector, c2::Float64; prior_std::Float64=3.0)
    return phi_circle(theta, c2) + 0.5 * dot(theta, theta) / prior_std^2
end

# ----- Reference grid -----

function circle_reference_grid(;
    c2::Float64=0.09, prior_std::Float64=3.0,
    xlim::Tuple{Float64,Float64}=(-3.0, 3.0), ylim::Tuple{Float64,Float64}=(-3.0, 3.0),
    n_grid::Int=500,
)
    xs = collect(range(xlim[1], xlim[2], length=n_grid))
    ys = collect(range(ylim[1], ylim[2], length=n_grid))
    dx, dy = xs[2] - xs[1], ys[2] - ys[1]
    x_grid = repeat(xs, 1, n_grid)
    y_grid = repeat(ys', n_grid, 1)

    r2 = x_grid.^2 .+ y_grid.^2
    potential = 0.5 .* ((1.0 .- r2).^2 ./ c2 .+ r2 ./ prior_std^2)

    weights = exp.(-(potential .- minimum(potential)))
    density = weights ./ (sum(weights) * dx * dy)

    mean_ref = [sum(x_grid .* weights) / sum(weights), sum(y_grid .* weights) / sum(weights)]
    map_idx = argmin(potential)
    map_ref = [x_grid[map_idx], y_grid[map_idx]]

    return (c2=c2, prior_std=prior_std, theta_dim=2, xs=xs, ys=ys,
            x_grid=x_grid, y_grid=y_grid, density=density, potential=potential,
            mean_ref=mean_ref, map_ref=map_ref)
end

# ----- Observation model -----

function circle_observation_model(c2::Float64, prior_std::Float64)
    y_aug = vcat(circle_observation(), zeros(Float64, 2))
    sigma_y = Array(Diagonal(vcat(fill(c2, 1), fill(prior_std^2, 2))))
    forward(theta) = circle_augmented_forward(theta)
    return forward, y_aug, sigma_y
end

function ensemble_cov(theta_ens::AbstractMatrix)
    theta_mean = mean(theta_ens, dims=2)
    anomalies = (theta_ens .- theta_mean) ./ sqrt(size(theta_ens, 2) - 1)
    return anomalies * anomalies'
end

# ----- Error computation -----

function circle_run_errors(ekiobj_or_wrapper, problem; is_cmaes::Bool=false)
    theta_hist = theta_history(ekiobj_or_wrapper, problem.theta_dim)
    means = [dropdims(mean(t, dims=2), dims=2) for t in theta_hist]
    n_hist = length(means)

    map_errors = zeros(Float64, n_hist)
    observation_errors = zeros(Float64, n_hist)
    potentials = zeros(Float64, n_hist)
    y_obs = circle_observation()

    for i in 1:n_hist
        map_errors[i] = norm(means[i] - problem.map_ref) / max(norm(problem.map_ref), 1e-8)
        y_pred = circle_forward(means[i])
        observation_errors[i] = norm(y_pred - y_obs) / max(norm(y_obs), 1e-8)
        potentials[i] = phi_posterior_circle(means[i], problem.c2; prior_std=problem.prior_std)
    end

    return (means=means, map_errors=map_errors, observation_errors=observation_errors, potentials=potentials)
end

# ----- Plotting -----

function plot_circle_summary(problem, run_results, save_file::String)
    n_rows = length(run_results)
    fig, axs = PyPlot.subplots(nrows=n_rows, ncols=4, figsize=(22, 4.5 * n_rows), squeeze=false)

    for (r_ind, row) in enumerate(run_results)
        ax_shape = axs[r_ind, 1]
        ax_shape.contour(problem.x_grid, problem.y_grid, problem.density, 12, colors="gray", linewidths=0.9)
        ax_shape.scatter(problem.mean_ref[1], problem.mean_ref[2], marker="*", s=110, color="black", label="ref. mean")

        for (f_ind, case) in enumerate(row.cases)
            errors = case.errors
            iters = 0:length(errors.map_errors)-1
            style = plot_style(f_ind)
            markevery = max(1, div(length(iters), 8))

            if !haskey(case, :is_cmaes) || !case.is_cmaes
                theta_hist = theta_history(case.ekiobj, problem.theta_dim)
                theta_final = theta_hist[end]
                ax_shape.scatter(theta_final[1, :], theta_final[2, :], s=12, alpha=0.45,
                                 color=style.color, label=case.label)
            end

            axs[r_ind, 2].semilogy(iters, errors.map_errors;
                color=style.color, linestyle=style.linestyle, marker=style.marker,
                fillstyle="none", markevery=markevery, label=case.label)
            axs[r_ind, 3].semilogy(iters, errors.observation_errors;
                color=style.color, linestyle=style.linestyle, marker=style.marker,
                fillstyle="none", markevery=markevery, label=case.label)
            axs[r_ind, 4].semilogy(iters, errors.potentials;
                color=style.color, linestyle=style.linestyle, marker=style.marker,
                fillstyle="none", markevery=markevery, label=case.label)
        end

        ax_shape.set_title(row.label)
        ax_shape.set_xlabel("theta_1"); ax_shape.set_ylabel("theta_2")
        ax_shape.set_xlim(extrema(problem.xs)); ax_shape.set_ylim(extrema(problem.ys))
        ax_shape.legend(fontsize=7, loc="best")

        axs[r_ind, 2].set_title("theta error (MAP)")
        axs[r_ind, 3].set_title("y error")
        axs[r_ind, 4].set_title("Phi_R")
        for col in 2:4
            axs[r_ind, col].set_xlabel("Iterations"); axs[r_ind, col].grid()
            axs[r_ind, col].legend(fontsize=7, loc="best")
        end
    end

    fig.tight_layout()
    ensure_parent_dir(save_file)
    fig.savefig(save_file, dpi=180)
    PyPlot.close(fig)
    return save_file
end

# ----- Main run function -----

function run_circle_inflation_eki(;
    c2::Float64=0.09, prior_std::Float64=3.0, n_grid::Int=500,
    n_ens::Int=50, n_iter::Int=500,
    inflation_dt::Vector{Float64}=[0.8], dropout_rate::Float64=0.5,
    dropout_trust_radius::Float64=5.0,
    method_configs::Vector{MethodConfig}=default_method_configs(),
    include_cmaes::Bool=true,
    seed::Int=42,
    save_prefix::String="",
    output_file::String="",
    save_plots::Bool=true,
)
    all(0.0 < dt < 1.0 for dt in inflation_dt) || error("inflation_dt must satisfy 0 < dt < 1")
    if isempty(save_prefix)
        save_prefix = joinpath(@__DIR__, "Figs", "Circle_InflationEKI",
            config_summary_label(method_configs, inflation_dt, dropout_rate))
    end
    if isempty(output_file)
        output_file = save_prefix * ".jls"
    end

    problem = circle_reference_grid(c2=c2, prior_std=prior_std, n_grid=n_grid)
    forward, y_aug, sigma_y = circle_observation_model(problem.c2, prior_std)
    theta_dim = problem.theta_dim
    Random.seed!(seed)
    theta0 = prior_std .* randn(theta_dim, n_ens)
    run_results = NamedTuple[]

    dt = first(inflation_dt)

    # --- No inflation runs ---
    no_inf_cases = NamedTuple[]
    for (f_ind, config) in enumerate(method_configs)
        case = run_single_eki_method(forward, theta0, sigma_y, y_aug, config, problem;
            dt=dt, inflation=false, n_iter=n_iter, dropout_rate=dropout_rate,
            dropout_trust_radius=dropout_trust_radius,
            seed=seed + f_ind, error_fn=circle_run_errors)
        push!(no_inf_cases, (; case.ekiobj, case.errors, case.label, is_cmaes=false))
    end
    push!(run_results, (label="no inflation", cases=no_inf_cases))

    # --- Inflation runs ---
    for dt_val in inflation_dt
        inf_cases = NamedTuple[]
        for (f_ind, config) in enumerate(method_configs)
            case = run_single_eki_method(forward, theta0, sigma_y, y_aug, config, problem;
                dt=dt_val, inflation=true, n_iter=n_iter, dropout_rate=dropout_rate,
                dropout_trust_radius=dropout_trust_radius,
                seed=seed + 1000 + f_ind, error_fn=circle_run_errors)
            push!(inf_cases, (; case.ekiobj, case.errors, case.label, is_cmaes=false))
        end
        push!(run_results, (label="inflation dt=$(dt_val)", cases=inf_cases))
    end

    # --- CMA-ES ---
    if include_cmaes
        objective(theta) = phi_posterior_circle(theta, problem.c2; prior_std=problem.prior_std)
        cma_case = run_cmaes_method(objective, dropdims(mean(theta0, dims=2), dims=2), problem;
            sigma0=1.0, n_iter=n_iter, seed=seed + 2000, error_fn=circle_run_errors)
        push!(run_results, (label="CMA-ES", cases=[(; cma_case.cma_result, errors=cma_case.errors,
                            label="CMA-ES", is_cmaes=true)]))
    end

    plot_files = String[]
    if save_plots
        push!(plot_files, plot_circle_summary(problem, run_results, save_prefix * ".png"))
    end

    result = (problem=problem, run_results=run_results, method_configs=method_configs,
              c2=c2, prior_std=prior_std, theta_dim=theta_dim, n_grid=n_grid,
              n_ens=n_ens, n_iter=n_iter, inflation_dt=inflation_dt, dropout_rate=dropout_rate,
              output_file=output_file, plot_files=plot_files)

    ensure_parent_dir(output_file)
    serialize(output_file, result)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_circle_inflation_eki()
    @info "Finished Circle inflation EKI" result.output_file result.plot_files
end
