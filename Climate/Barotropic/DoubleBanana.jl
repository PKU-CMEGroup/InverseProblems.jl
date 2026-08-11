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

# ----- Double banana problem (2D) -----
# G(θ) = [log(λ(θ₂-θ₁²)² + (1-θ₁)²); θ₁; θ₂]
# y = [log(λ+1); 0; 0], ση = [0.3; 1; 1]

double_banana_observation(lambda::Float64) = [log(lambda + 1.0), 0.0, 0.0]

function double_banana_forward(theta::AbstractVector, lambda::Float64)
    r = lambda * (theta[2] - theta[1]^2)^2 + (1.0 - theta[1])^2
    return [log(r), theta[1], theta[2]]
end

function phi_double_banana(theta::AbstractVector, lambda::Float64, sigma_eta::AbstractVector)
    residual = double_banana_forward(theta, lambda) .- double_banana_observation(lambda)
    return 0.5 * sum((residual ./ sigma_eta).^2)
end

# ----- Reference grid -----

function double_banana_reference_grid(;
    lambda::Float64=100.0, sigma_eta::Vector{Float64}=[0.3, 1.0, 1.0],
    xlim::Tuple{Float64,Float64}=(-3.0, 3.0), ylim::Tuple{Float64,Float64}=(-3.0, 3.0),
    n_grid::Int=500,
)
    xs = collect(range(xlim[1], xlim[2], length=n_grid))
    ys = collect(range(ylim[1], ylim[2], length=n_grid))
    dx, dy = xs[2] - xs[1], ys[2] - ys[1]
    x_grid = repeat(xs, 1, n_grid)
    y_grid = repeat(ys', n_grid, 1)

    potential = zeros(Float64, n_grid, n_grid)
    for i in 1:n_grid, j in 1:n_grid
        potential[i, j] = phi_double_banana([x_grid[i, j], y_grid[i, j]], lambda, sigma_eta)
    end

    weights = exp.(-(potential .- minimum(potential)))
    density = weights ./ (sum(weights) * dx * dy)

    mean_ref = [sum(x_grid .* weights) / sum(weights), sum(y_grid .* weights) / sum(weights)]
    map_idx = argmin(potential)
    map_ref = [x_grid[map_idx], y_grid[map_idx]]

    return (lambda=lambda, sigma_eta=sigma_eta, theta_dim=2, xs=xs, ys=ys,
            x_grid=x_grid, y_grid=y_grid, density=density, potential=potential,
            mean_ref=mean_ref, map_ref=map_ref)
end

# ----- Observation model -----

function double_banana_observation_model(lambda::Float64, sigma_eta::Vector{Float64})
    y = double_banana_observation(lambda)
    sigma_y = Array(Diagonal(sigma_eta.^2))
    forward(theta) = double_banana_forward(theta, lambda)
    return forward, y, sigma_y
end

function ensemble_cov(theta_ens::AbstractMatrix)
    theta_mean = mean(theta_ens, dims=2)
    anomalies = (theta_ens .- theta_mean) ./ sqrt(size(theta_ens, 2) - 1)
    return anomalies * anomalies'
end

# ----- Error computation -----

function double_banana_run_errors(ekiobj_or_wrapper, problem; is_cmaes::Bool=false)
    theta_hist = theta_history(ekiobj_or_wrapper, problem.theta_dim)
    means = [dropdims(mean(t, dims=2), dims=2) for t in theta_hist]
    n_hist = length(means)

    map_errors = zeros(Float64, n_hist)
    observation_errors = zeros(Float64, n_hist)
    potentials = zeros(Float64, n_hist)
    y_obs = double_banana_observation(problem.lambda)

    for i in 1:n_hist
        map_errors[i] = norm(means[i] - problem.map_ref) / max(norm(problem.map_ref), 1e-8)
        y_pred = double_banana_forward(means[i], problem.lambda)
        observation_errors[i] = norm(y_pred - y_obs) / max(norm(y_obs), 1e-8)
        potentials[i] = phi_double_banana(means[i], problem.lambda, problem.sigma_eta)
    end

    return (means=means, map_errors=map_errors, observation_errors=observation_errors, potentials=potentials)
end

# ----- Plotting -----

function plot_double_banana_summary(problem, run_results, save_file::String)
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

        ax_shape.set_title(row.label * ", lambda = $(problem.lambda)")
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

function run_double_banana_inflation_eki(;
    lambda::Float64=100.0, sigma_eta::Vector{Float64}=[0.3, 1.0, 1.0],
    prior_std::Float64=2.0, n_grid::Int=500,
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
        save_prefix = joinpath(@__DIR__, "Figs", "DoubleBanana_InflationEKI",
            config_summary_label(method_configs, inflation_dt, dropout_rate))
    end
    if isempty(output_file)
        output_file = save_prefix * ".jls"
    end

    problem = double_banana_reference_grid(lambda=lambda, sigma_eta=sigma_eta, n_grid=n_grid)
    forward, y_aug, sigma_y = double_banana_observation_model(problem.lambda, sigma_eta)
    theta_dim = problem.theta_dim
    Random.seed!(seed)
    theta0 = prior_std .* randn(theta_dim, n_ens)
    run_results = NamedTuple[]

    dt = first(inflation_dt)

    # --- No inflation ---
    no_inf_cases = NamedTuple[]
    for (f_ind, config) in enumerate(method_configs)
        case = run_single_eki_method(forward, theta0, sigma_y, y_aug, config, problem;
            dt=dt, inflation=false, n_iter=n_iter, dropout_rate=dropout_rate,
            dropout_trust_radius=dropout_trust_radius,
            seed=seed + f_ind, error_fn=double_banana_run_errors)
        push!(no_inf_cases, (; case.ekiobj, case.errors, case.label, is_cmaes=false))
    end
    push!(run_results, (label="no inflation", cases=no_inf_cases))

    # --- Inflation ---
    for dt_val in inflation_dt
        inf_cases = NamedTuple[]
        for (f_ind, config) in enumerate(method_configs)
            case = run_single_eki_method(forward, theta0, sigma_y, y_aug, config, problem;
                dt=dt_val, inflation=true, n_iter=n_iter, dropout_rate=dropout_rate,
                dropout_trust_radius=dropout_trust_radius,
                seed=seed + 1000 + f_ind, error_fn=double_banana_run_errors)
            push!(inf_cases, (; case.ekiobj, case.errors, case.label, is_cmaes=false))
        end
        push!(run_results, (label="inflation dt=$(dt_val)", cases=inf_cases))
    end

    # --- CMA-ES ---
    if include_cmaes
        objective(theta) = phi_double_banana(theta, problem.lambda, sigma_eta)
        cma_case = run_cmaes_method(objective, dropdims(mean(theta0, dims=2), dims=2), problem;
            sigma0=1.0, n_iter=n_iter, seed=seed + 2000, error_fn=double_banana_run_errors)
        push!(run_results, (label="CMA-ES", cases=[(; cma_case.cma_result, errors=cma_case.errors,
                            label="CMA-ES", is_cmaes=true)]))
    end

    plot_files = String[]
    if save_plots
        push!(plot_files, plot_double_banana_summary(problem, run_results, save_prefix * ".png"))
    end

    result = (problem=problem, run_results=run_results, method_configs=method_configs,
              lambda=lambda, sigma_eta=sigma_eta, prior_std=prior_std,
              theta_dim=theta_dim, n_grid=n_grid, n_ens=n_ens, n_iter=n_iter,
              inflation_dt=inflation_dt, dropout_rate=dropout_rate,
              output_file=output_file, plot_files=plot_files)

    ensure_parent_dir(output_file)
    serialize(output_file, result)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_double_banana_inflation_eki()
    @info "Finished Double Banana inflation EKI" result.output_file result.plot_files
end
