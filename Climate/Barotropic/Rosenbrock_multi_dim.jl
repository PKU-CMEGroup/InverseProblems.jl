using LinearAlgebra
using Random
using Serialization
using Statistics
using PyPlot

include("../../Inversion/InflationEKI.jl")

const THETA_FIELD = Symbol("theta")
const SIGMA_Y_SQRT_FIELD = Symbol("Sigma_y_sqrt")

function theta_history(ekiobj, theta_dim=nothing)
    theta = hasproperty(ekiobj, THETA_FIELD) ?
        getfield(ekiobj, THETA_FIELD) :
        getfield(ekiobj, Symbol("θ"))

    if theta_dim === nothing || size(theta[1], 1) == theta_dim
        return theta
    elseif size(theta[1], 2) == theta_dim
        return [permutedims(theta_i) for theta_i in theta]
    else
        error("Cannot determine ensemble orientation for theta size $(size(theta[1]))")
    end
end

function sigma_y_sqrt(ekiobj::EKIObj)
    return hasproperty(ekiobj, SIGMA_Y_SQRT_FIELD) ?
        getfield(ekiobj, SIGMA_Y_SQRT_FIELD) :
        getfield(ekiobj, Symbol("Σ_y_sqrt"))
end

function check_rosenbrock_dimension(theta_dim::Int)
    theta_dim >= 2 || throw(
        ArgumentError(
            "The chained Rosenbrock function requires theta_dim >= 2, " *
            "but theta_dim = $theta_dim."
        )
    )
    return nothing
end

function rosenbrock_observation(theta_dim::Int)
    check_rosenbrock_dimension(theta_dim)
    observation_dim = 2 * (theta_dim - 1)
    y = zeros(Float64, observation_dim)
    y[2:2:end] .= 1.0
    return y
end

function rosenbrock_forward(theta::AbstractVector, c1::Float64)
    theta_dim = length(theta)
    check_rosenbrock_dimension(theta_dim)

    y_pred = zeros(Float64, 2 * (theta_dim - 1))
    for i in 1:(theta_dim - 1)
        j = 2*i - 1
        y_pred[j] = 10.0 * (theta[i + 1] - c1 * theta[i]^2)
        y_pred[j+1] = theta[i]
    end
    return y_pred
end

function rosenbrock_augmented_forward(theta::AbstractVector, c1::Float64)
    return vcat(rosenbrock_forward(theta, c1), theta)
end

function phi_rosenbrock(theta::AbstractVector, c1::Float64, c2::Float64)
    residual = rosenbrock_forward(theta, c1) .- rosenbrock_observation(length(theta))
    return 0.5 * dot(residual, residual) / c2
end

function phi_posterior_rosenbrock(
    theta::AbstractVector,
    c1::Float64,
    c2::Float64;
    prior_std::Float64=10.0,
)
    return phi_rosenbrock(theta, c1, c2) + 0.5 * dot(theta, theta) / prior_std^2
end

function normalize_message(message)
    max_message = maximum(message)
    return max_message == 0.0 ? message : message ./ max_message
end

function rosenbrock_reference_grid(;
    c1::Float64,
    c2::Float64=1.0,
    prior_std::Float64=10.0,
    theta_dim::Int=100,
    xlim::Tuple{Float64,Float64}=(-8.0, 8.0),
    ylim::Tuple{Float64,Float64}=(-4.0, 16.0),
    n_grid::Int=700,
)
    check_rosenbrock_dimension(theta_dim)

    xs = collect(range(xlim[1], xlim[2], length=n_grid))
    ys = collect(range(ylim[1], ylim[2], length=n_grid))
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    x_grid = repeat(xs, 1, n_grid)
    y_grid = repeat(ys', n_grid, 1)

    prior_x = exp.(-0.5 .* xs.^2 ./ prior_std^2)
    prior_y = exp.(-0.5 .* ys.^2 ./ prior_std^2)
    node_x = exp.(-0.5 .* (1.0 .- xs).^2 ./ c2)
    node_y = exp.(-0.5 .* (1.0 .- ys).^2 ./ c2)

    transition_xy = exp.(-50.0 / c2 .* (y_grid .- c1 .* x_grid.^2).^2)

    current_state = reshape(ys, n_grid, 1)
    next_state = reshape(ys, 1, n_grid)
    transition_yy = exp.(-50.0 / c2 .* (next_state .- c1 .* current_state.^2).^2)

    backward = Vector{Vector{Float64}}(undef, theta_dim)
    backward[theta_dim] = ones(Float64, n_grid)
    for i in (theta_dim - 1):-1:2
        backward[i] = node_y .* (transition_yy * (prior_y .* backward[i + 1])) .* dy
        backward[i] = normalize_message(backward[i])
    end
    backward[1] = node_x .* (transition_xy * (prior_y .* backward[2])) .* dy
    backward[1] = normalize_message(backward[1])

    forward_messages = Vector{Vector{Float64}}(undef, theta_dim)
    forward_messages[1] = ones(Float64, n_grid)
    forward_messages[2] = (transition_xy' * (prior_x .* node_x)) .* dx
    forward_messages[2] = normalize_message(forward_messages[2])
    for i in 2:(theta_dim - 1)
        forward_messages[i + 1] = (transition_yy' * (forward_messages[i] .* prior_y .* node_y)) .* dy
        forward_messages[i + 1] = normalize_message(forward_messages[i + 1])
    end

    weights = transition_xy .* reshape(prior_x .* node_x, n_grid, 1) .* reshape(prior_y .* backward[2], 1, n_grid)
    density = weights ./ (sum(weights) * dx * dy)
    mean_pair_ref = [
        sum(x_grid .* weights) / sum(weights),
        sum(y_grid .* weights) / sum(weights),
    ]
    potential = -log.(max.(weights, floatmin(Float64)))
    map_idx = argmin(potential)
    map_pair_ref = [x_grid[map_idx], y_grid[map_idx]]

    mean_ref = zeros(Float64, theta_dim)
    marginal_1 = prior_x .* backward[1]
    mean_ref[1] = sum(xs .* marginal_1) / sum(marginal_1)
    for i in 2:theta_dim
        marginal_i = forward_messages[i] .* prior_y .* backward[i]
        mean_ref[i] = sum(ys .* marginal_i) / sum(marginal_i)
    end
    map_ref = ones(Float64, theta_dim)

    return (
        c1=c1,
        c2=c2,
        prior_std=prior_std,
        theta_dim=theta_dim,
        xs=xs,
        ys=ys,
        x_grid=x_grid,
        y_grid=y_grid,
        density=density,
        potential=potential,
        mean_pair_ref=mean_pair_ref,
        map_pair_ref=map_pair_ref,
        mean_ref=mean_ref,
        map_ref=map_ref,
    )
end

function rosenbrock_observation_model(c1::Float64, c2::Float64, prior_std::Float64, theta_dim::Int)
    check_rosenbrock_dimension(theta_dim)
    observation_dim = 2 * (theta_dim - 1)
    y_aug = vcat(rosenbrock_observation(theta_dim), zeros(Float64, theta_dim))
    sigma_y = Array(Diagonal(vcat(fill(c2, observation_dim), fill(prior_std^2, theta_dim))))
    forward(theta) = rosenbrock_augmented_forward(theta, c1)
    return forward, y_aug, sigma_y
end

function ensemble_cov(theta_ens::AbstractMatrix)
    theta_mean = mean(theta_ens, dims=2)
    anomalies = (theta_ens .- theta_mean) ./ sqrt(size(theta_ens, 2) - 1)
    return anomalies * anomalies'
end

function ensemble_means(ekiobj::EKIObj)
    return [dropdims(mean(theta_ens, dims=2), dims=2) for theta_ens in theta_history(ekiobj)]
end

function rosenbrock_run_errors(ekiobj, problem)
    means = [dropdims(mean(theta_ens, dims=2), dims=2) for theta_ens in theta_history(ekiobj, problem.theta_dim)]
    theta_ens_hist = theta_history(ekiobj, problem.theta_dim)
    n_hist = length(means)

    mean_errors = zeros(Float64, n_hist)
    map_errors = zeros(Float64, n_hist)
    observation_errors = zeros(Float64, n_hist)
    potentials = zeros(Float64, n_hist)
    covariance_norms = zeros(Float64, n_hist)
    y_obs = rosenbrock_observation(problem.theta_dim)

    for i in 1:n_hist
        mean_errors[i] = norm(means[i] - problem.mean_ref) / norm(problem.mean_ref)
        map_errors[i] = norm(means[i] - problem.map_ref) / norm(problem.map_ref)
        observation_errors[i] = norm(rosenbrock_forward(means[i], problem.c1) - y_obs) / norm(y_obs)
        potentials[i] = phi_posterior_rosenbrock(
            means[i],
            problem.c1,
            problem.c2;
            prior_std=problem.prior_std,
        )
        covariance_norms[i] = norm(ensemble_cov(theta_ens_hist[i]))
    end

    return (
        means=means,
        mean_errors=mean_errors,
        map_errors=map_errors,
        observation_errors=observation_errors,
        potentials=potentials,
        covariance_norms=covariance_norms,
    )
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

function ensure_parent_dir(path::String)
    dir = dirname(path)
    if !isempty(dir)
        mkpath(dir)
    end
end

function plot_rosenbrock_summary(problem, run_results, save_file::String)
    fig, axs = PyPlot.subplots(nrows=length(run_results), ncols=4, figsize=(20, 4.5 * length(run_results)), squeeze=false)

    for (r_ind, row) in enumerate(run_results)
        ax_shape = axs[r_ind, 1]
        ax_shape.contour(problem.x_grid, problem.y_grid, problem.density, 12, colors="gray", linewidths=0.9)
        ax_shape.scatter(problem.mean_ref[1], problem.mean_ref[2], marker="*", s=110, color="black", label="reference mean")

        for (f_ind, run) in enumerate(row.cases)
            errors = run.errors
            iters = 0:length(errors.mean_errors)-1
            style = plot_style(f_ind)
            markevery = max(1, div(length(iters), 8))

            theta_final = theta_history(run.ekiobj, problem.theta_dim)[end]
            ax_shape.scatter(theta_final[1, :], theta_final[2, :], s=12, alpha=0.45, color=style.color, label=run.label)

            axs[r_ind, 2].semilogy(
                iters,
                errors.mean_errors;
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                fillstyle="none",
                markevery=markevery,
                label=run.label,
            )
            axs[r_ind, 3].semilogy(
                iters,
                errors.observation_errors;
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                fillstyle="none",
                markevery=markevery,
                label=run.label,
            )
            axs[r_ind, 4].semilogy(
                iters,
                errors.potentials;
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                fillstyle="none",
                markevery=markevery,
                label=run.label,
            )
        end

        ax_shape.set_title(row.label * ", c = $(problem.c1)")
        ax_shape.set_xlabel("theta_1")
        ax_shape.set_ylabel("theta_2")
        ax_shape.set_xlim(extrema(problem.xs))
        ax_shape.set_ylim(extrema(problem.ys))
        ax_shape.legend(fontsize=8, loc="best")

        axs[r_ind, 2].set_title("theta error")
        axs[r_ind, 2].set_xlabel("Iterations")
        axs[r_ind, 2].set_ylabel("Rel. error to E[theta | y]")

        axs[r_ind, 3].set_title("y error")
        axs[r_ind, 3].set_xlabel("Iterations")
        axs[r_ind, 3].set_ylabel("Rel. observation error")

        axs[r_ind, 4].set_title("Phi_R")
        axs[r_ind, 4].set_xlabel("Iterations")
        axs[r_ind, 4].set_ylabel("Phi_R(theta mean)")

        for col in 2:4
            axs[r_ind, col].grid()
            axs[r_ind, col].legend(fontsize=8, loc="best")
        end
    end

    fig.tight_layout()
    ensure_parent_dir(save_file)
    fig.savefig(save_file, dpi=180)
    PyPlot.close(fig)
    return save_file
end

function run_rosenbrock_inflation_eki(;
    c1::Float64=1.0,
    c2::Float64=1.0,
    prior_std::Float64=10.0,
    theta_dim::Int=100,
    n_grid::Int=700,
    n_ens::Int=50,
    n_iter::Int=500,
    inflation_dt::Vector{Float64}=[0.2, 0.8],
    dropout_rate::Float64=0.5,
    filter_types::Vector{String}=["ETKI", "dropout-ETKI", "DEKI"],
    seed::Int=42,
    save_prefix::String=joinpath(@__DIR__, "Figs", "Rosenbrock_InflationEKI", "$(join(filter_types, "_"))_c$(c1)_prior$(prior_std)_dim$(theta_dim)_ens$(n_ens)_iter$(n_iter)_dt$(inflation_dt)_dropout$(dropout_rate)"),
    output_file::String=save_prefix * ".jls",
    save_plots::Bool=true,
)
    all(0.0 < dt < 1.0 for dt in inflation_dt) || error("inflation_dt must satisfy 0 < inflation_dt < 1")
    check_rosenbrock_dimension(theta_dim)

    problem = rosenbrock_reference_grid(
        c1=c1,
        c2=c2,
        prior_std=prior_std,
        theta_dim=theta_dim,
        n_grid=n_grid,
    )

    forward, y_aug, sigma_y = rosenbrock_observation_model(problem.c1, c2, prior_std, theta_dim)
    Random.seed!(seed)
    theta0 = prior_std .* randn(theta_dim, n_ens)
    run_results = NamedTuple[]

    no_inflation_cases = NamedTuple[]
    for (f_ind, filter_type) in enumerate(filter_types)
        Random.seed!(seed + f_ind)
        @info "Running Rosenbrock EKI without inflation" filter_type c1=problem.c1 n_ens n_iter
        ekiobj = EKI_Run(
            forward,
            copy(theta0),
            sigma_y,
            y_aug;
            filter_type=filter_type,
            Δτ=first(inflation_dt),
            N_iter=n_iter,
            dropout_rate=dropout_rate,
            inflation=false,
        )
        errors = rosenbrock_run_errors(ekiobj, problem)
        push!(no_inflation_cases, (
            ekiobj=ekiobj,
            errors=errors,
            filter_type=filter_type,
            inflation_dt=nothing,
            label=filter_type,
            c1=problem.c1,
        ))
    end
    push!(run_results, (
        label="no inflation",
        inflation_dt=nothing,
        cases=no_inflation_cases,
    ))

    for (dt_ind, dt) in enumerate(inflation_dt)
        inflation_cases = NamedTuple[]
        for (f_ind, filter_type) in enumerate(filter_types)
            Random.seed!(seed + 1000dt_ind + f_ind)
            @info "Running Rosenbrock inflation EKI" filter_type c1=problem.c1 n_ens n_iter inflation_dt=dt dropout_rate
            ekiobj = EKI_Run(
                forward,
                copy(theta0),
                sigma_y,
                y_aug;
                filter_type=filter_type,
                Δτ=dt,
                N_iter=n_iter,
                dropout_rate=dropout_rate,
                inflation=true,
            )
            errors = rosenbrock_run_errors(ekiobj, problem)
            push!(inflation_cases, (
                ekiobj=ekiobj,
                errors=errors,
                filter_type=filter_type,
                inflation_dt=dt,
                label=filter_type,
                c1=problem.c1,
            ))

            @info "Finished Rosenbrock run" filter_type c1=problem.c1 inflation_dt=dt final_mean=errors.means[end] rel_mean_error=errors.mean_errors[end]
        end
        push!(run_results, (
            label="inflation dt=$(dt)",
            inflation_dt=dt,
            cases=inflation_cases,
        ))
    end

    plot_files = String[]
    if save_plots
        push!(plot_files, plot_rosenbrock_summary(problem, run_results, save_prefix * ".png"))
    end

    result = (
        problem=problem,
        run_results=run_results,
        filter_types=filter_types,
        c2=c2,
        prior_std=prior_std,
        theta_dim=theta_dim,
        n_grid=n_grid,
        n_ens=n_ens,
        n_iter=n_iter,
        inflation_dt=inflation_dt,
        dropout_rate=dropout_rate,
        output_file=output_file,
        plot_files=plot_files,
    )

    ensure_parent_dir(output_file)
    serialize(output_file, result)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_rosenbrock_inflation_eki()
    @info "Finished Rosenbrock inflation EKI" result.output_file result.plot_files
end
