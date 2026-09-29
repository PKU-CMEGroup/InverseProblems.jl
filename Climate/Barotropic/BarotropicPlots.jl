using LinearAlgebra
using Statistics
using PyPlot

function barotropic_default_plot_prefix(output_file::String)
    return splitext(output_file)[1]
end

function barotropic_meshgrid(mesh::Spectral_Spherical_Mesh)
    lon_deg = mesh.λc .* 180 / pi
    lat_deg = mesh.θc .* 180 / pi
    return repeat(lon_deg, 1, length(lat_deg)), repeat(lat_deg, 1, length(lon_deg))'
end

function barotropic_save_plot(fig, save_file::String)
    fig.tight_layout()
    mkpath(dirname(save_file))
    fig.savefig(save_file, dpi=180)
    PyPlot.close(fig)
    return save_file
end

function barotropic_plot_field!(fig, ax, mesh, grid_dat; title="", clim=nothing, cmap="viridis", obs_coord=nothing)
    x_grid, y_grid = barotropic_meshgrid(mesh)
    values = grid_dat[:, :, 1]

    if isnothing(clim)
        im = ax.pcolormesh(x_grid, y_grid, values, shading="gouraud", cmap=cmap)
    else
        im = ax.pcolormesh(x_grid, y_grid, values, shading="gouraud", vmin=clim[1], vmax=clim[2], cmap=cmap)
    end

    if !isnothing(obs_coord)
        lon_deg = mesh.λc .* 180 / pi
        lat_deg = mesh.θc .* 180 / pi
        ax.scatter(lon_deg[obs_coord[:, 1]], lat_deg[obs_coord[:, 2]], color="black", s=8)
    end

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.82)
    return im
end

function barotropic_save_single_field(mesh, grid_dat, save_file; title="", clim=nothing, cmap="viridis", obs_coord=nothing)
    fig, axs = PyPlot.subplots(nrows=1, ncols=1, figsize=(7, 4), squeeze=false)
    barotropic_plot_field!(fig, axs[1, 1], mesh, grid_dat; title=title, clim=clim, cmap=cmap, obs_coord=obs_coord)
    return barotropic_save_plot(fig, save_file)
end

function barotropic_plot_initial_condition(sparam::Setup_Param, save_file::String)
    grid_vor_pert = sparam.grid_vor .- sparam.grid_vor_b
    fields = [
        ("Background zonal velocity", sparam.grid_u_b, "viridis"),
        ("Background vorticity", sparam.grid_vor_b, "viridis"),
        ("Vorticity perturbation", grid_vor_pert, "RdBu_r"),
        ("Initial vorticity", sparam.grid_vor, "viridis"),
    ]

    fig, axs = PyPlot.subplots(nrows=2, ncols=2, figsize=(12, 7), squeeze=false)
    axes = [axs[1, 1], axs[1, 2], axs[2, 1], axs[2, 2]]
    for i in 1:4
        title, field, cmap = fields[i]
        clim = (minimum(field), maximum(field))
        barotropic_plot_field!(fig, axes[i], sparam.mesh, field; title=title, clim=clim, cmap=cmap)
    end
    return barotropic_save_plot(fig, save_file)
end

function barotropic_plot_observation_frames(
    sparam::Setup_Param,
    truth,
    save_file::String;
    forward::Function=Barotropic_Main,
)
    _, obs_raw_data = forward(sparam, truth)
    vel_u_frames = obs_raw_data["vel_u"]
    nframes = length(vel_u_frames)
    clim = (minimum(minimum.(vel_u_frames)), maximum(maximum.(vel_u_frames)))

    fig, axs = PyPlot.subplots(nrows=1, ncols=nframes, figsize=(6nframes, 4), squeeze=false)
    for i in 1:nframes
        obs_hour = round(i * sparam.obs_time / 3600; digits=2)
        barotropic_plot_field!(
            fig,
            axs[1, i],
            sparam.mesh,
            vel_u_frames[i];
            title="Zonal velocity, T=$(obs_hour)h",
            clim=clim,
            cmap="viridis",
            obs_coord=sparam.obs_coord,
        )
    end
    return barotropic_save_plot(fig, save_file)
end

function barotropic_plot_recovered_vorticity(sparam::Setup_Param, recovered, save_file::String)
    nfields = length(recovered) + 1
    fig, axs = PyPlot.subplots(nrows=1, ncols=nfields, figsize=(6nfields, 4), squeeze=false)
    clim = (minimum(sparam.grid_vor), maximum(sparam.grid_vor))

    barotropic_plot_field!(fig, axs[1, 1], sparam.mesh, sparam.grid_vor; title="Truth", clim=clim, cmap="viridis")
    for (i, item) in enumerate(recovered)
        label, grid_vor_est = item
        barotropic_plot_field!(fig, axs[1, i + 1], sparam.mesh, grid_vor_est; title=label, clim=clim, cmap="viridis")
    end

    return barotropic_save_plot(fig, save_file)
end

function barotropic_plot_convergence(vorticity_errors, observation_errors, save_file::String; method_label::String)
    fig, axs = PyPlot.subplots(nrows=1, ncols=2, figsize=(10, 4), squeeze=false)
    iterations = collect(1:length(vorticity_errors))

    axs[1, 1].plot(iterations, vorticity_errors, marker="o", label=method_label)
    axs[1, 1].set_xlabel("Iterations")
    axs[1, 1].set_ylabel("Relative L2 error")
    axs[1, 1].legend()

    axs[1, 2].semilogy(iterations, observation_errors, marker="o", label=method_label)
    axs[1, 2].set_xlabel("Iterations")
    axs[1, 2].set_ylabel("Relative observation error")
    axs[1, 2].legend()

    return barotropic_save_plot(fig, save_file)
end

function barotropic_plot_history(values, save_file::String; method_label::String, ylabel::String, first_iteration::Int)
    iterations = first_iteration:(first_iteration + length(values) - 1)
    fig, ax = PyPlot.subplots(nrows=1, ncols=1, figsize=(6, 4), squeeze=false)
    ax[1, 1].plot(iterations, values, linestyle="--", marker="o", fillstyle="none", label=method_label)
    ax[1, 1].set_xlabel("Iterations")
    ax[1, 1].set_ylabel(ylabel)
    ax[1, 1].grid()
    ax[1, 1].legend()
    return barotropic_save_plot(fig, save_file)
end

barotropic_plot_covariance_norm(values, save_file::String; method_label::String) =
    barotropic_plot_history(values, save_file; method_label, ylabel="Frobenius norm of covariance", first_iteration=0)

barotropic_plot_optimization_error(values, save_file::String; method_label::String) =
    barotropic_plot_history(values, save_file; method_label, ylabel="Optimization error", first_iteration=1)

function inflated_eki_plot_prefix(output_file::String, filter_type::String)
    default_prefix = barotropic_default_plot_prefix(output_file)
    prefix_name = basename(default_prefix)
    if startswith(prefix_name, "InflatedEKI_Barotropic")
        prefix_name = replace(
            prefix_name,
            "InflatedEKI_Barotropic" => filter_type * "_Barotropic";
            count=1,
        )
    end
    return joinpath(dirname(default_prefix), prefix_name)
end

function write_inflated_eki_plots(result; method_label::String, plot_prefix::String)
    plot_files = barotropic_write_standard_plots(
        result;
        method_label=method_label,
        plot_prefix=plot_prefix,
    )
    push!(plot_files, barotropic_plot_optimization_error(
        result.optimization_errors,
        plot_prefix * "_opt_errors.png";
        method_label=method_label,
    ))
    push!(plot_files, barotropic_plot_covariance_norm(
        result.covariance_norms,
        plot_prefix * "_cov_norm.png";
        method_label=method_label,
    ))
    return plot_files
end

function barotropic_vorticity_errors(sparam::Setup_Param, tau_history, reconstruct_func::Function)
    errors = zeros(Float64, length(tau_history))
    for i in eachindex(tau_history)
        grid_vor_i = reconstruct_func(sparam, tau_history[i])
        errors[i] = norm(grid_vor_i - sparam.grid_vor) / norm(sparam.grid_vor)
    end
    return errors
end

function barotropic_observation_errors(y_obs, y_pred_history)
    errors = zeros(Float64, length(y_pred_history))
    for i in eachindex(y_pred_history)
        errors[i] = norm(y_obs - y_pred_history[i]) / norm(y_obs)
    end
    return errors
end

function barotropic_write_standard_plots(
    result;
    method_label::String,
    plot_prefix::String,
    observation_forward::Function=Barotropic_Main,
    observation_truth=result.tau_ref,
)
    plot_files = String[]
    push!(plot_files, barotropic_plot_initial_condition(result.sparam, plot_prefix * "_initial_condition.png"))
    push!(plot_files, barotropic_plot_observation_frames(
        result.sparam,
        observation_truth,
        plot_prefix * "_zonal_velocity_observations.png";
        forward=observation_forward,
    ))
    push!(plot_files, barotropic_plot_recovered_vorticity(result.sparam, [(method_label, result.grid_vor_est)], plot_prefix * "_recovered_vorticity.png"))
    push!(plot_files, barotropic_plot_convergence(
        result.vorticity_errors,
        result.observation_errors,
        plot_prefix * "_convergence.png";
        method_label=method_label,
    ))
    return plot_files
end

function grid_dropout_eaki_plot_vorticity_std(sparam::Setup_Param, theta_ensemble, save_file::String)
    grid_std = reshape(vec(std(theta_ensemble; dims=2)), size(sparam.grid_vor))
    return barotropic_save_single_field(
        sparam.mesh,
        grid_std,
        save_file;
        title="Final vorticity standard deviation",
        clim=(0.0, maximum(grid_std)),
        cmap="viridis",
    )
end

function write_grid_dropout_eaki_plots(result, final_ensemble; method_label::String, plot_prefix::String)
    plot_files = barotropic_write_standard_plots(
        result;
        method_label=method_label,
        plot_prefix=plot_prefix,
        observation_forward=Barotropic_Main_Grid,
        observation_truth=result.sparam.grid_vor,
    )
    push!(plot_files, barotropic_plot_optimization_error(
        result.optimization_errors,
        plot_prefix * "_opt_errors.png";
        method_label=method_label,
    ))
    push!(plot_files, barotropic_plot_covariance_norm(
        result.covariance_norms,
        plot_prefix * "_cov_norm.png";
        method_label=method_label,
    ))
    push!(plot_files, grid_dropout_eaki_plot_vorticity_std(
        result.sparam,
        final_ensemble,
        plot_prefix * "_vorticity_std.png",
    ))
    return plot_files
end
