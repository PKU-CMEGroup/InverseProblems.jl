using LinearAlgebra
using Random
using Serialization
using Statistics
using PyPlot

include("Barotropic.jl")
include("../../Inversion/LowRankAugmentedDropoutEAKI.jl")
include("Section52Plots.jl")

using .LowRankAugmentedDropoutEAKI: LowRankPrior, prior_coordinates,
    prior_coordinate_dimension,
    low_rank_augmented_inner, EKI_Run_Low_Rank_Prior, low_rank_opt_errors,
    EKI_Run, opt_errors

"""
    Barotropic_Main_Grid(sparam, grid_vor0)

Run the barotropic model from a grid-valued initial vorticity field.

Unlike `Barotropic_Main`, this entry point does not interpret the inverse
variable as truncated spherical-harmonic coefficients. The caller provides the
full `nlon x nlat x 1` initial vorticity grid, which is transformed to the
model's spectral state before time stepping.
"""
function Barotropic_Main_Grid(sparam::Setup_Param, grid_vor0)
    name = "Barotropic"
    num_fourier, nlat = sparam.num_fourier, sparam.nθ
    nd = 1
    num_spherical = num_fourier + 1
    nlon = 2nlat

    radius = sparam.radius
    omega = sparam.omega

    mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nlat, nlon, nd, radius)
    sinlat = mesh.sinθ

    atmo_data = Atmo_Data(name, nlon, nlat, nd, false, false, false, false, sinlat, radius, omega)

    damping_order = 4
    damping_coef = 1.e-04
    robert_coef = 0.04
    implicit_coef = 0.0

    start_time = 0
    model_dt = sparam.Δt
    obs_time = sparam.obs_time
    end_time = sparam.end_time
    init_step = true

    integrator = Filtered_Leapfrog(
        robert_coef,
        damping_order,
        damping_coef,
        mesh.laplacian_eig,
        implicit_coef,
        model_dt,
        init_step,
        start_time,
        end_time,
    )

    dyn_data = Dyn_Data(name, num_fourier, num_spherical, nlon, nlat, nd, 0, 0)
    grid_u, grid_v = dyn_data.grid_u_c, dyn_data.grid_v_c
    spe_vor_c, spe_div_c = dyn_data.spe_vor_c, dyn_data.spe_div_c
    grid_vor, grid_div = dyn_data.grid_vor, dyn_data.grid_div

    spe_div_c .= 0.0
    grid_div .= 0.0

    # This is the only new connection point to the PDE solver: the inverse
    # variable is already a full grid field, so we use the existing "grid_vor"
    # initializer. It converts grid_vor0 to the T85 spectral state used by the
    # barotropic time integrator.
    Barotropic_ω0!(mesh, "grid_vor", grid_vor0, spe_vor_c, grid_vor; radius=radius)
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c, spe_div_c, grid_u, grid_v)

    obs_data = Dict("vel_u"=>Array{Float64,3}[], "vel_v"=>Array{Float64,3}[], "vor"=>Array{Float64,3}[])

    nt = Int64(end_time / model_dt)
    time = start_time
    Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
    Update_Init_Step!(integrator)
    time += model_dt

    for i in 2:nt
        Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
        time += model_dt
        if time % obs_time == 0
            push!(obs_data["vel_u"], copy(grid_u))
            push!(obs_data["vor"], copy(grid_vor))
        end
    end

    return mesh, obs_data
end

function section52_grid_shape(sparam::Setup_Param)
    return length(sparam.mesh.λc), length(sparam.mesh.θc), 1
end

function section52_grid_parameter_dimension(sparam::Setup_Param)
    nlon, nlat, _ = section52_grid_shape(sparam)
    return nlon * nlat
end

function section52_check_grid_parameterization(parameterization::String)
    parameterization in ("perturbation_grid", "full_grid") ||
        error("parameterization must be \"perturbation_grid\" or \"full_grid\".")
end

function section52_grid_prior_mean(sparam::Setup_Param; parameterization::String="perturbation_grid")
    section52_check_grid_parameterization(parameterization)
    nparam = section52_grid_parameter_dimension(sparam)
    if parameterization == "perturbation_grid"
        # Preferred high-dimensional parameterization:
        #   theta = initial vorticity perturbation on the grid,
        #   zeta0 = zeta_b + reshape(theta).
        # The prior mean is therefore zero perturbation.
        return zeros(Float64, nparam)
    else
        # Alternative parameterization:
        #   theta = the full initial vorticity grid itself.
        # Then the prior mean is the background vorticity field.
        return vec(copy(sparam.grid_vor_b))
    end
end

function section52_grid_truth_vector(sparam::Setup_Param; parameterization::String="perturbation_grid")
    section52_check_grid_parameterization(parameterization)
    if parameterization == "perturbation_grid"
        # Store the true unknown in the same coordinates used by dropout-EAKI.
        return vec(copy(sparam.grid_vor .- sparam.grid_vor_b))
    else
        return vec(copy(sparam.grid_vor))
    end
end

function section52_project_grid_vorticity_to_model_space(sparam::Setup_Param, grid_vor)
    spe_vor = similar(sparam.spe_vor)
    grid_projected = similar(sparam.grid_vor)
    spe_vor .= 0.0
    grid_projected .= 0.0

    # The model evolves only the modes represented by its spectral truncation.
    # This grid -> spectral -> grid pass removes grid-scale components that the
    # T85 solver cannot actually carry forward.
    Trans_Grid_To_Spherical!(sparam.mesh, grid_vor, spe_vor)
    Trans_Spherical_To_Grid!(sparam.mesh, spe_vor, grid_projected)

    return grid_projected
end

function section52_grid_vector_to_vorticity(
    sparam::Setup_Param,
    θ::AbstractVector;
    parameterization::String="perturbation_grid",
    project_to_model_space::Bool=false,
)
    section52_check_grid_parameterization(parameterization)
    nlon, nlat, nlev = section52_grid_shape(sparam)
    @assert nlev == 1
    @assert length(θ) == nlon * nlat "θ must have length $(nlon * nlat)."

    # dropout-EAKI sees θ as a vector. The physical model needs a 3D grid
    # array, matching the existing NNGCM storage convention: nlon x nlat x level.
    θ_grid = reshape(θ, nlon, nlat, 1)
    grid_vor0 = Array{Float64,3}(undef, nlon, nlat, 1)

    if parameterization == "perturbation_grid"
        grid_vor0 .= sparam.grid_vor_b
        grid_vor0 .+= θ_grid
    else
        grid_vor0 .= θ_grid
    end

    return project_to_model_space ? section52_project_grid_vorticity_to_model_space(sparam, grid_vor0) : grid_vor0
end

function section52_vorticity_to_grid_vector(
    sparam::Setup_Param,
    grid_vor;
    parameterization::String="perturbation_grid",
)
    section52_check_grid_parameterization(parameterization)
    if parameterization == "perturbation_grid"
        return vec(copy(grid_vor .- sparam.grid_vor_b))
    else
        return vec(copy(grid_vor))
    end
end

function section52_project_grid_parameter_to_model_space(
    sparam::Setup_Param,
    θ::AbstractVector;
    parameterization::String="perturbation_grid",
)
    grid_vor0 = section52_grid_vector_to_vorticity(
        sparam,
        θ;
        parameterization=parameterization,
        project_to_model_space=false,
    )
    grid_projected = section52_project_grid_vorticity_to_model_space(sparam, grid_vor0)
    return section52_vorticity_to_grid_vector(sparam, grid_projected; parameterization=parameterization)
end

function barotropic_u_forward_grid_dropout_eaki(
    sparam::Setup_Param,
    θ::AbstractVector;
    parameterization::String="perturbation_grid",
)
    # This is the high-dimensional forward map G(θ). It converts one
    # grid-valued initial condition to the pointwise zonal wind observations.
    grid_vor0 = section52_grid_vector_to_vorticity(
        sparam,
        θ;
        parameterization=parameterization,
        project_to_model_space=false,
    )
    _, obs_raw_data = Barotropic_Main_Grid(sparam, grid_vor0)
    return convert_obs(
        sparam.obs_coord,
        obs_raw_data;
        antisymmetric=sparam.antisymmetric,
        obs_name="vel_u",
    )
end

function build_section52_grid_dropout_eaki_problem(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=7,
    trunc_N_true::Int=num_fourier,
    radius::Float64=6371.2e3,
    omega::Float64=7.292e-5,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    obs_seed::Int=42,
    parameterization::String="perturbation_grid",
)
    section52_check_grid_parameterization(parameterization)
    nlon = 2nlat
    n_y = nobs * n_obs_frames

    obs_coord = zeros(Int, nobs, 2)
    Random.seed!(obs_seed)
    obs_coord[:, 1] .= rand(1:nlon-1, nobs)
    obs_coord[:, 2] .= rand(div(nlat, 2)+1:nlat-1, nobs)

    mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b,
    grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, init_data =
        Barotropic_Init(
            num_fourier,
            nlat;
            trunc_N=trunc_N_true,
            radius=radius,
            m=perturbation_wavenumber,
            A=perturbation_amplitude,
            symmetric=false,
        )

    sparam = Setup_Param(
        num_fourier,
        nlat,
        model_dt,
        end_time,
        n_obs_frames,
        obs_coord,
        false,
        n_y,
        trunc_N,
        mesh,
        grid_u_b,
        grid_v_b,
        grid_vor_b,
        spe_vor_b,
        grid_u,
        grid_v,
        grid_vor,
        spe_vor,
        init_data;
        radius=radius,
        omega=omega,
    )

    θ_ref = section52_grid_truth_vector(sparam; parameterization=parameterization)
    return sparam, θ_ref
end

function make_noisy_observations_grid_dropout_eaki(
    sparam::Setup_Param,
    θ_ref;
    parameterization::String="perturbation_grid",
    noise_level::Float64=0.05,
    noise_floor::Float64=1.0e-8,
    noise_seed::Int=123,
)
    y_ref = barotropic_u_forward_grid_dropout_eaki(
        sparam,
        θ_ref;
        parameterization=parameterization,
    )
    Random.seed!(noise_seed)
    y_obs = y_ref .+ noise_level .* y_ref .* randn(length(y_ref))
    noise_std = max.(abs.(noise_level .* y_ref), noise_floor)
    Σ_y = Array(Diagonal(noise_std .^ 2))

    return y_ref, y_obs, Σ_y
end

function reconstruct_initial_vorticity_grid_dropout_eaki(sparam::Setup_Param, tau)
    spe_vor = copy(sparam.spe_vor)
    grid_vor = copy(sparam.grid_vor)
    Barotropic_ω0!(
        sparam.mesh,
        "spec_vor",
        tau,
        spe_vor,
        grid_vor;
        spe_vor_b=sparam.spe_vor_b,
        radius=sparam.radius,
    )
    return grid_vor
end

function section52_grid_prior_covariance_factor(sparam::Setup_Param, prior_cov_sqrt)
    U = zeros(
        Float64,
        section52_grid_parameter_dimension(sparam),
        size(prior_cov_sqrt, 2),
    )
    for j in axes(U, 2)
        grid_vor_j = reconstruct_initial_vorticity_grid_dropout_eaki(
            sparam,
            view(prior_cov_sqrt, :, j),
        )
        U[:, j] .= section52_vorticity_to_grid_vector(sparam, grid_vor_j)
    end
    return U
end

function section52_grid_to_prior_coefficients(
    sparam::Setup_Param,
    theta::AbstractVector,
    trunc_N::Int=sparam.trunc_N,
)
    length(theta) == section52_grid_parameter_dimension(sparam) ||
        throw(DimensionMismatch("grid parameter has the wrong length"))
    1 <= trunc_N <= sparam.num_fourier ||
        throw(ArgumentError("trunc_N must lie in 1:$(sparam.num_fourier)"))
    grid_vor = reshape(collect(theta), section52_grid_shape(sparam))
    spe_vor = similar(sparam.spe_vor)
    fill!(spe_vor, 0)
    Trans_Grid_To_Spherical!(sparam.mesh, grid_vor, spe_vor)
    return spe_to_param(spe_vor, trunc_N; radius=sparam.radius)
end

function section52_grid_to_prior_coefficients(
    sparam::Setup_Param,
    theta::AbstractMatrix,
    trunc_N::Int=sparam.trunc_N,
)
    coefficients = Matrix{Float64}(undef, trunc_N * (trunc_N + 2), size(theta, 2))
    for j in axes(theta, 2)
        coefficients[:, j] .= section52_grid_to_prior_coefficients(
            sparam, view(theta, :, j), trunc_N,
        )
    end
    return coefficients
end

function section52_grid_prior(
    sparam::Setup_Param;
    prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
)
    covariance = isnothing(prior_cov) ?
        barotropic_power_law_prior_cov(sparam.trunc_N; sigma=10.0, alpha=0.0) :
        prior_cov
    # covariance = barotropic_heat_prior_cov(
    #     sparam.trunc_N; sigma=10.0, beta=0.2, variance_floor=1.0e-14,)
    prior_cov_mat, prior_cov_sqrt = barotropic_prior_covariance_factor(
        sparam.trunc_N;
        prior_cov=covariance,
    )
    coordinates(theta) = prior_cov_sqrt \
        section52_grid_to_prior_coefficients(sparam, theta, sparam.trunc_N)
    prior = LowRankPrior(
        section52_grid_prior_mean(sparam),
        size(prior_cov_sqrt, 2),
        coordinates,
    )
    return prior, prior_cov_mat, prior_cov_sqrt
end

function section52_initial_grid_ensemble(
    sparam::Setup_Param,
    n_ens::Int;
    init_trunc_N::Int=sparam.trunc_N,
    init_prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    project_to_model_space::Bool=true,
    ensemble_seed::Int=123,
)
    1 <= init_trunc_N <= sparam.num_fourier ||
        throw(ArgumentError("init_trunc_N must lie in 1:$(sparam.num_fourier)"))
    covariance = isnothing(init_prior_cov) ?
        barotropic_power_law_prior_cov(init_trunc_N; sigma=1.0, alpha=0.0) :
        init_prior_cov
    init_prior_cov_mat, init_prior_cov_sqrt = barotropic_prior_covariance_factor(
        init_trunc_N;
        prior_cov=covariance,
    )
    U_init = section52_grid_prior_covariance_factor(sparam, init_prior_cov_sqrt)

    rng = MersenneTwister(ensemble_seed)
    xi0 = randn(rng, size(init_prior_cov_sqrt, 2), n_ens)
    τ0 = init_prior_cov_sqrt * xi0
    θ0 = U_init * xi0
    if project_to_model_space
        for j in axes(θ0, 2)
            θ0[:, j] .= section52_project_grid_parameter_to_model_space(
                sparam,
                θ0[:, j],
            )
        end
    end

    return θ0, init_prior_cov_mat, init_prior_cov_sqrt, U_init, τ0
end


function section52_grid_lowrank_covariance_norm(θ::AbstractMatrix)
    size(θ, 2) <= 1 && return 0.0
    θ_mean = mean(θ, dims=2)
    Z = (θ .- θ_mean) ./ sqrt(size(θ, 2) - 1)

    # The sample covariance is C = Z*Z', which would be N_θ x N_θ.
    # Its nonzero eigenvalues are the same as those of the small Gram matrix
    # Z'*Z. Therefore the Frobenius norm of C can be computed from Z'*Z,
    # an n_ens x n_ens matrix, without ever storing C.
    return norm(Z' * Z)
end

function EKI_Run_Grid_Prior(
    forward::Function,
    θ0::Array{FT,2},
    Σ_y::Array{FT,2},
    y::Array{FT,1},
    prior::LowRankPrior{FT};
    filter_type::String="dropout-EAKI",
    Δτ::FT=FT(0.5),
    N_iter::Int=50,
    dropout_rate::FT=FT(0.5),
    inflation::Bool=true,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::FT=one(FT),
) where FT<:AbstractFloat
    return EKI_Run_Low_Rank_Prior(
        forward, θ0, Σ_y, y, prior;
        filter_type=filter_type,
        Δτ=Δτ,
        N_iter=N_iter,
        dropout_rate=dropout_rate,
        inflation=inflation,
        dropout_correction_mode=dropout_correction_mode,
        joint_dropout_weight=joint_dropout_weight,
    )
end

function section52_grid_optimization_errors(
    ekiobj,
    prior::Union{LowRankPrior,Nothing},
)
    return isnothing(prior) ? opt_errors(ekiobj)[2:end] :
           low_rank_opt_errors(ekiobj, prior)[2:end]
end

function section52_plot_grid_optimization_error(optimization_errors, save_file::String; method_label::String)
    iterations = collect(1:length(optimization_errors))
    fig, ax = PyPlot.subplots(nrows=1, ncols=1, figsize=(6, 4), squeeze=false)

    ax[1, 1].plot(iterations, optimization_errors, linestyle="--", marker="o", fillstyle="none", label=method_label)
    ax[1, 1].set_xlabel("Iterations")
    ax[1, 1].set_ylabel("Optimization error")
    ax[1, 1].grid()
    ax[1, 1].legend()

    fig.tight_layout()
    mkpath(dirname(save_file))
    fig.savefig(save_file, dpi=180)
    PyPlot.close(fig)
    return save_file
end

function section52_plot_grid_observation_frames(sparam::Setup_Param, grid_vor_ref, save_file::String)
    _, obs_raw_data = Barotropic_Main_Grid(sparam, grid_vor_ref)
    vel_u_frames = obs_raw_data["vel_u"]
    nframes = length(vel_u_frames)
    clim = (minimum(minimum.(vel_u_frames)), maximum(maximum.(vel_u_frames)))

    fig, axs = PyPlot.subplots(nrows=1, ncols=nframes, figsize=(6nframes, 4), squeeze=false)
    for i in 1:nframes
        obs_hour = round(i * sparam.obs_time / 3600; digits=2)
        section52_plot_field!(
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
    fig.tight_layout()
    mkpath(dirname(save_file))
    fig.savefig(save_file, dpi=180)
    PyPlot.close(fig)
    return save_file
end

function section52_plot_grid_vorticity_std(sparam::Setup_Param, theta_ensemble, save_file::String)
    grid_std = reshape(vec(std(theta_ensemble; dims=2)), size(sparam.grid_vor))
    return section52_save_single_field(
        sparam.mesh,
        grid_std,
        save_file;
        title="Final vorticity standard deviation",
        clim=(0.0, maximum(grid_std)),
        cmap="viridis",
    )
end

function section52_write_grid_dropout_eaki_plots(result, final_ensemble; method_label::String, plot_prefix::String)
    plot_files = String[]
    push!(plot_files, section52_plot_initial_condition(result.sparam, plot_prefix * "_initial_condition.png"))
    push!(plot_files, section52_plot_grid_observation_frames(
        result.sparam,
        result.sparam.grid_vor,
        plot_prefix * "_zonal_velocity_observations.png",
    ))
    push!(plot_files, section52_plot_recovered_vorticity(
        result.sparam,
        [(method_label, result.grid_vor_est)],
        plot_prefix * "_recovered_vorticity.png",
    ))
    push!(plot_files, section52_plot_convergence(
        result.vorticity_errors,
        result.observation_errors,
        plot_prefix * "_convergence.png";
        method_label=method_label,
    ))
    push!(plot_files, section52_plot_grid_optimization_error(
        result.optimization_errors,
        plot_prefix * "_opt_errors.png";
        method_label=method_label,
    ))
    push!(plot_files, section52_plot_covariance_norm(
        result.covariance_norms,
        plot_prefix * "_cov_norm.png";
        method_label=method_label,
    ))
    push!(plot_files, section52_plot_grid_vorticity_std(
        result.sparam,
        final_ensemble,
        plot_prefix * "_vorticity_std.png",
    ))
    return plot_files
end

function run_section52_grid_dropout_eaki(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=7,
    init_trunc_N::Int=trunc_N,
    n_iter::Int=20,
    n_ens::Int=30,
    inflation_dt::Float64=0.2,
    dropout_rate::Float64=0.5,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::Float64=1.0,
    noise_level::Float64=0.05,
    noise_seed::Int=123,
    obs_seed::Int=42,
    ensemble_seed::Int=123,
    prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    init_prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    project_initial_ensemble::Bool=true,
    output_file::String=joinpath(@__DIR__, "Figs", "GridDropoutEAKI_Barotropic_Section52.jls"),
    save_plots::Bool=true,
    plot_prefix::Union{String,Nothing}=nothing,
    inflation::Bool=true,
    use_prior_augmentation::Bool=true,
    store_ensemble_history::Bool=false,
)
    sparam, θ_ref = build_section52_grid_dropout_eaki_problem(
        num_fourier=num_fourier,
        nlat=nlat,
        model_dt=model_dt,
        end_time=end_time,
        n_obs_frames=n_obs_frames,
        nobs=nobs,
        trunc_N=trunc_N,
        obs_seed=obs_seed,
        perturbation_wavenumber=perturbation_wavenumber,
        perturbation_amplitude=perturbation_amplitude,
    )
    y_ref, y_obs, Σ_y = make_noisy_observations_grid_dropout_eaki(
        sparam,
        θ_ref;
        noise_level=noise_level,
        noise_seed=noise_seed,
    )

    grid_prior, prior_cov_mat, prior_cov_sqrt = section52_grid_prior(
        sparam;
        prior_cov=prior_cov,
    )
    θ0, init_prior_cov_mat, init_prior_cov_sqrt, grid_init_cov_sqrt, τ0 = section52_initial_grid_ensemble(
        sparam,
        n_ens;
        init_trunc_N=init_trunc_N,
        init_prior_cov=init_prior_cov,
        project_to_model_space=project_initial_ensemble,
        ensemble_seed=ensemble_seed,
    )

    forward(θ) = barotropic_u_forward_grid_dropout_eaki(sparam, θ)

    grid_prior_mean = section52_grid_prior_mean(sparam)
    grid_ekiobj = if use_prior_augmentation
        EKI_Run_Grid_Prior(
            forward,
            θ0,
            Σ_y,
            y_obs,
            grid_prior;
            filter_type="dropout-EAKI",
            Δτ=inflation_dt,
            N_iter=n_iter,
            dropout_rate=dropout_rate,
            inflation=inflation,
            dropout_correction_mode=dropout_correction_mode,
            joint_dropout_weight=joint_dropout_weight,
        )
    else
        EKI_Run(
            forward,
            θ0,
            Σ_y,
            y_obs;
            filter_type="dropout-EAKI",
            Δτ=inflation_dt,
            N_iter=n_iter,
            dropout_rate=dropout_rate,
            inflation=inflation,
            dropout_correction_mode=dropout_correction_mode,
            joint_dropout_weight=joint_dropout_weight,
        )
    end

    θ_history = [vec(mean(θ, dims=2)) for θ in grid_ekiobj.θ[2:end]]
    reconstruct_grid(sparam, θ) = section52_grid_vector_to_vorticity(
        sparam,
        θ;
        project_to_model_space=true,
    )
    vorticity_errors = section52_vorticity_errors(sparam, θ_history, reconstruct_grid)
    data_y_pred = vec.(grid_ekiobj.y_pred[2:end])
    observation_errors = section52_observation_errors(y_obs, data_y_pred)
    optimization_errors = section52_grid_optimization_errors(
        grid_ekiobj,
        use_prior_augmentation ? grid_prior : nothing,
    )
    covariance_norms = section52_grid_lowrank_covariance_norm.(grid_ekiobj.θ)

    θ_est = vec(mean(grid_ekiobj.θ[end], dims=2))
    grid_vor_est = section52_grid_vector_to_vorticity(
        sparam,
        θ_est;
        project_to_model_space=true,
    )
    rel_vorticity_error = norm(grid_vor_est - sparam.grid_vor) / norm(sparam.grid_vor)
    rel_observation_error = norm(y_obs - data_y_pred[end]) / norm(y_obs)

    result = (
        sparam=sparam,
        theta_ref=θ_ref,
        y_ref=y_ref,
        y_obs=y_obs,
        obs_cov=Σ_y,
        prior_mean=zeros(Float64, size(prior_cov_sqrt, 1)),
        prior_cov=prior_cov_mat,
        prior_cov_sqrt=prior_cov_sqrt,
        init_prior_mean=zeros(Float64, size(init_prior_cov_sqrt, 1)),
        init_prior_cov=init_prior_cov_mat,
        init_prior_cov_sqrt=init_prior_cov_sqrt,
        grid_prior_mean=grid_prior_mean,
        grid_prior_cov_sqrt=nothing,
        grid_prior_representation="spectral_coordinates",
        grid_prior_rank=prior_coordinate_dimension(grid_prior),
        grid_init_cov_sqrt=grid_init_cov_sqrt,
        augmented_observation_dimension=length(y_obs) + (use_prior_augmentation ? length(grid_prior_mean) : 0),
        augmented_whitened_dimension=length(y_obs) + (use_prior_augmentation ? prior_coordinate_dimension(grid_prior) : 0),
        tau0=τ0,
        theta0=θ0,
        grid_ekiobj=store_ensemble_history ? grid_ekiobj : nothing,
        theta_est=θ_est,
        grid_vor_est=grid_vor_est,
        vorticity_errors=vorticity_errors,
        observation_errors=observation_errors,
        optimization_errors=optimization_errors,
        covariance_norms=covariance_norms,
        rel_vorticity_error=rel_vorticity_error,
        rel_observation_error=rel_observation_error,
        filter_type="dropout-EAKI",
        dropout_rate=dropout_rate,
        init_trunc_N=init_trunc_N,
        parameterization="perturbation_grid",
        project_initial_ensemble=project_initial_ensemble,
        used_prior_augmentation=use_prior_augmentation,
        output_file=output_file,
        plot_files=String[],
        dropout_correction_mode=dropout_correction_mode,
        joint_dropout_weight=joint_dropout_weight,
    )

    if save_plots
        prefix = isnothing(plot_prefix) ? section52_default_plot_prefix(output_file) : plot_prefix
        plot_files = section52_write_grid_dropout_eaki_plots(
            result,
            grid_ekiobj.θ[end];
            method_label="dropout-EAKI grid",
            plot_prefix=prefix,
        )
        result = merge(result, (plot_files=plot_files,))
    end

    mkpath(dirname(output_file))
    serialize(output_file, result)

    return result
end

function run_section52_grid_dropout_eaki_smoke_test(;
    output_file::String=joinpath(@__DIR__, "Figs", "GridDropoutEAKI_Barotropic_smoke.jls"),
    save_plots::Bool=false,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::Float64=1.0,
)
    return run_section52_grid_dropout_eaki(
        num_fourier=8,
        nlat=16,
        model_dt=1800,
        end_time=3600,
        n_obs_frames=1,
        nobs=4,
        trunc_N=2,
        n_iter=1,
        n_ens=4,
        perturbation_wavenumber=2.0,
        output_file=output_file,
        save_plots=save_plots,
        dropout_correction_mode=dropout_correction_mode,
        joint_dropout_weight=joint_dropout_weight,
        store_ensemble_history=true,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_section52_grid_dropout_eaki_smoke_test()
    @info "Finished Section 5.2 grid dropout-EAKI smoke test" result.rel_vorticity_error result.rel_observation_error result.output_file
end
