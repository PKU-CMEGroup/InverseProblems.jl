using LinearAlgebra
using Random
using Serialization
using Statistics

include("Barotropic.jl")
include("../../Inversion/InflatedEKI.jl")
include("BarotropicPlots.jl")


function barotropic_u_forward_inflated_eki(sparam::Setup_Param, tau)
    _, obs_raw_data = Barotropic_Main(sparam, tau)
    return convert_obs(
        sparam.obs_coord,
        obs_raw_data;
        antisymmetric=sparam.antisymmetric,
        obs_name="vel_u",
    )
end

function barotropic_u_prior_forward_inflated_eki(sparam::Setup_Param, tau)
    return vcat(barotropic_u_forward_inflated_eki(sparam, tau), tau)
end

function augment_observations_with_prior_inflated_eki(y_obs, obs_cov, prior_mean, prior_cov)
    y_aug = vcat(y_obs, prior_mean)
    if obs_cov isa Diagonal && prior_cov isa Diagonal
        return y_aug, Diagonal(vcat(diag(obs_cov), diag(prior_cov)))
    end
    n_param = length(prior_mean)
    obs_cov_aug = zeros(Float64, length(y_obs) + n_param, length(y_obs) + n_param)
    obs_cov_aug[1:length(y_obs), 1:length(y_obs)] .= obs_cov
    obs_cov_aug[length(y_obs)+1:end, length(y_obs)+1:end] .= prior_cov
    return y_aug, obs_cov_aug
end

function inflated_eki_prior_covariance_factor(n_param::Int, prior_cov)
    covariance = isnothing(prior_cov) ? Diagonal(fill(9.0, n_param)) : prior_cov
    size(covariance) == (n_param, n_param) ||
        throw(DimensionMismatch("prior covariance must be $(n_param) x $(n_param)"))
    covariance_sqrt = covariance isa Diagonal ?
        Diagonal(sqrt.(diag(covariance))) : cholesky(Symmetric(covariance)).L
    return covariance, covariance_sqrt
end

function build_inflated_eki_problem(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=85,
    radius::Float64=6371.2e3,
    omega::Float64=7.292e-5,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    obs_seed::Int=42,
)
    nlon = 2nlat
    n_y = nobs * n_obs_frames

    obs_coord = zeros(Int, nobs, 2)
    Random.seed!(obs_seed)
    obs_coord[:, 1] .= rand(1:nlon-1, nobs)
    obs_coord[:, 2] .= rand(div(nlat, 2)+1:nlat-1, nobs)

    mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b,
    grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, tau_ref =
        Barotropic_Init(
            num_fourier,
            nlat;
            trunc_N=trunc_N,
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
        tau_ref;
        radius=radius,
        omega=omega,
    )

    return sparam, tau_ref
end

function make_noisy_observations_inflated_eki(
    sparam::Setup_Param,
    tau_ref;
    noise_level::Float64=0.05,
    noise_floor::Float64=1.0e-8,
    noise_seed::Int=123,
)
    y_ref = barotropic_u_forward_inflated_eki(sparam, tau_ref)
    Random.seed!(noise_seed)
    y_obs = y_ref .+ noise_level .* y_ref .* randn(length(y_ref))
    noise_std = max.(abs.(noise_level .* y_ref), noise_floor)
    obs_cov = Diagonal(noise_std .^ 2)

    return y_ref, y_obs, obs_cov
end

function reconstruct_initial_vorticity_inflated_eki(sparam::Setup_Param, tau)
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

function inflated_eki_initial_ensemble(
    sparam::Setup_Param,
    n_ens::Int;
    init_trunc_N::Int=7,
    init_prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    project_to_model_space::Bool=true,
    ensemble_seed::Int=123,
)
    1 <= init_trunc_N <= sparam.trunc_N ||
        throw(ArgumentError("init_trunc_N must lie in 1:$(sparam.trunc_N)"))
    init_n_param = init_trunc_N * (init_trunc_N + 2)
    init_prior_cov_mat, init_prior_cov_sqrt = inflated_eki_prior_covariance_factor(
        init_n_param,
        init_prior_cov,
    )

    rng = MersenneTwister(ensemble_seed)
    tau0 = init_prior_cov_sqrt * randn(rng, size(init_prior_cov_sqrt, 2), n_ens)
    theta0 = zeros(Float64, sparam.N_θ, n_ens)
    if project_to_model_space
        for j in axes(theta0, 2)
            spe_vor = param_to_spe(
                view(tau0, :, j),
                sparam.num_fourier;
                radius=sparam.radius,
            )
            theta0[:, j] .= spe_to_param(
                spe_vor,
                sparam.trunc_N;
                radius=sparam.radius,
            )
        end
    else
        theta0[1:size(tau0, 1), :] .= tau0
    end

    return theta0, init_prior_cov_mat, init_prior_cov_sqrt, tau0
end

function inflated_ensemble_mean(ekiobj::EKIObj)
    return dropdims(mean(ekiobj.θ[end], dims=2), dims=2)
end

function inflated_ensemble_covariance_norm(θ::AbstractMatrix)
    size(θ, 2) <= 1 && return 0.0
    Z = (θ .- mean(θ, dims=2)) ./ sqrt(size(θ, 2) - 1)
    return norm(Z' * Z)
end

function run_inflated_eki(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=85,
    init_trunc_N::Int=7,
    n_iter::Int=20,
    n_ens::Int=60,
    inflation_dt::Float64=0.5,
    noise_level::Float64=0.05,
    noise_seed::Int=123,
    obs_seed::Int=42,
    ensemble_seed::Int=123,
    prior_mean::Union{Vector{Float64},Nothing}=nothing,
    prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    init_prior_cov::Union{Diagonal,Matrix,Nothing}=nothing,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    filter_type::String="dropout-EAKI",
    dropout_rate::Float64=0.3,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::Float64=1.0,
    project_initial_ensemble::Bool=true,
    output_file::String=joinpath(@__DIR__, "Figs", "InflatedEKI_Barotropic.jls"),
    save_plots::Bool=true,
    plot_prefix::Union{String,Nothing}=nothing,
    inflation::Bool=true,
)
    sparam, tau_ref = build_inflated_eki_problem(
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
    y_ref, y_obs, obs_cov = make_noisy_observations_inflated_eki(
        sparam,
        tau_ref;
        noise_level=noise_level,
        noise_seed=noise_seed,
    )

    n_param = sparam.N_θ
    prior_mean_vec = isnothing(prior_mean) ? zeros(Float64, n_param) : copy(prior_mean)
    prior_cov_mat, prior_cov_sqrt_mat =
        inflated_eki_prior_covariance_factor(n_param, prior_cov)
    @assert length(prior_mean_vec) == n_param "prior_mean must have length $(n_param)."
    @assert size(prior_cov_mat) == (n_param, n_param) "prior_cov must be $(n_param) x $(n_param)."

    y_inflated_eki, obs_cov_inflated_eki =
        augment_observations_with_prior_inflated_eki(y_obs, obs_cov, prior_mean_vec, prior_cov_mat)

    θ0, init_prior_cov_mat, init_prior_cov_sqrt, tau0 = inflated_eki_initial_ensemble(
        sparam,
        n_ens;
        init_trunc_N=init_trunc_N,
        init_prior_cov=init_prior_cov,
        project_to_model_space=project_initial_ensemble,
        ensemble_seed=ensemble_seed,
    )
    forward(tau) = barotropic_u_prior_forward_inflated_eki(sparam, tau)

    inflated_ekiobj = EKI_Run(
        forward,
        θ0,
        obs_cov_inflated_eki,
        y_inflated_eki;
        filter_type=filter_type,
        Δτ=inflation_dt,
        N_iter=n_iter,
        dropout_rate=dropout_rate,
        inflation=inflation,
        dropout_correction_mode=dropout_correction_mode,
        joint_dropout_weight=joint_dropout_weight,
    )

    tau_history = [dropdims(mean(θ, dims=2), dims=2) for θ in inflated_ekiobj.θ[2:end]]
    vorticity_errors = barotropic_vorticity_errors(sparam, tau_history, reconstruct_initial_vorticity_inflated_eki)
    data_y_pred = [
        dropdims(mean(y_pred[1:length(y_obs), :], dims=2), dims=2)
        for y_pred in inflated_ekiobj.y_pred[2:end]
    ]
    observation_errors = barotropic_observation_errors(y_obs, data_y_pred)
    optimization_errors = opt_errors(inflated_ekiobj)[2:end]
    covariance_norms = inflated_ensemble_covariance_norm.(inflated_ekiobj.θ)

    tau_est = inflated_ensemble_mean(inflated_ekiobj)
    grid_vor_est = reconstruct_initial_vorticity_inflated_eki(sparam, tau_est)
    rel_vorticity_error = norm(grid_vor_est - sparam.grid_vor) / norm(sparam.grid_vor)
    rel_observation_error = norm(y_obs - data_y_pred[end]) / norm(y_obs)

    result = (
        sparam=sparam,
        tau_ref=tau_ref,
        y_ref=y_ref,
        y_obs=y_obs,
        obs_cov=obs_cov,
        y_inflated_eki=y_inflated_eki,
        obs_cov_inflated_eki=obs_cov_inflated_eki,
        prior_mean=prior_mean_vec,
        prior_cov=prior_cov_mat,
        prior_cov_sqrt=prior_cov_sqrt_mat,
        init_trunc_N=init_trunc_N,
        init_prior_cov=init_prior_cov_mat,
        init_prior_cov_sqrt=init_prior_cov_sqrt,
        tau0=tau0,
        theta0=θ0,
        project_initial_ensemble=project_initial_ensemble,
        inflated_ekiobj=inflated_ekiobj,
        tau_est=tau_est,
        grid_vor_est=grid_vor_est,
        vorticity_errors=vorticity_errors,
        observation_errors=observation_errors,
        optimization_errors=optimization_errors,
        covariance_norms=covariance_norms,
        rel_vorticity_error=rel_vorticity_error,
        rel_observation_error=rel_observation_error,
        filter_type=filter_type,
        dropout_rate=dropout_rate,
        dropout_correction_mode=dropout_correction_mode,
        joint_dropout_weight=joint_dropout_weight,
        output_file=output_file,
        plot_files=String[],
    )

    if save_plots
        prefix = isnothing(plot_prefix) ? inflated_eki_plot_prefix(output_file, filter_type) : plot_prefix
        plot_files = write_inflated_eki_plots(
            result;
            method_label=filter_type,
            plot_prefix=prefix,
        )
        result = merge(result, (plot_files=plot_files,))
    end

    mkpath(dirname(output_file))
    serialize(output_file, result)

    return result
end

function run_inflated_eki_comparison(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    inflation_dt::Float64=0.5,
    noise_level::Float64=0.05,
    noise_seed::Int=123,
    obs_seed::Int=42,
    ensemble_seed::Int=123,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    filter_type::String="dropout-EAKI",
    dropout_rate::Float64=0.3,
    dropout_correction_mode::String="joint",
    joint_dropout_weight::Float64=1.0,
    project_initial_ensemble::Bool=true,
    output_file::String=joinpath(@__DIR__, "Figs", "InflatedEKI_Barotropic_comparison.jls"),
    save_plots::Bool=true,
    plot_prefix::Union{String,Nothing}=nothing,
    inflation::Bool=true,
)
    trunc_N = 7
    n_param = trunc_N * (trunc_N + 2)
    prior_mean = zeros(Float64, n_param)
    covariance = Diagonal(fill(9.0, n_param))
    specifications = (
        (id="Nens30_Niter20", label="J=30, Iteration=20", n_ens=30, n_iter=20),
        (id="Nens60_Niter10", label="J=60, Iteration=10", n_ens=60, n_iter=10),
    )

    output_stem = splitext(output_file)[1]
    cases = NamedTuple[]
    for specification in specifications
        case_output_file = output_stem * "_" * specification.id * ".jls"
        case_result = run_inflated_eki(
            num_fourier=num_fourier,
            nlat=nlat,
            model_dt=model_dt,
            end_time=end_time,
            n_obs_frames=n_obs_frames,
            nobs=nobs,
            trunc_N=trunc_N,
            init_trunc_N=trunc_N,
            n_iter=specification.n_iter,
            n_ens=specification.n_ens,
            inflation_dt=inflation_dt,
            noise_level=noise_level,
            noise_seed=noise_seed,
            obs_seed=obs_seed,
            ensemble_seed=ensemble_seed,
            prior_mean=prior_mean,
            prior_cov=covariance,
            init_prior_cov=covariance,
            perturbation_wavenumber=perturbation_wavenumber,
            perturbation_amplitude=perturbation_amplitude,
            filter_type=filter_type,
            dropout_rate=dropout_rate,
            dropout_correction_mode=dropout_correction_mode,
            joint_dropout_weight=joint_dropout_weight,
            project_initial_ensemble=project_initial_ensemble,
            output_file=case_output_file,
            save_plots=false,
            inflation=inflation,
        )
        push!(cases, (
            label=specification.label,
            specification=specification,
            result=case_result,
        ))
    end

    @assert cases[1].result.tau_ref == cases[2].result.tau_ref
    @assert cases[1].result.y_obs == cases[2].result.y_obs
    @assert diag(cases[1].result.prior_cov) == diag(covariance)
    @assert diag(cases[2].result.prior_cov) == diag(covariance)
    @assert diag(cases[1].result.init_prior_cov) == diag(covariance)
    @assert diag(cases[2].result.init_prior_cov) == diag(covariance)

    comparison = (
        cases=cases,
        specifications=specifications,
        trunc_N=trunc_N,
        init_trunc_N=trunc_N,
        prior_mean=prior_mean,
        prior_cov=covariance,
        init_prior_cov=covariance,
        output_file=output_file,
        plot_files=String[],
    )

    if save_plots
        prefix = isnothing(plot_prefix) ? inflated_eki_plot_prefix(output_file, filter_type) : plot_prefix
        plot_files = write_inflated_eki_comparison_plots(cases; plot_prefix=prefix)
        comparison = merge(comparison, (plot_files=plot_files,))
    end

    mkpath(dirname(output_file))
    serialize(output_file, comparison)
    return comparison
end
