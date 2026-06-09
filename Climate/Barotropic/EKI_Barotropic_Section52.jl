using LinearAlgebra
using Random
using Serialization
using Statistics
using PyPlot

include("Barotropic.jl")
include("../../Inversion/KI.jl")
include("Section52Plots.jl")

"""
    barotropic_u_forward_eki(sparam, tau)

Forward map for the Section 5.2 barotropic inverse problem.

`tau` is the vector of triangularly truncated spherical-harmonic coefficients
for the initial vorticity perturbation. The observable is the pointwise zonal
velocity `u`, matching Section 5.2 of arXiv:2102.10677.
"""
function barotropic_u_forward_eki(sparam::Setup_Param, tau)
    _, obs_raw_data = Barotropic_Main(sparam, tau)
    return convert_obs(
        sparam.obs_coord,
        obs_raw_data;
        antisymmetric=sparam.antisymmetric,
        obs_name="vel_u",
    )
end

function barotropic_u_ensemble_eki(sparam::Setup_Param, tau_ens::AbstractMatrix)
    n_ens = size(tau_ens, 1)
    g_ens = zeros(Float64, n_ens, sparam.N_y)

    Threads.@threads for i in 1:n_ens
        g_ens[i, :] .= barotropic_u_forward_eki(sparam, tau_ens[i, :])
    end

    return g_ens
end

function build_section52_eki_problem(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=7,
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

function make_noisy_observations_eki(
    sparam::Setup_Param,
    tau_ref;
    noise_level::Float64=0.05,
    noise_floor::Float64=1.0e-8,
    noise_seed::Int=123,
)
    y_ref = barotropic_u_forward_eki(sparam, tau_ref)
    Random.seed!(noise_seed)
    y_obs = y_ref .+ noise_level .* y_ref .* randn(length(y_ref))
    noise_std = max.(abs.(noise_level .* y_ref), noise_floor)
    obs_cov = Array(Diagonal(noise_std .^ 2))

    return y_ref, y_obs, obs_cov
end

function reconstruct_initial_vorticity_eki(sparam::Setup_Param, tau)
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

function ensemble_mean(ekiobj::EKIObj)
    return dropdims(mean(ekiobj.θ[end], dims=1), dims=1)
end

function eki_plot_prefix(output_file::String, filter_type::String)
    default_prefix = section52_default_plot_prefix(output_file)
    prefix_dir = dirname(default_prefix)
    prefix_name = basename(default_prefix)
    prefix_name = startswith(prefix_name, "EKI_Barotropic") ?
        replace(prefix_name, "EKI_Barotropic" => filter_type * "_Barotropic"; count=1) :
        prefix_name
    return joinpath(prefix_dir, prefix_name)
end

function plot_param_iter(ekiobj::EKIObj{FT, IT}, θ_ref::Array{FT,1}, θ_ref_names::Array{String}) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = [dropdims(mean(θ, dims=1), dims=1) for θ in ekiobj.θ]
    θθ_cov = [cov(θ, dims=1, corrected=true) for θ in ekiobj.θ]
    
    N_iter = length(θ_mean) - 1
    ites = Array(LinRange(1, N_iter+1, N_iter+1))
    
    θ_mean_arr = abs.(hcat(θ_mean...))
    
    N_θ = length(θ_ref)
    θθ_std_arr = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        for j = 1:N_θ
            θθ_std_arr[j, i] = sqrt(θθ_cov[i][j,j])
        end
    end
    
    for i = 1:N_θ
        PyPlot.errorbar(ites, θ_mean_arr[i,:], yerr=3.0*θθ_std_arr[i,:], fmt="--o", fillstyle="none", label=θ_ref_names[i])   
        PyPlot.plot(ites, fill(θ_ref[i], N_iter+1), "--", color="gray")
    end
    
    PyPlot.xlabel("Iterations")
    PyPlot.legend()
    PyPlot.tight_layout()
end

function plot_opt_errors(ekiobj::EKIObj{FT, IT}, 
    θ_ref::Union{Array{FT,1}, Nothing} = nothing, 
    transform_func::Union{Function, Nothing} = nothing) where {FT<:AbstractFloat, IT<:Int}
    
    θ_mean = [dropdims(mean(θ, dims=1), dims=1) for θ in ekiobj.θ]
    θθ_cov = [cov(θ, dims=1, corrected=true) for θ in ekiobj.θ]
    y_pred = ekiobj.y_pred
    Σ_η = ekiobj.Σ_η
    y = ekiobj.y

    N_iter = length(θ_mean) - 1
    
    ites = Array(LinRange(1, N_iter, N_iter))
    N_subfigs = (θ_ref === nothing ? 2 : 3)

    errors = zeros(Float64, N_subfigs, N_iter)
    fig, ax = PyPlot.subplots(ncols=N_subfigs, figsize=(N_subfigs*6,6))
    for i = 1:N_iter
        errors[N_subfigs - 1, i] = 0.5*(y - y_pred[i])'*(Σ_η\(y - y_pred[i]))
        errors[N_subfigs, i]     = norm(θθ_cov[i])
        
        if N_subfigs == 3
            errors[1, i] = norm(θ_ref - (transform_func === nothing ? θ_mean[i] : transform_func(θ_mean[i])))/norm(θ_ref)
        end
        
    end

    markevery = max(div(N_iter, 10), 1)
    ax[N_subfigs - 1].plot(ites, errors[N_subfigs - 1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs - 1].set_xlabel("Iterations")
    ax[N_subfigs - 1].set_ylabel("Optimization error")
    ax[N_subfigs - 1].grid()
    
    ax[N_subfigs].plot(ites, errors[N_subfigs, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
    ax[N_subfigs].set_xlabel("Iterations")
    ax[N_subfigs].set_ylabel("Frobenius norm of the covariance")
    ax[N_subfigs].grid()
    if N_subfigs == 3
        ax[1].set_xlabel("Iterations")
        ax[1].plot(ites, errors[1, :], linestyle="--", marker="o", fillstyle="none", markevery=markevery)
        ax[1].set_ylabel("L₂ norm error")
        ax[1].grid()
    end
end

function run_section52_eki(;
    num_fourier::Int=85,
    nlat::Int=256,
    model_dt::Int=1800,
    end_time::Int=86400,
    n_obs_frames::Int=2,
    nobs::Int=50,
    trunc_N::Int=7,
    n_iter::Int=20,
    n_ens::Union{Int,Nothing}=nothing,
    alpha_reg::Float64=0.5,
    eki_dt::Float64=0.5,
    update_freq::Int=0,
    noise_level::Float64=0.05,
    noise_seed::Int=123,
    obs_seed::Int=42,
    ensemble_seed::Int=123,
    perturbation_wavenumber::Float64=4.0,
    perturbation_amplitude::Float64=8.0e-5,
    filter_type::String="EAKI",
    output_file::String=joinpath(@__DIR__, "Figs", "EKI_Barotropic_Section52.jls"),
    save_plots::Bool=true,
    plot_prefix::Union{String,Nothing}=nothing,
)
    sparam, tau_ref = build_section52_eki_problem(
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
    y_ref, y_obs, obs_cov = make_noisy_observations_eki(
        sparam,
        tau_ref;
        noise_level=noise_level,
        noise_seed=noise_seed,
    )

    n_param = sparam.N_θ
    n_ens = isnothing(n_ens) ? 2n_param + 1 : n_ens

    tau0_mean = zeros(Float64, n_param)
    tau0_cov_sqrt = Matrix{Float64}(I, n_param, n_param)
    prior_mean = zeros(Float64, n_param)
    prior_cov_sqrt = Matrix{Float64}(I, n_param, n_param)

    Random.seed!(ensemble_seed)
    ekiobj = EKIObj(
        filter_type,
        ["tau"],
        n_ens,
        tau0_mean,
        tau0_cov_sqrt,
        prior_mean,
        prior_cov_sqrt,
        y_obs,
        obs_cov,
        eki_dt,
        alpha_reg,
        update_freq,
    )

    ens_func(tau_ens) = barotropic_u_ensemble_eki(sparam, tau_ens)

    for iter in 1:n_iter
        @info "EKI iteration" iter n_iter
        update_ensemble!(ekiobj, ens_func)
    end

    tau_history = [dropdims(mean(θ, dims=1), dims=1) for θ in ekiobj.θ[2:end]]
    vorticity_errors = section52_vorticity_errors(sparam, tau_history, reconstruct_initial_vorticity_eki)
    observation_errors = section52_observation_errors(y_obs, ekiobj.y_pred)
    covariance_norms = [norm(cov(θ, dims=1, corrected=true)) for θ in ekiobj.θ]

    tau_est = ensemble_mean(ekiobj)
    grid_vor_est = reconstruct_initial_vorticity_eki(sparam, tau_est)
    rel_vorticity_error = norm(grid_vor_est - sparam.grid_vor) / norm(sparam.grid_vor)
    rel_observation_error = norm(y_obs - ekiobj.y_pred[end]) / norm(y_obs)

    result = (
        sparam=sparam,
        tau_ref=tau_ref,
        y_ref=y_ref,
        y_obs=y_obs,
        obs_cov=obs_cov,
        ekiobj=ekiobj,
        tau_est=tau_est,
        grid_vor_est=grid_vor_est,
        vorticity_errors=vorticity_errors,
        observation_errors=observation_errors,
        covariance_norms=covariance_norms,
        rel_vorticity_error=rel_vorticity_error,
        rel_observation_error=rel_observation_error,
        output_file=output_file,
        plot_files=String[],
    )

    if save_plots
        prefix = isnothing(plot_prefix) ? eki_plot_prefix(output_file, filter_type) : plot_prefix
        plot_files = section52_write_standard_plots(result; method_label=filter_type, plot_prefix=prefix)
        
        # Call plot_param_iter and save
        fig_param = PyPlot.figure()
        θ_ref_names = ["tau_$i" for i in 1:length(result.tau_ref)]
        plot_param_iter(result.ekiobj, result.tau_ref, θ_ref_names)
        PyPlot.savefig(prefix * "_param_iter.png", dpi=180)
        PyPlot.close(fig_param)
        push!(plot_files, prefix * "_param_iter.png")
        
        # Call plot_opt_errors and save
        plot_opt_errors(result.ekiobj, result.tau_ref)
        PyPlot.savefig(prefix * "_opt_errors.png", dpi=180)
        PyPlot.close()
        push!(plot_files, prefix * "_opt_errors.png")

        push!(plot_files, section52_plot_covariance_norm(
            result.covariance_norms,
            prefix * "_cov_norm.png";
            method_label=filter_type,
        ))
        
        result = merge(result, (plot_files=plot_files,))
    end

    mkpath(dirname(output_file))
    serialize(output_file, result)

    return result
end

function run_section52_eki_smoke_test(;
    output_file::String=joinpath(@__DIR__, "Figs", "EKI_Barotropic_smoke.jls"),
)
    return run_section52_eki(
        num_fourier=8,
        nlat=16,
        model_dt=1800,
        end_time=3600,
        n_obs_frames=1,
        nobs=4,
        trunc_N=2,
        n_iter=1,
        n_ens=17,
        perturbation_wavenumber=2.0,
        output_file=output_file,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_section52_eki_smoke_test()
    @info "Finished Section 5.2 EKI smoke test" result.rel_vorticity_error result.rel_observation_error result.output_file
end
