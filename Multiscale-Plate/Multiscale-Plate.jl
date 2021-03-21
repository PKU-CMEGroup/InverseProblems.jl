using NNFEM
using JLD2
using Statistics
using Random
using LinearAlgebra
using Distributions



include("../Plot.jl")
include("CommonFuncs.jl")
include("../RExKI.jl")
include("../ModelError/Misfit2Diagcov.jl")
mutable struct Params
    θ_name::Array{String, 1}
    θ_func::Function
    dθ_func::Function
    θ_func_inv::Function
    matlaw::String

    ρ::Float64
    tids::Array{Int64,1}
    force_scale::Float64
    
    NT::Int64
    T::Float64
    
    n_tids::Int64
    n_obs_point::Int64
    n_obs::Int64
    n_data::Int64
end

function Params(matlaw::String, tids::Array{Int64,1}, n_obs_point::Int64 = 2, n_obs_time::Int64 = 200, T::Float64 = 200.0, NT::Int64 = 200)
    if matlaw == "PlaneStressPlasticity"
        θ_name = ["E",   "nu",  "sigmaY", "K"] 
        θ_func = (θ)-> [θ[1]*1.0e+5, 1/(2+3exp(θ[2])),  θ[3]*1.0e+3, θ[4]*1.0e+4]
        θ_func_inv = (θ)-> [θ[1]/1.0e+5, log((1 - 2θ[2])/(3θ[2])),  θ[3]/1.0e+3, θ[4]/1.0e+4]
        dθ_func = (θ)-> [1.0e+5        0                             0            0;
                         0      -3exp(θ[2])/(2+3exp(θ[2]))^2         0            0;
                         0             0                           1.0e+3         0;
                         0             0                              0         1.0e+4]
        # θ_func = (θ)-> [θ[1]*1.0e+5, θ[2]*0.02,  θ[3]*1.0e+3, θ[4]*1.0e+4]
        # θ_func_inv = (θ)-> [θ[1]/1.0e+5, θ[2]/0.02,  θ[3]/1.0e+3, θ[4]/1.0e+4]
        # dθ_func = (θ)-> [1.0e+5        0               0            0;
        #                     0         0.02             0            0;
        #                     0             0         1.0e+3          0;
        #                     0             0            0         1.0e+4]

    else
        error("unrecognized matlaw: ", matlaw)
    end

    fiber_fraction = 0.25
    ρ = 4.5*(1 - fiber_fraction) + 3.2*fiber_fraction
    
    force_scale = 5.0
    n_tids = length(tids)
    n_data = 2n_obs_point * div(n_obs_time, ΔNT) * n_tids
    
    return Params(θ_name, θ_func, dθ_func, θ_func_inv, matlaw, ρ, tids, force_scale, NT, T,  n_tids, n_obs_point, n_obs_point * n_obs_time, n_data)
end

function Foward(phys_params::Params, θ::Array{Float64,1})
    θ_func, ρ, tids, force_scale, n_data = phys_params.θ_func, phys_params.ρ, phys_params.tids, phys_params.force_scale, phys_params.n_data
    matlaw = phys_params.matlaw
    n_obs = div(n_data, length(tids))
    obs = zeros(Float64, n_data)
    
    for tid = 1:length(tids)
        _, data = Run_Homogenized(θ, θ_func, matlaw, ρ, tids[tid], force_scale)
        obs[(tid-1)*n_obs+1:tid*n_obs] = data[:][ΔNT:ΔNT:end]
    end
    
    return obs
end


function Ensemble(phys_params::Params,  params_i::Array{Float64, 2})
    n_data = phys_params.n_data
    
    N_ens,  N_θ = size(params_i)
    
    g_ens = zeros(Float64, N_ens,  n_data)
    
    Threads.@threads for i = 1:N_ens 
        # g: N_ens x N_data
        g_ens[i, :] .= Foward(phys_params, params_i[i, :])
    end
    
    return g_ens
end



function ExKI(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, N_iter::Int64 = 100)
    
    
    parameter_names = ["E"]
    
    ens_func(θ_ens) = Ensemble(phys_params, θ_ens)
    
    
    exkiobj = ExKIObj(parameter_names,
    θ0_bar, 
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    update_cov = 1
    PLOT = true
    for i in 1:N_iter
        @info size(exkiobj.g_t),  phys_params.NT, 4
        update_ensemble!(exkiobj, ens_func) 
        @info "θ: ", exkiobj.θ_bar[end]
        @info "θθ_cov: ", exkiobj.θθ_cov[end]
        @info "norm(θ) :", norm(exkiobj.θ_bar[end]), "norm(θθ_cov) :", norm(exkiobj.θθ_cov[end]) 
        
        @info "F error of data_mismatch :", (exkiobj.g_bar[end] - exkiobj.g_t)'*(exkiobj.obs_cov\(exkiobj.g_bar[end] - exkiobj.g_t))
        
        
        if (update_cov > 0) && (i%update_cov == 0) 
            exkiobj.θθ_cov[1] = copy(exkiobj.θθ_cov[end])
        end


        if PLOT 
            ExKI_Plot(phys_params, exkiobj, i)
        end
        
    end
    
    return exkiobj
end

function ExKI_Plot(phys_params, exkiobj, i)
    T, NT, n_tids, n_data, matlaw = phys_params.T, phys_params.NT, phys_params.n_tids, phys_params.n_data, phys_params.matlaw
         
    n_obs_point = 2 # top left and right corners
    n_obs_time = div(n_data, 2n_obs_point*n_tids)
    
    obs_ref = reshape(exkiobj.g_t,    n_obs_time ,  2n_obs_point, n_tids)
    obs_init = reshape(exkiobj.g_bar[1], n_obs_time ,  2n_obs_point, n_tids)
    obs = reshape(exkiobj.g_bar[i], n_obs_time ,  2n_obs_point, n_tids)

    fig_disp, ax_disp = PyPlot.subplots(nrows=2, ncols = 4,  sharex=true, sharey=true, figsize=(24,12))
    ts = LinRange(0, T, NT+1)
    L_scale, t_scale = scales[1], scales[3]


    for disp_id = 1:2
        # first tid first point
        ax_disp[disp_id,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,disp_id, 1]*L_scale, "-o", color="grey",fillstyle="none", label = "Observation")
        ax_disp[disp_id,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_init[:,   disp_id, 1]*L_scale, "-g",label = "UKI (initial)", markevery=20)
        ax_disp[disp_id,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs[ :,   disp_id, 1]*L_scale, "-r*",label = "UKI", markevery=20)
        ax_disp[disp_id,1].set
        # first tid second point
        ax_disp[disp_id,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,disp_id+2, 1]*L_scale, "-o", color="grey",fillstyle="none", label = "Observation")
        ax_disp[disp_id,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_init[:,    disp_id+2, 1]*L_scale, "-g",label = "UKI (initial)", markevery=20)
        ax_disp[disp_id,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs[:,    disp_id+2, 1]*L_scale, "-r*",label = "UKI", markevery=20)

        # second tid first point
        ax_disp[disp_id,3].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,disp_id, 2]*L_scale, "-o", color="grey",fillstyle="none", label = "Observation")
        ax_disp[disp_id,3].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_init[ :,   disp_id, 2]*L_scale, "-g", label = "UKI (initial)", markevery=20)
        ax_disp[disp_id,3].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs[ :,   disp_id, 2]*L_scale, "-r*", label = "UKI", markevery=20)
         # second tid second point
        ax_disp[disp_id,4].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,disp_id+2, 2]*L_scale, "-o", color="grey",fillstyle="none", label = "Observation")
        ax_disp[disp_id,4].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_init[:,    disp_id+2, 2]*L_scale, "-g", label = "UKI (initial)", markevery=20)
        ax_disp[disp_id,4].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs[:,    disp_id+2, 2]*L_scale, "-r*", label = "UKI", markevery=20)

        
    end
    
    ax_disp[1, 1].set_ylabel("X-Displacement (cm)")
    ax_disp[2, 1].set_ylabel("Y-Displacement (cm)")
    ax_disp[2,1].set_xlabel("Time (s)")
    ax_disp[2,2].set_xlabel("Time (s)")
    ax_disp[2,3].set_xlabel("Time (s)")
    ax_disp[2,4].set_xlabel("Time (s)")





    ax_disp[1,1].legend()
    fig_disp.tight_layout()
    fig_disp.savefig("Plate_disp_test"*string(i)*".png")
    close(fig_disp)
end
function Multiscale_Test(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, 
    N_iter::Int64;
    ki_file = nothing)
    
    if ki_file === nothing
        kiobj = ExKI(phys_params, t_mean, t_cov,  θ0_bar, θθ0_cov, α_reg, N_iter)
        @save "exkiobj.dat" kiobj
    else
        @load "exkiobj.dat" kiobj
    end
    
    # optimization related plots
    fig_ite, ax_ite = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    
    ites = Array(1:N_iter)
    errors = zeros(Float64, (3, N_iter))
    for i in ites
        errors[1, i] = NaN
        errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
        errors[3, i] = norm(kiobj.θθ_cov[i])   
        
    end
    errors[3, 1] = norm(θθ0_cov) 
    ax_ite[1].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[2].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
    
    ax_ite[1].set_xlabel("Iterations")
    ax_ite[1].set_ylabel("Optimization error")
    ax_ite[1].grid(true)
    ax_ite[2].set_xlabel("Iterations")
    ax_ite[2].set_ylabel("Frobenius norm")
    ax_ite[2].grid(true)
    
    fig_ite.tight_layout()
    fig_ite.savefig("Plate-error.png")
    close("all")
    
    # parameter plot
    N_θ = length(θ0_bar)
    θ_bar_arr = hcat(kiobj.θ_bar...)
    θθ_std = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        θ_bar_arr[:,i] = phys_params.θ_func(θ_bar_arr[:,i])
        θθ_cov = phys_params.dθ_func(θ_bar_arr[:,i]) * kiobj.θθ_cov[i] * phys_params.dθ_func(θ_bar_arr[:,i])' 
        for j = 1:N_θ
            θθ_std[j, i] = sqrt(θθ_cov[j,j])
        end
    end


    stress_scale = scales[2]

    errorbar(ites, θ_bar_arr[1,ites] * stress_scale, yerr=3.0*θθ_std[1,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"E~(GPa)")
    errorbar(ites, θ_bar_arr[2,ites], yerr=3.0*θθ_std[2,ites], fmt="--o",fillstyle="none", label=L"ν") 
    errorbar(ites, θ_bar_arr[3,ites] * stress_scale, yerr=3.0*θθ_std[3,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"σ_Y~(GPa)")
    errorbar(ites, θ_bar_arr[4,ites] * stress_scale, yerr=3.0*θθ_std[4,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"K~(GPa)")
    semilogy()
    xlabel("Iterations")
    legend()
    tight_layout()
    savefig("Plate_theta.png")
    close("all")
    
    return kiobj
end

function Multiscale_Test_Plot(phys_params::Params, 
    t_mean::Array{Float64,1}, t_cov::Array{Float64,2}, 
    θ0_bar::Array{Float64,1}, θθ0_cov::Array{Float64,2}, 
    α_reg::Float64, 
    N_iter::Int64;
    ki_file)
    

    @load "exkiobj.dat" kiobj
    ExKI_Plot(phys_params, kiobj, N_iter)
    # optimization related plots
    fig_ite, ax_ite = PyPlot.subplots(ncols = 2, nrows=1, sharex=false, sharey=false, figsize=(12,6))
    
    ites = Array(1:N_iter)
    errors = zeros(Float64, (3, N_iter))
    for i in ites
        errors[1, i] = NaN
        errors[2, i] = 0.5*(kiobj.g_bar[i] - kiobj.g_t)'*(kiobj.obs_cov\(kiobj.g_bar[i] - kiobj.g_t))
        errors[3, i] = norm(kiobj.θθ_cov[i])   
        
    end
    errors[3, 1] = norm(θθ0_cov) 
    ax_ite[1].semilogy(ites, errors[2, :], "-o", fillstyle="none", markevery=2 )
    ax_ite[2].semilogy(ites, errors[3, :], "-o", fillstyle="none", markevery=2 )
    
    ax_ite[1].set_xlabel("Iterations")
    ax_ite[1].set_ylabel("Optimization error")
    ax_ite[1].grid(true)
    ax_ite[2].set_xlabel("Iterations")
    ax_ite[2].set_ylabel("Frobenius norm")
    ax_ite[2].grid(true)
    
    fig_ite.tight_layout()
    fig_ite.savefig("Plate-error.png")
    close("all")
    
    # parameter plot
    N_θ = length(θ0_bar)
    θ_bar_arr = hcat(kiobj.θ_bar...)
    θθ_std = zeros(Float64, (N_θ, N_iter+1))
    for i = 1:N_iter+1
        θ_bar_arr[:,i] = phys_params.θ_func(θ_bar_arr[:,i])
        θθ_cov = phys_params.dθ_func(θ_bar_arr[:,i]) * kiobj.θθ_cov[i] * phys_params.dθ_func(θ_bar_arr[:,i])' 
        for j = 1:N_θ
            θθ_std[j, i] = sqrt(θθ_cov[j,j])
        end
    end

    
    stress_scale = scales[2]

    errorbar(ites, θ_bar_arr[1,ites] * stress_scale, yerr=3.0*θθ_std[1,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"E~(GPa)")
    errorbar(ites, θ_bar_arr[2,ites], yerr=3.0*θθ_std[2,ites], fmt="--o",fillstyle="none", label=L"ν") 
    errorbar(ites, θ_bar_arr[3,ites] * stress_scale, yerr=3.0*θθ_std[3,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"σ_Y~(GPa)")
    errorbar(ites, θ_bar_arr[4,ites] * stress_scale, yerr=3.0*θθ_std[4,ites]* stress_scale, fmt="--o",fillstyle="none", label=L"K~(GPa)")
    semilogy()
    xlabel("Iterations")
    legend()
    tight_layout()
    savefig("Plate_theta.png")
    close("all")
    
    return kiobj
end



function prediction(phys_params, kiobj, θ_mean, θθ_cov, obs_noise_level, porder::Int64=2, tid::Int64=300, force_scale::Float64=0.5, fiber_size::Int64=5)
    # test on 300
    
    @load "Data/order$porder/obs$(tid)_$(force_scale)_$(fiber_size).jld2" obs
    obs_ref = obs[ΔNT:ΔNT:end, :]
    
    
    θ_scale, matlaw, ρ, force_scale, n_tids, n_obs = phys_params.θ_func, phys_params.matlaw, phys_params.ρ, phys_params.force_scale, phys_params.n_tids, phys_params.n_obs
    
    # only visulize the first point
    NT, T = phys_params.NT, phys_params.T
    ts = LinRange(0, T, NT+1)
    
    # optimization related plots
    fig_disp, ax_disp = PyPlot.subplots(ncols = 2, nrows=2, sharex=true, sharey=true, figsize=(12,8))
    
    

    θθ_cov = (θθ_cov+θθ_cov')/2 
    θ_p = construct_sigma_ensemble(kiobj, θ_mean, θθ_cov)
    N_ens = kiobj.N_ens

    n_tids, n_obs_point, n_data = phys_params.n_tids, phys_params.n_obs_point, phys_params.n_data
    
    n_obs_time = div(n_data, 2n_obs_point * n_tids)

    obs = zeros(Float64, N_ens, n_obs_time * 2n_obs_point)

    # todo hard code
    nx, ny = 10, 5
    p_ids = [(nx*porder+1)*(ny*porder+1); (nx*porder+1)*(ny*porder) + div(nx*porder, 2) + 1]
    

    Threads.@threads for i = 1:N_ens
        θ = θ_p[i, :]

        @info "θ is ", θ
        
        obs[i, :] = Run_Homogenized(θ, phys_params.θ_func, matlaw, ρ, tid, force_scale; 
                                    T=200.0, NT=200, nx=nx, ny=ny, porder=2, p_ids=p_ids)[2][ΔNT:ΔNT:end]
    end

    obs_mean = obs[1, :]

    obs_cov  = construct_cov(kiobj,  obs, obs_mean)  #+ Array(Diagonal( obs_noise_level^2 * obs_mean.^2)) 
    obs_std = sqrt.(diag(obs_cov))

    obs_mean = reshape(obs_mean, n_obs_time , 2n_obs_point)
    obs_std = reshape(obs_std, n_obs_time , 2n_obs_point)

    markevery = 20
    L_scale, t_scale = scales[1], scales[3]

    @info size(ts[ΔNT+1:ΔNT:end]), size(obs_ref)
    # top right x
    ax_disp[1,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,1]*L_scale, "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[1,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_mean[:,1]*L_scale, "-*r",  markevery = markevery, label="UKI")
    ax_disp[1,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,1] + 3obs_std[:,1])*L_scale,  "--r")
    ax_disp[1,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,1] - 3obs_std[:,1])*L_scale,  "--r")
    # top middle x
    ax_disp[1,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:, 3]*L_scale, "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[1,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_mean[:,3]*L_scale, "-*r",  markevery = markevery, label="UKI")
    ax_disp[1,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,3] + 3obs_std[:,3])*L_scale,   "--r")
    ax_disp[1,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,3] - 3obs_std[:,3])*L_scale,   "--r")
    # top right y
    ax_disp[2,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:,2]*L_scale, "--o", color="grey", fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[2,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_mean[:,2]*L_scale, "-*r",  markevery = markevery, label="UKI")
    ax_disp[2,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,2]+ 3obs_std[:,2])*L_scale,   "--r")
    ax_disp[2,1].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,2]- 3obs_std[:,2])*L_scale,   "--r")
    # top middle y
    ax_disp[2,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_ref[:, 4]*L_scale, "--o", color="grey",  fillstyle="none", label="Reference", markevery = markevery)
    ax_disp[2,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, obs_mean[:,4]*L_scale, "-*r",    markevery = markevery, label="UKI")
    ax_disp[2,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,4]+ 3obs_std[:,4])*L_scale,   "--r")
    ax_disp[2,2].plot(ts[ΔNT+1:ΔNT:end]*t_scale, (obs_mean[:,4]- 3obs_std[:,4])*L_scale,  "--r")
    
    
    ax_disp[2,1].set_xlabel("Time (s)")
    ax_disp[2,2].set_xlabel("Time (s)")
    ax_disp[1,1].set_ylabel("X-Displacement (cm)")
    ax_disp[2,1].set_ylabel("Y-Displacement (cm)")
    ax_disp[1,1].legend()
    
    
    fig_disp.tight_layout()
    fig_disp.savefig("Plate_disp-$(tid).png")
    close(fig_disp)
end

# driver_test()

# error("stop")

tids = [100; 102]
porder = 2
fiber_size = 5
force_scale = 5.0
T = 200.0
NT = 200
n_obs_time = NT
n_obs_point = 2 # top left and right corners
obs_noise_level = 0.05
t_mean_noiseless = []
for i = 1:length(tids)
    @load "Data/order$porder/obs$(tids[i])_$(force_scale)_$(fiber_size).jld2" obs 
    obs = obs[:, :][:]

    # todo skip some
    obs = obs[ΔNT:ΔNT:end]

    push!(t_mean_noiseless, obs)
end

t_mean_noiseless = vcat(t_mean_noiseless...)



# first choice of the observation covariance, only observation error 
t_cov = Array(Diagonal(obs_noise_level^2 * t_mean_noiseless.^2))
# add 5 percents observation noise
Random.seed!(123); 
t_mean = copy(t_mean_noiseless)
for i = 1:length(t_mean)
    noise = obs_noise_level*t_mean[i] *  (rand(Uniform(0, 2))-1) #
    t_mean[i] += noise
end

# todo misspecified observation covariance estimation
# first choice of the observation covariance, only observation error 
t_cov = Array(Diagonal(   fill(0.0001^2, length(t_mean))   ))
# add 5 percents observation noise



α_reg = 1.0
N_iter = 20
matlaw = "PlaneStressPlasticity"

if matlaw == "PlaneStressPlasticity"

    phys_params = Params("PlaneStressPlasticity", tids, n_obs_point, n_obs_time, T, NT)
    N_θ = 4
    E, ν = 1e+6, 0.2
    θ0_bar =  phys_params.θ_func_inv([E; ν; 0.97e+4; 1e+5])
    θθ0_cov = Array(Diagonal(fill(1.0, N_θ))) 
    # todo remove ki_file = "exkData/exkiobj.dat"
    # kiobj_perf = Multiscale_Test(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter; ki_file = "exkData/exkiobj.dat")
    kiobj_perf = Multiscale_Test(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter)

else
    error("unrecognized matlaw: ", matlaw)
end


# update t_cov
data_misfit = (kiobj_perf.g_bar[end] - t_mean)
n_dm = length(kiobj_perf.g_bar[end] - t_mean)

diag_cov = Misfit2Diagcov(2, data_misfit, t_mean)
t_cov = Array(Diagonal(diag_cov))



# todo use the commented line
kiobj = Multiscale_Test(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter)
kiobj = Multiscale_Test_Plot(phys_params, t_mean, t_cov, θ0_bar, θθ0_cov, α_reg, N_iter; ki_file = "exkiobj.dat")

Ny_Nθ = n_dm/length(kiobj.θ_bar[end])
tid = 203
prediction(phys_params, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end]*Ny_Nθ, obs_noise_level, porder, tid, force_scale, fiber_size)

tid = 300
prediction(phys_params, kiobj, kiobj.θ_bar[end], kiobj.θθ_cov[end]*Ny_Nθ, obs_noise_level, porder, tid, force_scale, fiber_size)


#todo change ΔNT = 1 or 5 in CommonFuncs, to change the observation freqency 