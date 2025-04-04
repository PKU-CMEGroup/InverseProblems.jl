using NNGCM
using LinearAlgebra
using Random
using Distributions
using JLD2
include("Barotropic.jl")
include("../../Inversion/Plot.jl")
include("../../Inversion/DF_GMVI.jl")

num_fourier, nθ = 42, 64 #85, 128
Δt, end_time =  1800, 86400
n_obs_frames = 2
obs_time, nobs = Int64(end_time/n_obs_frames), 100
antisymmetric = false
trunc_N = 10
N_θ = (trunc_N+2)*trunc_N
N_y = nobs*n_obs_frames 
N_f = N_y + N_θ 
# already initialized
barotropic = Setup_Param(num_fourier, nθ, Δt, end_time, n_obs_frames, nobs, antisymmetric, N_y, trunc_N);



# Generate reference observation
# mesh, obs_raw_data = Barotropic_Main(barotropic, nothing; init_type = "truth");
mesh, obs_raw_data = Barotropic_Main(barotropic, barotropic.init_data; init_type = "spec_vor");


# Plot observation data
obs_coord = barotropic.obs_coord
n_obs_frames = barotropic.n_obs_frames
antisymmetric = barotropic.antisymmetric
for i_obs = 1:n_obs_frames
    Lat_Lon_Pcolormesh(mesh, obs_raw_data["vel_u"][i_obs], 1, obs_coord; save_file_name =   "Figs/Barotropic_u-"*string(i_obs)*".pdf", cmap = "viridis", antisymmetric=antisymmetric)
    Lat_Lon_Pcolormesh(mesh, obs_raw_data["vor"][i_obs], 1, obs_coord; save_file_name =   "Figs/Barotropic_vor-"*string(i_obs)*".pdf", cmap = "viridis", antisymmetric=antisymmetric)
end     




# compute posterior distribution by UKI
N_iter = 20
N_modes = 3
θ0_w  = fill(1.0, N_modes)/N_modes


μ_0 = zeros(Float64, N_θ)  # prior/initial mean 
θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)
Random.seed!(63);
σ_0 = 10.0
for i = 1:N_modes
    θ0_mean[i, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[i, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
end

########################### CHEATING ############
DEBUG = false
if DEBUG
    grid_vor_mirror = -barotropic.grid_vor[:, end:-1:1,  :]
    spe_vor_mirror = similar(barotropic.spe_vor_b)
    Trans_Grid_To_Spherical!(mesh, grid_vor_mirror, spe_vor_mirror)
    mesh, obs_raw_data_mirror = Barotropic_Main(barotropic, grid_vor_mirror; init_type = "grid_vor");
    init_data_mirror = spe_to_param(spe_vor_mirror-barotropic.spe_vor_b, barotropic.trunc_N; radius=barotropic.radius)

    θ0_mean[1, :]    .= barotropic.init_data
    θ0_mean[2, :]    .= init_data_mirror
end
###################################################





y_noiseless = convert_obs(barotropic.obs_coord, obs_raw_data; antisymmetric=barotropic.antisymmetric)

σ_η = 1.0e-6
Random.seed!(123);
y_obs = y_noiseless + 0.0*rand(Normal(0, σ_η), N_y)
Σ_η = Array(Diagonal(fill(σ_η^2, N_y)))



Δt = 0.5
T = N_iter * Δt
func_F(x) = barotropic_F(barotropic, (y_obs, μ_0, σ_η, σ_0), x)

@time df_gmviobj = DF_GMVI_Run(func_F, T, N_iter, θ0_w, θ0_mean, θθ0_cov;
                               sqrt_matrix_type = "Cholesky",
                               # setup for Gaussian mixture part
                               quadrature_type_GM = "mean_point",
                               N_f = N_f,
                               quadrature_type = "unscented_transform",
                               c_weight_BIP = 1.0e-3,
                               w_min=1e-8)
@save "df_gmviobj.jld2" df_gmviobj





function plot_field(mesh::Spectral_Spherical_Mesh, grid_dat::Array{Float64,3}, level::Int64, clim, ax; cmap="viridis")
    
    λc, θc = mesh.λc, mesh.θc
    nλ, nθ = length(λc), length(θc)
    λc_deg, θc_deg = λc*180/pi, θc*180/pi
    
    X,Y = repeat(λc_deg, 1, nθ), repeat(θc_deg, 1, nλ)'
    
    
    return ax.pcolormesh(X, Y, grid_dat[:,:,level], shading= "gouraud", clim=clim, cmap=cmap)
    
end


N_ens = 2N_θ + 1
# visulize the log permeability field
fig_vor, ax_vor = PyPlot.subplots(ncols = 4, sharex=true, sharey=true, figsize=(20,5))
for ax in ax_vor ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(barotropic.grid_vor), maximum(barotropic.grid_vor))

plot_field(mesh, barotropic.grid_vor, 1, color_lim, ax_vor[1]) 
ax_vor[1].set_title("Truth")

spe_vor, grid_vor = copy(barotropic.spe_vor), copy(barotropic.grid_vor)


Barotropic_ω0!(mesh, "spec_vor", df_gmviobj.x_mean[N_iter][1,:], spe_vor, grid_vor; spe_vor_b = barotropic.spe_vor_b)
plot_field(mesh, grid_vor, 1,  color_lim, ax_vor[2]) 
ax_vor[2].set_title("Mode 1")

Barotropic_ω0!(mesh, "spec_vor", df_gmviobj.x_mean[N_iter][2,:], spe_vor, grid_vor; spe_vor_b = barotropic.spe_vor_b)
plot_field(mesh, grid_vor, 1,  color_lim, ax_vor[3]) 
ax_vor[3].set_title("Mode 2")

Barotropic_ω0!(mesh, "spec_vor", df_gmviobj.x_mean[N_iter][3,:], spe_vor, grid_vor; spe_vor_b = barotropic.spe_vor_b)
plot_field(mesh, grid_vor, 1,  color_lim, ax_vor[4]) 
ax_vor[4].set_title("Mode 3")


fig_vor.tight_layout()
fig_vor.savefig("Barotropic-2D-vor.pdf")





N_ens = 2N_θ + 1
fig, (ax1, ax2, ax3, ax4) = PyPlot.subplots(ncols=4, figsize=(20,5))
ites = Array(LinRange(0, N_iter-1, N_iter))
errors = zeros(Float64, (3, N_iter, N_modes))
spe_vor, grid_vor = copy(barotropic.spe_vor), copy(barotropic.grid_vor)

for m = 1:N_modes
    for i = 1:N_iter
        if m == N_modes
            grid_vor_truth = barotropic.grid_vor
        else
            grid_vor_truth = barotropic.grid_vor[:, end:-1:1]
        end
        
        
        Barotropic_ω0!(mesh, "spec_vor", df_gmviobj.x_mean[i][m,:], spe_vor, grid_vor; spe_vor_b = barotropic.spe_vor_b)
        errors[1, i, m] = norm(grid_vor_truth - grid_vor)/norm(grid_vor_truth)
        errors[2, i, m] = 0.5*(df_gmviobj.y_pred[i][m,:] - df_gmviobj.y)'*(df_gmviobj.Σ_η\(df_gmviobj.y_pred[i][m,:] - df_gmviobj.y))
        errors[3, i, m] = norm(df_gmviobj.θθ_cov[i][m,:,:])
    end
end

linestyles = ["o"; "x"; "s"]
markevery = 5
for m = 1: N_modes
    ax1.plot(ites, errors[1, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Rel. error of a(x)")
ax1.legend()

for m = 1: N_modes
    ax2.semilogy(ites, errors[2, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax2.set_xlabel("Iterations")
ax2.set_ylabel(L"\Phi_R")
ax2.legend()

for m = 1: N_modes
    ax3.plot(ites, errors[3, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Frobenius norm of covariance")
ax3.legend()


θ_w = exp.(hcat(df_gmviobj.logθ_w...))
for m = 1: N_modes
    ax4.plot(ites, θ_w[m, 1:N_iter], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax4.set_xlabel("Iterations")
ax4.set_ylabel("Weights")
ax4.legend()
fig.tight_layout()
fig.savefig("Barotropic-2D-convergence.pdf")


fig, ax = PyPlot.subplots(ncols=1, figsize=(16,5))
θ_ref = barotropic.init_data

n_ind = 16
θ_ind = Array(1:n_ind)
ax.scatter(θ_ind, θ_ref[θ_ind], s = 100, marker="x", color="black", label="Truth")
for m = 1:N_modes
    ax.scatter(θ_ind, df_gmviobj.x_mean[N_iter][m,θ_ind], s = 50, marker="o", color="C"*string(m), facecolors="none", label="Mode "*string(m))
end

Nx = 1000
for i in θ_ind
    θ_min = minimum(df_gmviobj.x_mean[N_iter][:,i] .- 3sqrt.(df_gmviobj.θθ_cov[N_iter][:,i,i]))
    θ_max = maximum(df_gmviobj.x_mean[N_iter][:,i] .+ 3sqrt.(df_gmviobj.θθ_cov[N_iter][:,i,i]))
        
    xxs = zeros(N_modes, Nx)  
    zzs = zeros(N_modes, Nx)  
    for m =1:N_modes
        xxs[m, :], zzs[m, :] = Gaussian_1d(df_gmviobj.x_mean[N_iter][m,i], df_gmviobj.θθ_cov[N_iter][m,i,i], Nx, θ_min, θ_max)
        zzs[m, :] *= exp(df_gmviobj.logθ_w[N_iter][m]) * 3
    end
    label = nothing
    if i == 1
        label = "GMKI"
    end
    ax.plot(sum(zzs, dims=1)' .+ i, xxs[1,:], linestyle="-", color="C0", fillstyle="none", label=label)
    ax.plot(fill(i, Nx), xxs[1,:], linestyle=":", color="black", fillstyle="none")
        
end
ax.set_xticks(θ_ind)
ax.set_xlabel(L"\theta" * " indices")
ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
fig.tight_layout()
fig.savefig("Barotropic-2D-density.pdf")

