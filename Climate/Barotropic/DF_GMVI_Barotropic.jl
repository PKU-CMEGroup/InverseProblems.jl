using NNGCM
using LinearAlgebra
using Random
using Distributions
using JLD2
include("Barotropic.jl")
include("../../Inversion/Plot.jl")
include("../../Inversion/DF_GMVI.jl")

num_fourier, nθ = 42, 64 #85, 128
nλ = 2nθ
Δt, end_time =  1800, 86400
n_obs_frames = 2
obs_time, nobs = Int64(end_time/n_obs_frames), 50
antisymmetric = false
trunc_N = 8
N_θ = (trunc_N+2)*trunc_N
N_y = nobs*n_obs_frames 
N_f = N_y + N_θ 

obs_coord = zeros(Int64, nobs, 2)
Random.seed!(42)
obs_coord[:,1], obs_coord[:, 2] = rand(1:nλ-1, nobs), rand(1:div(nθ,2), nobs)
# obs_coord[:,1], obs_coord[:, 2] = rand(1:nλ-1, nobs), rand(1:nθ-1, nobs)

# Initialization for false results 
spe_mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b, grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, init_data = 
Barotropic_Init(num_fourier, nθ; trunc_N = trunc_N, radius = 6371.2e3, m = 4.0, θ0 = 45.0 * pi / 180,  θw = 15.0 * pi / 180.0, A = 0.0, symmetric = false)
barotropic_A0 = Setup_Param(num_fourier, nθ, Δt, end_time, n_obs_frames, obs_coord, antisymmetric, N_y, trunc_N, spe_mesh,
                         grid_u_b, grid_v_b, grid_vor_b, spe_vor_b,      # background velocity/vorticity profiles
                         grid_u, grid_v, grid_vor, spe_vor, init_data);
# Generate reference observation
_, obs_raw_data_A0 = Barotropic_Main(barotropic_A0, barotropic_A0.init_data);




# Initialization for truth resutls
spe_mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b, grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, init_data = 
Barotropic_Init(num_fourier, nθ; trunc_N = trunc_N, radius = 6371.2e3, m = 2.0, θ0 = 45.0 * pi / 180,  θw = 15.0 * pi / 180.0, A = 4.0e-5, symmetric = false)
barotropic = Setup_Param(num_fourier, nθ, Δt, end_time, n_obs_frames, obs_coord, antisymmetric, N_y, trunc_N, spe_mesh,
                         grid_u_b, grid_v_b, grid_vor_b, spe_vor_b,      # background velocity/vorticity profiles
                         grid_u, grid_v, grid_vor, spe_vor, init_data);
# Generate reference observation
spe_mesh, obs_raw_data = Barotropic_Main(barotropic, barotropic.init_data);
   

# Plot solutions    
Lat_Lon_Pcolormesh(spe_mesh, grid_u_b,  1; save_file_name = "Figs/Barotropic_u_backgroud.pdf", cmap = "viridis")
Lat_Lon_Pcolormesh(spe_mesh, grid_vor_b,  1; save_file_name = "Figs/Barotropic_vor_backgroud.pdf", cmap = "viridis")
Lat_Lon_Pcolormesh(spe_mesh, grid_vor_pert, 1; save_file_name = "Figs/Barotropic_vor_pert0.pdf", cmap = "viridis")
Lat_Lon_Pcolormesh(spe_mesh, grid_vor, 1; save_file_name = "Figs/Barotropic_vor0.pdf", cmap = "viridis")
Lat_Lon_Pcolormesh(spe_mesh, grid_u, 1; save_file_name = "Figs/Barotropic_vel_u0.pdf", cmap = "viridis")
    


# Plot observation data
obs_coord = barotropic.obs_coord
n_obs_frames = barotropic.n_obs_frames
antisymmetric = barotropic.antisymmetric
for i_obs = 1:n_obs_frames
    Lat_Lon_Pcolormesh(spe_mesh, obs_raw_data["vel_u"][i_obs], 1, obs_coord; save_file_name =   "Figs/Barotropic_u-"*string(i_obs)*".pdf", cmap = "viridis", antisymmetric=antisymmetric)
    Lat_Lon_Pcolormesh(spe_mesh, obs_raw_data["vor"][i_obs], 1, obs_coord; save_file_name =   "Figs/Barotropic_vor-"*string(i_obs)*".pdf", cmap = "viridis", antisymmetric=antisymmetric)
end     

# exit()


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
    Trans_Grid_To_Spherical!(spe_mesh, grid_vor_mirror, spe_vor_mirror)
    spe_mesh, obs_raw_data_mirror = Barotropic_Main(barotropic, grid_vor_mirror; init_type = "grid_vor");
    init_data_mirror = spe_to_param(spe_vor_mirror-barotropic.spe_vor_b, barotropic.trunc_N; radius=barotropic.radius)

    θ0_mean[1, :]    .= barotropic.init_data
    θ0_mean[2, :]    .= init_data_mirror
end
###################################################





y_noiseless_A0 = convert_obs(barotropic.obs_coord, obs_raw_data_A0; antisymmetric=barotropic.antisymmetric)
y_noiseless = convert_obs(barotropic.obs_coord, obs_raw_data; antisymmetric=barotropic.antisymmetric)

σ_η = 1.0e-6
Random.seed!(123);
y_obs = y_noiseless + 0.0*rand(Normal(0, σ_η), N_y)
Σ_η = Array(Diagonal(fill(σ_η^2, N_y)))



Δt = 0.5
T = N_iter * Δt
func_F(x) = barotropic_F(barotropic, (y_obs, μ_0, σ_η, σ_0), x)
@info "Phi_R of two potential modes: ", 0.5*func_F(barotropic_A0.init_data)'*func_F(barotropic_A0.init_data), 0.5*func_F(barotropic.init_data)'*func_F(barotropic.init_data)


@time df_gmviobj = DF_GMVI_Run(func_F, T, N_iter, θ0_w, θ0_mean, θθ0_cov;
                               sqrt_matrix_type = "Cholesky",
                               # setup for Gaussian mixture part
                               quadrature_type_GM = "mean_point",
                               N_f = N_f,
                               quadrature_type = "unscented_transform",
                               c_weight_BIP = 1.0e-3,
                               w_min=1e-8)
@save "df_gmviobj.jld2" df_gmviobj


# df_gmviobj = load("df_gmviobj.jld2")["df_gmviobj"]





plot_results(barotropic, df_gmviobj)