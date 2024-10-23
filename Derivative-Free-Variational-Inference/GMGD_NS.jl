using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2

include("../Inversion/Plot.jl")
include("../Inversion/GMGD.jl")
include("../Fluid/Spectral-Navier-Stokes.jl")
include("../Fluid/Spectral-Mesh.jl")


ν = 1.0e-2                                      # viscosity
N, L = 128, 2*pi                                # resolution and domain size 
ub, vb = 0.0, 2*pi                              # background velocity 
method="Crank-Nicolson"                         # RK4 or Crank-Nicolson
N_t = 2500;                                     # time step
T = 0.5;                                        # final time
obs_ΔNx, obs_ΔNy, obs_ΔNt = 8, 16, 1250         # observation
symmetric = true
σ_0 = sqrt(2)*pi
N_KL = 128
N_θ = 128

seed=22
Random.seed!(seed);

mesh = Spectral_Mesh(N, N, L, L)
s_param = Setup_Param(ν, ub, vb,  
    N, L,  
    method, N_t, T,
    obs_ΔNx, obs_ΔNy, obs_ΔNt; 
    symmetric = symmetric,
    N_ω0_θ = N_θ,
    N_ω0_ref = N_KL,
    f = (x, y) -> (0, cos(4*x)),
    σ = σ_0,
    ω0_seed=seed)
ω0_ref = s_param.ω0_ref
ω0_ref_mirror = -ω0_ref[[1;end:-1:2], :]
# generate observation data
y_noiseless = forward_helper(s_param, ω0_ref; symmetric=true, save_file_name="NS", vmin=-5.0, vmax=5.0);
y_noiseless_mirror = forward_helper(s_param, ω0_ref_mirror; symmetric=true, save_file_name="NS_mirror", vmin=-5.0, vmax=5.0);
@info "y - y_mirror = ", norm(y_noiseless - y_noiseless_mirror)

# compute posterior distribution by GMKI
N_iter = 50
N_modes = 3
θ0_w  = fill(1.0, N_modes)/N_modes



θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)
for i = 1:N_modes
    θ0_mean[i, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[i, :, :] .= Array(Diagonal(fill(1.0^2, N_θ)))
end
μ_0 = zeros(Float64, N_θ)  # prior/initial mean 
Σ_0 = Array(Diagonal(fill(σ_0^2, N_θ)))  # prior/initial covariance


σ_η = 0.1
N_y = length(y_noiseless)
y = y_noiseless + rand(Normal(0, σ_η), N_y)
Σ_η = Array(Diagonal(fill(σ_η^2, N_y)))


### Augment the system
s_param.N_y = N_f = N_y + N_θ


dt = 0.5
T = dt*N_iter
func_args = (y, μ_0, σ_η, σ_0)
func_F(x) = NS_F(s_param, func_args, x)
        
gmgdobj = GMGD_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        θ0_w, θ0_mean, θθ0_cov;
        sqrt_matrix_type = "Cholesky",
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = true, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1.0e-3,
        w_min=1e-10)
@save "gmgdobj-NS.jld2" gmgdobj




#################################################################################################################################################

function plot_field(mesh::Spectral_Mesh, grid_dat::Array{Float64,2}, clim, ax; cmap="viridis")
    
    N_x, N_y = mesh.N_x, mesh.N_y
    xx, yy = mesh.xx, mesh.yy
    X,Y = repeat(xx, 1, N_y), repeat(yy, 1, N_x)'
    
    return ax.pcolormesh(X, Y, grid_dat, shading= "gouraud", clim=clim, cmap=cmap)
end


N_ens = 2N_θ + 1
# visulize the log permeability field
fig_vor, ax_vor = PyPlot.subplots(ncols = 5, sharex=true, sharey=true, figsize=(20,4))
for ax in ax_vor ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(s_param.ω0_ref), maximum(s_param.ω0_ref))

plot_field(mesh, s_param.ω0_ref, color_lim, ax_vor[1]) 
ax_vor[1].set_title("Truth")

plot_field(mesh, -s_param.ω0_ref[[1;end:-1:2], :], color_lim, ax_vor[2]) 
ax_vor[2].set_title("Truth (mirrored)")

grid_vor = Random_Field_From_Theta(mesh, gmgdobj.x_mean[N_iter][1,:], s_param.seq_pairs)   
plot_field(mesh, grid_vor,  color_lim, ax_vor[3]) 
ax_vor[3].set_title("Mode 1")

grid_vor = Random_Field_From_Theta(mesh, gmgdobj.x_mean[N_iter][2,:], s_param.seq_pairs)   
plot_field(mesh, grid_vor,  color_lim, ax_vor[4]) 
ax_vor[4].set_title("Mode 2")

grid_vor = Random_Field_From_Theta(mesh, gmgdobj.x_mean[N_iter][3,:], s_param.seq_pairs)   
plot_field(mesh, grid_vor,  color_lim, ax_vor[5]) 
ax_vor[5].set_title("Mode 3")


fig_vor.tight_layout()
fig_vor.savefig("NS-2D-vor.pdf")





N_ens = 2N_θ + 1
fig, (ax1, ax2, ax3, ax4) = PyPlot.subplots(ncols=4, figsize=(20,5))
ites = Array(LinRange(0, N_iter-1, N_iter))
errors = zeros(Float64, (3, N_iter, N_modes))

for m = 1:N_modes
    for i = 1:N_iter
        if m in [2,3]
            grid_vor_truth = s_param.ω0_ref
        else
            grid_vor_truth = -s_param.ω0_ref[[1;end:-1:2], :]
        end
        
        grid_vor = Random_Field_From_Theta(mesh, gmgdobj.x_mean[i][m,:], s_param.seq_pairs)   

        errors[1, i, m] = norm(grid_vor_truth - grid_vor)/norm(grid_vor_truth)
        errors[2, i, m] = gmgdobj.phi_r_pred[i][m]
        errors[3, i, m] = norm(gmgdobj.xx_cov[i][m,:,:])
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


x_w = exp.(hcat(gmgdobj.logx_w...))
for m = 1: N_modes
    ax4.plot(ites, x_w[m, 1:N_iter], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "mode "*string(m))
end
ax4.set_xlabel("Iterations")
ax4.set_ylabel("Weights")
ax4.legend()
fig.tight_layout()
fig.savefig("NS-2D-convergence.pdf")


fig, ax = PyPlot.subplots(ncols=1, figsize=(16,5))
θ_ref = s_param.ω0_θ_ref

n_ind = 16
θ_ind = Array(1:n_ind)
ax.scatter(θ_ind, θ_ref[θ_ind], s = 100, marker="x", color="black", label="Truth")
for m = 1:N_modes
    ax.scatter(θ_ind, gmgdobj.x_mean[N_iter][m,θ_ind], s = 50, marker="o", color="C"*string(m), facecolors="none", label="Mode "*string(m))
end

Nx = 1000
for i in θ_ind
    θ_min = minimum(gmgdobj.x_mean[N_iter][:,i] .- 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
    θ_max = maximum(gmgdobj.x_mean[N_iter][:,i] .+ 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
        
    xxs = zeros(N_modes, Nx)  
    zzs = zeros(N_modes, Nx)  
    for m =1:N_modes
        xxs[m, :], zzs[m, :] = Gaussian_1d(gmgdobj.x_mean[N_iter][m,i], gmgdobj.xx_cov[N_iter][m,i,i], Nx, θ_min, θ_max)
        zzs[m, :] *= exp(gmgdobj.logx_w[N_iter][m])
    end
    label = nothing
    if i == 1
        label = "DF-GMGD"
    end
    ax.plot(sum(zzs, dims=1)' .+ i, xxs[1,:], linestyle="-", color="C0", fillstyle="none", label=label)
    ax.plot(fill(i, Nx), xxs[1,:], linestyle=":", color="black", fillstyle="none")
        
end
ax.set_xticks(θ_ind)
ax.set_xlabel(L"\theta" * " indices")
ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
fig.tight_layout()
fig.savefig("NS-2D-density.pdf")

