using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2

include("../Inversion/Plot.jl")
include("../Inversion/GMBBVI.jl")
include("../Inversion/AnnealingInitialize.jl")
include("../Fluid/Darcy-2D.jl")

BLAS.set_num_threads(1)

#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
# function compute_f_2d(yy::Array{FT, 1}) where {FT<:AbstractFloat}
#     N = length(yy)
#     f_2d = zeros(FT, N, N)

#     for i = 1:N
#             f_2d[:,i] .= 1000.0 * (2.0 .+ cos(4pi*yy[i]) * sin.(pi*yy))
#     end
#     return f_2d
# end

#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
function compute_f_2d(yy::Array{FT, 1}) where {FT<:AbstractFloat}
    N = length(yy)
    f_2d = zeros(FT, N, N)
    for i = 1:N
        if (yy[i] <= 4/6)
            f_2d[:,i] .= 1000.0
        elseif (yy[i] >= 4/6 && yy[i] <= 5/6)
            f_2d[:,i] .= 2000.0
        elseif (yy[i] >= 5/6)
            f_2d[:,i] .= 3000.0
        end
    end
    return f_2d
end




#=
Compute observation values with symmetry consideration
=#
function compute_obs(darcy::Setup_Param{FT, IT}, h_2d::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[darcy.x_locs, darcy.y_locs] 
    
    Nx_o, Ny_o = size(obs_2d)
    
    obs_2d_sym = (obs_2d[1:div(Nx_o+1, 2), :] + obs_2d[end:-1:div(Nx_o, 2)+1, :]) / 2.0
    
    # obs_2d_sym = (obs_2d[:, 1:div(Ny_o+1, 2)] + obs_2d[:, end:-1:div(Ny_o, 2)+1]) / 2.0
    
    return obs_2d_sym[:]
end


function plot_field(darcy::Setup_Param{FT, IT}, u_2d::Array{FT, 2}, plot_obs::Bool,  filename::String = "None") where {FT<:AbstractFloat, IT<:Int}
    N = darcy.N
    xx = darcy.xx

    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    pcolormesh(X, Y, u_2d, cmap="viridis")
    colorbar()

    if plot_obs
        x_obs, y_obs = X[darcy.x_locs[1:div(length(darcy.x_locs)+1,2)], darcy.y_locs][:], Y[darcy.x_locs[1:div(length(darcy.x_locs)+1,2)], darcy.y_locs][:] 
        scatter(x_obs, y_obs, color="black")
        
        x_obs, y_obs = X[darcy.x_locs[div(length(darcy.x_locs)+1,2)+1:end], darcy.y_locs][:], Y[darcy.x_locs[div(length(darcy.x_locs)+1,2)+1:end], darcy.y_locs][:] 
        scatter(x_obs, y_obs, color="black", facecolors="none")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
    end
end

function obs_with_refined_mesh(N, L, N_KL, obs_ΔNx, obs_ΔNy, N_θ, d, τ, σ_0, seed; refine_factor::Int = 2)
    N_refined = refine_factor * (N - 1) + 1
    L = L
    obs_ΔNx_refined, obs_ΔNy_refined = obs_ΔNx * refine_factor, obs_ΔNy * refine_factor
    darcy_refined = Setup_Param(N_refined, L, N_KL, obs_ΔNx_refined, obs_ΔNy_refined, N_θ, d, τ, σ_0; seed=seed)
    κ_2d_refined = exp.(darcy_refined.logκ_2d)
    h_2d_refined = solve_Darcy_2D(darcy_refined, κ_2d_refined)
    y_noiseless_refined = compute_obs(darcy_refined, h_2d_refined)

    figure(1)
    plot_field(darcy_refined, h_2d_refined, true, "Darcy-2D-obs.pdf")
    figure(2)
    plot_field(darcy_refined, darcy_refined.logκ_2d, false, "Darcy-2D-logk-ref.pdf")

    return y_noiseless_refined

end 

seed = 111
N, L = 81, 1.0
obs_ΔNx, obs_ΔNy = 5, 5
d = 2.0
τ = 3.0
N_KL = 32
N_θ = 32
σ_0 = 5.0
darcy = Setup_Param(N, L, N_KL, obs_ΔNx, obs_ΔNy, N_θ, d, τ, σ_0; seed = seed)

y_noiseless = obs_with_refined_mesh(N, L, N_KL, obs_ΔNx, obs_ΔNy, N_θ, d, τ, σ_0, seed; refine_factor = 3)
# @save "Darcy-2D-truth.jld2" darcy y_noiseless

@info "Darcy Problem with N=", N, "N_KL=", N_KL
@info "length of y_noiseless: ", length(y_noiseless)
@info "number of observation points: ", (1+div(length(darcy.x_locs),2))*length(darcy.y_locs)


@load "Darcy-2D-truth.jld2"  darcy y_noiseless
    
    


# GMBBVI initialization
N_iter = 500
N_y = length(y_noiseless)
σ_η = 0.25
Σ_η = σ_η^2 * Array(Diagonal(fill(1.0, N_y)))
Random.seed!(123);
y = y_noiseless + rand(Normal(0, σ_η), N_y)

 
 
μ_0 = zeros(Float64, N_θ)  # prior/initial mean 
Σ_0 = Array(Diagonal(fill(σ_0^2, N_θ)))  # prior/initial covariance
darcy.N_y = N_f = (N_y + N_θ)


# Gaussian mixture parameters initialization
N_modes = 5
θ0_w  = fill(1.0, N_modes)/N_modes
θ0_mean, θθ0_cov  = zeros(N_modes, N_θ), zeros(N_modes, N_θ, N_θ)



Random.seed!(3);
for im = 1:N_modes
    θ0_mean[im, :]    .= rand(Normal(0, σ_0), N_θ) 
    θθ0_cov[im, :, :] .= Array(Diagonal(fill(σ_0^2, N_θ)))
end
for im = 1:div(N_modes,2)
    θ0_mean[N_modes-im+1, :]    .= -θ0_mean[im, :]
end


dt = 0.9
func_args = (y, μ_0, σ_η, σ_0)
func_F(x) = darcy_F(darcy, func_args, x)
func_Phi(x) = 0.5 * norm(darcy_F(darcy, func_args, x))^2
N_ens = 4*N_θ


# gmgdobj = Gaussian_mixture_GMBBVI(
#         func_Phi,
#         θ0_w, θ0_mean, θθ0_cov;
#         N_iter = N_iter,
#         dt = dt,
#         N_ens = N_ens,
#         scheduler_type = "stable_cos_decay",
#         quadrature_type = "random_sampling"
#        ) 
# @save "gmgdobj-Darcy.jld2" gmgdobj


gmgdobj = load("gmgdobj-Darcy.jld2")["gmgdobj"]


fig, (ax1, ax2, ax3, ax4) = PyPlot.subplots(ncols=4, figsize=(20,5))
ites = Array(LinRange(0, N_iter-1, N_iter))
errors = zeros(Float64, (3, N_iter, N_modes))

logκ_2d_truth = darcy.logκ_2d
logκ_2d_mirror = darcy.logκ_2d[end:-1:1, :]

mark_truth = Vector{Bool}(undef, N_modes)   
# True: close to truth; False: close to mirrored truth

for m = 1:N_modes
    logκ_2d = compute_logκ_2d(darcy, gmgdobj.x_mean[end][m,:])
         
    d1 = norm(logκ_2d - logκ_2d_truth)
    d2 = norm(logκ_2d - logκ_2d_mirror)
    mark_truth[m] = (d1 < d2)

end

for m = 1:N_modes
    for i = 1:N_iter

        logκ_2d = compute_logκ_2d(darcy, gmgdobj.x_mean[i][m,:])
        
        if mark_truth[m] == true
            errors[1, i, m] = norm(logκ_2d - logκ_2d_truth)/ norm(darcy.logκ_2d)
        else
            errors[1, i, m] = norm(logκ_2d - logκ_2d_mirror)/ norm(darcy.logκ_2d)
        end
       
        errors[2, i, m] = func_Phi(gmgdobj.x_mean[i][m,:])
        errors[3, i, m] = norm(gmgdobj.xx_cov[i][m,:,:])
    end
end

linestyles = ["o"; "x"; "s"; "*"; "^"; "v"; "<"; ">"]
markevery = 20
for m = 1: N_modes
    if mark_truth[m] == true
        label = "Mode "*string(m)*" (truth)"
    else
        label = "Mode "*string(m)*" (mirrored)"
    end
    ax1.plot(ites, errors[1, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= label)
end
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Rel. error of log a(x)")
ax1.legend()

for m = 1: N_modes
    ax2.semilogy(ites, errors[2, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "Mode "*string(m))
end
ax2.set_xlabel("Iterations")
ax2.set_ylabel(L"\Phi_R")
ax2.legend()

for m = 1: N_modes
    ax3.plot(ites, errors[3, :, m], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "Mode "*string(m))
end
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Frobenius norm of covariance")
ax3.legend()


x_w = exp.(hcat(gmgdobj.logx_w...))
for m = 1: N_modes
    ax4.plot(ites, x_w[m, 1:N_iter], marker=linestyles[m], color = "C"*string(m), fillstyle="none", markevery=markevery, label= "Mode "*string(m))
end
ax4.set_xlabel("Iterations")
ax4.set_ylabel("Weights")
ax4.legend()
fig.tight_layout()
fig.savefig("Darcy-2D-convergence.pdf")


fig, ax = PyPlot.subplots(ncols=1, figsize=(16,5))
θ_ref = darcy.θ_ref

n_ind = 16
θ_ind = Array(1:n_ind)
ax.scatter(θ_ind, θ_ref[θ_ind], s = 100, marker="x", color="black", label="Truth")
for m = 1:N_modes
    ax.scatter(θ_ind, gmgdobj.x_mean[N_iter][m,θ_ind], s = 50, marker="o", color="C"*string(m), facecolors="none", label="Mode "*string(m))
end

Nx = 1000
scale = 1
for i in θ_ind
    x_min = minimum(gmgdobj.x_mean[N_iter][:,i] .- 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
    x_max = maximum(gmgdobj.x_mean[N_iter][:,i] .+ 3sqrt.(gmgdobj.xx_cov[N_iter][:,i,i]))
        
    xxs = zeros(N_modes, Nx)  
    zzs = zeros(N_modes, Nx)  
    for m =1:N_modes
        xxs[m, :], zzs[m, :] = Gaussian_1d(gmgdobj.x_mean[N_iter][m,i], gmgdobj.xx_cov[N_iter][m,i,i], Nx, x_min, x_max)
        zzs[m, :] *= exp(gmgdobj.logx_w[N_iter][m])

        @info "ind ", i, " mode ", m, " mean ", gmgdobj.x_mean[N_iter][m,i],  " std ", sqrt(gmgdobj.xx_cov[N_iter][m,i,i])
    end
    
    label = nothing
    if i == 1
        label = "GMBBVI"
    end
    ax.plot(sum(zzs, dims=1)'/scale .+ i, xxs[1,:], linestyle="-", color="C0", fillstyle="none", label=label)
    ax.plot(fill(i, Nx), xxs[1,:], linestyle=":", color="black", fillstyle="none")
        
end
ax.set_xticks(θ_ind)
ax.set_xlabel(L"\theta" * " indices")
ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5))
fig.tight_layout()
# fig.savefig("Darcy-2D-density.pdf")
fig.savefig("Darcy-2D-density-GMBBVI.pdf")



truth_im_ind = findall(mark_truth)
mirror_im_ind = findall(!, mark_truth)

truth_modes_center = sum(x_w[truth_im_ind, end] .* gmgdobj.x_mean[end][truth_im_ind, :], dims=1) / sum(x_w[truth_im_ind, end])
mirror_modes_center = sum(x_w[mirror_im_ind, end] .* gmgdobj.x_mean[end][mirror_im_ind, :], dims=1) / sum(x_w[mirror_im_ind, end])
truth_modes_center = vec(truth_modes_center)
mirror_modes_center = vec(mirror_modes_center)

fig_logk, ax_logk = PyPlot.subplots(ncols=4, sharex=true, sharey=true, figsize=(16,4))
for ax in ax_logk ;  ax.set_xticks([]) ; ax.set_yticks([]) ; end
color_lim = (minimum(darcy.logκ_2d), maximum(darcy.logκ_2d))

plot_field(darcy, darcy.logκ_2d, color_lim, ax_logk[1]) 
ax_logk[1].set_title("Truth")
plot_field(darcy, darcy.logκ_2d[end:-1:1, :],  color_lim, ax_logk[2]) 
ax_logk[2].set_title("Truth (mirrored)")
plot_field(darcy, compute_logκ_2d(darcy, truth_modes_center),  color_lim, ax_logk[3])
ax_logk[3].set_title("Mean of modes close to truth")
plot_field(darcy, compute_logκ_2d(darcy, mirror_modes_center),  color_lim, ax_logk[4])  
ax_logk[4].set_title("Mean of modes close to mirror")


fig_logk.tight_layout()
fig_logk.savefig("Darcy-2D-logk.pdf")

