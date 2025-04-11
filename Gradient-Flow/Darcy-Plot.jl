using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2
using ForwardDiff
using NPZ
using KernelDensity
include("../Inversion/Plot.jl")
include("Darcy-1D.jl")


Random.seed!(42);
N, L = 128, 1.0
obs_ΔN = 16
d = 2.0
τ = 3.0
N_KL = N_θ = 16
ση = 1.0
σprior = 10.0
darcy = Darcy(N, L, N_KL, obs_ΔN, N_θ, ση, σprior, d, τ)
θ_ref = darcy.θ_ref
k = exp.(darcy.logk)
h = darcy.h_ref
# observation
y = darcy.y_obs


###################################################################################
fig, ax = PyPlot.subplots(ncols=3, sharey=true, figsize=(10,4))
N_x, xx = darcy.N_x, darcy.xx
ax[1].plot(darcy.f, xx)
ax[1].set_ylabel("x")
ax[1].set_xlabel("f(x)")
ax[2].plot(darcy.h_ref, xx)
obs_locs = darcy.obs_locs
x_obs = xx[obs_locs]
ax[2].scatter(darcy.y_obs, x_obs, color="black")
ax[2].set_xlabel("p(x)")
ax[3].plot(k, xx)
ax[3].set_xlabel("a(x, θ)")
ax[1].invert_yaxis()
ax[2].invert_yaxis()
ax[3].invert_yaxis()
fig.subplots_adjust(bottom=0.15,top=0.95,left=0.08,right=0.98,hspace=0.2)
fig.savefig("Darcy-1D-solution.pdf")

###################################################################################

us = npzread("us.npy")
N_iter_MCMC, N_θ = size(us)
n_burn_in = div(N_iter_MCMC, 4)



θ_cov_ind = Array(1:div(N_θ*(N_θ+1), 2))
mcmc_θiθj_cov = zeros(Float64, N_θ, N_θ)
mcmc_θ_mean = mean(us[n_burn_in:N_iter_MCMC, :], dims=1)[:]
mcmc_θθ_std = std(us[n_burn_in:N_iter_MCMC, :], dims=1)[:]

for i = 1:N_θ
    for j = i:N_θ
        mcmc_θiθj_cov[i,j] = (us[n_burn_in:N_iter_MCMC, i] .- mcmc_θ_mean[i])' * (us[n_burn_in:N_iter_MCMC, j] .- mcmc_θ_mean[j])/(N_iter_MCMC - n_burn_in)
        mcmc_θiθj_cov[j,i] = mcmc_θiθj_cov[i,j]
    end
end



fig, ax = PyPlot.subplots(figsize=(7,6))
im = ax.pcolormesh(mcmc_θiθj_cov, cmap="binary")
ax.set_xticks([0, 4, 8, 12, 16]) 
ax.set_xticklabels(["0", "4", "8", "12", "16"])
ax.set_yticks([0, 4, 8, 12, 16]) 
ax.set_yticklabels(["0", "4", "8", "12", "16"])

fig.colorbar(im)
fig.savefig("Darcy-1D-cov.pdf")



####################################################################################
# Evaluate MCMC
accept = 0
for i = 1:N_iter_MCMC-1
    if norm(us[i + 1,:] - us[i,:]) > 1e-10
        global accept += 1
    end
end
@info "The accept rate is ", accept/N_iter_MCMC
mcmc_θ_mean_temp = mean(us[div(N_iter_MCMC,2):N_iter_MCMC, :], dims=1)[:]
@info  "Error/Relative error indicators of MCMC: " norm(mcmc_θ_mean_temp - mcmc_θ_mean), norm(mcmc_θ_mean_temp - mcmc_θ_mean)/norm(mcmc_θ_mean)

###################################################################################

fig, ax = PyPlot.subplots(ncols=2, figsize=(16,8))
ngd_mean = npzread("GFR_mean.npy")   # Ntheta 0:Nt
ngd_cov = npzread("GFR_cov.npy") 
gd_mean = npzread("GD_mean.npy") 
gd_cov = npzread("GD_cov.npy") 
wgd_mean = npzread("GW_mean.npy") 
wgd_cov = npzread("GW_cov.npy")


Stein_θ = npzread("Stein_theta.npy")
Stein_prec_θ = npzread("Stein_prec_theta.npy")
Wasserstein_θ =  npzread("Wasserstein_theta.npy")
Wasserstein_prec_θ = npzread("Wasserstein_prec_theta.npy") 
       
N_θ, N_t = size(ngd_mean) 
N_ens, N_θ, N_t = size(Stein_θ) 
Stein_mean, Stein_prec_mean, Wasserstein_mean, Wasserstein_prec_mean = zeros(N_θ, N_t), zeros(N_θ, N_t), zeros(N_θ, N_t), zeros(N_θ, N_t)
Stein_cov, Stein_prec_cov, Wasserstein_cov, Wasserstein_prec_cov = zeros(N_θ, N_θ, N_t), zeros(N_θ, N_θ, N_t), zeros(N_θ, N_θ, N_t), zeros(N_θ, N_θ, N_t)
for i = 1:N_t
    Stein_mean[:, i] = mean(Stein_θ[:,:,i], dims=1)
    Stein_prec_mean[:, i] = mean(Stein_prec_θ[:,:,i], dims=1)
    Wasserstein_mean[:, i] = mean(Wasserstein_θ[:,:,i], dims=1)
    Wasserstein_prec_mean[:, i] = mean(Wasserstein_prec_θ[:,:,i], dims=1)
    
  
    Stein_cov[:,:, i] = (Stein_θ[:,:,i] - ones(N_ens)*Stein_mean[:, i]')' * (Stein_θ[:,:,i] - ones(N_ens)*Stein_mean[:, i]') / (N_ens - 1)
    Stein_prec_cov[:,:, i] = (Stein_prec_θ[:,:,i] - ones(N_ens)*Stein_prec_mean[:, i]')' * (Stein_prec_θ[:,:,i] - ones(N_ens)*Stein_prec_mean[:, i]') / (N_ens - 1)
    Wasserstein_cov[:,:, i] = (Wasserstein_θ[:,:,i] - ones(N_ens)*Wasserstein_mean[:, i]')' * (Wasserstein_θ[:,:,i] - ones(N_ens)*Wasserstein_mean[:, i]') / (N_ens - 1)
    Wasserstein_prec_cov[:,:, i] = (Wasserstein_prec_θ[:,:,i] - ones(N_ens)*Wasserstein_prec_mean[:, i]')' * (Wasserstein_prec_θ[:,:,i] - ones(N_ens)*Wasserstein_prec_mean[:, i]') / (N_ens - 1)
end    


mean_all, std_all = zeros(7, N_θ), zeros(7, N_θ)

mean_all[1,:] =  Wasserstein_mean[:, N_t]  
mean_all[2,:] =  Wasserstein_prec_mean[:, N_t]  
mean_all[3,:] =  Stein_mean[:, N_t]  
mean_all[4,:] =  Stein_prec_mean[:, N_t] 
mean_all[5,:] =  gd_mean[:,  N_t]  
mean_all[6,:] =  ngd_mean[:, N_t]  
mean_all[7,:] =  wgd_mean[:, N_t]  

std_all[1,:] =  sqrt.(diag(Wasserstein_cov[:, :, N_t])) 
std_all[2,:] =  sqrt.(diag(Wasserstein_prec_cov[:, :, N_t])) 
std_all[3,:] =  sqrt.(diag(Stein_cov[:, :, N_t])) 
std_all[4,:] =  sqrt.(diag(Stein_prec_cov[:, :, N_t])) 
std_all[5,:] =  sqrt.(diag(gd_cov[:, :, N_t])) 
std_all[6,:] =  sqrt.(diag(ngd_cov[:, :, N_t])) 
std_all[7,:] =  sqrt.(diag(wgd_cov[:, :, N_t])) 

mean_error, cov_error = zeros(7, N_t), zeros(7, N_t)
for i = 1:N_t
    
    mean_error[1,i] = norm(Wasserstein_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    mean_error[2,i] = norm(Wasserstein_prec_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    mean_error[3,i] = norm(Stein_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    mean_error[4,i] = norm(Stein_prec_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)  
    mean_error[5,i] = norm(gd_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    mean_error[6,i] = norm(ngd_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    mean_error[7,i] = norm(wgd_mean[:, i] - mcmc_θ_mean)/norm(mcmc_θ_mean)
    
    
    cov_error[1,i] = norm(Wasserstein_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
    cov_error[2,i] = norm(Wasserstein_prec_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
    cov_error[3,i] = norm(Stein_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
    cov_error[4,i] = norm(Stein_prec_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
    cov_error[5,i] = norm(gd_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)    
    cov_error[6,i] = norm(ngd_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
    cov_error[7,i] = norm(wgd_cov[:, :, i] - mcmc_θiθj_cov)/norm(mcmc_θiθj_cov)
end

    
    

markers = ["--s", "-.s", "--x", "--x", "--*", "--o", "--s"]
colors  = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
labels  = [ "Wasserstein GF", "Affine invariant Wasserstein GF", 
           "Stein GF", "Affine invariant Stein GF", "Gassian approximate GF", "Gassian approximate Fisher-Rao GF",
           "Gassian approximate Wasserstein GF"]
indt = Array(0:N_t-1)
fig, ax = PyPlot.subplots(ncols=2, figsize=(13, 6))
for i = 1:7
    ax[1].semilogy(indt, mean_error[i,:], markers[i], label=labels[i], color=colors[i], markevery=500, fillstyle="none")
    ax[2].semilogy(indt, cov_error[i,:], markers[i], label=labels[i], color=colors[i], markevery=500, fillstyle="none")
end

y_min, y_max = ax[1].get_ylim(); ax[1].set_ylim(0.9e-2, y_max)
y_min, y_max = ax[2].get_ylim(); ax[2].set_ylim(0.9e-2, y_max)
ax[1].grid("on")
ax[2].grid("on")
ax[1].set_xlabel("Number of iterations")
ax[1].set_ylabel("Rel. mean error")
ax[2].set_xlabel("Number of iterations")
ax[2].set_ylabel("Rel. covariance error")
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles,labels,loc = "upper center",bbox_to_anchor=(0.5,0.99),ncol=3)
fig.subplots_adjust(bottom=0.12,top=0.8,left=0.08,right=0.98,hspace=0.2)
fig.savefig("Darcy-1D-error.pdf")



# ###################################################################################

θ_ind = Array(1:N_θ)
fig, ax = PyPlot.subplots(figsize=(13,6))

ax.plot(θ_ind , mcmc_θ_mean,"-D", color="black", label="MCMC")
ax.plot(θ_ind , mcmc_θ_mean + 3.0*mcmc_θθ_std, "--", color ="black")
ax.plot(θ_ind , mcmc_θ_mean - 3.0*mcmc_θθ_std,  "--", color ="black")
ax.plot(θ_ind , θ_ref, "--v", color="C9", fillstyle="none", label="Reference")

for i = 1:7
    ax.plot(θ_ind, mean_all[i,:],                  markers[i],  label=labels[i],   color=colors[i], fillstyle="none")
    ax.plot(θ_ind, mean_all[i,:] + 3*std_all[i,:],  "--",   color=colors[i],  fillstyle="none")
    ax.plot(θ_ind, mean_all[i,:] - 3*std_all[i,:],  "--",    color=colors[i], fillstyle="none")
end

ax.grid("on")


ax.legend(loc = "upper center",bbox_to_anchor=(0.5,1.28),ncol=3)
# plot MCMC results 
ax.set_xlabel("θ indices")
fig.subplots_adjust(bottom=0.12,top=0.8,left=0.05,right=0.98,hspace=0.2)
fig.savefig("Darcy-1D-theta.pdf")












