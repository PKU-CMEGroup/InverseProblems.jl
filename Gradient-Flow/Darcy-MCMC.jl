using LinearAlgebra
using Distributions
using Random
using SparseArrays
using JLD2
using ForwardDiff
using NPZ
include("../Inversion/RWMCMC.jl")
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



# compute posterior distribution by MCMC
# (uninformative) prior mean and covariance
μ0 =  θ_ref #
Σ0 = Array(Diagonal(fill(σprior^2, N_θ)))

N_iter_MCMC , n_burn_in= 2*10^8, 5*10^7


us = PCN_Run(arg -> logρ_likelihood(arg, darcy), μ0, Σ0, 0.008, N_iter_MCMC)

npzwrite("us.npy", us)