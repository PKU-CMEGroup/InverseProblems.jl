using LinearAlgebra
using Random
using Distributions
include("../Inversion/Plot.jl")
include("../Inversion/GaussianMixture.jl")
include("../Inversion/GMBBVI.jl")
include("../Inversion/GMKI.jl")
include("../Inversion/GMWVI.jl")
include("../Inversion/AnnealingInitialize.jl")
include("./MultiModal.jl")

mutable struct Setup_Param{IT<:Int}
    θ_names::Array{String,1}
    N_θ::IT
    N_y::IT
end

function Setup_Param(N_θ::IT, N_y::IT) where {IT<:Int}
    return Setup_Param(["θ"], N_θ, N_y)
end

function visualization(ax, objs, func_Phi, Gtype::String;
    name_array = nothing, Nx = 100, Ny = 100, x_lim = [-3.0, 3.0], y_lim = [-3.0, 3.0], N_iter = 200)
    
    # posterior_mean, posterior_cov = multimodal_moments(Gtype)

    x_min, x_max = x_lim 
    y_min, y_max = y_lim

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
    Z_ref = posterior_2d(func_Phi, X, Y, "func_Phi")
    
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)

    N_obj = length(objs)
    errors = zeros(N_obj, N_iter+1, 1) 

    for (i_obj, obj) in enumerate(objs)
        @info "i_obj: $i_obj", typeof(obj)
        for iter = 0:N_iter
            N_modes = obj.N_modes
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1][:,1:2]
            xx_cov = obj.xx_cov[iter+1][:,1:2,1:2]

            Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)

            # ρ_gm_mean, ρ_gm_cov = compute_ρ_gm_moments(x_w, x_mean, xx_cov)

            errors[i_obj, iter+1, 1] = norm(Z - Z_ref,1)*dx*dy
            # errors[i_obj, iter+1, 2] = norm(ρ_gm_mean - posterior_mean, 2)
            # errors[i_obj, iter+1, 3] = norm(ρ_gm_cov - posterior_cov, 2)/norm(posterior_cov, 2)
            
            if iter == N_iter  
                # plot the outcome of the first trial 
                ax[i_obj + 1].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                ax[i_obj + 1].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey", alpha=0.5) 
                ax[i_obj + 1].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none", alpha=0.5)

                ax[i_obj + 1].set_xlim(x_lim)
                ax[i_obj + 1].set_ylim(y_lim)
            end
        end
    end


    for i = 1:N_obj     
        
        if name_array !== nothing
            ax[end].semilogy(Array(0:N_iter), errors[i,:,1], label = name_array[i])    
        else
            ax[end].semilogy(Array(0:N_iter), errors[i,:,1])    
        end

    end

    if name_array !== nothing  ax[end].legend()  end

end

Random.seed!(123);

fig, ax = PyPlot.subplots(nrows=3, ncols=5, sharex=false, sharey=false, figsize=(20,12))


N_modes = 40 # number of modes in Gaussian mixture
N_iter = 500
N_x = 2
N_ens = 4 * N_x
quadrature_type = "random_sampling"

x0_w  = ones(N_modes)/N_modes
μ0, Σ0 = zeros(N_x), Matrix(I(N_x)) 
x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
for im = 1:N_modes
    x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
    xx0_cov[im, :, :] .= Σ0
end


@info "Running: circle density"
Gtype = "Circle"
ση = [0.3; ones(N_x-2)]
A = [1.0 0.0; 0.0 1.0]
y = [1.0; zeros(N_x-2)]

func_marginal_args = (y[1:1], ση[1:1], A , Gtype)
func_Phi_marginal(x) = Phi(x, func_marginal_args)

func_args = (y, ση, A, Gtype)
func_Phi(x) = Phi(x, func_args)
func_dPhi(x) = dPhi(x, func_args)
func_F(x) = F(x, func_args)

# x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = initialize_with_annealing(func_Phi, x0_w, x0_mean, xx0_cov; N_ens = N_ens, scheduler_type = "exponential_decay", N_iter=500)

x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = x0_w, x0_mean, xx0_cov

obj1 = Gaussian_mixture_GMBBVI(func_Phi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 0.9, N_ens = N_ens, quadrature_type = quadrature_type)
obj2 = Gaussian_mixture_WGFVI(func_dPhi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 1.0e-3, Hessian_correct_GM=false)[1]

s_param = Setup_Param(N_x, size(y,1)); Δt = 0.5
obj3 = Gaussian_mixture_GMKI(s_param, func_F, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal, N_iter, Δt) 

objs = [obj1, obj2, obj3]
visualization(ax[1,:], objs, func_Phi_marginal, Gtype; name_array = ["GMBBVI", "WGFVI", "GMKI"], x_lim = [-3.0, 3.0], y_lim = [-3.0, 3.0], N_iter = N_iter)



@info "Running: banana shape density"
Gtype = "Banana"
ση = [sqrt(10.0); sqrt(10.0); ones(N_x-2)]
λ = 10.0
y = [0.0; 1.0; zeros(N_x-2)]

func_marginal_args = (y[1:2], ση[1:2], λ, Gtype)
func_Phi_marginal(x) = Phi(x, func_marginal_args)

func_args = (y, ση, λ, Gtype)
func_Phi(x) = Phi(x, func_args)
func_dPhi(x) = dPhi(x, func_args)
func_F(x) = F(x, func_args)

# x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = initialize_with_annealing(func_Phi, x0_w, x0_mean, xx0_cov; N_ens = N_ens, scheduler_type = "exponential_decay", N_iter = 500)

x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = x0_w, x0_mean, xx0_cov
obj1 = Gaussian_mixture_GMBBVI(func_Phi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 0.9, N_ens = N_ens, quadrature_type = quadrature_type)
obj2 = Gaussian_mixture_WGFVI(func_dPhi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 2.0e-4, Hessian_correct_GM=false)[1]

s_param = Setup_Param(N_x, size(y,1)); Δt = 0.5
obj3 = Gaussian_mixture_GMKI(s_param, func_F, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal, N_iter, Δt) 

objs = [obj1, obj2, obj3]

visualization(ax[2,:], objs, func_Phi_marginal, Gtype; name_array = ["GMBBVI", "WGFVI", "GMKI"], x_lim = [-3.0, 3.0], y_lim = [-2.0, 10.0], N_iter = N_iter)


@info "Running: Funnel example"
Gtype = "Funnel"
ση = ones(N_x)
A = Diagonal(ones(N_x-1))
y = zeros(N_x)

func_marginal_args = (y[1:2], ση[1:2], A[1,1], Gtype)
func_Phi_marginal(x) = Phi(x, func_marginal_args)

func_args = (y, ση, A, Gtype)
func_Phi(x) = Phi(x, func_args)
func_dPhi(x) = dPhi(x, func_args)
func_F(x) = F(x, func_args)

x0_w_anneal, x0_mean_anneal, xx0_cov_anneal = x0_w, x0_mean, xx0_cov
obj1 = Gaussian_mixture_GMBBVI(func_Phi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 0.9, N_ens = N_ens, quadrature_type = quadrature_type)
obj2 = Gaussian_mixture_WGFVI(func_dPhi, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal;
                N_iter = N_iter, dt = 1.0e-3, Hessian_correct_GM=false)[1]

s_param = Setup_Param(N_x, size(y,1)); Δt = 0.5
obj3 = Gaussian_mixture_GMKI(s_param, func_F, x0_w_anneal, x0_mean_anneal, xx0_cov_anneal, N_iter, Δt) 
objs = [obj1, obj2, obj3]

visualization(ax[3,:], objs, func_Phi_marginal, Gtype; name_array = ["GMBBVI", "WGFVI", "GMKI"], x_lim =  [-10.0, 10.0], y_lim = [-20.0, 20.0], N_iter = N_iter)


ax[1,1].set_title("Reference", fontsize=15)
ax[1,2].set_title(L"GMBBVI", fontsize=15)
ax[1,3].set_title(L"WGFVI", fontsize=15)
ax[1,4].set_title(L"GMKI", fontsize=15)
ax[1,5].set_title("TV distance", fontsize=15)


fig.tight_layout()
fig.savefig("GMBBVI-Comparison.pdf")