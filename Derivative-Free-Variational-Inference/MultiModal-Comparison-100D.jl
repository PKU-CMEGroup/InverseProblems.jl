using Random
using Distributions
using KernelDensity
include("../Inversion/AffineInvariantMCMC.jl")
include("./MultiModal.jl")
include("../Inversion/GaussianMixture.jl")
include("../Inversion/GMBBVI.jl")
include("../Inversion/Plot.jl")



function visualization_comparison_100d(ax, obj_BBVI= nothing, obj_MCMC = nothing; Nx = 200, Ny = 200, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0],
    func_F = nothing, func_Phi = nothing, bandwidth=nothing, make_label::Bool=false, N_iter=500)
    
    
    x_min, x_max = x_lim
    y_min, y_max = y_lim
    boundary=((x_lim[1],x_lim[2]),(y_lim[1],y_lim[2]))

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'

    Z_ref = (func_Phi === nothing ? posterior_2d(func_F, X, Y, "func_F") : posterior_2d(func_Phi, X, Y, "func_Phi"))
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)


    if obj_MCMC!=nothing
        error = zeros(length(obj_MCMC),N_iter+1)
        last_n_iters = 10  #use the last "last_n_iters" iterations to draw pictures

        for (i,ens) in enumerate(obj_MCMC)
            for iter = 0:N_iter

                if bandwidth==nothing
                    kde_iter=kde(ens[:,:,iter+1]'; boundary=boundary, npoints=(Nx,Ny))
                else
                    kde_iter=kde(ens[:,:,iter+1]'; boundary=boundary, npoints=(Nx,Ny), bandwidth=bandwidth)
                end

                Z = kde_iter.density/(sum(kde_iter.density)*dx*dy)
                error[i,iter+1] = norm(Z - Z_ref,1)*dx*dy
                
                if iter == N_iter
                    
                    last_ens = hcat([ens[:,:,i] for i in N_iter+2-last_n_iters:N_iter+1]...)
                    last_ens_number =size(last_ens,2)

                    if bandwidth==nothing
                        kde_last=kde(last_ens'; boundary=boundary, npoints=(Nx,Ny))
                    else
                        kde_last=kde(last_ens'; boundary=boundary, npoints=(Nx,Ny), bandwidth=bandwidth)
                    end

                    Z = kde_last.density/(sum(kde_last.density)*dx*dy)

                    ax[i+1].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                    ax[i+1].scatter(last_ens[1,1:max(1,div(last_ens_number,1000)):end], last_ens[2,1:max(1,div(last_ens_number,1000)):end], marker=".", color="red", s=10, alpha=100/last_ens_number)
                    ax[i+1].set_xlim(x_lim)
                    ax[i+1].set_ylim(y_lim)

                end
            end
        end
        label = ["J="*string(size(ens,2))  for ens in obj_MCMC ]
    end



    if obj_BBVI !=nothing
        error = zeros(length(obj_BBVI),N_iter+1)
        last_n_iters = 10  #use the last "last_n_iters" iterations to draw pictures

        for (i, obj) in enumerate(obj_BBVI)
            for iter = 0:N_iter
                x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
                x_mean = obj.x_mean[iter+1][:,1:2]
                xx_cov = obj.xx_cov[iter+1][:,1:2,1:2]
                Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
                error[iter+1] = norm(Z - Z_ref,1)*dx*dy
                
                if iter == N_iter
                
                    ax[i+1].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                    ax[i+1].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey", alpha=0.5) 
                    ax[i+1].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none", alpha=0.5)
                
                end
            end
        end
        label = ["J="*string(obj.N_ens)  for obj in obj_BBVI ]
    end

    
    ax[5].semilogy(Array(0:N_iter), error', label=label)   
    
    if make_label==true  ax[5].legend()  end

    ymin, ymax = ax[5].get_ylim()

end





fig, ax = PyPlot.subplots(nrows=2, ncols=5, sharex=false, sharey=false, figsize=(20,6))

# Problem setup
N_iter = 1000
Nx, Ny = 100, 100

ση = 1.0
Gtype = "Gaussian"
N_x = 100
A = [1.0 1.0; 1.0 2.0]
y = [0.0; 1.0; zeros(N_x-2)]
func_args = (y, ση, A , Gtype)
func_F(x) = F(x, func_args)
func_F_marginal(x) = F(x, (y[1:2], ση, A , Gtype))
func_Phi(x) = 0.5*norm(func_F(x))^2
log_prob(x) = logrho(x, func_args)
func_prob(x)= exp(log_prob(x))
μ0, Σ0 = zeros(N_x), 1*Diagonal(ones(N_x))



#########BBVI
N_modes = 40
x0_w  = ones(N_modes)/N_modes
Random.seed!(111);
N_ens_array = [256, 512, 1024]
N_ens_max = N_ens_array[end]
x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
for im = 1:N_modes
    x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
    xx0_cov[im, :, :] .= Σ0
end
dt = 0.5
BBVI = [Gaussian_mixture_BBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = dt, N_ens=N_ens)
        for N_ens in N_ens_array]
visualization_comparison_100d(ax[1, :], BBVI , nothing; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F_marginal, 
    bandwidth=(0.32,0.22), make_label=true,  N_iter= N_iter)








# #########MCMC
Random.seed!(111);
N_ens_array = [1024, 4096, 16384]
N_ens_max = N_ens_array[end]
ens_0 = zeros(N_x,N_ens_max)
for j = 1:N_ens_max
    ens_0[:,j]  = rand(MvNormal(zeros(N_x), Σ0)) + μ0
end 

ens_MCMC = [ Run_StretchMove(ens_0[:,1:N_ens], func_prob; output="History", N_iter=N_iter)[1:2,:,:]  for N_ens in N_ens_array ]
y_2d = y[1:2]
func_args = (y_2d, ση, A , Gtype)
func_F(x) = F(x, func_args)
visualization_comparison_100d(ax[2,:], nothing, ens_MCMC ; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F_marginal, 
    bandwidth=(0.32,0.22), make_label=true,  N_iter= N_iter)



    
fig.tight_layout()
fig.savefig("MultiModal-Comparison-100D.pdf")








