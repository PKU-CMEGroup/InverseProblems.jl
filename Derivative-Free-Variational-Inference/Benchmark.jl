using LinearAlgebra
using Random
using Distributions
using KernelDensity
using PyCall
using ForwardDiff
include("../Inversion/AffineInvariantMCMC.jl")
include("../Inversion/Plot.jl")
include("../Inversion/GaussianMixture.jl")
include("../Inversion/GMBBVI.jl")
include("../Inversion/AnnealingInitialize.jl")
include("./MultiModal.jl")

function auto_add_python_path(target_file="WALNUTS.py")
    py_sys = pyimport("sys")
    curr = pwd() 
    
    for i in 1:4
        for (root, dirs, files) in walkdir(curr)
            if target_file in files
                if !(root in py_sys.path)
                    pushfirst!(PyVector(py_sys."path"), root)
                    @info "Found and added Python path: $root"
                    return true
                end
            end
        end
        curr = dirname(curr) 
    end
    @warn "Could not find $target_file automatically."
    return false
end

auto_add_python_path("WALNUTS.py")

const wn = pyimport("WALNUTS")
const ai = pyimport("adaptiveIntegrators")
const emcee = pyimport("emcee")

function run_walnuts_python(target_phi_julia, q0, n_x, n_iter)
    function target_func_py(q)
        val = -target_phi_julia(q)
        grad = -ForwardDiff.gradient(target_phi_julia, q)
        return (val, grad) 
    end

    samples_w, diag_w = wn.WALNUTS(
        target_func_py, q0, 
        generated = (q -> q),
        integrator = ai.adaptLeapFrogR2P,
        M=12, H0=0.3, delta0=0.3, numIter=1000, warmupIter=1000,
        adaptDeltaTarget=0.6, recordOrbitStats=false
    )

    q_curr = samples_w[:, end]
    H_curr = diag_w[end, 16]    
    delta_curr = diag_w[end, 19] 

    samples, _, _, _ = wn.WALNUTS(
        target_func_py, q_curr,
        generated = (q -> q),
        integrator = ai.adaptLeapFrogR2P,
        M=12, H0=H_curr, delta0=delta_curr,
        numIter=n_iter, warmupIter=0,
        recordOrbitStats=true
    )

    return samples 
end


function run_emcee_python(target_phi_julia, ens0, n_iter; drop_rate=0.8, thin::Int=20, ndim_save::Int=2)
    function log_prob(q)
        val = -target_phi_julia(q)
        return val
    end

    ndim, nwalkers = size(ens0)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(ens0', n_iter)
    samples = sampler.get_chain(flat=true, discard=Int(round(drop_rate*n_iter)), thin=thin)[:, 1:ndim_save]    
    return samples'  # (2, n_sampler)
end


function calculate_sample_stats(samples)
    if isempty(samples) || size(samples, 2) < 2
        return fill(NaN, size(samples, 1)), fill(NaN, size(samples, 1), size(samples, 1))
    end
    total_mean = mean(samples, dims=2)[:, 1]
    total_cov = cov(samples') 
    return total_mean, total_cov
end


function visualization(ax, obj_GMBBVI, ens_MCMC, ens_WALNUTS; Nx = 200, Ny = 200, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0],
        func_F = nothing, func_Phi = nothing, bandwidth=nothing, make_label::Bool=false, N_iter=500, Num=5000, Gtype="Gaussian")

        x_min, x_max = x_lim
        y_min, y_max = y_lim

        xx = LinRange(x_min, x_max, Nx)
        yy = LinRange(y_min, y_max, Ny)
        dx, dy = xx[2] - xx[1], yy[2] - yy[1]
        X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'

        Z_ref = (func_Phi === nothing ? posterior_2d(func_F, X, Y, "func_F") : posterior_2d(func_Phi, X, Y, "func_Phi"))
        color_lim = (minimum(Z_ref), maximum(Z_ref))
        ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)

        mean_val, cov_val = multimodal_moments(Gtype)
        text_str = "μ:[$(round(mean_val[1], digits=2)), $(round(mean_val[2], digits=2))], σ²: [$(round(cov_val[1,1], digits=2)), $(round(cov_val[2,2], digits=2))]"
        ax[1].text(0.05, 0.95, text_str, transform=ax[1].transAxes, fontsize=8,
                                   verticalalignment="top", bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.7))

        error = zeros(2, N_iter+1)

        if !isnothing(obj_GMBBVI)
                for iter = 0:N_iter
                x_w = exp.(obj_GMBBVI.logx_w[iter+1]); x_w /= sum(x_w)
                x_mean = obj_GMBBVI.x_mean[iter+1][:,1:2]
                xx_cov = obj_GMBBVI.xx_cov[iter+1][:,1:2,1:2]
                Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
                error[1, iter+1] = norm(Z - Z_ref,1)*dx*dy
                
                if iter == N_iter
                        ax[2].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                        ax[2].scatter([obj_GMBBVI.x_mean[1][:,1];], [obj_GMBBVI.x_mean[1][:,2];], marker="x", color="grey", alpha=0.5) 
                        ax[2].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none", alpha=0.5)

                        mean_val, cov_val = compute_ρ_gm_moments(x_w, x_mean, xx_cov)
                        text_str = "μ:[$(round(mean_val[1], digits=2)), $(round(mean_val[2], digits=2))], σ²: [$(round(cov_val[1,1], digits=2)), $(round(cov_val[2,2], digits=2))]"
                        ax[2].text(0.05, 0.95, text_str, transform=ax[2].transAxes, fontsize=8,
                                   verticalalignment="top", bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.7))
                end
                end
        end

        if !isnothing(ens_MCMC)
            
                # Only the final 10,000 samples will be visualized, to avoid potential challenges during the plotting process.
                samples_2d = ens_MCMC[1:2, end-10000:end] 
                
                boundary = ((x_lim[1],x_lim[2]),(y_lim[1],y_lim[2]))
                if bandwidth===nothing
                        kde_iter=kde(samples_2d';boundary=boundary,npoints=(Nx,Ny))
                else
                        kde_iter=kde(samples_2d';boundary=boundary,npoints=(Nx,Ny),bandwidth=bandwidth) 
                end

                Z = kde_iter.density
                sum_Z = sum(Z)
                if sum_Z > 0 Z ./= (sum_Z*dx*dy) end
                
    
                ax[3].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                ax[3].scatter(samples_2d[1,:], samples_2d[2,:], marker=".", color="red", s=10, alpha=0.1)
                ax[3].set_xlim(x_lim)
                ax[3].set_ylim(y_lim)
                
                mean_val, cov_val = calculate_sample_stats(samples_2d)
                if !any(isnan, mean_val)
                    text_str = "μ:[$(round(mean_val[1], digits=2)), $(round(mean_val[2], digits=2))], σ²: [$(round(cov_val[1,1], digits=2)), $(round(cov_val[2,2], digits=2))]"
                    ax[3].text(0.05, 0.95, text_str, transform=ax[3].transAxes, fontsize=8,
                                verticalalignment="top", bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.7))
                end
                
        end
        
        # WALNUTS Error Calculation
        if !isnothing(ens_WALNUTS)
                boundary=((x_lim[1],x_lim[2]),(y_lim[1],y_lim[2]))
                samples_per_iter = div(Num, N_iter)
                total_samples = size(ens_WALNUTS, 2)
                for iter = 0:N_iter
                        n_samples_end = min((iter + 1) * samples_per_iter, total_samples)
                        if n_samples_end < 2
                                error[2, iter+1] = NaN
                                continue
                        end
                        
                        samples_iter_2d = ens_WALNUTS[1:2, 1:n_samples_end]

                        if bandwidth===nothing
                                kde_iter=kde(samples_iter_2d';boundary=boundary,npoints=(Nx,Ny))
                        else
                                kde_iter=kde(samples_iter_2d';boundary=boundary,npoints=(Nx,Ny),bandwidth=bandwidth)
                        end
        
                        Z = kde_iter.density
                        sum_Z = sum(Z)
                        if sum_Z > 0 Z ./= (sum_Z*dx*dy) end
                        error[2, iter+1] = norm(Z - Z_ref,1)*dx*dy
                end
        end
    
        # Helper for plotting final MCMC-type distributions
        function plot_mcmc_on_ax(ax_plt, samples, title)
            if samples === nothing || size(samples, 2) < 20
                ax_plt.text(0.5, 0.5, "No Samples", ha="center", va="center")
                ax_plt.set_title(title)
                ax_plt.set_xlim(x_lim)
                ax_plt.set_ylim(y_lim)
                return
            end
        
            ax_plt.set_title(title)
        
            burn_in = div(size(samples, 2), 2)
            
            samples_to_plot = samples[1:2, burn_in+1:end]
        
            if size(samples_to_plot, 2) < 2
                ax_plt.text(0.5, 0.5, "No Samples Post Burn-in", ha="center", va="center")
                return
            end

            kde_boundary = ((x_lim[1], x_lim[2]), (y_lim[1], y_lim[2]))
        
            kde_res = kde(samples_to_plot', boundary=kde_boundary, npoints=(Nx, Ny), bandwidth=bandwidth)
            Z_kde = kde_res.density
            if sum(Z_kde) > 0 Z_kde ./= sum(Z_kde) * dx * dy end
            ax_plt.pcolormesh(X, Y, Z_kde, cmap="viridis", vmin=color_lim[1], vmax=color_lim[2])
        
            alpha_val = min(1.0, 100 / size(samples_to_plot, 2))
            ax_plt.scatter(samples_to_plot[1, :], samples_to_plot[2, :], 
                           marker=".", color="red", s=10, alpha=alpha_val)

            mean_val, cov_val = calculate_sample_stats(samples_to_plot)
            
            if !any(isnan, mean_val) && !any(isnan, cov_val)
                text_str = "μ: [$(round(mean_val[1], digits=2)), $(round(mean_val[2], digits=2))], σ²: [$(round(cov_val[1,1], digits=2)), $(round(cov_val[2,2], digits=2))]"
                ax_plt.text(0.05, 0.95, text_str, transform=ax_plt.transAxes, fontsize=8,
                           verticalalignment="top", bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"white", "alpha"=>0.7))
            end
        
            ax_plt.set_xlim(x_lim)
            ax_plt.set_ylim(y_lim)
        end

    plot_mcmc_on_ax(ax[4], ens_WALNUTS, "WALNUTS")

    ax[5].semilogy(Array(0:N_iter), error', label=["GMBBVI","WALNUTS"])   

    if make_label==true
            ax[5].legend()
    end

    ymin, ymax = ax[5].get_ylim()
    if ymin > 0.1
            ax[5].set_ylim(0.1, ymax)
    end
end

Random.seed!(111);
Nx, Ny = 100, 100

N_modes = 40 
N_x = 50
N_bbvi_sample = 4*N_x
N_ens = 500  
Num = 5000   # WALNUTS
N_iter_GMBBVI = 500   # GMBBVI
N_iter_emcee = 20000   # emcee

fig, ax = PyPlot.subplots(nrows=5, ncols=5, sharex=false, sharey=false, figsize=(20,20))


dt = 0.9
quadrature_type = "random_sampling"

x0_w  = ones(N_modes)/N_modes
μ0, Σ0 = zeros(N_x), Matrix(I(N_x)) 
x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
for im = 1:N_modes
    x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
    xx0_cov[im, :, :] .= Σ0
end

q0 = randn(N_x)
ens_0=zeros(N_x,N_ens)
for j = 1:N_ens
    ens_0[:,j]  = rand(MvNormal(zeros(N_x), Σ0)) + μ0
end

@info "Running: Gaussian"
Gtype = "Gaussian"
ση = [1.0; 1.0; ones(N_x-2)]
y = [0.0; 1.0; zeros(N_x-2)]
arg = [1.0 1.0; 1.0 2.0]
args = (y, ση, arg, Gtype)
func_Phi(x) = Phi(x, args)
func_prob(x) = exp(-Phi(x, args))
func_Phi_marginal(x) = Phi(x, (y[1:2], ση[1:2], arg, Gtype))

t_gmbbvi = @elapsed obj_GMBBVI = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter=N_iter_GMBBVI, dt=dt, N_ens=N_bbvi_sample)
t_mcmc = @elapsed obj_MCMC = run_emcee_python(func_Phi, ens_0, N_iter_emcee)
t_walnuts = @elapsed obj_WALNUTS = run_walnuts_python(func_Phi, q0, N_x, Num)

@info "Gaussian - Time: GMBBVI: $(t_gmbbvi)s, MCMC: $(t_mcmc)s, WALNUTS: $(t_walnuts)s"
visualization(ax[1,:], obj_GMBBVI, obj_MCMC, obj_WALNUTS;
                          Nx=Nx, Ny=Ny, x_lim=[-4.0, 4.0], y_lim=[-3.0, 5.0], func_Phi=func_Phi_marginal, 
                          bandwidth=(0.06,0.11), make_label=true, N_iter=N_iter_GMBBVI, Num = Num, Gtype = Gtype)

@info "Running: Four mode"
Gtype = "Four_modes"
ση = 1.0
y = [4.2297; 4.2297; 0.5; 0.0; zeros(N_x-2)]
args = (y, ση, 0, Gtype)
func_Phi(x)= Phi(x, args)
func_prob(x)=exp(-Phi(x, args))
func_Phi_marginal(x) = Phi(x, (y[1:4], ση, 0, Gtype))

t_gmbbvi = @elapsed obj_GMBBVI = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter=N_iter_GMBBVI, dt=dt, N_ens=N_bbvi_sample)
t_mcmc = @elapsed obj_MCMC = run_emcee_python(func_Phi, ens_0, N_iter_emcee)
t_walnuts = @elapsed obj_WALNUTS = run_walnuts_python(func_Phi, q0, N_x, Num)


@info "Four_modes - Time: GMBBVI: $(t_gmbbvi)s, MCMC: $(t_mcmc)s, WALNUTS: $(t_walnuts)s"
visualization(ax[2,:], obj_GMBBVI, obj_MCMC, obj_WALNUTS;
                          Nx=Nx, Ny=Ny, x_lim=[-4.0, 4.0], y_lim=[-4.0, 4.0], func_Phi=func_Phi_marginal, 
                          bandwidth=(0.06,0.11), make_label=false, N_iter=N_iter_GMBBVI, Num = Num, Gtype = Gtype)


@info "Running: Circle"
Gtype = "Circle"
ση = [0.3; ones(N_x-2)]
y = [1.0; zeros(N_x-2)]
arg = Matrix(I, 2, 2)
args = (y, ση, arg, Gtype)
func_Phi(x) = Phi(x, args)
func_prob(x) = exp(-Phi(x, args))
func_Phi_marginal(x) = Phi(x, (y[1:1], ση[1:1], arg, Gtype))

t_gmbbvi = @elapsed obj_GMBBVI = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter=N_iter_GMBBVI, dt=dt, N_ens=N_bbvi_sample)
t_mcmc = @elapsed obj_MCMC = run_emcee_python(func_Phi, ens_0, N_iter_emcee)
t_walnuts = @elapsed obj_WALNUTS = run_walnuts_python(func_Phi, q0, N_x, Num)

@info "Circle - Time: GMBBVI: $(t_gmbbvi)s, MCMC: $(t_mcmc)s, WALNUTS: $(t_walnuts)s"
visualization(ax[3,:], obj_GMBBVI, obj_MCMC, obj_WALNUTS;
                          Nx=Nx, Ny=Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_Phi=func_Phi_marginal, 
                          bandwidth=(0.06,0.11), make_label=false, N_iter=N_iter_GMBBVI, Num = Num, Gtype = Gtype)

@info "Running: Banana"
Gtype = "Banana"
ση = [sqrt(10.0); sqrt(10.0); ones(N_x-2)]
y = [0.0; 1.0; zeros(N_x-2)]
arg = 10.0
args = (y, ση, arg, Gtype)
func_Phi(x) = Phi(x, args)
func_prob(x) = exp(-Phi(x, args))
func_Phi_marginal(x) = Phi(x, (y[1:2], ση[1:2], arg, Gtype))

t_gmbbvi = @elapsed obj_GMBBVI = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter=N_iter_GMBBVI, dt=dt, N_ens=N_bbvi_sample)
t_mcmc = @elapsed obj_MCMC = run_emcee_python(func_Phi, ens_0, N_iter_emcee)
t_walnuts = @elapsed obj_WALNUTS = run_walnuts_python(func_Phi, q0, N_x, Num)

@info "Banana - Time: GMBBVI: $(t_gmbbvi)s, MCMC: $(t_mcmc)s, WALNUTS: $(t_walnuts)s"
visualization(ax[4,:], obj_GMBBVI, obj_MCMC, obj_WALNUTS;
                          Nx=Nx, Ny=Ny, x_lim=[-4.0, 4.0], y_lim=[-2.0, 15.0], func_Phi=func_Phi_marginal, 
                          bandwidth=(0.06,0.11), make_label=false, N_iter=N_iter_GMBBVI, Num = Num, Gtype = Gtype)

@info "Running: Funnel"
Gtype = "Funnel"
ση = ones(N_x)
y = zeros(N_x)
arg = Matrix(I, N_x-1, N_x-1)
args = (y, ση, arg, Gtype)
func_Phi(x) = Phi(x, args)
func_prob(x) = exp(-Phi(x, args))
func_Phi_marginal(x) = Phi(x, (y[1:2], ση[1:2], Matrix(I, 1, 1), Gtype))

t_gmbbvi = @elapsed obj_GMBBVI = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov; N_iter=N_iter_GMBBVI, dt=dt, N_ens=N_bbvi_sample)
t_mcmc = @elapsed obj_MCMC = run_emcee_python(func_Phi, ens_0, N_iter_emcee)
t_walnuts = @elapsed obj_WALNUTS = run_walnuts_python(func_Phi, q0, N_x, Num)

@info "Funnel - Time: GMBBVI: $(t_gmbbvi)s, MCMC: $(t_mcmc)s, WALNUTS: $(t_walnuts)s"
visualization(ax[5,:], obj_GMBBVI, obj_MCMC, obj_WALNUTS;
                          Nx=Nx, Ny=Ny, x_lim=[-8.0, 8.0], y_lim=[-10.0, 10.0], func_Phi=func_Phi_marginal, 
                          bandwidth=(0.06,0.11), make_label=false, N_iter=N_iter_GMBBVI, Num = Num, Gtype = Gtype)

for r in 1:5
    ax[r,1].set_title("Reference", fontsize=15)
    ax[r,2].set_title("GMBBVI", fontsize=15)
    ax[r,3].set_title("Stretch_Move MCMC", fontsize=15)
    ax[r,4].set_title("WALNUTS", fontsize=15)
    ax[r,5].set_title("TV Error", fontsize=15)
end

fig.tight_layout()
fig.savefig("Benchmark_$(N_x)_Dim.pdf")
