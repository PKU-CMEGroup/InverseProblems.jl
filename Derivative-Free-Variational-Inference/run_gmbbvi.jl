using LinearAlgebra
using Random
using Distributions
using NPZ
include("../Inversion/GaussianMixture.jl")
include("../Inversion/GMBBVI.jl")
include("../Inversion/AnnealingInitialize.jl")
include("./MultiModal.jl")

output_dir = "data_gmbbvi_new"
if !isdir(output_dir)
    mkdir(output_dir)
end

Random.seed!(123);

dim_test = [2, 10, 50]
n_trials = 10
n_iter = 2000 
dt = 0.5
n_modes = 40

# Grid for L1 (Still 2D for density map comparison)
Nx, Ny = 200, 200
x_lim = [-10.0, 15.0]
y_lim = [-20.0, 20.0]
xx = range(x_lim[1], stop=x_lim[2], length=Nx)
yy = range(y_lim[1], stop=y_lim[2], length=Ny)
dx, dy = step(xx), step(yy)
X_grid = repeat(xx, 1, Ny)
Y_grid = repeat(yy, 1, Nx)'

for (idx, N_x) in enumerate(dim_test)
    println("\n" * "="^40)
    println("Running GMBBVI for Dimension: $N_x")
    println("="^40)

    N_ens = 4 * N_x
    
    Gtype = "Funnel"
    ση = ones(N_x)
    A = Diagonal(ones(N_x-1))
    y = zeros(N_x)
    func_args = (y, ση, A, Gtype)
    func_Phi(x) = Phi(x, func_args)
    func_marginal_args = (y[1:2], ση[1:2], A[1,1], Gtype)
    func_Phi_marginal(x) = Phi(x, func_marginal_args)
    Z_ref = posterior_2d(func_Phi_marginal, X_grid, Y_grid, "func_Phi")

    posterior_mean, posterior_cov = multimodal_moments(Gtype)

    for i_trial in 1:n_trials
        println("  -> Trial $i_trial / $n_trials")
        Random.seed!(i_trial * 100)

        x0_w  = ones(n_modes)/n_modes
        μ0, Σ0 = [0; zeros(N_x-1)], Matrix(I(N_x)) 
        x0_mean, xx0_cov = zeros(n_modes, N_x), zeros(n_modes, N_x, N_x)
        for im = 1:n_modes
            x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
            xx0_cov[im, :, :] .= Σ0
        end

        result_obj = Gaussian_mixture_GMBBVI(func_Phi, x0_w, x0_mean, xx0_cov;
            N_iter = n_iter, dt = dt, N_ens = N_ens)
        
        err_mean_hist = zeros(n_iter + 1)
        err_cov_hist = zeros(n_iter + 1)
        err_l1_hist = zeros(n_iter + 1)

        for iter = 0:n_iter
            logx_w = result_obj.logx_w[iter+1]
            x_w = exp.(logx_w)
            x_w /= sum(x_w)
            x_mean = result_obj.x_mean[iter+1]
            xx_cov = result_obj.xx_cov[iter+1]

            curr_mean, curr_cov = compute_ρ_gm_moments(x_w, x_mean, xx_cov)
            
            # --- MODIFIED: Compute Error for Dimension 1 ONLY ---
            # Julia uses 1-based indexing. Dim 1 is index 1.
            err_mean_hist[iter+1] = abs(curr_mean[1] - posterior_mean[1])
            err_cov_hist[iter+1]  = abs(curr_cov[1, 1] - posterior_cov[1, 1]) / abs(posterior_cov[1, 1])
            
            # L1 Error is still computed on the 2D grid for density comparison
            x_mean_2d = x_mean[:, 1:2]
            xx_cov_2d = xx_cov[:, 1:2, 1:2]
            Z_est = Gaussian_mixture_2d(x_w, x_mean_2d, xx_cov_2d, X_grid, Y_grid)
            err_l1_hist[iter+1] = norm(Z_est - Z_ref, 1) * dx * dy
        end

        final_w = exp.(result_obj.logx_w[end])
        final_w /= sum(final_w)
        
        filename = joinpath(output_dir, "dim_$(N_x)_trial_$(i_trial-1).npz")
        
        # --- SAVE METADATA ---
        npzwrite(filename, Dict(
            "error_mean" => err_mean_hist,
            "error_cov" => err_cov_hist,
            "error_L1" => err_l1_hist,
            "final_w" => final_w,
            "x0_mean_2d" => x0_mean[:, 1:2], 
            "final_mu_2d" => result_obj.x_mean[end][:, 1:2], 
            "final_cov_2d" => result_obj.xx_cov[end][:, 1:2, 1:2],
            "total_iter" => n_iter 
        ))
    end
end
println("GMBBVI done.")