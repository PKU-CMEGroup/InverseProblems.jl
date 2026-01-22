ENV["PYTHON"] = expanduser("~/.conda/envs/my_walnuts_env/bin/python") 

using PyPlot
using LinearAlgebra
using Random
using Distributions
using KernelDensity
using NPZ
using Statistics

using PyCall
statsmodels_kde = pyimport("statsmodels.nonparametric.kernel_density")
KDEMultivariate = statsmodels_kde.KDEMultivariate

include("../Inversion/GMBBVI.jl")
include("../Inversion/Plot.jl")
include("../Inversion/GaussianMixture.jl")
include("./MultiModal.jl")

# --- Configuration ---
dim_test = [2, 10, 50]
n_trials = 10
walnuts_dir = "data_walnuts"
gmbbvi_dir = "data_gmbbvi"

# --- Grid Setup ---
Nx, Ny = 200, 200
x_lim = [-10.0, 10.0]
y_lim = [-20.0, 20.0]
xx = range(x_lim[1], stop=x_lim[2], length=Nx)
yy = range(y_lim[1], stop=y_lim[2], length=Ny)
dx, dy = step(xx), step(yy)
X = repeat(xx, 1, Ny)
Y = repeat(yy, 1, Nx)'

# Helper to plot error with shading
function plot_err_with_std(ax, data_list, color, label, linestyle="-", x_vals=nothing)
    if isempty(data_list) return end
    mat = reduce(hcat, data_list)' 
    mu = vec(mean(mat, dims=1))
    sigma = vec(std(mat, dims=1))
    
    xs = (x_vals === nothing) ? (0:length(mu)-1) : x_vals
    
    ax.semilogy(xs, mu, color=color, linestyle=linestyle, label=label, linewidth=1.5)
    ax.fill_between(xs, mu .- sigma, mu .+ sigma, color=color, alpha=0.15, linewidth=0)
end

# --- Main Plotting Loop ---
fig, axes = PyPlot.subplots(nrows=length(dim_test), ncols=5, sharex=false, sharey=false, figsize=(20, 12))
#PyPlot.subplots_adjust(wspace=0.35, hspace=0.3)

for (idx, N_x) in enumerate(dim_test)
    println("Processing Dimension: $N_x")
    ax_row = axes[idx, :] 
    
    for ax in ax_row
        ax.tick_params(axis="both", which="major", labelsize=15)
    end

    # ---------------------------------------------------------
    # 1. Reference Density
    # ---------------------------------------------------------
    Gtype = "Funnel"
    ση = ones(N_x)
    A = Diagonal(ones(N_x-1))
    y = zeros(N_x)
    func_marginal_args = (y[1:2], ση[1:2], A[1,1], Gtype)
    func_Phi_marginal(x) = Phi(x, func_marginal_args)
    Z_ref = posterior_2d(func_Phi_marginal, X, Y, "func_Phi")
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    
    ax_row[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim, shading="auto")
    ax_row[1].set_xlim(x_lim)
    ax_row[1].set_ylim(y_lim)
    ax_row[1].set_title("Reference", fontsize=15)

    # ---------------------------------------------------------
    # Load Data & Prepare Average/Last structures
    # ---------------------------------------------------------
    
    gm_me, gm_ce, gm_l1 = [], [], []
    wn_me, wn_ce, wn_l1 = [], [], []
    
    last_gm_data = nothing
    last_wn_data = nothing
    
    x_marg = collect(range(-15, 15, length=400))
    y_gm_accum = zeros(length(x_marg)) 
    
    wn_total_iter = 0
    wn_chunk_size = 2000
    
    for t in 0:(n_trials-1)
        # --- Load GMBBVI ---
        fn_gm = joinpath(gmbbvi_dir, "dim_$(N_x)_trial_$(t).npz")
        if isfile(fn_gm)
            data = npzread(fn_gm)
            push!(gm_me, data["error_mean"])
            push!(gm_ce, data["error_cov"])
            push!(gm_l1, data["error_L1"])
            
            w, mu, cov = data["final_w"], data["final_mu_2d"], data["final_cov_2d"]
            for k in 1:length(w)
                y_gm_accum .+= w[k] .* pdf.(Normal(mu[k, 1], sqrt(cov[k, 1, 1])), x_marg)
            end
            
            if t == n_trials - 1
                last_gm_data = data
            end
        end

        # --- Load WALNUTS ---
        fn_wn = joinpath(walnuts_dir, "dim_$(N_x)_trial_$(t).npz")
        if isfile(fn_wn)
            data = npzread(fn_wn)
            push!(wn_me, data["mean_err"])
            push!(wn_ce, data["cov_err"])
            if haskey(data, "l1_err_chunk")
                push!(wn_l1, data["l1_err_chunk"])
            end
            
            if haskey(data, "chunk_size") wn_chunk_size = data["chunk_size"] end
            if haskey(data, "total_iter") wn_total_iter = data["total_iter"] end
            
            if t == n_trials - 1
                last_wn_data = data
            end
        end
    end

    # ---------------------------------------------------------
    # 2. GMBBVI - Visualization of LAST Trial
    # ---------------------------------------------------------
    if last_gm_data !== nothing
        w = last_gm_data["final_w"]
        mu = last_gm_data["final_mu_2d"]
        cov = last_gm_data["final_cov_2d"]
        
        Z_gm_last = Gaussian_mixture_2d(w, mu, cov, X, Y)
        ax_row[2].pcolormesh(X, Y, Z_gm_last, cmap="viridis", clim=color_lim, shading="auto")
        
        ax_row[2].scatter(mu[:, 1], mu[:, 2], marker="o", color="red", facecolors="none", alpha=0.6, s=20)
        if haskey(last_gm_data, "x0_mean_2d")
            x0_mu = last_gm_data["x0_mean_2d"]
            ax_row[2].scatter(x0_mu[:, 1], x0_mu[:, 2], marker="x", color="grey", alpha=0.4, s=15)
        end
    end
    ax_row[2].set_xlim(x_lim)
    ax_row[2].set_ylim(y_lim)
    ax_row[2].set_title("GMBBVI", fontsize=15)

    # ---------------------------------------------------------
    # 3. WALNUTS - Visualization of LAST Trial (KDE)
    # ---------------------------------------------------------
    if last_wn_data !== nothing
        wn_samples = last_wn_data["samples"]
        n_kde_samples = min(size(wn_samples, 2), 5000) 
        kde_input = wn_samples[:, (end-n_kde_samples+1):end]'
        
        kde = KDEMultivariate(data=kde_input, var_type="cc", bw="cv_ml")
        grid_points = hcat(vec(X), vec(Y))
        Z_wn_flat = kde.pdf(grid_points)
        Z_wn = reshape(Z_wn_flat, size(X))
        
        ax_row[3].pcolormesh(X, Y, Z_wn, cmap="viridis", shading="auto")
        
        n_scatter = min(size(wn_samples, 2), 500)
        scatter_pts = wn_samples[:, (end-n_scatter+1):end]
        ax_row[3].scatter(scatter_pts[1, :], scatter_pts[2, :], marker=".", color="red", s=3, alpha=0.2)
    end
    ax_row[3].set_xlim(x_lim)
    ax_row[3].set_ylim(y_lim)
    ax_row[3].set_title("WALNUTS", fontsize=15)

    # ---------------------------------------------------------
    # 4. Marginal Dim 1
    # ---------------------------------------------------------
    y_true = pdf.(Normal(0, 3.0), x_marg)
    ax_row[4].plot(x_marg, y_true, "k--", linewidth=2, label="Reference", alpha=0.6)

    # --- GMBBVI Marginal (Last Trial) ---
    if last_gm_data !== nothing
        w = last_gm_data["final_w"]
        mu = last_gm_data["final_mu_2d"]
        cov = last_gm_data["final_cov_2d"]
        
        y_gm_last = zeros(length(x_marg))
        for k in 1:length(w)
            y_gm_last .+= w[k] .* pdf.(Normal(mu[k, 1], sqrt(cov[k, 1, 1])), x_marg)
        end
        
        ax_row[4].plot(x_marg, y_gm_last, color="red", linestyle="-", label="GMBBVI")
    end

    # --- WALNUTS Marginal (Last Trial) ---
    if last_wn_data !== nothing
        samples = last_wn_data["samples"]
        samples_dim1_last = samples[1, :]
        
        ax_row[4].hist(samples_dim1_last, bins=100, density=true, color="blue", alpha=0.2, label="WALNUTS")
    end
    
    ax_row[4].legend(fontsize=10, loc="upper right")
    ax_row[4].set_title(raw"$\theta_{(1)}$ marginal density", fontsize=15)

    # ---------------------------------------------------------
    # 5. Convergence
    # ---------------------------------------------------------
    ax_err = ax_row[5]
    ax_err.grid(true, which="both", linestyle="--", alpha=0.3)
    
    plot_err_with_std(ax_err, gm_me, "red",       "GMBBVI mean err.", "-")
    plot_err_with_std(ax_err, gm_ce, "magenta",   "GMBBVI covariance err.",  "-")
    plot_err_with_std(ax_err, gm_l1, "darkorange","GMBBVI TV distance",   "-")
    
    ax_err.set_xlabel("GMBBVI iterations", color="black", fontsize=15)
    
    # WALNUTS Colors (Twin Axis)
    ax_err_twin = ax_err.twiny()
    ax_err_twin.tick_params(labelsize=15)
    
    if !isempty(wn_me)
        if wn_total_iter == 0 wn_total_iter = length(wn_me[1]) end
        x_wn = range(0, stop=wn_total_iter, length=length(wn_me[1]))
        
        plot_err_with_std(ax_err_twin, wn_me, "blue",      "WALNUTS mean err.", "-", x_wn)
        plot_err_with_std(ax_err_twin, wn_ce, "cyan",      "WALNUTS covariance err.",  "-", x_wn)
    end
    
    ax_err_twin.set_xlabel("WALNUTS iterations", color="black", fontsize=15)
    ax_err_twin.tick_params(axis="x", labelcolor="black")

    lines_1, labels_1 = ax_err.get_legend_handles_labels()
    lines_2, labels_2 = ax_err_twin.get_legend_handles_labels()
    ax_err.legend(vcat(lines_1, lines_2), vcat(labels_1, labels_2), fontsize=10, loc="lower right", framealpha=0.9)
end

fig.tight_layout()
fig.savefig("combined_funnel_comparison_final_avg_marg.pdf")
println("Done. Saved to funnel_comparison.pdf")
close(fig)