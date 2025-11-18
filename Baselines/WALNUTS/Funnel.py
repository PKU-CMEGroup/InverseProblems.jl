import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.nonparametric.kernel_density import KDEMultivariate

if (sys.path[-1].find("WALNUTSpy") == -1):
    try:
        import WALNUTS as wn
        import targetDistr as td
        import adaptiveIntegrators as ai
    except ImportError:
        print("ERROR: WALNUTS path not found or incorrect.")
        print("Please adjust the sys.path.append line to point to your WALNUTSpy directory.")
        sys.exit(1)

# --- Matplotlib Global Parameters ---
rcParams = {
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
}
plt.rcParams.update(rcParams)

np.random.seed(seed=0)
n = 10
target_func = td.funneln
target_name = f"funnel-{n}D"

nIter = 200000

q0 = np.zeros(n)
q0[0] = 3.0 * np.random.normal()
for i in range(n - 1):
    q0[i + 1] = np.exp(0.5 * q0[0]) * np.random.normal()


def gen(q):
    return (np.array([q[0], q[1]]))


# --- 3. Warmup Phase ---
print("--- Starting Warmup Phase ---")
warmup_samples, warmup_diagnostics = wn.WALNUTS(target_func,
                                                q0,
                                                generated=gen,
                                                integrator=ai.adaptLeapFrogR2P,
                                                M=12, H0=0.3, delta0=0.3, numIter=1000,
                                                warmupIter=1000,
                                                adaptDeltaTarget=0.6,
                                                recordOrbitStats=False
                                                )
print("--- Warmup Finished ---\n")

q_current = warmup_samples[:, -1]
H_current = warmup_diagnostics[-1, 15]
delta_current = warmup_diagnostics[-1, 18]

# --- 4. Main Sampling Phase ---
print(f"--- Starting Main Sampling Phase for {nIter} iterations ---")
start_time = time.time()

all_samples = []
all_diagnostics = []
chunk_size = 1000
num_chunks = int(np.ceil(nIter / chunk_size))

for i in range(num_chunks):
    current_iter_in_chunk = min(chunk_size, nIter - i * chunk_size)
    if current_iter_in_chunk <= 0:
        break

    samples_chunk, diagnostics_chunk, _, _ = wn.WALNUTS(
        target_func,
        q_current,
        generated=gen,
        integrator=ai.adaptLeapFrogR2P,
        M=12, H0=H_current, delta0=delta_current,
        numIter=current_iter_in_chunk,
        warmupIter=0,
        recordOrbitStats=True
    )

    all_samples.append(samples_chunk)
    all_diagnostics.append(diagnostics_chunk)

    q_current = samples_chunk[:, -1]
    H_current = diagnostics_chunk[-1, 15]
    delta_current = diagnostics_chunk[-1, 18]

    iterations_done = (i + 1) * chunk_size if (i + 1) < num_chunks else nIter
    total_time_so_far = time.time() - start_time
    print(f"Completed {iterations_done}/{nIter} iterations. Total time elapsed: {total_time_so_far:.2f} seconds")

final_samples = np.concatenate(all_samples, axis=1)
final_diagnostics = np.concatenate(all_diagnostics, axis=0)
total_time = time.time() - start_time

print("--- Main Sampling Finished ---")
print(f"Total sampling time for {nIter} iterations: {total_time:.2f} seconds\n")


def visualize_walnuts_results(samples, target_name, n_dims):
    samples_to_plot = samples[:2, :]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5), dpi=100)
    ax1, ax2, ax3, ax4 = axes

    x_lim = [-10, 15]
    y_lim = [-20, 20]
    n_bins_grid = 200

    xx = np.linspace(x_lim[0], x_lim[1], n_bins_grid)
    yy = np.linspace(y_lim[0], y_lim[1], n_bins_grid)
    X, Y = np.meshgrid(xx, yy)

    def log_marginal_funnel_2d(q0, q1):
        log_p_q0 = norm.logpdf(q0, loc=0, scale=3.0)
        scale_q1 = np.exp(0.5 * q0) + 1e-9
        log_p_q1_given_q0 = norm.logpdf(q1, loc=0, scale=scale_q1)
        return log_p_q0 + log_p_q1_given_q0

    log_p_grid = log_marginal_funnel_2d(X, Y)

    p_grid = np.exp(log_p_grid - np.nanmax(log_p_grid))
    Z_ref = p_grid.reshape(X.shape)
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    Z_ref /= np.nansum(Z_ref) * dx * dy

    color_lim = (Z_ref.min(), Z_ref.max())

    ax1.pcolormesh(X, Y, Z_ref, cmap='viridis', shading='auto', vmin=color_lim[0], vmax=color_lim[1])
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)

    num_total_samples = samples_to_plot.shape[1]
    max_points_to_plot = 500
    if num_total_samples > max_points_to_plot:
        scatter_samples = samples_to_plot[:, -max_points_to_plot:]
    else:
        scatter_samples = samples_to_plot

    data_for_kde = samples_to_plot.T
    kde_samples_max = 5000
    if data_for_kde.shape[0] > kde_samples_max:
        data_for_kde_subset = data_for_kde[-kde_samples_max:, :]
    else:
        data_for_kde_subset = data_for_kde

    kde = KDEMultivariate(data=data_for_kde_subset, var_type='cc', bw='cv_ml')

    grid_points_2d = np.vstack([X.ravel(), Y.ravel()])
    Z_kde = kde.pdf(grid_points_2d.T).reshape(X.shape)

    Z_kde /= np.nansum(Z_kde) * dx * dy

    ax2.pcolormesh(X, Y, Z_kde, cmap='viridis', shading='auto', vmin=color_lim[0], vmax=color_lim[1])

    ax2.scatter(scatter_samples[0, :], scatter_samples[1, :],
                marker='.', color='red', s=3, alpha=0.1)

    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)

    true_mean_full = np.zeros(n_dims)
    true_cov_diag = [9.0] + [np.exp(4.5)] * (n_dims - 1)
    true_cov_full = np.diag(true_cov_diag)
    true_mean_to_compare = true_mean_full[:2]
    true_cov_to_compare = true_cov_full[:2, :2]

    n_iter_samples = samples_to_plot.shape[1]
    d = samples_to_plot.shape[0]

    running_mean_err = np.zeros(n_iter_samples)
    running_cov_err = np.zeros(n_iter_samples)

    current_mean = np.zeros(d)
    M2 = np.zeros((d, d))

    for k in range(n_iter_samples):
        x = samples_to_plot[:, k]
        count = k + 1

        delta = x - current_mean
        current_mean += delta / count
        delta2 = x - current_mean
        M2 += np.outer(delta, delta2)

        if count < 2:
            current_cov = np.zeros((d, d))
        else:
            current_cov = M2 / (count - 1)

        running_mean_err[k] = np.linalg.norm(current_mean - true_mean_to_compare)
        running_cov_err[k] = np.linalg.norm(current_cov - true_cov_to_compare) / np.linalg.norm(true_cov_to_compare)

    iters = np.arange(1, n_iter_samples + 1)
    ax3.semilogy(iters, running_mean_err)
    ax3.set_xlabel("Number of Samples")
    ax3.grid(False)

    ax4.semilogy(iters, running_cov_err)
    ax4.set_xlabel("Number of Samples")
    ax4.grid(False)

    plt.tight_layout()
    plt.savefig("funnel.pdf")
    plt.close()


samples_for_moments = final_samples
est_mean_final = np.mean(samples_for_moments, axis=1)
est_cov_final = np.cov(samples_for_moments)
print("--- Final Estimated Moments (from main sampling phase) ---")
print(f"Estimated Mean (first 2 dims): {est_mean_final[:2]}")
print(f"Estimated Covariance (2x2 block):\n{est_cov_final[:2, :2]}")
print("-" * 55 + "\n")

visualize_walnuts_results(final_samples, target_name, n_dims=n)