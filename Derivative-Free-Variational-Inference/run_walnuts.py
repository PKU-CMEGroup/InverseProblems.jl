import sys
import time
import numpy as np
from scipy.stats import norm, gaussian_kde
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
walnuts_path = os.path.join(parent_dir, "WALNUTS")

if walnuts_path not in sys.path:
    sys.path.append(walnuts_path)
    print(f"Added path: {walnuts_path}")

try:
    import WALNUTS as wn
    import targetDistr as td
    import adaptiveIntegrators as ai
except ImportError:
    print("ERROR: WALNUTS path not found.")
    sys.exit(1)

def get_funnel_ref_density(X, Y, dx, dy):
    log_p_x = norm.logpdf(X, loc=0, scale=3.0)
    scale_y = np.exp(0.5 * X)
    log_p_y_given_x = norm.logpdf(Y, loc=0, scale=scale_y)
    log_p = log_p_x + log_p_y_given_x
    Z = np.exp(log_p)
    return Z / (np.nansum(Z) * dx * dy)

def run_simulation():
    # --- Configuration ---
    dim_test = [2, 10, 50]
    n_trials = 10
    n_iter = 16000  
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data_walnuts")    
    os.makedirs(output_dir, exist_ok=True)

    # --- Setup Grid for L1 Error ---
    n_bins = 200
    xx = np.linspace(-10, 15, n_bins)
    yy = np.linspace(-20, 20, n_bins)
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    X_grid, Y_grid = np.meshgrid(xx, yy)
    Z_ref = get_funnel_ref_density(X_grid, Y_grid, dx, dy)
    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel()]) 

    for idx, n in enumerate(dim_test):
        print(f"\n{'='*20} Running WALNUTS for Dimension: {n} {'='*20}")

        # True stats for Funnel (Dim 0 is x ~ N(0, 3^2))
        true_mean_dim0 = 0.0
        true_var_dim0 = 9.0

        for trial in range(n_trials):
            print(f"--- Dimension {n} | Trial {trial + 1}/{n_trials} ---")
            
            np.random.seed(seed=trial * 100)
            target_func = td.funneln

            # Initialization
            q0 = np.zeros(n)
            q0[0] = 3.0 * np.random.normal()
            for i in range(n - 1):
                q0[i + 1] = np.exp(0.5 * q0[0]) * np.random.normal()

            def gen(q):
                return (np.array([q[0], q[1]]))

            # Warmup
            warmup_samples, warmup_diagnostics = wn.WALNUTS(
                target_func, q0, generated=gen, integrator=ai.adaptLeapFrogR2P,
                M=12, H0=0.3, delta0=0.3, numIter=1000,
                warmupIter=1000, adaptDeltaTarget=0.6, recordOrbitStats=False
            )
            
            q_current = warmup_samples[:, -1]
            H_current = warmup_diagnostics[-1, 15]
            delta_current = warmup_diagnostics[-1, 18]

            # Main Sampling
            chunk_size = 2000
            num_chunks = int(np.ceil(n_iter / chunk_size))
            
            samples_list = [] 
            curr_mean = np.zeros(2) 
            M2 = np.zeros((2, 2))
            count = 0
            
            running_mean_err = []
            running_cov_err = []
            running_l1_err = [] 
            
            start_time = time.time()

            for i in range(num_chunks):
                current_iter_in_chunk = min(chunk_size, n_iter - i * chunk_size)
                
                samples_chunk, diagnostics_chunk, _, _ = wn.WALNUTS(
                    target_func, q_current, generated=gen,
                    integrator=ai.adaptLeapFrogR2P,
                    M=12, H0=H_current, delta0=delta_current,
                    numIter=current_iter_in_chunk,
                    warmupIter=0,
                    recordOrbitStats=True
                )
                
                samples_list.append(samples_chunk)
                
                q_current = samples_chunk[:, -1]
                H_current = diagnostics_chunk[-1, 15]
                delta_current = diagnostics_chunk[-1, 18]

                # 1. Online Mean/Cov Error (DIM 0 ONLY)
                k_chunk = samples_chunk.shape[1]
                for k in range(k_chunk):
                    x = samples_chunk[:, k] # x is 2D (projected by gen)
                    count += 1
                    delta = x - curr_mean
                    curr_mean += delta / count
                    delta2 = x - curr_mean
                    M2 += np.outer(delta, delta2)
                    
                    if count > 1:
                        curr_cov = M2 / (count - 1)
                        # Error for Dimension 0 ONLY
                        me = np.abs(curr_mean[0] - true_mean_dim0)
                        ce = np.abs(curr_cov[0, 0] - true_var_dim0) / true_var_dim0
                        
                        running_mean_err.append(me)
                        running_cov_err.append(ce)
                    else:
                        running_mean_err.append(0.0)
                        running_cov_err.append(0.0)
                
                # 2. L1 Error (Using ALL history)
                # --- FIX: Slice to first 2 dimensions ---
                # This ensures KDE dimensionality matches grid_coords (2D)
                # even if the simulation dimension 'n' is high (10, 50).
                history_all = np.concatenate(samples_list, axis=1)
                history_2d = history_all[:2, :] 
                
                try:
                    kde = gaussian_kde(history_2d)
                    Z_est_flat = kde(grid_coords)
                    Z_est = Z_est_flat.reshape(X_grid.shape)
                    Z_est /= (np.nansum(Z_est) * dx * dy)
                    l1 = np.nansum(np.abs(Z_ref - Z_est)) * dx * dy
                except Exception as e:
                    # Print error once to help debugging if it still fails
                    if i == 0: print(f"  Warning: KDE failed: {e}")
                    l1 = running_l1_err[-1] if len(running_l1_err) > 0 else 1.0
                
                running_l1_err.append(l1)

            total_time = time.time() - start_time
            print(f"  Finished Trial {trial+1}. Time: {total_time:.2f}s")
            
            final_samples = np.concatenate(samples_list, axis=1)
            
            np.savez(os.path.join(output_dir, f"dim_{n}_trial_{trial}.npz"), 
                     samples=final_samples, 
                     mean_err=np.array(running_mean_err),
                     cov_err=np.array(running_cov_err),
                     l1_err_chunk=np.array(running_l1_err),
                     chunk_size=chunk_size,
                     total_iter=n_iter) 

    print("\nAll WALNUTS simulations completed.")

if __name__ == "__main__":
    run_simulation()