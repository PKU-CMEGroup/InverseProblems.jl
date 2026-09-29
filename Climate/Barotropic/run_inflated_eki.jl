include("InflatedEKI_Barotropic.jl")

result = run_inflated_eki(
    filter_type="dropout-EAKI",      
    dropout_correction_mode="joint",
    model_dt=1200,
    inflation_dt=0.5,          
    dropout_rate=0.5,
    num_fourier=85,
    nlat=256,
    trunc_N=20,
    init_trunc_N=20,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    n_iter=10,
    n_ens=60,
    project_initial_ensemble=true,
)
