include("InflatedEKI_Barotropic.jl")

# Paired T7 experiment. Both cases use prior = initialization = N(0, 9I_63).
result = run_inflated_eki_comparison(
    filter_type="dropout-EAKI",      
    dropout_correction_mode="joint",
    model_dt=1800,
    inflation_dt=0.5,          
    dropout_rate=0.5,
    num_fourier=85,
    nlat=256,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    project_initial_ensemble=true,
)
