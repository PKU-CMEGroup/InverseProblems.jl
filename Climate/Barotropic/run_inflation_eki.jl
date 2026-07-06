include("InflationEKI_Barotropic_Section52.jl")

result = run_section52_inflation_eki(
    filter_type="drop-ETKI",      
    inflation_dt=0.2,          
    dropout_rate=0.3,
    num_fourier=85,
    nlat=256,
    trunc_N=7,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    n_iter=20,
)
