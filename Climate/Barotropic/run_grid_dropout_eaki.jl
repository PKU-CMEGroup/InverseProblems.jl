include("GridDropoutEAKI_Barotropic_Section52.jl")

result = run_section52_grid_dropout_eaki(
    num_fourier=85,
    nlat=256,
    trunc_N=85,
    init_trunc_N=7,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    n_iter=10,
    n_ens=60,
    inflation_dt=0.5,
    dropout_rate=0.5,
    project_initial_ensemble=true,
    store_ensemble_history=false,
)
