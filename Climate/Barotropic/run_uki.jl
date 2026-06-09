include("UKI_Barotropic_Section52.jl")

result = run_section52_uki(
    num_fourier=85,
    nlat=256,
    trunc_N=7,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    n_iter=20,
)