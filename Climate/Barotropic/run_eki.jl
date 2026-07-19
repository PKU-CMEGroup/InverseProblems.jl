include("EKI_Barotropic_Section52.jl")

result = run_section52_eki(
    filter_type="EKI",      # 或 "EKI", "EAKI"
    eki_dt=0.2,              # 这里就是 Δτ
    alpha_reg=1.0,
    update_freq=1,
    num_fourier=85,
    nlat=256,
    trunc_N=7,
    nobs=50,
    n_obs_frames=2,
    end_time=86400,
    n_iter=20,
    n_ens=30
)