using LinearAlgebra, Random, Statistics, Distributions

mutable struct EKIObj{FT<:AbstractFloat, IT<:Int}
    # Type of ensemble Kalman inversion: "EKI", "ETKI", "EAKI"
    filter_type::String
    # Ensemble of parameters at each iteration: θ ∈ R^(N_θ × N_ens)
    θ::Vector{Array{FT,2}}
    # Predicted observations for each ensemble
    y_pred::Vector{Array{FT,2}}
    # Observed data vector
    y::Array{FT,1}
    # Lower triangular sqrt of observation covariance
    Σ_y_sqrt::LowerTriangular{FT,Matrix{FT}}
    # Ensemble size
    N_ens::IT
    # Parameter dimension
    N_θ::IT
    # Observation dimension
    N_y::IT
    # Artificial time step
    Δτ::FT
    # Dropout rate for dropout optimization EAKI/ETKI
    dropout_rate::FT
end

# Constructor
function EKIObj(filter_type::String, θ0::Array{FT,2}, y_pred_0::Array{FT,2}, y0::Array{FT,1}, 
                        Σ_y::Array{FT,2}, Δτ::FT, dropout_rate::FT=FT(0.5)) where FT<:AbstractFloat
    θ = [θ0]
    y_pred = [y_pred_0]

    N_θ, N_ens = size(θ0)
    N_y = size(y0, 1)
    Σ_y_sqrt = LowerTriangular(cholesky(Σ_y).L)
    obj = EKIObj(filter_type, θ, y_pred, y0, Σ_y_sqrt, N_ens, N_θ, N_y, Δτ, dropout_rate)
    return obj
end

function is_dropout_optimization_filter(filter_type::String)
    return filter_type == "dropout-EAKI" ||
           filter_type == "dropout-ETKI" ||
           filter_type == "dropout-NF-EKI"
end

function active_svd_rank(s::AbstractVector{FT}) where FT<:AbstractFloat
    isempty(s) && return 0
    max_s = maximum(s)
    max_s == zero(FT) && return 0
    tol = max(eps(FT) * length(s) * max_s, FT(1e-12) * max_s)
    r = findlast(s .> tol)
    return isnothing(r) ? 0 : r
end

function dropout_optimization_mean(eki::EKIObj, forward::Function, m_hat::Array{FT,2},
                                   Z_hat::Array{FT,2}, Σ_y::Array{FT,2}) where FT<:AbstractFloat
    dropout_λ = 1 - eki.dropout_rate
    0 < dropout_λ <= 1 || error("dropout_rate must satisfy 0 <= dropout_rate < 1")

    θ_hat = m_hat .+ Z_hat * sqrt(eki.N_ens - 1)
    x_hat = forward(θ_hat)
    x_hat_mean = mean(x_hat, dims=2)
    Y_hat = (x_hat .- x_hat_mean) ./ sqrt(eki.N_ens - 1)

    m_sub = m_hat .+ reshape((Z_hat * Y_hat') *
                             ((Y_hat * Y_hat' + Σ_y) \ (eki.y .- vec(x_hat_mean))), :, 1)

    ρ = rand(Bernoulli(dropout_λ), eki.N_θ)
    while sum(ρ) == 0
        ρ = rand(Bernoulli(dropout_λ), eki.N_θ)
    end
    Z_tilde = reshape(ρ, :, 1) .* Z_hat

    θ_tilde = m_sub .+ Z_tilde * sqrt(eki.N_ens - 1)
    x_tilde = forward(θ_tilde)
    x_tilde_mean = mean(x_tilde, dims=2)
    Y_tilde = (x_tilde .- x_tilde_mean) ./ sqrt(eki.N_ens - 1)

    L = pinv(Y_hat' * (Σ_y \ Y_hat)) * (Y_hat' * (Σ_y \ Y_tilde))
    Z_tilde_⊥ = Z_tilde - Z_hat * L
    Y_tilde_⊥ = Y_tilde - Y_hat * L

    m_new = m_sub .+ reshape((Z_tilde_⊥ * Y_tilde_⊥') *
                             ((Y_tilde_⊥ * Y_tilde_⊥' + Σ_y) \ (eki.y .- vec(x_tilde_mean))), :, 1)
    return reshape(m_new, :, 1)
end

# Ensemble forward evaluation (parallel)
function ensemble_forward(forward::Function, θ::Array{FT,2}, N_y::Int) where FT<:AbstractFloat
    y_pred = zeros(FT, N_y, size(θ,2))
    Threads.@threads for j in 1:size(θ,2)
        y_pred[:,j] = forward(θ[:,j])
    end
    return y_pred
end

# Unified ensemble update function
function update_ensemble!(eki::EKIObj{FT}, forward::Function) where FT<:AbstractFloat

    # --- Prediction step: inflated mean-field dynamics ---
    filter_type = eki.filter_type
    θ_prev = eki.θ[end]
    mn = mean(θ_prev, dims=2)
    θb = mn .+ sqrt(1 / (1 - eki.Δτ)) .* (θ_prev .- mn)
    Σ_y_sqrt_n = sqrt(1/eki.Δτ) * eki.Σ_y_sqrt

    # --- Predicted observations ---
    xb = forward(θb)
    xbar = mean(xb, dims=2)
    y_obs = reshape(eki.y, :, 1)

    # --- Compute deviations ---
    Zb = (θb .- mn) ./ sqrt(eki.N_ens - 1)
    Yb = (xb .- xbar) ./ sqrt(eki.N_ens - 1)

    Σ_y_n = Σ_y_sqrt_n * Σ_y_sqrt_n'
        
    Cθx = Zb * Yb'
    Cxx  = Yb * Yb' + Σ_y_n

    # Kalman gain
    K = Cθx / Cxx

    # --- Analysis step based on filter_type ---
    if filter_type == "EKI" || filter_type == "NF-EKI"
    
        θ_new = similar(θb)
        if filter_type == "EKI" 
            ν = Σ_y_sqrt_n * rand(MvNormal(zeros(eki.N_y), I(eki.N_y)))
            θ_new = θb + K * (y_obs .- xb .- reshape(ν, :, 1))
        else
            θ_new = θb + K * (y_obs .- xb)
        end
        
    elseif filter_type == "dropout-NF-EKI"
        # Dropout optimization based on the noise-free EKI covariance update
        θ_hat = θb + K * (y_obs .- xb)
        m_hat = mean(θ_hat, dims=2)
        Z_hat = (θ_hat .- m_hat) ./ sqrt(eki.N_ens - 1)
        m_new = dropout_optimization_mean(eki, forward, m_hat, Z_hat, Σ_y_n)
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)

    elseif filter_type == "dropout-EAKI"
        # Dropout optimization based on the EAKI covariance update
        P, Db_sqrt, V = svd(Zb; full=false)
        m_hat = mn .+ reshape(K * (eki.y .- vec(xbar)), :, 1)
        r = active_svd_rank(Db_sqrt)
        if r == 0
            θ_new = m_hat .+ zeros(FT, eki.N_θ, size(θb, 2))
        else
            P_r = P[:,1:r]
            V_r = V[:,1:r]
            temp = Σ_y_sqrt_n \ Yb
            S = Symmetric(V_r' * inv(I + temp' * temp) * V_r)
            eig = eigen(S)
            U, D = eig.vectors, eig.values
            A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
            A2 = Diagonal(sqrt.(D)) * inv(Diagonal(Db_sqrt[1:r])) * P_r'
            Z_hat = A1 * (A2 * Zb)
            m_new = dropout_optimization_mean(eki, forward, m_hat, Z_hat, Σ_y_n)
            θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)
        end

    elseif filter_type == "dropout-ETKI"
        # Dropout optimization based on the ETKI covariance update
        temp = Σ_y_sqrt_n \ Yb
        eig = eigen(Symmetric(temp' * temp))
        P, D = eig.vectors, eig.values
        T = P * inv(sqrt.(I + Diagonal(D))) * P'
        Z_hat = Zb * T
        m_hat = mn .+ reshape(K * (eki.y .- vec(xbar)), :, 1)
        m_new = dropout_optimization_mean(eki, forward, m_hat, Z_hat, Σ_y_n)
        θ_new = m_new .+ Z_hat * sqrt(eki.N_ens - 1)

    
    elseif filter_type == "EAKI"
        # Ensemble Adjustment Kalman Inversion
        # Compact SVD of Zb
        P, Db_sqrt, V = svd(Zb; full=false)
        mn1 = mn .+ reshape(K * (eki.y .- vec(xbar)), :, 1)
        r = active_svd_rank(Db_sqrt)
        if r == 0
            θ_new = mn1 .+ zeros(FT, eki.N_θ, size(θb, 2))
        else
            P_r = P[:,1:r]
            V_r = V[:,1:r]
            temp = Σ_y_sqrt_n \ Yb
            S = Symmetric(V_r' * inv(I + temp' * temp) * V_r)   # r × r
            eig = eigen(S)
            U, D = eig.vectors, eig.values
            A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
            A2 = Diagonal(sqrt.(D)) * inv(Diagonal(Db_sqrt[1:r])) * P_r'
            θ_new = A1 * (A2 * (θb .- mn)) .+ mn1
        end

    elseif filter_type == "ETKI"
        # Ensemble Transform Kalman Inversion
        temp = Σ_y_sqrt_n \ Yb
        eig = eigen(Symmetric(temp' * temp))
        P, D = eig.vectors, eig.values
        T = P * inv(sqrt.(I + Diagonal(D))) * P'
        mn1 = mn .+ reshape(K * (eki.y .- vec(xbar)), :, 1)
        Z_new = Zb * T
        θ_new = mn1 .+ Z_new * sqrt(eki.N_ens - 1)
    else
        error("Unknown filter_type: $(filter_type)")
    end

    size(θ_new) == size(θ_prev) || error("θ_new has size $(size(θ_new)), expected $(size(θ_prev))")
    push!(eki.θ, θ_new)
    push!(eki.y_pred, xb)
    return eki
end

# Main EKI run
function EKI_Run(forward::Function, θ0::Array{FT,2}, Σ_y::Array{FT,2}, y::Array{FT,1};
                 filter_type::String="EKI", Δτ::FT=0.5, N_iter::Int=50, forward_parallel::Bool=false,
                 dropout_rate::FT=FT(0.5)) where FT<:AbstractFloat
    N_y = size(y, 1)    

    # Obtain forward function for parallel evaluation
    func(x) = forward_parallel ? forward(x) : ensemble_forward(forward, x, N_y)

    y_pred_0 = func(θ0)
    # EKI obj initialization
    obj = EKIObj(filter_type, θ0, y_pred_0, y, Σ_y, Δτ, dropout_rate)
    @info "Running ", filter_type, " with ensemble size ", size(θ0,2)
    for n in 1:N_iter
        if n % max(1, div(N_iter,10)) == 0
            @info ("Iteration ", n, "/", N_iter)
        end
        update_ensemble!(obj, func)
    end
    return obj
end

# Optional: compute errors vs true observations
function opt_errors(eki::EKIObj)
    N_iter = length(eki.θ) - 1
    errors = zeros(Float64, N_iter+1)
    for i in 0:N_iter
        res = eki.y_pred[i+1] .- eki.y
        errors[i+1] = 0.5 * norm(eki.Σ_y_sqrt \ res)^2 / eki.N_ens
    end
    return errors
end