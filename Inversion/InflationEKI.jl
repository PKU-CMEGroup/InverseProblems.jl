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
end

# Constructor
function EKIObj(filter_type::String, θ0::Array{FT,2}, y_pred_0::Array{FT,2}, y0::Array{FT,1}, 
                        Σ_y::Array{FT,2}, Δτ::FT) where FT<:AbstractFloat
    θ = [θ0]
    y_pred = [y_pred_0]

    N_θ, N_ens = size(θ0)
    N_y = size(y0, 1)
    Σ_y_sqrt = LowerTriangular(cholesky(Σ_y).L)
    obj = EKIObj(filter_type, θ, y_pred, y0, Σ_y_sqrt, N_ens, N_θ, N_y, Δτ)
    return obj
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
function update_ensemble!(eki::EKIObj, forward::Function)

    # --- Prediction step: inflated mean-field dynamics ---
    θ_prev = eki.θ[end]
    mn = mean(θ_prev, dims=2)
    θb = mn .+ sqrt(1 / (1 - eki.Δτ)) .* (θ_prev .- mn)
    Σ_y_sqrt_n = sqrt(1/eki.Δτ) * eki.Σ_y_sqrt

    # --- Predicted observations ---
    xb = forward(θb)
    xbar = mean(xb, dims=2)

    # --- Compute deviations ---
    Zb = (θb .- mn) ./ sqrt(eki.N_ens - 1)
    Yb = (xb .- xbar) ./ sqrt(eki.N_ens - 1)

    Σ_y_n = Σ_y_sqrt_n * Σ_y_sqrt_n'
        
    Cθx = Zb * Yb'
    Cxx  = Yb * Yb' + Σ_y_n

    # Kalman gain
    K = Cθx / Cxx

    # --- Analysis step based on filter_type ---
    if eki.filter_type == "EKI" || eki.filter_type == "NF-EKI"
    
        θ_new = similar(θb)
        if eki.filter_type == "EKI" 
            ν = Σ_y_sqrt_n * rand(MvNormal(zeros(eki.N_y), I(eki.N_y)))
            θ_new = θb + K * (-xb .+ eki.y .- ν)
        else
            θ_new = θb + K * (-xb .+ eki.y)
        end
        
    
    elseif eki.filter_type == "EAKI"
        # Ensemble Adjustment Kalman Inversion
        # Compact SVD of Zb
        P, Db_sqrt, V = svd(Zb; full=false)
        r = findlast(Db_sqrt .> 1e-6)
        P_r = P[:,1:r]
        V_r = V[:,1:r]
        temp = Σ_y_sqrt_n \ Yb
        S = Symmetric(V_r' * inv(I + temp' * temp) * V_r)   # r × r
        eig = eigen(S)
        U, D = eig.vectors, eig.values
        A1 = P_r * Diagonal(Db_sqrt[1:r]) * U
        A2 = Diagonal(sqrt.(D)) * inv(Diagonal(Db_sqrt[1:r])) * P_r'
        mn1 = mn + K * (eki.y .- xbar)
        θ_new = A1 * (A2 * (θb .- mn)) .+ mn1

    elseif eki.filter_type == "ETKI"
        # Ensemble Transform Kalman Inversion
        temp = Σ_y_sqrt_n \ Yb
        eig = eigen(Symmetric(temp' * temp))
        P, D = eig.vectors, eig.values
        T = P * inv(sqrt.(I + Diagonal(D))) * P'
        mn1 = mn + K * (eki.y .- xbar)
        Z_new = Zb * T
        θ_new = mn1 .+ Z_new * sqrt(eki.N_ens - 1)
    else
        error("Unknown filter_type: $(eki.filter_type)")
    end

    push!(eki.θ, θ_new)
    push!(eki.y_pred, xb)
    return eki
end

# Main EKI run
function EKI_Run(forward::Function, θ0::Array{FT,2}, Σ_y::Array{FT,2}, y::Array{FT,1};
                 filter_type::String="EKI", Δτ::FT=0.5, N_iter::Int=50, forward_parallel::Bool=false) where FT<:AbstractFloat
    N_y = size(y, 1)    

    # Obtain forward function for parallel evaluation
    func(x) = forward_parallel ? forward(θ0) : ensemble_forward(forward, x, N_y)

    y_pred_0 = func(θ0)
    # EKI obj initialization
    obj = EKIObj(filter_type, θ0, y_pred_0, y, Σ_y, Δτ)
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