using LinearAlgebra
using ForwardDiff
using Random
using Distributions
using Sobol
# quadrature points
# C = √C √Cᵀ
# xᵢ=m+√C cᵢ , here cᵢ ∈ Rᴺ
# ∫f(x)N(θ;m,C) = ∑ wᵢf(xᵢ)
#              
# c_weights = [c₁;c₂;...;c_{N_ens}]
# mean_weights = [w₁;w₂;...;w_{N_ens}]
# x₁ᵀ                mᵀ           c₁ᵀ 
# x₂ᵀ           =    mᵀ    +      c₂ᵀ             √Cᵀ
# ⋮                  ⋮             ⋮
# x_{N_ens}ᵀ         mᵀ           c_{N_ens}ᵀ  
mutable struct HybridQMCHandler
    s::SobolSeq         
    iter::Int           
    switch_iter::Int    
    N_ens::Int          
    N_x::Int    
    z_buffer::Matrix{Float64}
end

function generate_quadrature_rule(N_x, quadrature_type; c_weight=sqrt(N_x), N_ens = -1, switch_iter = -1)
    if quadrature_type == "mean_point"
        N_ens = 1
        c_weights    = zeros(N_x, N_ens)
        mean_weights = ones(N_ens)
    elseif quadrature_type == "random_sampling"
        c_weights = nothing
        mean_weights =  ones(N_ens)/N_ens
    elseif quadrature_type == "hybrid_qmc"
        if N_ens == -1 || switch_iter == -1
            error("For hybrid_qmc, N_ens and switch_iter must be provided!")
        end
        
        s = SobolSeq(N_x)
        skip(s, N_ens * N_x)
        
        z_buf = zeros(Float64, N_x, N_ens)
        
        c_weights = HybridQMCHandler(s, 0, switch_iter, N_ens, N_x, z_buf)
        mean_weights = ones(N_ens)/N_ens
        
    elseif  quadrature_type == "unscented_transform"
        N_ens = 2N_x+1
        c_weights = zeros(N_x, N_ens)
        for i = 1:N_x
            c_weights[i, i+1]      =  c_weight
            c_weights[i, N_x+i+1]  = -c_weight
        end
        mean_weights = fill(1/(2.0*c_weight^2), N_ens)
        # warning: when c_weight <= sqrt(N_x), the weight is negative
        # the optimal is sqrt(3), 
        # Julier, S. J., Uhlmann, J. K., & Durrant-Whyte, H. F. (2000). A new method for nonlinear transformation of means and covariances in filters and estimators. 
        mean_weights[1] = 1 - N_x/c_weight^2
    
    elseif quadrature_type == "cubature_transform_o3"
        N_ens = 2N_x
        c_weight = sqrt(N_x)
        c_weights = zeros(N_x, N_ens)
        for i = 1:N_x
            c_weights[i, i]          =  c_weight
            c_weights[i, N_x+i]      =  -c_weight
        end
        mean_weights = ones(N_ens)/N_ens 

    elseif quadrature_type == "cubature_transform_o5"
        # High-degree cubature Kalman filter
        # Bin Jia, Ming Xin, Yang Cheng
        N_ens = 2N_x*N_x + 1
        c_weights    = zeros(N_x, N_ens)
        mean_weights = ones(N_ens)

        mean_weights[1] = 2.0/(N_x + 2)

        for i = 1:N_x
            c_weights[i, 1+i]          =  sqrt(N_x+2)
            c_weights[i, 1+N_x+i]      =  -sqrt(N_x+2)
            mean_weights[1+i] = mean_weights[1+N_x+i] = (4 - N_x)/(2*(N_x+2)^2)
        end
        ind = div(N_x*(N_x - 1),2)
        for i = 1: N_x
            for j = i+1:N_x
                c_weights[i, 2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)],      c_weights[j, 2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)]        =   sqrt(N_x/2+1),  sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+ind+div((2N_x-i)*(i-1),2)+(j-i)],  c_weights[j, 2N_x+ind+1+div((2N_x-i)*(i-1),2)+(j-i)]    =  -sqrt(N_x/2+1), -sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+2ind+div((2N_x-i)*(i-1),2)+(j-i)], c_weights[j, 2N_x+2ind+1+div((2N_x-i)*(i-1),2)+(j-i)]   =   sqrt(N_x/2+1), -sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+3ind+div((2N_x-i)*(i-1),2)+(j-i)], c_weights[j, 2N_x+3ind+1+div((2N_x-i)*(i-1),2)+(j-i)]   =  -sqrt(N_x/2+1),  sqrt(N_x/2+1)
                
                mean_weights[2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)]      = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+ind+div((2N_x-i)*(i-1),2)+(j-i)]  = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+2ind+div((2N_x-i)*(i-1),2)+(j-i)] = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+3ind+div((2N_x-i)*(i-1),2)+(j-i)] = 1.0/(N_x+2)^2
            end
        end

    else 
        print("cubature tansform with quadrature type ", quadrature_type, " has not implemented.")

    end

    return N_ens, c_weights, mean_weights
end

function compute_sqrt_matrix(C; type="Cholesky")
    if type == "Cholesky"
        L = cholesky(Hermitian(C)).L
        sqrt_cov, inv_sqrt_cov = L,  inv(L) 
    elseif type == "SVD"
        U, D, _ = svd(Hermitian(C))
        sqrt_cov, inv_sqrt_cov = U*Diagonal(sqrt.(D)),  Diagonal(sqrt.(1.0./D))*U' 
    else
        print("Type ", type, " for computing sqrt matrix has not implemented.")
    end
    return sqrt_cov, inv_sqrt_cov
end

function construct_ensemble(x_mean, sqrt_cov; c_weights = nothing, N_ens = 100)

    if isa(c_weights, HybridQMCHandler)
        handler = c_weights
        handler.iter += 1
        
        z = handler.z_buffer 
        
        if handler.iter <= handler.switch_iter
            for i in 1:handler.N_ens
                next!(handler.s, view(z, :, i)) 
            end
            @. z = quantile(Normal(), z)
            xs = (sqrt_cov * z)' .+ x_mean'
            return xs
            
        else
            randn!(z) 
            z_mean = mean(z, dims=2)
            z .-= z_mean
            L_emp = cholesky(z * z'/handler.N_ens + I*1e-12).L
            z_corrected = L_emp \ z 
            xs = ones(handler.N_ens)*x_mean' + (sqrt_cov * z_corrected)'
            return xs
        end
    end

            
    if c_weights === nothing
        N_x = size(x_mean,1)
        # generate random weights on the fly
        c_weights = rand(Normal(0, 1), N_x, N_ens)
        
        # enforce symmetry
        # if N_ens %2 == 0
        #     c_weights[:, div(N_ens, 2)+1:end] = -c_weights[:, 1:div(N_ens, 2)]
        # else
        #     c_weights[:, 1] .= 0.0
        #     c_weights[:, div(N_ens, 2)+2:end] = -c_weights[:, 2:div(N_ens, 2)+1]
        # end
        # # enforce empirical covariance, no need
        # L = cholesky(c_weights * c_weights'/N_ens).L
        # c_weights = L\c_weights
        
        c_mean = c_weights * ones(N_ens) / N_ens
        c_weights .-= c_mean
        L = cholesky(c_weights * c_weights'/N_ens).L
        c_weights = L\c_weights
        
        xs = ones(N_ens)*x_mean' + (sqrt_cov * c_weights)'
    else
        N_ens = size(c_weights,2)
        xs = ones(N_ens)*x_mean' + (sqrt_cov * c_weights)'
    end

    return xs
end


# Derivative Free
function compute_expectation_BIP(x_mean, inv_sqrt_cov, V, c_weight)

    N_x = length(x_mean)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = 0.0, zeros(N_x), zeros(N_x, N_x)

    
    α = c_weight
    N_ens, N_f = size(V)
    a = zeros(N_x, N_f)
    b = zeros(N_x, N_f)
    c = zeros(N_f)
    
    c = V[1, :]
    for i = 1:N_x
        a[i, :] = (V[i+1, :] + V[i+N_x+1, :] - 2*V[1, :])/(2*α^2)
        b[i, :] = (V[i+1, :] - V[i+N_x+1, :])/(2*α)
    end
    ATA = a * a'
    BTB = b * b'
    BTA = b * a'
    BTc, ATc = b * c, a * c
    cTc = c' * c

    # Ignore second order effect, mean point approximation
    # Φᵣ_mean = 1/2*(sum(ATA) + 2*tr(ATA) + 2*sum(ATc) + tr(BTB) + cTc)
    Φᵣ_mean = 1/2*(cTc)
    
    # Ignore second order effect, mean point approximation
    # ∇Φᵣ_mean = inv_sqrt_cov'*(sum(BTA,dims=2) + 2*diag(BTA) + BTc)
    ∇Φᵣ_mean = inv_sqrt_cov'*(BTc)

    # Keep SPD part, with mean point approximation
    # ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(2*dropdims(sum(ATA, dims=2), dims=2) + 4*diag(ATA) + 2*ATc) + BTB)*inv_sqrt_cov
    ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(6*diag(ATA)) + BTB)*inv_sqrt_cov
          
    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end



function compute_expectation(V, ∇V, ∇²V, mean_weights)

    N_ens, N_x = size(∇V)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = 0.0, zeros(N_x), zeros(N_x, N_x)

    # general sampling problem Φᵣ = V 
    Φᵣ_mean   = mean_weights' * V
    ∇Φᵣ_mean  = ∇V' * mean_weights
    ∇²Φᵣ_mean = sum(mean_weights[i] * ∇²V[i, :, :] for i = 1:length(mean_weights))

    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end


