using LinearAlgebra
using ForwardDiff


function G(θ, arg, Gtype = "Gaussian")
    K = ones(length(θ)-2,2)
    if Gtype == "Gaussian"
        A = arg
        return [A*θ[1:2]; θ[3:end]-K*θ[1:2]]
    elseif Gtype == "Logconcave"
        λ = arg
        return [(sqrt(λ)*θ[1] - θ[2]); θ[2]^2; θ[3:end]-K*θ[1:2]] 
    elseif Gtype == "Four_modes"
        return [(θ[1]- θ[2])^2 ; (θ[1] + θ[2])^2; θ[1:2]; θ[3:end]-K*θ[1:2]]
        
    elseif Gtype == "Circle"
        A = arg
        return [θ[1:2]'*A*θ[1:2]; θ[3:end]-K*θ[1:2]]
    elseif Gtype == "Banana"
        λ = arg
        return [λ*(θ[2] - θ[1]^2); θ[1]; θ[3:end]-K*θ[1:2]]
    elseif Gtype == "Double_banana"
        λ = arg
        return [log( λ*(θ[2] - θ[1]^2)^2 + (1 - θ[1])^2 ); θ[1]; θ[2]; θ[3:end]-K*θ[1:2]]
    else
        print("Error in function G")
    end
end


function F(θ, args)
    y, ση, arg, Gtype = args
    Gθ = G(θ, arg, Gtype )
    return (y - Gθ) ./ ση
end


function Phi(θ, args)
    y, ση, arg, Gtype = args
    Gθ = G(θ, arg, Gtype )
    F = (y - Gθ) ./ ση

    return F'*F/2.0
end


function logrho(θ, args)
    Fθ = F(θ, args)
    return -0.5*norm(Fθ)^2
end


function dPhi(θ, args)
     return -logrho(θ, args), 
           -ForwardDiff.gradient(x -> logrho(x, args), θ), 
           -ForwardDiff.hessian(x -> logrho(x, args), θ)
end


function Gaussian_mixture_VI(func_dPhi, func_F, w0, μ0, Σ0; N_iter = 100, dt = 1.0e-3, Hessian_correct_GM=true)

    N_modes, N_θ = size(μ0)
    

    
    T =  N_iter * dt
    N_modes = 1
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    
    objs = []

    if func_dPhi !== nothing
        gmgdobj = GMVI_Run(
        func_dPhi, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        quadrature_type = "mean_point",
        Hessian_correct_GM = Hessian_correct_GM)
        
        push!(objs, gmgdobj)

    end

    if func_F !== nothing
        N_f = length(func_F(ones(N_θ)))
        gmgdobj_BIP = DF_GMVI_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        Hessian_correct_GM = Hessian_correct_GM, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1.0e-3,
        w_min=1e-8)
        
        push!(objs, gmgdobj_BIP)

    end

    return objs
end

# ===================================================================
#               新增代码：为HMC提供独立接口
# ===================================================================

function ∇U_for_HMC(θ, args)
    return ForwardDiff.gradient(x -> Phi(x, args), θ)
end
