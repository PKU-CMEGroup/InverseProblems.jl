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
    elseif Gtype == "Funnel"
        A = arg
        return [θ[1]/3 + 3*(length(θ)-1)/2; A * θ[2:end] / exp(θ[1]/2)]
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

function log_Gaussian_mixture(x, args)
    x_w, x_mean, inv_sqrt_x_cov = args
    # C = L L.T
    # C^-1 = L^-TL^-1
    N_x = size(x_mean, 2)
    ρ = 0
    exponents = [-0.5*(x-x_mean[im,:])'*(inv_sqrt_x_cov[im]'*inv_sqrt_x_cov[im]*(x-x_mean[im,:])) for im =1:length(x_w)]
    mexponent = maximum(exponents)
    for im = 1:length(x_w)
        ρ += x_w[im]*exp(exponents[im] - mexponent)*det(inv_sqrt_x_cov[im])
    end
    return  log(ρ) + mexponent - N_x/2*log(2*π)
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


