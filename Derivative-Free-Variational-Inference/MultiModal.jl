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

function multimodal_moments(Gtype::String)
    if Gtype == "Circle"
        return ([0, 0], [0.5002314411516613 0; 0 0.5002314411516613])
    elseif Gtype == "Banana"
        return ([0.9999999820951503, 10.999999610616223],
        [10.999999610635019 30.999992090185334; 30.999992090185334 361.09983180877657])
    elseif Gtype == "Funnel"
        return ([0, 0], 
        [9.0 0; 0 54.67959866084672])
    elseif Gtype == "Gaussian_mixture"
        return ([-1.205, -0.865],
        [17.279671070293197 0.6809403064876811; 0.6809403064876809 15.33721856523279])
    else
        @error "Error in function G"
    end
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

function Gaussian_mixture_args(;N_x::Int=2)
    x_w_ref = [0.12, 0.05, 0.1, 0.06, 0.12, 0.08, 0.08, 0.12, 0.12, 0.15]
    x_mean_0 = [-5.0 5.0; 4.0 6.0; 4.0 3.0; -0.25 0.75; 0.75 -0.75; -3.5 0; -2 -2; -6 -3; -6 -5; 4 -6]
    inv_sqrt_xx_cov_0 = [
        [1.155 0.0; 1.54 1.9245], [0.8944 0.0; 0.0 2.582],
        [1.414 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0],
        [1.581 0.0; 0.0 1.581], [1.0 0.0; 0.0 2.0],
        [2.0 0.0; 0.0 1.0], [1.414 0.0; -0.970 1.213],
        [1.0 0.0; 0.825 1.8334], [0.6325 0.0; 0.0 2.8284] ]
    x_mean_ref = zeros(10,N_x)
    inv_sqrt_xx_cov_ref = []
    for im = 1:10
        x_mean_ref[im,1:2] = x_mean_0[im,:]
        x_mean_ref[im,3:N_x] .= im
        inv_sqrt_xx_cov = ones(N_x,N_x)
        inv_sqrt_xx_cov[1:2,1:2] =  inv_sqrt_xx_cov_0[im]
        push!(inv_sqrt_xx_cov_ref, tril(inv_sqrt_xx_cov)) 
    end
    return x_w_ref, x_mean_ref, inv_sqrt_xx_cov_ref
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


