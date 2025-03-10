using Random
using Statistics
using Distributions
using LinearAlgebra
using DocStringExtensions


# compute Gaussian density without 1/(2π^N_x/2) at x
# input : Gaussian mean, the inverse of sqrt(cov), and x
function Gaussian_density_helper(x_mean::Array{FT,1}, inv_sqrt_xx_cov, x::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((x - x_mean)'* (inv_sqrt_xx_cov'*inv_sqrt_xx_cov*(x - x_mean)) )) * abs(det(inv_sqrt_xx_cov))
end


# compute derivatives of Gaussian mixture density (e.g, ρ, ∇ρ, ∇²ρ) without 1/(2π^N_x/2) at x
# input : Gaussian mixture weights, means, the inverse of sqrt(covs), and x
# input : Hessian_correct_GM , whether correct ∇²ρ computation
function Gaussian_mixture_density_derivatives(x_w::Array{FT,1}, x_mean::Array{FT,2}, inv_sqrt_xx_cov, x::Array{FT,1}, Hessian_correct_GM::Bool) where {FT<:AbstractFloat}
    N_modes, N_x = size(x_mean)

    ρ = 0.0
    ∇ρ = zeros(N_x)
    ∇²ρ = zeros(N_x, N_x)
   
    for i = 1:N_modes
        temp = inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]*(x_mean[i,:] - x)
        ρᵢ   = Gaussian_density_helper(x_mean[i,:], inv_sqrt_xx_cov[i], x)
    
        ρ   += x_w[i]*ρᵢ
        ∇ρ  += x_w[i]*ρᵢ*temp
        ∇²ρ += (Hessian_correct_GM ? x_w[i]*ρᵢ*( temp * temp') : x_w[i]*ρᵢ*( temp * temp' - inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]))
    end

    return ρ, ∇ρ, ∇²ρ
end


# compute derivatives of log Gaussian mixture density (e.g, log(ρ), ∇log(ρ), ∇²log(ρ)) at N_modes by N_ens points xp
# input : Gaussian mixture weights, means, the inverse of sqrt(covs)
# input : Hessian_correct_GM ,  whether correct ∇²ρ computation
function compute_logρ_gm(x_p, x_w, x_mean, inv_sqrt_xx_cov, Hessian_correct_GM)
    N_modes, N_ens, N_x = size(x_p)
    logρ = zeros(N_modes, N_ens)
    ∇logρ = zeros(N_modes, N_ens, N_x)
    ∇²logρ = zeros(N_modes, N_ens, N_x, N_x)
    for im = 1:N_modes
        for i = 1:N_ens
            ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x_p[im, i, :], Hessian_correct_GM)
            
            logρ[im, i]         =   log(  ρ  ) - N_x/2.0 * log(2π)
            ∇logρ[im, i, :]     =   ∇ρ/ρ
            ∇²logρ[im, i, :, :] =  (Hessian_correct_GM ? ∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ' - inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im] : (∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ'))
        end
    end

    return logρ, ∇logρ, ∇²logρ
end



# compute expectaion of derivatives of Gaussian mixture density (e.g, ρ, ∇ρ, ∇²ρ) with respect to each mode
# input : Gaussian mixture weights, means, the inverse of sqrt(covs)
# input : c_weights_GM, mean_weights_GM, N_ens_GM
# input : Hessian_correct_GM , whether correct ∇²ρ computation
function compute_logρ_gm_expectation(x_w, x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM, N_ens_GM, Hessian_correct_GM)
    x_w = x_w / sum(x_w)
    N_modes, N_x = size(x_mean)
    if c_weights_GM !== nothing
        N_ens_GM =  size(c_weights_GM, 2)
    end
    xs = zeros(N_modes, N_ens_GM, N_x)
    logρ_mean, ∇logρ_mean, ∇²logρ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes
        xs[im,:,:] = construct_ensemble(x_mean[im, :], sqrt_xx_cov[im]; c_weights = c_weights_GM, N_ens = N_ens_GM)
    end
    
    logρ, ∇logρ, ∇²logρ = compute_logρ_gm(xs, x_w, x_mean, inv_sqrt_xx_cov, Hessian_correct_GM)
   
    for im = 1:N_modes
        logρ_mean[im], ∇logρ_mean[im,:], ∇²logρ_mean[im,:,:] = compute_expectation(logρ[im,:], ∇logρ[im,:,:], ∇²logρ[im,:,:,:], mean_weights_GM)
    end

    return  logρ_mean, ∇logρ_mean, ∇²logρ_mean
end
   




###### Plot function 

function Gaussian_density_1d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, xx) where {FT<:AbstractFloat}
    dx = [xx[:]' ;] - repeat(x_mean, 1, length(xx))
    return exp.( -1/2*(dx .* (inv_sqrt_xx_cov'*(inv_sqrt_xx_cov*dx)))) .* abs(det(inv_sqrt_xx_cov))
end

function Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
    
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    
    # 1d Gaussian plot
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    
    for im = 1:N_modes
        y .+= x_w[im]*Gaussian_density_1d(x_mean[im,:], inv_sqrt_xx_cov[im], xx)'
    end

    y = y/(sum(y)*dx)
    
    return y 
end


 
# compute posterior for any 1D test
# when func_type is "func_F", func returns F(x)
# when func_type is "func_Phi", func returns Phi_r(x)
function posterior_1d(func, xx, func_type)
    @assert(func_type == "func_F" || func_type == "func_Phi")
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    for i = 1:N_x
        fx = func([xx[i];])
        y[i] = (func_type == "func_F" ? exp(-fx'*fx/2) : exp(-fx))
    end
    y /= (sum(y)*dx) 

    return y
end
    


function visualization_1d(ax; Nx=2000, x_lim=[-4.0,4.0], func_F = nothing, func_Phi = nothing, objs=nothing, label=nothing)

    # visualization 
    x_min, x_max = x_lim
    
    xx = LinRange(x_min, x_max, Nx)
    dx = xx[2] - xx[1] 
    
    yy_ref = (func_Phi === nothing ? posterior_1d(func_F, xx, "func_F") : posterior_1d(func_Phi, xx, "func_Phi"))
    color_lim = (minimum(yy_ref), maximum(yy_ref))
    
    for i = 1:length(objs)
        ax[i].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
    end       
   
    N_obj = length(objs)
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1]
            xx_cov = obj.xx_cov[iter+1]
            yy = Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
            error[iobj, iter+1] = norm(yy - yy_ref,1)*dx
            
            if iter == N_iter
                ax[iobj].plot(xx, yy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
                N_modes = size(x_mean, 1)
                

                ax[iobj].scatter(obj.x_mean[1], exp.(obj.logx_w[1]), marker="x", color="grey") 
                ax[iobj].scatter(x_mean, x_w, marker="o", color="red", facecolors="none")

            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+1].semilogy(Array(0:N_iter), error[i_obj, :], 
            label=(label===nothing ? label : label*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")" ))
    end
    if label!==nothing 
       ax[N_obj+1].legend()
    end
end



function Gaussian_density_2d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, X, Y) where {FT<:AbstractFloat}
    dx = [X[:]' ; Y[:]'] - repeat(x_mean, 1, length(X))
    return reshape( exp.( -1/2*sum(dx .* ((inv_sqrt_xx_cov'*inv_sqrt_xx_cov)*dx), dims=1)) .* abs(det(inv_sqrt_xx_cov)), size(X))
end

function Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    # 2d Gaussian plot
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for im = 1:N_modes
        Z .+= x_w[im]*Gaussian_density_2d(x_mean[im,:], inv_sqrt_xx_cov[im], X, Y)
    end
    Z = Z/(sum(Z)*dx*dy)
    return Z
    
end


function posterior_2d(func, X, Y, func_type)
    @assert(func_type == "func_F" || func_type == "func_Phi")
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            fx = func([X[i,j] ; Y[i,j]])
            Z[i,j] = (func_type == "func_F" ? exp(-fx'*fx/2) : exp(-fx))
        end
    end
    Z /= (sum(Z)*dx*dy) 

    return Z
end



function visualization_2d(ax; Nx=2000, Ny=2000, x_lim=[-4.0,4.0], y_lim=[-4.0,4.0], func_F = nothing, func_Phi = nothing, objs=nothing, label=nothing)
    
    # visualization 
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'   #'

    Z_ref = (func_Phi === nothing ? posterior_2d(func_F, X, Y, "func_F") : posterior_2d(func_Phi, X, Y, "func_Phi"))
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)

    N_obj = length(objs)
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1][:,1:2]
            xx_cov = obj.xx_cov[iter+1][:,1:2,1:2]
            Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
            error[iobj, iter+1] = norm(Z - Z_ref,1)*dx*dy
            
            if iter == N_iter
                    
                ax[1+iobj].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                N_modes = size(x_mean, 1)
                ax[1+iobj].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey", alpha=0.5) 
                ax[1+iobj].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none", alpha=0.5)
               
            end
        end
        
    end

    if isa(label, String)
        label = fill(label, N_obj)
    end
    for i_obj = 1:N_obj
        ax[N_obj+2].semilogy(Array(0:N_iter), error[i_obj, :], 
                        label=(label===nothing ? label : label[i_obj]*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")" ))   
    end
    # Get the current y-axis limits
    ymin, ymax = ax[N_obj+2].get_ylim()
    # Ensure the lower bound of y-ticks is below 0.1
    if ymin > 0.1
        ax[N_obj+2].set_ylim(0.1, ymax)  # Set the lower limit to a value below 0.1
    end
    if label!==nothing 
       ax[N_obj+2].legend()
    end
   
end




