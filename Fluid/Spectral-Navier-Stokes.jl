using LinearAlgebra
using Random
using Distributions

include("Spectral-Mesh.jl")


"""
Navier stokes equation with double periodicity
U = (u,v) and p denote velocity and pressure 
∂U/∂t + U⋅∇U - νΔU + ∇p = f
∇⋅U = 0

Let ω = ∇×U, we have the vorticity equation
∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f

ω = 1/(N_xN_y)∑_{kx, ky}  ω_hat[kx,ky]  e^{i (2π/Lx kx x + 2π/Ly ky y)}

∂ω_hat/∂t + FFT[(U⋅∇)ω] + ν( (2πkx/Lx)² + (2πky/Ly)² )ω_hat  = 2π/Lx kx fy_hat - 2π/Ly ky fx_hat

The domain is [0,Lx][0,Ly]
"""
mutable struct Spectral_NS_Solver
    mesh::Spectral_Mesh

    ν::Float64

    fx::Union{Array{Float64, 2}, Nothing}
    fy::Union{Array{Float64, 2}, Nothing}
    curl_f_hat::Array{ComplexF64, 2}
    stochastic_forcing::Bool

    ub::Float64
    vb::Float64
    # Since the streamfunction is assumed to be periodic 
    # U = curl ψ  + Ub = [∂ψ/∂y + ub, -∂ψ/∂x + vb]

    ω_hat::Array{ComplexF64, 2}
    u_hat::Array{ComplexF64, 2}
    v_hat::Array{ComplexF64, 2}
    p_hat::Array{ComplexF64, 2}

    ω::Array{Float64, 2}
    u::Array{Float64, 2}
    v::Array{Float64, 2}
    p::Array{Float64, 2}

    Δω_hat::Array{ComplexF64, 2}
    δω_hat::Array{ComplexF64, 2}

    k1::Array{ComplexF64, 2}
    k2::Array{ComplexF64, 2}
    k3::Array{ComplexF64, 2}
    k4::Array{ComplexF64, 2}
    
    
end


# constructor of the Spectral_NS_Solver
#
# There are several forcing intialization approaches
# first check curl_f_hat
# then check curl_f, compute curl_f_hat
# then check fx and fy, compute curl_f_hat
# set curl_f_hat to 0
#
# There are velocity(vorticity) intialization approaches
# * initialize u0 and v0 components, user should make sure they are incompressible div ⋅ (u0 , v0) = 0
# * initialize ω0 and mean backgroud velocity ub and vb
#
function Spectral_NS_Solver(mesh::Spectral_Mesh, ν::Float64;
    fx::Union{Array{Float64, 2}, Nothing} = nothing, 
    fy::Union{Array{Float64, 2}, Nothing} = nothing, 
    curl_f::Union{Array{Float64, 2}, Nothing} = nothing,
    curl_f_hat::Union{Array{ComplexF64, 2}, Nothing} = nothing,
    stochastic_forcing::Bool = false,
    u0::Union{Array{Float64, 2}, Nothing} = nothing, 
    v0::Union{Array{Float64, 2}, Nothing} = nothing, 
    ω0::Union{Array{Float64, 2}, Nothing} = nothing, 
    ub::Union{Float64, Nothing} = nothing, 
    vb::Union{Float64, Nothing} = nothing)    
    N_x, N_y = mesh.N_x, mesh.N_y
    
    
    if curl_f_hat === nothing
        curl_f_hat = zeros(ComplexF64, N_x, N_y)
        
        if curl_f === nothing
            if fx === nothing
                fx = zeros(Float64, N_x, N_y)
            end
            if fy === nothing
                fy = zeros(Float64, N_x, N_y)
            end 
            Apply_Curl!(mesh, fx, fy, curl_f_hat) 
            
        else
            Trans_Grid_To_Spectral!(mesh, curl_f, curl_f_hat)
            

        end
    end
   
    
    
    u_hat = zeros(ComplexF64, N_x, N_y)
    v_hat = zeros(ComplexF64, N_x, N_y)
    ω_hat = zeros(ComplexF64, N_x, N_y)
    p_hat = zeros(ComplexF64, N_x, N_y)
    u = zeros(Float64, N_x, N_y)
    v = zeros(Float64, N_x, N_y)
    ω = zeros(Float64, N_x, N_y)
    p = zeros(Float64, N_x, N_y)
    if ω0 === nothing
    
        u .= u0
        v .= v0
        
        Trans_Grid_To_Spectral!(mesh, u, u_hat)
        Trans_Grid_To_Spectral!(mesh, v, v_hat)
        
        ub, vb = u_hat[1,1]/(N_x*N_y), v_hat[1,1]/(N_x*N_y)
        
        Vor_From_UV_Spectral!(mesh, u_hat, v_hat, ω_hat, ω)
        
    else
        ω = zeros(Float64, N_x, N_y)
        ω .= ω0
        ω_hat = zeros(ComplexF64, N_x, N_y)
        Trans_Grid_To_Spectral!(mesh, ω, ω_hat)
        
        
        UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)
        
        
        Trans_Spectral_To_Grid!(mesh, u_hat, u)
        Trans_Spectral_To_Grid!(mesh, v_hat, v)
    end
    
    
    δω_hat = zeros(ComplexF64, N_x, N_y)
    Δω_hat = zeros(ComplexF64, N_x, N_y)
    
    k1 = zeros(ComplexF64, N_x, N_y)
    k2 = zeros(ComplexF64, N_x, N_y)
    k3 = zeros(ComplexF64, N_x, N_y)
    k4 = zeros(ComplexF64, N_x, N_y)
    
    Spectral_NS_Solver(mesh, ν, fx, fy, curl_f_hat, stochastic_forcing, ub, vb, ω_hat, u_hat, v_hat, p_hat, ω, u, v, p, Δω_hat, δω_hat, k1, k2, k3, k4)
end


# compute convective and diffusive stable time step 
function Stable_Δt(mesh::Spectral_Mesh, ν::Float64, u::Array{Float64,2}, v::Array{Float64,2})
    Δx, Δy = mesh.Δx, mesh.Δy
    Δ = min(Δx, Δy)
    u_max, v_max = maximum(abs.(u)), maximum(abs.(v))


    Δt = min(Δx/u_max, Δy/v_max, Δ^2/(2*ν))

   
    return Δt
end

# ∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f
# ∂ω_hat/∂t = -FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )ω_hat  + curl_f_hat
# Compute the right hand-side
function Explicit_Residual!(self::Spectral_NS_Solver, ω_hat::Array{ComplexF64, 2}, δω_hat::Array{ComplexF64, 2})
    mesh = self.mesh
    ub, vb = self.ub, self.vb
    Compute_Horizontal_Advection!(mesh, ω_hat, δω_hat, ub, vb)

    δω_hat .= self.curl_f_hat - δω_hat

    Δω_hat = self.Δω_hat

    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat
end

# Also the Crank-Nicolson
# ∂ω/∂t + (U⋅∇)ω - νΔω  = ∇×f
# (ω_hat(n+1) - ω_hat(n))/Δt  = -FFT[(U⋅∇)ω] + ν(- ((2πkx/Lx)² + (2πky/Ly)²) )(ω_hat(n)+ω_hat(n+1))/2  + curl_f_hat
# [1 - νΔt(-(2πkx/Lx)² + -(2πky/Ly)²)/2] (ω_hat(n+1)-ω_hat(n))/Δt = -FFT[(U⋅∇)ω] + ν(-((2πkx/Lx)² + (2πky/Ly)²))ω_hat(n)/2  + curl_f_hat
# compute (ω_hat(n+1)-ω_hat(n))/Δt
function Semi_Implicit_Residual!(self::Spectral_NS_Solver, ω_hat::Array{ComplexF64, 2}, Δt::Float64, δω_hat::Array{ComplexF64, 2})

    mesh = self.mesh
    ub, vb = self.ub, self.vb
    Compute_Horizontal_Advection!(mesh, ω_hat, δω_hat, ub, vb)

    curl_f_hat = self.stochastic_forcing ? Stochastic_Forcing(mesh, self.curl_f_hat, Δt) : self.curl_f_hat
    δω_hat .= curl_f_hat - δω_hat

    Δω_hat = self.Δω_hat
    
    Apply_Laplacian!(mesh, ω_hat, Δω_hat)

    δω_hat .+= self.ν * Δω_hat

    δω_hat ./= (1.0 .- 0.5*self.ν*Δt * mesh.laplacian_eigs)

end


# Update ω, u_hat, v_hat, u, v from ω_hat
function Update_Grid_Vars!(self::Spectral_NS_Solver, compute_pressure::Bool = false)

    ω_hat, u_hat, v_hat = self.ω_hat, self.u_hat, self.v_hat
    ω, u, v = self.ω, self.u, self.v
    ub, vb = self.ub, self.vb

    mesh = self.mesh

    Trans_Spectral_To_Grid!(mesh, ω_hat, ω)
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)
    Trans_Spectral_To_Grid!(mesh, u_hat, u)
    Trans_Spectral_To_Grid!(mesh, v_hat, v)

    # Compute pressure
    p, p_hat = self.p, self.p_hat
    if compute_pressure
        fx, fy = self.fx, self.fy
        u_x, u_y, v_x, v_y, Δp = similar(ω), similar(ω), similar(ω), similar(ω), similar(ω)
        ∇f_hat, Δp_hat = similar(ω_hat), similar(ω_hat)

        Apply_Gradient!(mesh, u_hat, u_x, u_y)
        Apply_Gradient!(mesh, v_hat, v_x, v_y)
        Apply_Gradient!(mesh, fx, fy, ∇f_hat)
        
        Δp .= 2(u_x.*v_y - v_x.*u_y)
        Trans_Grid_To_Spectral!(mesh, Δp, Δp_hat)
        
        Δp_hat .+= ∇f_hat
        Solve_Laplacian!(mesh, Δp_hat, p_hat) 
        
        Trans_Spectral_To_Grid!(mesh, p_hat, p)
    end


    return ω, u, v, p
    
end

# Compute pressure from ω_hat and f_x, f_y
function Compute_Pressure(self::Spectral_NS_Solver, ω_hat, f_x, f_y)

    
    u_hat, v_hat = self.u_hat, self.v_hat
    ω, u, v = self.ω, self.u, self.v
    ub, vb = self.ub, self.vb

    mesh = self.mesh
    Trans_Spectral_To_Grid!(mesh, ω_hat, ω)
    UV_Spectral_From_Vor!(mesh, ω_hat, u_hat, v_hat, ub, vb)


end



function Solve!(self::Spectral_NS_Solver, Δt::Float64, method::String)
    ω_hat, δω_hat = self.ω_hat, self.δω_hat

    if method == "Crank-Nicolson"
        Semi_Implicit_Residual!(self, ω_hat, Δt, δω_hat)
        ω_hat .+= Δt*δω_hat
    elseif method == "RK4"
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4 
        Explicit_Residual!(self, ω_hat,  k1)
        Explicit_Residual!(self, ω_hat + Δt/2.0*k1, k2)
        Explicit_Residual!(self, ω_hat + Δt/2.0*k2, k3)
        Explicit_Residual!(self, ω_hat + Δt*k3, k4)
        
        ω_hat .+= Δt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    end
end






#########################
mutable struct Setup_Param
    # physics
    ν::Float64        # dynamic viscosity
    ub::Float64       # background velocity 
    vb::Float64
    fx::Union{Array{Float64, 2}, Nothing}
    fy::Union{Array{Float64, 2}, Nothing}
    curl_f::Union{Array{Float64, 2}, Nothing}

    # discretization
    N::Int64            # number of grid points for both x and y directions (including both ends)
    L::Float64            # computational domain [0, L]×[0, L]
    Δx::Float64
    xx::Array{Float64, 1}  # uniform grid [0, Δx, 2Δx ... L]

    method::String    # time discretization method 
    N_t::Int64        # number of time steps
    T::Float64        # total time

    # observation locations is tensor product x_locs × y_locs
    x_locs::Array{Int64, 1}
    y_locs::Array{Int64, 1}
    t_locs::Array{Int64, 1}
    symmetric::Bool
    t_average::Bool

    # for parameterization
    seq_pairs::Array{Int64, 2} # indexing eigen pairs of KL expansion, length(seq_pairs) >= N_θ 
    ω0_θ_ref::Union{Array{Float64, 1}, Nothing}   # coefficient of these truth modes (N_KL > 0)
    ω0_ref::Union{Array{Float64, 2}, Nothing}
    curl_f_θ_ref::Union{Array{Float64, 1}, Nothing}
    curl_f_ref::Union{Array{Float64, 2}, Nothing}

    
    # inverse parameters
    θ_names::Array{String, 1}
    N_θ::Int64   
    N_y::Int64   
        
end

function Setup_Param(ν::Float64, ub::Float64, vb::Float64,  
    N::Int64, L::Float64,  
    method::String, N_t::Int64, T::Float64,
    obs_ΔNx::Int64, obs_ΔNy::Int64, obs_ΔNt::Int64;
    symmetric::Bool = false,  # observation locations are symmetric (to test multiple modes)
    t_average::Bool = false,  # observation is the averaged in-time
    N_ω0_θ::Int64 = 0,  # initial condition parameter number
    N_curl_f_θ::Int64 = 0,   # forcing parameter number
    N_ω0_ref::Int64 = 0,  # generate the initial condition with N_ω0_ref modes, 0 indicating all modes  
    N_curl_f_ref::Int64 = 0,       # generate the forcing with N_f_ref modes, 0 indicating all modes
    f::Union{Function, Nothing} = nothing,  # fixed forcing, this will overwrite N_f_ref
    σ::Float64 = sqrt(2)*pi, ω0_seed::Int64=123, f_seed::Int64=42)
    
    #observation
    if symmetric
        x_locs = Array(obs_ΔNx+1:obs_ΔNx:div(N,2))
        y_locs = Array(obs_ΔNy+1:obs_ΔNy:div(N,2)) #Array(1:obs_ΔNy:N)
    else
        x_locs = Array(1:obs_ΔNx:N)
        y_locs = Array(1:obs_ΔNy:N)
    end
    t_locs = Array(obs_ΔNt:obs_ΔNt:N_t)

    N_y = (t_average ? length(x_locs)*length(y_locs) : length(x_locs)*length(y_locs)*length(t_locs))
    seq_pairs = Compute_Seq_Pairs(max(N_ω0_θ, N_curl_f_θ, N_ω0_ref, N_curl_f_ref))
    mesh = Spectral_Mesh(N, N, L, L)

    # initial condition 
    if N_ω0_ref > 0
        Random.seed!(ω0_seed);
        # The truth is generated with first N_ω0_ref modes
        ω0_θ_ref = rand(Normal(0, σ), N_ω0_ref)
        ω0_ref = Random_Field_From_Theta(mesh, ω0_θ_ref, seq_pairs) 
        ω0_θ_ref = ω0_θ_ref[1:N_ω0_θ]
    else
        # The truth is generated with all modes
        ω0_ref = Random_Field(mesh, σ; seed=ω0_seed)
        # project to obtain the first N_ω0_θ modes
        ω0_θ_ref = Theta_From_Random_Field(ω0_ref, N_ω0_θ, seq_pairs, mesh)
    end
    ω0_hat_ref = zeros(ComplexF64, N, N)
    Trans_Grid_To_Spectral!(mesh, ω0_ref, ω0_hat_ref)
    Trans_Spectral_To_Grid!(mesh, ω0_hat_ref, ω0_ref)


    
    # forcing  
    xx = Array(LinRange(0, L, N+1))
    fx = zeros(Float64, N, N)
    fy = zeros(Float64, N, N)
    if f !== nothing
        for i_x = 1:N
            for i_y = 1:N
                fx[i_x, i_y], fy[i_x, i_y] = f(xx[i_x], xx[i_y])
            end
        end
        curl_f_ref = nothing
        curl_f_θ_ref = nothing
    else
        # initial condition 
        if N_curl_f_ref > 0
            Random.seed!(f_seed);
            # The truth is generated with first N_ω0_ref modes
            curl_f_θ_ref = rand(Normal(0, σ), N_curl_f_ref)
            curl_f_ref = Random_Field_From_Theta(mesh, curl_f_θ_ref, seq_pairs) 
            curl_f_θ_ref = curl_f_θ_ref[1:N_ω0_θ]
        else
            # The truth is generated with all modes
            curl_f_ref = Random_Field(mesh, σ; seed=f_seed)
            # project to obtain the first N_ω0_θ modes
            curl_f_θ_ref = Construct_θ0(curl_f_ref, N_curl_f_θ, seq_pairs, mesh)
        end
        
        curl_f_hat_ref = zeros(ComplexF64, N, N)
        Trans_Grid_To_Spectral!(mesh, curl_f_ref, curl_f_hat_ref)
        Trans_Spectral_To_Grid!(mesh, curl_f_hat_ref, curl_f_ref)
    end


    θ_names = ["ω0-f"]
    N_θ = N_curl_f_θ + N_ω0_θ

    Setup_Param(ν, ub, vb, fx, fy, curl_f_ref, 
    N, L, L/N, xx,  
    method, N_t, T, 
    x_locs, y_locs, t_locs, symmetric, t_average,
    seq_pairs, ω0_θ_ref, ω0_ref, curl_f_θ_ref, curl_f_ref,
    θ_names, N_θ, N_y)
end


# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (kx + ky > 0 || (kx + ky == 0 && kx > 0)) 
function Z2_plus(kx, ky)
    return (ky > 0 || (ky == 0 && kx > 0)) 
end
function Z2_minus(kx, ky)
    return Z2_plus(-kx, -ky)
end

# Generate random numbers and initialize ω0 with all fourier modes
function Random_Field(mesh::Spectral_Mesh, σ::Float64; seed::Int64 = 123)
    # consider C = -Δ^{-1}
    # C u = λ u  => -u = λ Δ u
    # u = e^{ik⋅x}, λ Δ u = -λ |k|^2 u = - u => λ = 1/|k|^2
    # basis are cos(k⋅x)/√2π and sin(k⋅x)/√2π, k!=(0,0), zero mean.
    # for kx != 0 or ky != 0
    # ω = ∑ ak sin(k⋅x)/√2π|k|^2     if ky > 0 or (ky = 0 and kx > 0) 
    #   + ∑ bk cos(k⋅x)/√2π|k|^2,    otherwise
    #   = ∑ (bk/(2√2π|k|^2) - i ak/(2√2π|k|^2) )e^{ik⋅x} + (bk/(2√2π|k|^2) + i ak/(2√2π^2|k|^2) )e^{-ik⋅x}
    
    N_x , N_y = mesh.N_x, mesh.N_y
    kxx , kyy = mesh.kxx, mesh.kyy
    
    Random.seed!(seed);
    abks = rand(Normal(0, σ), N_x, N_y)
    
    ω0_hat = zeros(ComplexF64, N_x, N_y)
    ω0 = zeros(Float64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            kx, ky = kxx[ix], kyy[iy]
            abk = abks[ix, iy]
            if Z2_plus(kx, ky)
                #sin(kx)
                ω0_hat[ix, iy] -= abk*im/(2*sqrt(2)*pi*(kx^2+ky^2))
                # (ix=1 iy=1 => kx=0 ky=0) 1 => 1, i => n-i+2
                ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += abk*im/(2*sqrt(2)*pi*(kx^2+ky^2))  
            elseif Z2_minus(kx, ky)
                #cos(kx)
                ω0_hat[ix, iy] += abk/(2*sqrt(2)*pi*(kx^2+ky^2))
                # (ix=1 iy=1 => kx=0 ky=0) 1 => 1, i => n-i+2
                ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += abk/(2*sqrt(2)*pi*(kx^2+ky^2))  
            end
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, N_x*N_y * ω0_hat, ω0)
    
    #####
    # ω0_test = zeros(Float64, N_x, N_y)
    # X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    # Δx, Δy = mesh.Δx, mesh.Δy
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         X[ix, iy] = (ix-1)*Δx
    #         Y[ix, iy] = (iy-1)*Δy
    #     end
    # end
    
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         kx, ky = kxx[ix], kyy[iy]
    #         if (abs(kx) < N_x/3 && abs(ky) < N_y/3) 
    #             abk = abks[ix, iy]
    #             if (Z2_plus(kx, ky)) 
    #                 ω0_test .+= (abk * sin.(kx*X + ky*Y))/(sqrt(2)*pi*(kx^2 + ky^2))
    #             elseif (Z2_minus(kx, ky)) 
    #                 ω0_test .+= (abk * cos.(kx*X + ky*Y))/(sqrt(2)*pi*(kx^2 + ky^2))
    #             end 
    #         end
    #     end
    # end
    # @info  "Error", norm(ω0 - ω0_test)
    # error("Stop")
    
    
    
    ω0
end



# Generate stochastics forcing
function Stochastic_Forcing(mesh::Spectral_Mesh, curl_f_hat::Array{ComplexF64, 2}, Δt::Float64)
    
    N_x , N_y = mesh.N_x, mesh.N_y
    kxx , kyy = mesh.kxx, mesh.kyy
    
    dWk = rand(Normal(0, 1), N_x, N_y)/sqrt(Δt)
    
    curl_f_hat_dW = zeros(ComplexF64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            kx, ky = kxx[ix], kyy[iy]
            re_fk, im_fk = real(curl_f_hat[ix, iy]), imag(curl_f_hat[ix, iy])
            if Z2_plus(kx, ky)
                curl_f_hat_dW[ix, iy] -= im_fk*dWk[ix,iy]*im
                # (ix=1 iy=1 => kx=0 ky=0) 1 => 1, i => n-i+2
                curl_f_hat_dW[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += im_fk*dWk[ix,iy]*im 
            elseif Z2_minus(kx, ky)
                curl_f_hat_dW[ix, iy] += re_fk*dWk[ix,iy]
                # (ix=1 iy=1 => kx=0 ky=0) 1 => 1, i => n-i+2
                curl_f_hat_dW[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += re_fk*dWk[ix,iy]
            
            end
        end
    end
    curl_f_hat_dW .*= mesh.alias_filter
    return curl_f_hat_dW
end




# Generate ω field with energy decay function E
# kxx = [0,1,...,[N_x/2]-1, -[N_x/2], -[N_x/2]+1, ... -1]   
# kyy = [0,1,...,[N_y/2]-1, -[N_y/2], -[N_y/2]+1, ... -1]
function Random_Vor_Field(mesh::Spectral_Mesh, E::Function; seed::Int64 = 123)
    # u = ∑ uₖ e^{ik⋅x}  v = ∑ vₖ e^{ik⋅x}
    # ω = ∑ (vₖ kx - uₖ ky)i e^{ik⋅x}  =  ∑ ωₖ e^{ik⋅x} ; ωₖ = (vₖ kx - uₖ ky)i
    # ∇ ̇[u v] =  ∑ (uₖ kx + vₖ ky)ie^{ik⋅x} = 0
    # E(k) = 1/2(|uₖ|² + |vₖ|²) = 1/(2 k⋅k) [|vₖ kx - uₖ ky|² + |uₖ kx + vₖ ky|²] = |ωₖ|²/(2 k⋅k)
    # E(|k|) = π|k|(|uₖ|² + |vₖ|²) =  π|k||ωₖ|²/(k⋅k)
    
    # We first sample |ωₖ| ~ (E(|k|) |k| / π)^(1/2)
    # ωₖ = |ωₖ|exp{(η1(k) + η2(k)i) }
    
    N_x , N_y = mesh.N_x, mesh.N_y
    kxx , kyy = mesh.kxx, mesh.kyy
    
    Random.seed!(seed);
    η = rand(Uniform(0, 2π), N_x, N_y, 2)
    
    ω0_hat = zeros(ComplexF64, N_x, N_y)
    ω0 = zeros(Float64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            kx, ky = kxx[ix], kyy[iy]
            
            Eₖ = E(sqrt(kx^2 + ky^2))
            # Stable a posteriori LES of 2D turbulence using convolutional neural networks: 
            # Backscattering analysis and generalization to higher Re via transfer learning
            if kx ≥ 0 && ky ≥ 0
                ηk =  η[ix, iy, 1] + η[ix, iy, 2]
            elseif kx < 0 && ky ≥ 0
                ηk = -η[N_x-ix+2, iy, 1] + η[N_x-ix+2, iy, 2]
            elseif kx < 0 && ky < 0
                ηk = -η[N_x-ix+2, N_y-iy+2, 1] - η[N_x-ix+2, N_y-iy+2, 2]
            else # kx ≥0 & ky < 0
                ηk = η[ix, N_y-iy+2, 1] - η[ix, N_y-iy+2, 2]
            end
            ωₖ = sqrt( Eₖ*sqrt(kx^2 + ky^2)/π ) * exp(ηk*im)
            ω0_hat[ix, iy] = ωₖ    
        end
    end
    ω0_hat .*= mesh.alias_filter
    Trans_Spectral_To_Grid!(mesh, N_x*N_y * ω0_hat, ω0)
    ω0
end


# Generate ω field with energy decay function E
# kxx = [0,1,...,[N_x/2]-1, -[N_x/2], -[N_x/2]+1, ... -1]   
# kyy = [0,1,...,[N_y/2]-1, -[N_y/2], -[N_y/2]+1, ... -1]
function Compute_Energy_Spectrum(mesh::Spectral_Mesh, ω::Array{Float64,2})
    # u = ∑ uₖ e^{ik⋅x}  v = ∑ vₖ e^{ik⋅x}
    # ω = ∑ (vₖ kx - uₖ ky)i e^{ik⋅x}  =  ∑ ωₖ e^{ik⋅x} ; ωₖ = (vₖ kx - uₖ ky)i
    # ∇ ̇[u v] =  ∑ (uₖ kx + vₖ ky)ie^{ik⋅x} = 0
    # E(k) = 1/2(|uₖ|² + |vₖ|²) = 1/(2 k⋅k) [|vₖ kx - uₖ ky|² + |uₖ kx + vₖ ky|²] = |ωₖ|²/(2 k⋅k)
    # E(|k|) = π|k|(|uₖ|² + |vₖ|²) =  π|k||ωₖ|²/(k⋅k)
    
    # We first sample |ωₖ| ~ (E(|k|) |k| / π)^(1/2)
    # ωₖ = |ωₖ|exp{i (η1(k) + η2(k)i) }
    
    N_x , N_y = mesh.N_x, mesh.N_y
    kxx , kyy = mesh.kxx, mesh.kyy
    N_k = round(Int64, sqrt( (N_x/2)^2 + (N_y/2)^2 ) )
    E = zeros(Float64, N_k)

    ω_hat = zeros(ComplexF64, N_x, N_y)
    Trans_Grid_To_Spectral!(mesh, ω, ω_hat)

    for ix = 1:N_x
        for iy = 1:N_y
            if ix == 1 && iy == 1
                continue
            end

            kx, ky = kxx[ix], kyy[iy]
        
            k = sqrt(kx^2 + ky^2)
            
            E[round(Int64, k)] += abs2(ω_hat[ix, iy]) / (2*(kx^2 + ky^2)) 
        end
    end

    E/(N_x*N_y)^2
end


# For fourier modes, consider the zero-mean real field
# u = ∑ f_k e^{ik⋅x}, we have  f_{k} = conj(f_{-k})
# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1)
# with (ky > 0 || (kx + ky == 0 && kx > 0)) 
# And these fourier modes are sorted by 1/|k|^2 and the first N_KL modes 
# are computed in terms of (kx, ky) pair
function Compute_Seq_Pairs(N_KL::Int64)
    seq_pairs = zeros(Int64, N_KL, 2)
    trunc_N_x = trunc(Int64, sqrt(N_KL)) + 1
    
    seq_pairs = zeros(Int64, (2trunc_N_x+1)^2 - 1, 2)
    seq_pairs_mag = zeros(Int64, (2trunc_N_x+1)^2 - 1)
    
    seq_pairs_i = 0
    for kx = -trunc_N_x:trunc_N_x
        for ky = -trunc_N_x:trunc_N_x
            if (kx != 0 || ky != 0)
                seq_pairs_i += 1
                seq_pairs[seq_pairs_i, :] .= kx, ky
                seq_pairs_mag[seq_pairs_i] = kx^2 + ky^2
            end
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:N_KL, :]
end



# we can also use cos(k⋅x)/2π^2 and sin(k⋅x)/2π^2 as basis (L2 norm is 1) for zero-mean real fields
# ω = ∑ ak sin(k⋅x)/√2π|k|^2     if ky > 0 or (ky = 0 and kx > 0) 
#   + ∑ bk cos(k⋅x)/√2π|k|^2,    otherwise
#   = ∑ (bk/(2√2π|k|^2) - i ak/(2√2π|k|^2) )e^{ik⋅x} + (bk/(2√2π|k|^2) + i ak/(2√2π^2|k|^2) )e^{-ik⋅x}
    
function Random_Field_From_Theta(mesh::Spectral_Mesh, θ::Array{Float64,1}, seq_pairs::Array{Int64,2}; freq_or_not::Bool = false)
    
    N_x, N_y = mesh.N_x, mesh.N_y
    kxx, kyy = mesh.kxx, mesh.kyy
    
    ω0_hat = zeros(ComplexF64, N_x, N_y)
    ω0 = zeros(Float64, N_x, N_y)
    N_θ = length(θ)
    for i = 1:N_θ
        kx, ky = seq_pairs[i,:]
        ix = (kx >= 0 ? kx + 1 : N_x + kx + 1) 
        iy = (ky >= 0 ? ky + 1 : N_y + ky + 1) 
        if Z2_plus(kx, ky)
            ω0_hat[ix, iy] -= θ[i]*im/(2*sqrt(2)*pi*(kx^2+ky^2))
            # 1 => 1, i => n-i+2
            ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += θ[i]*im/(2*sqrt(2)*pi*(kx^2+ky^2))
        elseif Z2_minus(kx, ky)
            ω0_hat[ix, iy] += θ[i]/(2*sqrt(2)*pi*(kx^2+ky^2))
            # 1 => 1, i => n-i+2
            ω0_hat[(ix==1 ? 1 : N_x-ix+2), (iy==1 ? 1 : N_y-iy+2)] += θ[i]/(2*sqrt(2)*pi*(kx^2+ky^2))
        
        end
    end
    ω0_hat .*= mesh.alias_filter
    
    if freq_or_not
        return ω0_hat
    else
        return Trans_Spectral_To_Grid!(mesh, N_x*N_y * ω0_hat, ω0)
    end
    
#     ######
    # ω0_test = zeros(Float64, N_x, N_y)
    # X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    # Δx, Δy = mesh.Δx, mesh.Δy
    # for ix = 1:N_x
    #     for iy = 1:N_y
    #         X[ix, iy] = (ix-1)*Δx
    #         Y[ix, iy] = (iy-1)*Δy
    #     end
    # end


    # for i = 1:N_θ
    #     kx, ky = seq_pairs[i,:]
    #     @info kx, ky, N_x/3, N_y/3
    #     @assert((abs(kx) < N_x/3 && abs(ky) < N_y/3) && !((kx==0)&&(ky==0))) 
    #     abk = θ[i]
    #     if (Z2_plus(kx, ky)) 
    #         ω0_test .+= (abk * sin.(kx*X + ky*Y))/(sqrt(2)*pi*(kx^2 + ky^2))
    #     elseif (Z2_minus(kx, ky)) 
    #         ω0_test .+= (abk * cos.(kx*X + ky*Y))/(sqrt(2)*pi*(kx^2 + ky^2))
    #     end 
    # end

    # @info  "Error", norm(ω0 - ω0_test), norm(ω0) , norm(ω0_test)
    # error("Stop")
    # ω0
end

# project  ω0 on the first N_θ dominant modes
# namely, ∑ ak cos(k⋅x) + bk sin(k⋅x) with least square fitting
function Theta_From_Random_Field(ω0::Array{Float64, 2}, N_θ::Int64, seq_pairs::Array{Int64,2}, mesh::Spectral_Mesh)
    
    N_x , N_y = mesh.N_x, mesh.N_y
    Δx , Δy   = mesh.Lx/N_x, mesh.Ly/N_y
    # fit the dominant modes from data0 by solving A x = data0
    # ω = ∑ ak cos(k⋅x)/2π^2|k|^2 + ∑ bk sin(k⋅x)/2π^2|k|^2,    &  kx + ky > 0 or (kx + ky = 0 and kx > 0) 
    A = zeros(Float64, length(ω0), N_θ)
    X, Y = zeros(Float64, N_x, N_y), zeros(Float64, N_x, N_y)
    for ix = 1:N_x
        for iy = 1:N_y
            X[ix, iy] = (ix-1)*Δx
            Y[ix, iy] = (iy-1)*Δy
        end
    end
    
    for i = 1:Int64(N_θ/2)
        kx, ky = seq_pairs[i,:]
        if Z2_plus(kx,ky)
            A[:,i] = (sin.(kx*X + ky*Y)/(sqrt(2)*pi*(kx^2 + ky^2)))[:]
        elseif Z2_minus(kx,ky)
            A[:,i] = (cos.(kx*X + ky*Y)/(sqrt(2)*pi*(kx^2 + ky^2)))[:]
        else
            error("kx,ky = ", kx, ky)
        end
    end
    
    θ0 = A\ω0[:]

    return θ0
    
end



# generate truth (all Fourier modes) and observations
# observation are at frame 0, Δd_t, 2Δd_t, ... N_t
# with sparse points at Array(1:Δd_x:N_x) × Array(1:Δd_y:N_y)
function forward_helper(s_param::Setup_Param, ω0::Array{Float64,2}; symmetric::Bool = false, save_file_name::String="None", vmin=nothing, vmax=nothing)

    N_x = N_y = s_param.N
    Lx = Ly = s_param.L
    seq_pairs = s_param.seq_pairs
    method = s_param.method
    x_locs, y_locs, t_locs = s_param.x_locs, s_param.y_locs, s_param.t_locs
 
    mesh = Spectral_Mesh(N_x, N_y, Lx, Ly)
    
    solver = Spectral_NS_Solver(mesh, ν; fx = s_param.fx, fy = s_param.fy, ω0 = ω0, ub = s_param.ub, vb = s_param.vb)  
    Δt_max = Stable_Δt(mesh, ν, solver.u, solver.v)
    T, N_t = s_param.T, s_param.N_t
    Δt = T/N_t
    @assert(Δt <= Δt_max)
    # initialize observation 
    
    data = zeros(Float64, size(x_locs,1), size(y_locs,1), size(t_locs,1))
    
    obs_t_id = 1
    for i = 1:N_t
        Solve!(solver, Δt, method)
        if obs_t_id <= size(t_locs,1) && i == t_locs[obs_t_id]
            Update_Grid_Vars!(solver)
            if save_file_name != "None"
                Visual_Obs(mesh, solver.ω, x_locs, y_locs, "ω"; symmetric=symmetric, save_file_name=save_file_name*string(i)*".pdf", vmin=vmin, vmax=vmax)
            end
            if symmetric
                ω_mirror = solver.ω[[1;end:-1:2], :]
                data[:, :, obs_t_id] = (solver.ω[x_locs, y_locs] - ω_mirror[x_locs, y_locs])/2.0
            else
                data[:, :, obs_t_id] = solver.ω[x_locs, y_locs]
            end
            obs_t_id += 1
        end
    end
    
    return data[:]
    
    
end




function forward(s_param::Setup_Param, θ::Array{Float64,1})
    # θ = [ak; bk] and abk = [ak, bk], we have θ = abk[:] and abk = reshape(θ, N_θ/2, 2)
    N_x = N_y = s_param.N
    Lx = Ly = s_param.L
    seq_pairs = s_param.seq_pairs
    mesh = Spectral_Mesh(N_x, N_y, Lx, Ly)

    ω0 = Random_Field_From_Theta(mesh, θ, seq_pairs)   
    data = forward_helper(s_param, ω0; symmetric = s_param.symmetric)
    
    return data[:]
end




function aug_forward(s_param::Setup_Param, θ::Array{Float64,1})
    # θ = [ak; bk] and abk = [ak, bk], we have θ = abk[:] and abk = reshape(θ, N_θ/2, 2)
    

    data = forward(s_param, θ)
    
    return [data; θ]
end


function NS_F(s_param::Setup_Param, args, θ::Array{Float64, 1}) 
    y_obs, r₀, ση, σ₀ = args
    Gθ  = forward(s_param, θ)
    return [(y_obs  - Gθ)./ση; (r₀ - θ)./σ₀]

end








