using NNGCM
using LinearAlgebra
using Random
import PyPlot


mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int, CT<:Complex}
    # physics 170, 256, 1 #85, 128, 1  # 42, 64, 1
    θ_names::String
    num_fourier::IT  
    nθ::IT
    mesh::Spectral_Spherical_Mesh
    radius::FT 
    omega::FT
    
    # for parameterization
    Δt::IT
    end_time::IT
    n_obs_frames::IT
    obs_time::IT

    # background velocity/vorticity profiles
    grid_u_b::Array{FT,3}
    grid_v_b::Array{FT,3}
    grid_vor_b::Array{FT,3}
    spe_vor_b::Array{CT,3}
    
    # initial vorticity field
    grid_u::Array{FT,3}
    grid_v::Array{FT,3}
    grid_vor::Array{FT,3}
    spe_vor::Array{CT,3}
    init_data::Array{FT,1}
    
    # inverse parameters
    nobs::IT
    obs_coord::Array{IT, 2}
    antisymmetric::Bool
    N_y::IT
    trunc_N::IT
    N_θ::IT
end


function Setup_Param(num_fourier::IT,  nθ::IT, 
                     Δt::IT, end_time::IT, n_obs_frames::IT, nobs::IT, antisymmetric::Bool, N_y::IT, trunc_N::IT; radius::FT = 6371.2e3, omega::FT = 7.292e-5) where {FT<:AbstractFloat, IT<:Int}
    
    num_spherical = num_fourier + 1
    nλ = 2nθ
    N_θ = trunc_N^2 + 2*trunc_N
    obs_time = div(end_time, n_obs_frames)
    mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b, grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, init_data = Barotropic_Init(num_fourier, nθ; trunc_N=trunc_N, radius=radius)
    
    
    
    Lat_Lon_Pcolormesh(mesh, grid_u_b,  1; save_file_name = "Figs/Barotropic_u_backgroud.pdf", cmap = "viridis")
    Lat_Lon_Pcolormesh(mesh, grid_vor_b,  1; save_file_name = "Figs/Barotropic_vor_backgroud.pdf", cmap = "viridis")
    Lat_Lon_Pcolormesh(mesh, grid_vor_pert, 1; save_file_name = "Figs/Barotropic_vor_pert0.pdf", cmap = "viridis")
    Lat_Lon_Pcolormesh(mesh, grid_vor, 1; save_file_name = "Figs/Barotropic_vor0.pdf", cmap = "viridis")
    Lat_Lon_Pcolormesh(mesh, grid_u, 1; save_file_name = "Figs/Barotropic_vel_u0.pdf", cmap = "viridis")
    
    
    
    obs_coord = zeros(Int64, nobs, 2)
    Random.seed!(42)
    obs_coord[:,1], obs_coord[:, 2] = rand(1:nλ-1, nobs), rand(Int64(nθ/2)+1:nθ-1, nobs)

    
    Setup_Param("barotropic_flow", num_fourier, nθ, mesh, radius, omega,
        Δt, end_time, n_obs_frames, obs_time, 
        grid_u_b, grid_v_b, grid_vor_b, spe_vor_b, grid_u, 
        grid_v, grid_vor, spe_vor, init_data, 
        nobs, obs_coord, antisymmetric, N_y, trunc_N, N_θ)
end



# truncate the perturbation modes to trunc_N 
function Barotropic_Init(num_fourier::IT, nθ::IT; trunc_N::IT = num_fourier, radius::FT = 6371.2e3) where {FT<:AbstractFloat, IT<:Int}
    num_spherical = num_fourier + 1
    nλ = 2nθ
    nd = 1
    # Initialize mesh
    mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
    θc, λc = mesh.θc,  mesh.λc
    cosθ, sinθ = mesh.cosθ, mesh.sinθ
    grid_u_b = Array{FT, 3}(undef, nλ, nθ, 1)
    grid_v_b = Array{FT, 3}(undef, nλ, nθ, 1)
    spe_vor_b = Array{ComplexF64, 3}(undef, num_fourier+1, num_spherical+1, 1); spe_vor_b .= 0.0
    spe_div_b = Array{ComplexF64, 3}(undef, num_fourier+1, num_spherical+1, 1); spe_div_b .= 0.0
    grid_vor_b = Array{FT, 3}(undef, nλ, nθ, 1)
    grid_div_b = Array{FT, 3}(undef, nλ, nθ, 1)
    for i = 1:nλ
      grid_u_b[i, :, 1] .= 25 * cosθ - 30 * cosθ.^3 + 300 * sinθ.^2 .* cosθ.^6
    end
    grid_v_b .= 0.0 
    Vor_Div_From_Grid_UV!(mesh, grid_u_b, grid_v_b, spe_vor_b, spe_div_b) 
    Trans_Spherical_To_Grid!(mesh, spe_vor_b,  grid_vor_b)
    Trans_Spherical_To_Grid!(mesh, spe_div_b,  grid_div_b)
    
    
    grid_vor_pert = similar(grid_vor_b)
    grid_u = Array{FT, 3}(undef, nλ, nθ, 1)
    grid_v = Array{FT, 3}(undef, nλ, nθ, 1)
    # ! adding a perturbation to the vorticity
    m, θ0, θw, A = 4.0, 45.0 * pi / 180, 15.0 * pi / 180.0, 8.0e-5
    for i = 1:nλ
      for j = 1:nθ
        grid_vor_pert[i,j, 1] = A / 2.0 * cosθ[j] * exp(-((θc[j] - θ0) / θw)^2) * cos(m * λc[i])
      end
    end
    spe_vor_pert = similar(spe_vor_b); spe_vor_pert .= 0.0;
    Trans_Grid_To_Spherical!(mesh, grid_vor_pert,  spe_vor_pert) 
    
    grid_vor = grid_vor_b + grid_vor_pert
    spe_vor = similar(spe_vor_b); spe_vor .= 0.0;
    spe_zeros = similar(spe_vor_b); spe_zeros .= 0.0;
    Trans_Grid_To_Spherical!(mesh, grid_vor, spe_vor)
    UV_Grid_From_Vor_Div!(mesh, spe_vor,  spe_zeros, grid_u, grid_v)
    
      # F_1,2 F_2,2  
      # F_1,3 F_2,3 F_3,3
      #   ...
      # F_1,N+1 F_2,N+1 F_3,N+1, ..., F_N+1,N+1
    
      # n_init_data = length(init_data)
      # N = Int64(sqrt(1 + n_init_data) - 1)
#### DEBUG
#     n_init_data = (trunc_N+1)*(trunc_N+1) - 1
#     init_data = zeros(n_init_data)
#     m = 0
#     for n = 1:trunc_N
#       init_data[n] = spe_vor_pert[m+1,n+1,1] * radius
#     end
#     i_init_data = trunc_N
#     for m = 1:trunc_N
#       for n = m:trunc_N
#         # m = 1, 2, 3 .. m 
#         #     0  N  N-1...
#         #     N + N-1 ... N-m+2 
#         init_data[i_init_data + 1] =  real(spe_vor_pert[m+1,n+1,1])*radius
#         init_data[i_init_data + 2] = -imag(spe_vor_pert[m+1,n+1,1])*radius
#         i_init_data += 2
#       end 
#     end
#     @assert(norm(spe_to_param(spe_vor_pert, trunc_N; radius=radius) - init_data) ≈ 0.0 )
    init_data = spe_to_param(spe_vor_pert, trunc_N; radius=radius)
    
#     ##### 
#     DEBUG = false
#     if DEBUG
#         spe_vor0, grid_vor0 = similar(spe_vor_b), similar(grid_vor_b)
#         Barotropic_ω0!(mesh, "spec_vor", init_data, spe_vor0, grid_vor0; spe_vor_b = spe_vor_b, radius = radius)
#         @info "grid_vor rel. error = ", norm(grid_vor0 - grid_vor)/norm(grid_vor)
#         @info "spe_vor rel. error = ", norm(spe_vor0 - spe_vor)/norm(spe_vor)
#         Lat_Lon_Pcolormesh(mesh, grid_vor0, 1; save_file_name = "Figs/Debug_Barotropic_vor0.pdf", cmap = "viridis")
#         Lat_Lon_Pcolormesh(mesh, grid_vor - grid_vor0, 1; save_file_name = "Figs/Debug_Barotropic_vor0_diff.pdf", cmap = "viridis") 
#     end
     
    return mesh, grid_u_b, grid_v_b, grid_vor_b, spe_vor_b, grid_vor_pert, grid_u, grid_v, grid_vor, spe_vor, init_data
    
end

function spe_to_param(spe_vor, trunc_N; radius=6371.2e3)
    n_init_data = (trunc_N+1)*(trunc_N+1) - 1
    init_data = zeros(n_init_data)
    m = 0
    for n = 1:trunc_N
      init_data[n] = spe_vor[m+1,n+1,1] * radius
    end
    i_init_data = trunc_N
    for m = 1:trunc_N
      for n = m:trunc_N
        # m = 1, 2, 3 .. m 
        #     0  N  N-1...
        #     N + N-1 ... N-m+2 
        init_data[i_init_data + 1] =  real(spe_vor[m+1,n+1,1])*radius
        init_data[i_init_data + 2] = -imag(spe_vor[m+1,n+1,1])*radius
        i_init_data += 2
      end 
    end
    return init_data
end
function param_to_spe(init_data, num_fourier; radius=6371.2e3)
    num_spherical = num_fourier + 1
    spe_vor = Array{ComplexF64, 3}(undef, num_fourier+1, num_spherical+1, 1)
    spe_vor .= 0.0
    
    n_init_data = length(init_data)
    N = Int64(sqrt(1 + n_init_data) - 1)
    
    m = 0
    for n = 1:N
      spe_vor[m+1,n+1,1] += init_data[n]/radius
    end
    i_init_data = N
    for m = 1:N
      for n = m:N
        # m = 1, 2, 3 .. m 
        #     0  N  N-1...
        #     N + N-1 ... N-m+2 
        spe_vor[m+1,n+1,1] += (init_data[i_init_data+1] - init_data[i_init_data + 2]*im)/radius
        i_init_data += 2
      end 
    end
    return spe_vor
end
    


function Barotropic_Main(sparam::Setup_Param, init_data = nothing; init_type = "spec_vor")
  # the decay of a sinusoidal disturbance to a zonally symmetric flow 
  # that resembles that found in the upper troposphere in Northern winter.
  name = "Barotropic"
  num_fourier, nθ = sparam.num_fourier, sparam.nθ
  nd = 1 
  num_spherical = num_fourier + 1
  nλ = 2nθ
  
  radius = 6371.2e3
  omega = 7.292e-5
  
  # Initialize mesh
  mesh = sparam.mesh
  θc, λc = mesh.θc,  mesh.λc
  cosθ, sinθ = mesh.cosθ, mesh.sinθ
  
  
  # Initialize atmo_data
  atmo_data = Atmo_Data(name, nλ, nθ, nd, false, false, false, false, sinθ, radius, omega)
  
  # Initialize integrator
  damping_order = 4
  damping_coef = 1.e-04
  robert_coef  = 0.04 
  implicit_coef = 0.0
  
###########################################################################################
    
  start_time = 0 
  Δt = sparam.Δt
  obs_time = sparam.obs_time
  end_time = sparam.end_time
  ###########################################################################################
  
 
  
  init_step = true
  
  integrator = Filtered_Leapfrog(robert_coef, 
  damping_order, damping_coef, mesh.laplacian_eig,
  implicit_coef,
  Δt, init_step, start_time, end_time)
    
  ###########################################################################################
  # Initialization 
  # Initialize: 
  # dyn_data.spe_vor_c, dyn_data.spe_div_c = 0.0
  # dyn_data.grid_vor, dyn_data.grid_div = 0.0
  # dyn_data.grid_u_c, dyn_data.grid_v_c
  ###########################################################################################
  
  dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd, 0, 0)
  grid_u, grid_v = dyn_data.grid_u_c, dyn_data.grid_v_c
  spe_vor_c, spe_div_c = dyn_data.spe_vor_c, dyn_data.spe_div_c
  grid_vor, grid_div = dyn_data.grid_vor, dyn_data.grid_div
  
  grid_vor_b = nothing
  
    spe_div_c .= 0.0
    grid_div  .= 0.0
  if init_type == "truth"
    _, _, _, _, _, _, grid_u[:], grid_v[:], grid_vor[:], spe_vor_c[:], _ = Barotropic_Init(num_fourier, nθ; radius = radius) 
    
  else
    Barotropic_ω0!(mesh, init_type, init_data, spe_vor_c, grid_vor; spe_vor_b = sparam.spe_vor_b)
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c, spe_div_c, grid_u, grid_v)
  end
  
  spe_vor0 = copy(spe_vor_c); grid_vor0 = copy(grid_vor)
  
  
  obs_data = Dict("vel_u"=>Array{Float64, 3}[], "vel_v"=>Array{Float64, 3}[], "vor"=>Array{Float64, 3}[])
  
  
  NT = Int64(end_time / Δt)
  time = start_time
  Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
  Update_Init_Step!(integrator)
  time += Δt 
  for i = 2:NT
    Barotropic_Dynamics!(mesh, atmo_data, dyn_data, integrator)
    time += Δt
    if time % obs_time == 0
      push!(obs_data["vel_u"], copy(grid_u))
      push!(obs_data["vor"], copy(grid_vor))
    end
    
  end
  
  return mesh, obs_data
end






# initialize Barotropic flow ω0
# return spe_vor0, grid_vor0
# the initialization data is in init_data
# initialize with grid vorticity (init_type == "grid_vor")
# initialize with spectral vorticity (init_type == "spec_vor")
function Barotropic_ω0!(mesh, init_type::String, init_data, spe_vor0, grid_vor0; spe_vor_b = nothing, radius = 6371.2e3)
  
  
  if init_type == "grid_vor"
    grid_vor0[:] = copy(init_data)
    
    # grid_vor0 ./= radius
    
    Trans_Grid_To_Spherical!(mesh, grid_vor0, spe_vor0)
    
  elseif init_type == "spec_vor"
    # 
    spe_vor0 .= spe_vor_b
    # m = 0,   1, ... N
    # n = m, m+1, ... N
    # F_m,n = {F_0,1 F_0,2, F_0,3 ... F_0,N,    F_1,1 F_1,2, ... F_1,N, ..., F_N,N}
    # F_0,1 F_1,1  
    # F_0,2 F_1,2 F_2,2
    #   ...
    # F_0,N F_1,N F_2,N, ..., F_N,N
    # N + 2(N+1)N/2 = N^2 + 2N
    #### DEBUG
#     n_init_data = length(init_data)
#     N = Int64(sqrt(1 + n_init_data) - 1)
    
#     m = 0
#     for n = 1:N
#       spe_vor0[m+1,n+1,1] += init_data[n]/radius
#     end
#     i_init_data = N
#     for m = 1:N
#       for n = m:N
#         # m = 1, 2, 3 .. m 
#         #     0  N  N-1...
#         #     N + N-1 ... N-m+2 
#         spe_vor0[m+1,n+1,1] += (init_data[i_init_data+1] - init_data[i_init_data + 2]*im)/radius
#         i_init_data += 2
#       end 
#     end
#     @assert(i_init_data == n_init_data)
     
#     @assert( norm(param_to_spe_to(init_data, mesh.num_fourier; radius=radius) + spe_vor_b - spe_vor0) ≈ 0.0 )
    
        
    spe_vor0 .= spe_vor_b .+ param_to_spe(init_data, mesh.num_fourier; radius=radius)
        
    
    Trans_Spherical_To_Grid!(mesh, spe_vor0, grid_vor0)
  else 
    error("Initial condition type :", init_type,  " is not recognized.")
  end
  
    return spe_vor0, grid_vor0
end




function convert_obs(obs_coord, obs_raw_data; antisymmetric=false)
  # update obs
    obs_raw = obs_raw_data["vel_u"]
    nobs = size(obs_coord, 1)
    nframes = length(obs_raw)
    obs = zeros(Float64, nobs, nframes)
  
  
  for i = 1:nframes
    for j = 1:nobs
      if antisymmetric
        obs[j,i] = (obs_raw[i][obs_coord[j,1], obs_coord[j,2]] + obs_raw[i][obs_coord[j,1], end-obs_coord[j,2] + 1])/2.0
      else
        obs[j,i] = obs_raw[i][obs_coord[j,1], obs_coord[j,2]]
      end
    end
  end
  
  return obs[:]
end





function NNGCM.Lat_Lon_Pcolormesh(mesh::Spectral_Spherical_Mesh, grid_dat::Array{Float64,3}, level::Int64, obs_coord::Array{Int64, 2}; 
        vmin = nothing, vmax = nothing, save_file_name::String = "None", cmap="viridis", antisymmetric::Bool=false)
    
    λc, θc = mesh.λc, mesh.θc
    nλ, nθ = length(λc), length(θc)
    λc_deg, θc_deg = λc*180/pi, θc*180/pi
    
    X,Y = repeat(λc_deg, 1, nθ), repeat(θc_deg, 1, nλ)'
    
    
    PyPlot.pcolormesh(X, Y, grid_dat[:,:,level], shading= "gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
    PyPlot.colorbar()
    x_obs, y_obs = λc_deg[obs_coord[:,1]], θc_deg[obs_coord[:,2]]
    PyPlot.scatter(x_obs, y_obs, color="black")
    
    if antisymmetric
        x_obs_anti, y_obs_anti = λc_deg[obs_coord[:,1]], θc_deg[end + 1 .- obs_coord[:,2]]
        PyPlot.scatter(x_obs_anti, y_obs_anti, facecolors="none", edgecolors="black")
    end
    
    PyPlot.axis("equal")
    PyPlot.xlabel("Longitude")
    PyPlot.ylabel("Latitude")
    PyPlot.tight_layout()
    
    if save_file_name != "None"
        PyPlot.savefig(save_file_name)
        PyPlot.close("all")
    end
    
end
###########################################################################################
function aug_forward(barotropic::Setup_Param{FT, IT, CT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int, CT<:Complex}
      
    mesh, obs_raw_data = Barotropic_Main(barotropic, θ; init_type = "spec_vor")
    y = convert_obs(barotropic.obs_coord, obs_raw_data; antisymmetric=barotropic.antisymmetric)
    return [y ; θ]
end