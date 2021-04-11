include("Piston.jl")
include("Euler1D.jl")
# L is the length of the tube, assuming that the tube is at [0, L]
# N is the number of fluid ceils
# structure_inf = [xs_0, vs_0, ks, ms], which are parameters for structure
# fluid_inf =[gamma,ρ_l, v_l, p_l], which are parameters for the fluid
# mesh  = [L, N ,CFL]

function Solve(mesh_info, fluid_info, structure_info, time_info)
    L, N = mesh_info
    # constant flow velocity
    γ, ρ_0, v_0, p_0 = fluid_info
    Δt, T = time_info
    # assume the structure spring is attached at L/2
    ms, cs, ks, ds_0, vs_0, x0, motion, forced_motion_func = structure_info

    t = 0.0
    #####################################################
    # Initialize structure
    #####################################################
    
    as_0 = (p_0 - ks*ds_0 - cs*vs_0)/ms
    Q = [ds_0; vs_0; as_0]

    structure = Structure(ds_0, vs_0, as_0, x0, ms, cs, ks, motion, forced_motion_func, t)
    

    #####################################################
    # Initialize fluid
    #####################################################
    emb = [x0 + ds_0, vs_0]
    fluid = Euler1D(Int64(N), L, emb; FIVER_order = 2, γ = γ)
    init_func = (x) -> (x <= x0 ? (true, ρ_0, v_0, p_0) : (false, 0, 0, 0))
    Initial!(fluid, init_func)


    ################################################################
    if structure.motion == "AEROELASTIC"
        fext = p_0
        A6_First_Half_Step!(structure, fext, t, Δt)
    end

    while t < T

        # the predicted structure position and velocity at t + Δt
        emb = Send_Embeded_Disp(structure, t, Δt)
        
        # update fluid to t + Δt
        Fluid_Time_Advance!(fluid, emb, t, Δt)


        if structure.motion == "AEROELASTIC"
            # the corrected fluid load at t + Δt/2
            fext = Send_Force(fluid, t, Δt)

            # update structure to t + Δt
            Structure_Time_Advance!(structure, fext, t, Δt)
        end

        t = t + Δt
    end

    return fluid, structure
end

using PyPlot
L, N = 2.0, 200
mesh_info  = [L, N]
fluid_info = [1.4, 1.225, 0. , 1.0] 
ms, cs, ks = 1.0, 0.0, 1.0
structure_info = [ms, cs, ks, 0.0, 0.0, L/2, "AEROELASTIC", nothing] 

# function structure(t)
    
#     ds = 1.0/8.0 * t^4
#     vs = 1.0/2.0 * t^3
#     as = 3.0/2.0 * t^2

#     return [ds; vs; as]
# end

# structure_info = [ms, cs, ks, 0.0, 0.0, L/2, "FORCED", structure] 
Δt, T = 0.001, 0.5
time_info = [Δt, T]

fluid, piston = Solve(mesh_info, fluid_info, structure_info, time_info)
V = fluid.V
xx = fluid.xx

fig, ax = PyPlot.subplots(ncols=3, nrows=1, figsize=(18, 5))

ax[1].plot(xx, V[1, :], "-o", fillstyle = "none")
ax[2].plot(xx, V[2, :], "-o", fillstyle = "none")
ax[3].plot(xx, V[3, :], "-o", fillstyle = "none")
ax[1].set_xlabel("x")
ax[1].set_ylabel("ρ")
ax[2].set_xlabel("x")
ax[2].set_ylabel("v")
ax[3].set_xlabel("x")
ax[3].set_ylabel("p")
fig.tight_layout()