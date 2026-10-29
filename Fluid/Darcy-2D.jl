using JLD2
using Statistics
using LinearAlgebra
using Distributions
using Random
using SparseArrays




mutable struct Setup_Param{FT<:AbstractFloat, IT<:Int}
    # physics
    N::IT         # number of grid points for both x and y directions (including both ends)
    L::FT       # computational domain [0, L]×[0, L]
    Δx::FT
    xx::Array{FT, 1}  # uniform grid [0, Δx, 2Δx ... L]
    
    #for source term
    f_2d::Array{FT, 2}

    # for parameterization
    d::FT
    τ::FT


    
    
    #for parameterization
    N_KL::IT  # this is for generating the truth
    logκ_2d::Array{FT, 2}
    φ::Array{FT, 3}
    λ::Array{FT, 1}
    θ_ref::Array{FT, 1}


    # inverse parameters
    θ_names::Array{String, 1}
    N_θ::IT
    # observation locations is tensor product x_locs × y_locs
    x_locs::Array{IT, 1}
    y_locs::Array{IT, 1}

    N_y::IT
    
end


function Setup_Param(N::IT, L::FT, N_KL::IT, obs_ΔNx::IT, obs_ΔNy::IT, 
                     N_θ::IT, d::FT=2.0, τ::FT=3.0, σ0::FT=1.0; seed::IT=123)  where {FT<:AbstractFloat, IT<:Int}

    xx = Array(LinRange(0, L, N))
    Δx = xx[2] - xx[1]
    
    logκ_2d, φ, λ, θ_ref = generate_θ_KL(xx, N_KL, d, τ, σ0; seed=seed)
    f_2d = compute_f_2d(xx)

    x_locs = Array(obs_ΔNx+1:obs_ΔNx:N-obs_ΔNx)
    y_locs = Array(obs_ΔNy+1:obs_ΔNy:N-obs_ΔNy)
    N_y = length(x_locs)*length(y_locs)

    θ_names=["logκ"]

    Setup_Param(N, L, Δx, xx, f_2d, d, τ, N_KL, logκ_2d, φ, λ, θ_ref,  θ_names, N_θ, x_locs, y_locs, N_y)
end



#=
A hardcoding source function, 
which assumes the computational domain is
[0 1]×[0 1]
f(x,y) = f(y),
which dependes only on y
=#
function compute_f_2d(yy::Array{FT, 1}) where {FT<:AbstractFloat}
    N = length(yy)
    f_2d = zeros(FT, N, N)
    for i = 1:N
        if (yy[i] <= 4/6)
            f_2d[:,i] .= 1000.0
        elseif (yy[i] >= 4/6 && yy[i] <= 5/6)
            f_2d[:,i] .= 2000.0
        elseif (yy[i] >= 5/6)
            f_2d[:,i] .= 3000.0
        end
    end
    return f_2d
end



#=
Compute sorted pair (i, j), sorted by i^2 + j^2
with i≥0 and j≥0 and i+j>0

These pairs are used for Karhunen–Loève expansion
=#
function compute_seq_pairs(N_KL::IT) where {IT<:Int}
    seq_pairs = zeros(IT, N_KL, 2)
    trunc_Nx = trunc(IT, sqrt(2*N_KL)) + 1
    
    seq_pairs = zeros(IT, (trunc_Nx+1)^2 - 1, 2)
    seq_pairs_mag = zeros(IT, (trunc_Nx+1)^2 - 1)
    
    seq_pairs_i = 0
    for i = 0:trunc_Nx
        for j = 0:trunc_Nx
            if (i == 0 && j ==0)
                continue
            end
            seq_pairs_i += 1
            seq_pairs[seq_pairs_i, :] .= i, j
            seq_pairs_mag[seq_pairs_i] = i^2 + j^2
        end
    end
    
    seq_pairs = seq_pairs[sortperm(seq_pairs_mag), :]
    return seq_pairs[1:N_KL, :]
end


#=
Generate parameters for logk field, based on Karhunen–Loève expansion.
They include eigenfunctions φ, eigenvalues λ and the reference parameters θ_ref, 
and reference field logk_2d field

logκ = ∑ u_l √λ_l φ_l(x)                l = (l₁,l₂) ∈ Z^{0+}×Z^{0+} \ (0,0)

where φ_{l}(x) = √2 cos(πl₁x₁)             l₂ = 0
                 √2 cos(πl₂x₂)             l₁ = 0
                 2  cos(πl₁x₁)cos(πl₂x₂) 
      λ_{l} = (π^2l^2 + τ^2)^{-d} 

They can be sorted, where the eigenvalues λ_{l} are in descending order

generate_θ_KL function generates the summation of the first N_KL terms 
=#
function generate_θ_KL(xx::Array{FT,1}, N_KL::IT, d::FT=2.0, τ::FT=3.0, σ0::FT=1.0; seed::IT=123) where {FT<:AbstractFloat, IT<:Int}
    N = length(xx)
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    
    seq_pairs = compute_seq_pairs(N_KL)
    
    φ = zeros(FT, N_KL, N, N)
    λ = zeros(FT, N_KL)
    
    for i = 1:N_KL
        if (seq_pairs[i, 1] == 0 && seq_pairs[i, 2] == 0)
            φ[i, :, :] .= 1.0
        elseif (seq_pairs[i, 1] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 2]*Y))
        elseif (seq_pairs[i, 2] == 0)
            φ[i, :, :] = sqrt(2)*cos.(pi * (seq_pairs[i, 1]*X))
        else
            φ[i, :, :] = 2*cos.(pi * (seq_pairs[i, 1]*X)) .*  cos.(pi * (seq_pairs[i, 2]*Y))
        end

        λ[i] = (pi^2*(seq_pairs[i, 1]^2 + seq_pairs[i, 2]^2) + τ^2)^(-d)
    end
    

    Random.seed!(seed);
    θ_ref = rand(Normal(0, σ0), N_KL)

    logκ_2d = zeros(FT, N, N)
    for i = 1:N_KL
        logκ_2d .+= θ_ref[i]*sqrt(λ[i])*φ[i, :, :]
    end
    
    return logκ_2d, φ, λ, θ_ref
end



#=
Given θ, compute logk field as 
∑ θ[i] * sqrt(λ[i]) * φ[i, :, :]
=#
function compute_logκ_2d(darcy::Setup_Param{FT, IT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
    N, N_KL = darcy.N, darcy.N_KL
    λ, φ = darcy.λ, darcy.φ
    N_θ = length(θ)
    
    @assert(N_θ <= N_KL) 
    logκ_2d = zeros(FT, N, N)
    for i = 1:N_θ
        coefficient = θ[i] * sqrt(λ[i])
        @views logκ_2d .+= coefficient .* φ[i, :, :]
    end
    
    return logκ_2d
end


function compute_dκ_dθ(darcy::Setup_Param{FT, IT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
    N, N_KL = darcy.N, darcy.N_KL
    λ, φ = darcy.λ, darcy.φ
    N_θ = length(θ)
    
    @assert(N_θ <= N_KL) 
    logκ_2d = zeros(FT, N*N)
    dκ_dθ = zeros(FT, N*N, N_θ)

    for i = 1:N_θ
        logκ_2d .+= (θ[i] * sqrt(λ[i]) * φ[i, :, :])[:]
    end
    
    for i = 1:N_θ
        dκ_dθ[:, i] = (sqrt(λ[i]) * φ[i, :, :])[:] .* exp.(logκ_2d)
    end

    return dκ_dθ
end


#=
    return the unknow index for the grid point

    Since zero-Dirichlet boundary conditions are imposed on  
    all four edges, the freedoms are only on interior points

=#
function ind(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT<:AbstractFloat, IT<:Int}
    return (ix-1) + (iy-2)*(darcy.N - 2)
end

function ind_all(darcy::Setup_Param{FT, IT}, ix::IT, iy::IT) where {FT<:AbstractFloat, IT<:Int}
    return ix + (iy-1)*darcy.N
end

#=
    solve Darcy equation with finite difference method:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
=#
function solve_Darcy_2D(darcy::Setup_Param{FT, IT}, κ_2d::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    Δx, N = darcy.Δx, darcy.N
    n = N - 2
    N_unknowns = n^2
    N_entries = 5*N_unknowns - 4*n

    # The five-point stencil has a known, sorted CSC pattern. Construct it
    # directly to avoid small temporary vectors and duplicate COO diagonals.
    # Each call owns these arrays, so simultaneous samples share no workspace.
    colptr = Vector{IT}(undef, N_unknowns + 1)
    rowval = Vector{IT}(undef, N_entries)
    nzval = Vector{FT}(undef, N_entries)
    colptr[1] = 1
    entry = 1
    f_2d = darcy.f_2d
    𝓒 = Δx^2

    for iy = 2:N-1
        for ix = 2:N-1
            column = ind(darcy, ix, iy)
            top = FT((κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒)
            bottom = FT((κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒)
            right = FT((κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒)
            left = FT((κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒)

            if iy > 2
                rowval[entry] = column - n
                nzval[entry] = -bottom
                entry += 1
            end
            if ix > 2
                rowval[entry] = column - 1
                nzval[entry] = -left
                entry += 1
            end

            rowval[entry] = column
            nzval[entry] = top + bottom + right + left
            entry += 1

            if ix < N-1
                rowval[entry] = column + 1
                nzval[entry] = -right
                entry += 1
            end
            if iy < N-1
                rowval[entry] = column + n
                nzval[entry] = -top
                entry += 1
            end
            colptr[column + 1] = entry
        end
    end

    df = SparseMatrixCSC(N_unknowns, N_unknowns, colptr, rowval, nzval)
    h = df\vec(f_2d[2:N-1,2:N-1])
    
    h_2d = zeros(FT, N, N)
    h_2d[2:N-1,2:N-1] .= reshape(h, N-2, N-2) 
    
    return h_2d
end



#=
    the Darcy equation with finite difference method:
    -∇(κ∇h) = f
    with Dirichlet boundary condition, h=0 on ∂Ω
    The compute adjoint operator adjoint

    G(h, k) = 0   df(k)×h - f = 0
    ∂G/∂h , ∂G/∂k

    ∂G/∂h = df
    ∂G/∂k = ∂(df(k)×h)/∂k

=#
function adjoint_Darcy_2D(darcy::Setup_Param{FT, IT}, κ_2d::Array{FT,2}, h_2d::Array{FT,2}) where {FT<:AbstractFloat, IT<:Int}
    Δx, N = darcy.Δx, darcy.N
    dG_dk = zeros(Float64, (N-2)^2, N^2)

    indx = IT[]
    indy = IT[]
    vals = FT[]
    
    𝓒 = Δx^2
    for iy = 2:N-1
        for ix = 2:N-1
            
            ixy = ind(darcy, ix, iy) 
            
            #top
            if iy == N-1
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒)

                dG_dk[ixy, ind_all(darcy, ix, iy)]   += h_2d[ix,iy]/2/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy+1)] += h_2d[ix,iy]/2/𝓒
            else
                #ft = -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0 * (h_2d[ix,iy+1] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy+1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy+1])/2.0/𝓒])
                
                dG_dk[ixy, ind_all(darcy, ix, iy)]   -= (h_2d[ix,iy+1] - h_2d[ix,iy])/2.0/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy+1)] -= (h_2d[ix,iy+1] - h_2d[ix,iy])/2.0/𝓒
            end
            
            #bottom
            if iy == 2
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals,  (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒)

                dG_dk[ixy, ind_all(darcy, ix, iy)]   += h_2d[ix,iy]/2/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy-1)] += h_2d[ix,iy]/2/𝓒
            else
                #fb = (κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0 * (h_2d[ix,iy] - h_2d[ix,iy-1])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix, iy-1)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix, iy-1])/2.0/𝓒])

                dG_dk[ixy, ind_all(darcy, ix, iy)]   += (h_2d[ix,iy] - h_2d[ix,iy-1])/2.0/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy-1)] += (h_2d[ix,iy] - h_2d[ix,iy-1])/2.0/𝓒
            end
            
            #right
            if ix == N-1
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (0 - h_2d[ix,iy])
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒)

                dG_dk[ixy, ind_all(darcy, ix+1, iy)] += h_2d[ix,iy]/2/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy)]   += h_2d[ix,iy]/2/𝓒
            else
                #fr = -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0 * (h_2d[ix+1,iy] - h_2d[ix,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix+1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix+1, iy])/2.0/𝓒])

                dG_dk[ixy, ind_all(darcy, ix+1, iy)]   -= (h_2d[ix+1,iy] - h_2d[ix,iy])/2.0/𝓒
                dG_dk[ixy, ind_all(darcy, ix, iy)]     -= (h_2d[ix+1,iy] - h_2d[ix,iy])/2.0/𝓒
            end  
            
            #left
            if ix == 2
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - 0)
                push!(indx, ixy)
                push!(indy, ixy)
                push!(vals, (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒)

                dG_dk[ixy, ind_all(darcy, ix, iy)]   += h_2d[ix,iy]/2/𝓒
                dG_dk[ixy, ind_all(darcy, ix-1, iy)] += h_2d[ix,iy]/2/𝓒
            else
                #fl = (κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0 * (h_2d[ix,iy] - h_2d[ix-1,iy])
                append!(indx, [ixy, ixy])
                append!(indy, [ixy, ind(darcy, ix-1, iy)])
                append!(vals, [(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒, -(κ_2d[ix, iy] + κ_2d[ix-1, iy])/2.0/𝓒])

                dG_dk[ixy, ind_all(darcy, ix, iy)]   += (h_2d[ix,iy] - h_2d[ix-1,iy])/2.0/𝓒
                dG_dk[ixy, ind_all(darcy, ix-1, iy)] += (h_2d[ix,iy] - h_2d[ix-1,iy])/2.0/𝓒
            end
            
        end
    end
    
    
    df = sparse(indx, indy, vals, (N-2)^2, (N-2)^2)
    
    return df, dG_dk
end

#=
Compute observation values
=#
function compute_obs(darcy::Setup_Param{FT, IT}, h_2d::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    # X---X(1)---X(2) ... X(obs_N)---X
    obs_2d = h_2d[darcy.x_locs, darcy.y_locs] 
    
    return obs_2d[:]
end

#=
Compute observation values
=#
function dcompute_obs(darcy::Setup_Param{FT, IT}, h_2d::Array{FT, 2}) where {FT<:AbstractFloat, IT<:Int}
    # X---X(1)---X(2) ... X(obs_N)---X
    dobs_dh = zeros(Float64,  length(darcy.x_locs), length(darcy.y_locs),  (N-2)^2)
    for i = 1:length(darcy.x_locs)
        for j = 1:length(darcy.y_locs)
            dobs_dh[i, j , ind(darcy, darcy.x_locs[i], darcy.y_locs[j])] = 1.0
        end
    end
    return reshape(dobs_dh, length(darcy.x_locs)*length(darcy.y_locs),  (N-2)^2)
end

function plot_field(darcy::Setup_Param{FT, IT}, u_2d::Array{FT, 2}, plot_obs::Bool,  filename::String = "None") where {FT<:AbstractFloat, IT<:Int}
    N = darcy.N
    xx = darcy.xx

    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    pcolormesh(X, Y, u_2d, cmap="viridis")
    colorbar()

    if plot_obs
        x_obs, y_obs = X[darcy.x_locs, darcy.y_locs][:], Y[darcy.x_locs, darcy.y_locs][:] 
        scatter(x_obs, y_obs, color="black")
    end

    tight_layout()
    if filename != "None"
        savefig(filename)
    end
end

function plot_field(darcy::Setup_Param{FT, IT}, u_2d::Array{FT, 2},  clim, ax) where {FT<:AbstractFloat, IT<:Int}
    N = darcy.N
    xx = darcy.xx
    X,Y = repeat(xx, 1, N), repeat(xx, 1, N)'
    return ax.pcolormesh(X, Y, u_2d, cmap="viridis", clim=clim)
end




function forward(darcy::Setup_Param{FT, IT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
    
    logκ_2d = compute_logκ_2d(darcy, θ)
    κ_2d = exp.(logκ_2d)
    
    h_2d = solve_Darcy_2D(darcy, κ_2d)
    
    y = compute_obs(darcy, h_2d)
    return y
end


function dforward(darcy::Setup_Param{FT, IT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
    logκ_2d = compute_logκ_2d(darcy, θ)

    κ_2d = exp.(logκ_2d)

    h_2d = solve_Darcy_2D(darcy, κ_2d)

    df, dG_dk = adjoint_Darcy_2D(darcy, κ_2d, h_2d)

    dobs_dh = dcompute_obs(darcy, h_2d)

    dκ_dθ = compute_dκ_dθ(darcy, θ)

    dh_dθ = -df'\(dG_dk* dκ_dθ)

    dobs_dθ = dobs_dh*dh_dθ

    return dobs_dθ
end


function aug_forward(darcy::Setup_Param{FT, IT}, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
  
    
    y = forward(darcy, θ)
    return [y ; θ]
end

function darcy_F(darcy::Setup_Param{FT, IT}, args, θ::Array{FT, 1}) where {FT<:AbstractFloat, IT<:Int}
  
    y_obs, r₀, ση, σ₀ = args
    Gθ  = forward(darcy, θ)
    return [(y_obs  - Gθ)./ση; (r₀ - θ)./σ₀]

end


