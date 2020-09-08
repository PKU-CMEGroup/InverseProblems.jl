using JLD2
using Statistics
using LinearAlgebra

include("../Plot.jl")
include("../LRRUKI.jl")
include("../REnKI.jl")

function run_linear_ensemble(params_i, G)
    
    N_ens,  N_θ = size(params_i)
    g_ens = Vector{Float64}[]
    
    for i = 1:N_ens
        θ = params_i[i, :]
        # g: N_ens x N_data
        push!(g_ens, G*θ) 
    end
    
    return hcat(g_ens...)'
end




function LRRUKI_Run(t_mean, t_cov, θ_bar, θθ_cov,  G,  N_r, α_reg::Float64,  N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    lrrukiobj = LRRUKIObj(parameter_names,
    N_r,
    θ_bar, 
    θθ_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    for i in 1:N_iter
        
        update_ensemble!(lrrukiobj, ens_func) 
        
    end
    
    return lrrukiobj
    
end


function EnKI_Run(filter_type, t_mean, t_cov, θ0_bar, θθ0_cov,  G,  N_ens, α_reg::Float64, N_iter::Int64 = 100)
    parameter_names = ["θ"]
    
    ny, nθ = size(G)
    
    ens_func(θ_ens) = run_linear_ensemble(θ_ens, G)
    
    
    
    ekiobj = EnKIObj(filter_type, 
    parameter_names,
    N_ens, 
    θ0_bar,
    θθ0_cov,
    t_mean, # observation
    t_cov,
    α_reg)
    
    
    for i in 1:N_iter
        
        update_ensemble!(ekiobj, ens_func) 
        
    end
    
    return ekiobj
    
end


function Linear_Test(filter_type::String, problem_type::String, nθ::Int64 = 10, N_r::Int64 = 1, α_reg::Float64 = 1.0, N_ite::Int64 = 1000)
    
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(10.0^2, nθ)))     # standard deviation
    
    θθ0_cov = ones(Float64,  nθ, nθ) * 100 + Array(Diagonal(fill(1.0^2, nθ))) 
    
    G = zeros(Float64, nθ, nθ)
    if problem_type == "Hilbert"
        
        for i = 1:nθ
            for j = 1:nθ
                G[i,j] = 1/(i + j - 1)
            end
        end
        
    elseif problem_type == "Poisson_1D"
        for i = 1:nθ
            for j = 1:nθ
                if i == j
                    G[i,j] = 2
                elseif i == j-1 || i == j+1
                    G[i,j] = -1
                end
            end
        end
    elseif problem_type == "Identity"
        G = I
    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
    @info "cond(G) = " , cond(G)
    θ_ref = fill(1.0, nθ)
    t_mean = G*θ_ref 
    t_cov = Array(Diagonal(fill(0.5^2, nθ)))

    
    enkiobj = EnKI_Run(filter_type, t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    # error("stop enkiobj")
    lrrukiobj = LRRUKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_r, α_reg, N_ite)
    
    
    # Plot
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (2, N_ite+1))
    for i = 1:N_ite+1
        errors[1, i] = norm(lrrukiobj.θ_bar[i] .- 1.0)
        
        θ_bar = dropdims(mean(enkiobj.θ[i], dims=1), dims=1)
        errors[2, i] = norm(θ_bar .- 1.0)
        
    end

    errors ./= norm(ones(size(lrrukiobj.θ_bar[1], 1)))
    
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=50, label= "LRRUKI")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=50, label= "EnKI")
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    #ylim((0.1,15))
    grid("on")
    legend()
    tight_layout()
    
    savefig("Linear-"*string(nθ)*".pdf")
    close("all")
    
    return lrrukiobj
end


##########################
function Linear_Test(problem_type::String, nθ::Int64 = 10, N_r::Int64 = 1, α_reg::Float64 = 1.0, N_ite::Int64 = 1000)
    
    θ0_bar = zeros(Float64, nθ)  # mean 
    θθ0_cov = Array(Diagonal(fill(10.0^2, nθ)))     # standard deviation
    
    # θθ0_cov = ones(Float64,  nθ, nθ) * 100 + Array(Diagonal(fill(1.0^2, nθ))) 
    
    G = zeros(Float64, nθ, nθ)
    if problem_type == "Hilbert"
        
        for i = 1:nθ
            for j = 1:nθ
                G[i,j] = 1/(i + j - 1)
            end
        end
        
    elseif problem_type == "Poisson_1D"
        for i = 1:nθ
            for j = 1:nθ
                if i == j
                    G[i,j] = 2
                elseif i == j-1 || i == j+1
                    G[i,j] = -1
                end
            end
        end
    elseif problem_type == "Identity"
        G = I
    else
        error("Problem type : ", problem_type, " has not implemented!")
    end
    @info "cond(G) = " , cond(G)
    θ_ref = fill(1.0, nθ)
    t_mean = G*θ_ref 
    t_cov = Array(Diagonal(fill(1.0, nθ)))

    
    enkiobj = EnKI_Run("EnKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    eakiobj = EnKI_Run("EAKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    etkiobj = EnKI_Run("ETKI", t_mean, t_cov, θ0_bar, θθ0_cov, G, 2N_r+1, α_reg, N_ite)
    # error("stop enkiobj")
    # lrrukiobj = LRRUKI_Run(t_mean, t_cov, θ0_bar, θθ0_cov, G, N_r, α_reg, N_ite)
    
    
    # Plot
    ites = Array(LinRange(1, N_ite+1, N_ite+1))
    errors = zeros(Float64, (3, N_ite+1))
    for i = 1:N_ite+1
        θ_bar = dropdims(mean(enkiobj.θ[i], dims=1), dims=1)
        errors[1, i] = norm(θ_bar .- 1.0)

        θ_bar = dropdims(mean(eakiobj.θ[i], dims=1), dims=1)
        errors[2, i] = norm(θ_bar .- 1.0)

        θ_bar = dropdims(mean(etkiobj.θ[i], dims=1), dims=1)
        errors[3, i] = norm(θ_bar .- 1.0)


        # errors[1, i] = norm(lrrukiobj.θ_bar[i] .- 1.0)
        
        
        
    end

    errors ./= norm(θ_ref)
    
    
    semilogy(ites, errors[1, :], "--o", fillstyle="none", markevery=50, label= "EnKI")
    semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=50, label= "EAKI")
    semilogy(ites, errors[3, :], "--o", fillstyle="none", markevery=50, label= "ETKI")
    # semilogy(ites, errors[2, :], "--o", fillstyle="none", markevery=50, label= "EnKI")
    xlabel("Iterations")
    ylabel("\$L_2\$ norm error")
    #ylim((0.1,15))
    grid("on")
    legend()
    tight_layout()
    
    savefig("Linear-"*string(nθ)*".pdf")
    close("all")
    
    # return lrrukiobj
end



problem_type = "Poisson_1D"  #"Identity" #"Poisson_1D" 
filter_type = "ETKI"
Linear_Test(problem_type, 200, 10, 1.0, 1000)

@info "Finished!"




