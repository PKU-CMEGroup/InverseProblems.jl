using LinearAlgebra, Random, Distributions, Printf

export Run_NUTS_with_warmup

"""
    leapfrog(θ, r, ∇U, ϵ)

执行一步Leapfrog积分。 U是势能 (-logprob)。
"""
function leapfrog(θ::Vector, r::Vector, ∇U, ϵ::Float64)
    ∇θ = ∇U(θ)
    if any(!isfinite, ∇θ)
        return θ, r, false
    end

    r_new = r - 0.5 * ϵ * ∇θ
    θ_new = θ + ϵ * r_new

    ∇θ_new = ∇U(θ_new)
    if any(!isfinite, ∇θ_new)
        return θ_new, r_new, false
    end
    
    r_new = r_new - 0.5 * ϵ * ∇θ_new
    
    return θ_new, r_new, true
end

"""
    find_reasonable_epsilon(θ_init, U, ∇U; initial_epsilon=1.0)

寻找一个合理的初始步长 `ϵ`。
"""
function find_reasonable_epsilon(θ_init::Vector, U, ∇U; initial_epsilon=1.0)
    ϵ = initial_epsilon
    d = length(θ_init)
    r = randn(d)

    U_θ = U(θ_init)
    if !isfinite(U_θ) return 1e-3 end

    current_H = -U_θ - 0.5 * dot(r, r)

    θ_new, r_new, success = leapfrog(θ_init, r, ∇U, ϵ)

    if !success || any(!isfinite, θ_new) || any(!isfinite, r_new)
        for _ in 1:10
            ϵ *= 0.1
            if ϵ < 1e-9 break end
            θ_new, r_new, success = leapfrog(θ_init, r, ∇U, ϵ)
            if success && all(isfinite, θ_new) && all(isfinite, r_new) break end
        end
        if !success return 1e-3 end
    end

    U_θ_new = U(θ_new)
    if !isfinite(U_θ_new) return ϵ / 2.0 end

    new_H = -U_θ_new - 0.5 * dot(r_new, r_new)

    if !isfinite(new_H) || !isfinite(current_H) return 1e-3 end
    
    p_accept = exp(new_H - current_H)
    a = p_accept > 0.5 ? 1.0 : -1.0
    
    count = 0
    max_counts = 20
    while count < max_counts
        ϵ_cand = ϵ * (2.0^a)
        if !isfinite(ϵ_cand) || ϵ_cand <= 1e-10 || ϵ_cand > 1e2 break end
        ϵ = ϵ_cand

        θ_new, r_new, success = leapfrog(θ_init, r, ∇U, ϵ)
        if !success || any(!isfinite, θ_new) || any(!isfinite, r_new)
            ϵ /= (2.0^a); break
        end

        U_θ_new = U(θ_new)
        if !isfinite(U_θ_new) ϵ /= (2.0^a); break end

        new_H = -U_θ_new - 0.5 * dot(r_new, r_new)
        if !isfinite(new_H) ϵ /= (2.0^a); break end

        p_accept_check = exp(new_H - current_H)
        if a == 1.0 && p_accept_check < 0.5 break end
        if a == -1.0 && p_accept_check > 0.5 break end
        count += 1
    end
    
    return ϵ
end

"""
    build_tree(θ, r, logu, v, j, ϵ, U, ∇U)

为NUTS递归地构建二叉树。
"""
function build_tree(θ, r, logu, v, j, ϵ, U, ∇U)
    if j == 0
        θ_new, r_new, success = leapfrog(θ, r, ∇U, v * ϵ)
        if !success
            return θ, r, θ, r, θ, 0, false, 0.0, 0
        end

        U_val_new = U(θ_new)
        if !isfinite(U_val_new)
            return θ, r, θ, r, θ, 0, false, 0.0, 0
        end

        hamiltonian_new = -U_val_new - 0.5 * dot(r_new, r_new)
        if !isfinite(hamiltonian_new)
             return θ, r, θ, r, θ, 0, false, 0.0, 0
        end

        n_valid = (hamiltonian_new >= logu) ? 1 : 0
        s_valid = (hamiltonian_new >= logu - 1000)

        U_val_current = U(θ)
        hamiltonian_current = -U_val_current - 0.5 * dot(r, r)
        
        α = 0.0
        if isfinite(hamiltonian_current) && isfinite(hamiltonian_new)
            α = min(1.0, exp(hamiltonian_new - hamiltonian_current))
        end
        
        return θ_new, r_new, θ_new, r_new, θ_new, n_valid, s_valid, α, 1
    else
        θ_minus, r_minus, θ_plus, r_plus, θ_prime, n_prime, s_prime, α_prime, nα_prime = 
            build_tree(θ, r, logu, v, j - 1, ϵ, U, ∇U)

        if s_prime
            local θ_dprime, n_dprime, s_dprime, α_dprime, nα_dprime
            if v == -1
                θ_minus_subtree, r_minus_subtree, _, _, θ_dprime, n_dprime, s_dprime, α_dprime, nα_dprime = 
                    build_tree(θ_minus, r_minus, logu, v, j - 1, ϵ, U, ∇U)
                if s_dprime
                    θ_minus, r_minus = θ_minus_subtree, r_minus_subtree
                end
            else # v == 1
                _, _, θ_plus_subtree, r_plus_subtree, θ_dprime, n_dprime, s_dprime, α_dprime, nα_dprime =
                    build_tree(θ_plus, r_plus, logu, v, j - 1, ϵ, U, ∇U)
                if s_dprime
                    θ_plus, r_plus = θ_plus_subtree, r_plus_subtree
                end
            end

            if s_dprime
                if (n_prime + n_dprime) > 0 && rand() < n_dprime / (n_prime + n_dprime)
                    θ_prime = θ_dprime
                end
                α_prime += α_dprime
                nα_prime += nα_dprime
                n_prime += n_dprime
            end
            
            s_prime = s_dprime && 
                      (dot(θ_plus - θ_minus, r_minus) >= 0) &&
                      (dot(θ_plus - θ_minus, r_plus) >= 0)
        end

        return θ_minus, r_minus, θ_plus, r_plus, θ_prime, n_prime, s_prime, α_prime, nα_prime
    end
end


"""
    Run_NUTS_with_warmup(U, ∇U, θ_init; N_iter, N_warmup, δ_target)
"""
function Run_NUTS_with_warmup(U, ∇U, θ_init::Vector; N_iter=2000, N_warmup=1000, δ_target=0.65)
    γ = 0.05; t₀ = 10.0; κ = 0.75
    θ = copy(θ_init); d = length(θ)
    
    ϵ = find_reasonable_epsilon(θ, U, ∇U; initial_epsilon=0.1)
    if !isfinite(ϵ) || ϵ <= 1e-9
        @warn "NUTS: find_reasonable_epsilon returned invalid value ($ϵ). Resetting to 0.01."
        ϵ = 0.01
    end

    μ = log(10 * ϵ); H̄ = 0.0; ϵ̄ = 1.0
    samples = zeros(d, N_iter)
    

    for m in 1:N_iter
        if m % 5000 == 0 && m > 1 @info "NUTS iter = $m / $N_iter" end

        if any(!isfinite, θ)
            @error "NUTS Error: θ became non-finite at iter $m. Aborting."
            if m > 1 samples[:, m:end] .= samples[:, m-1] end
            break
        end
        if !isfinite(ϵ) || ϵ <= 1e-9 || ϵ > 1e2
            @warn "NUTS Warning: Epsilon invalid ($ϵ) at iter $m. Resetting."
            ϵ = 0.01; ϵ̄ = 1.0; μ = log(10*ϵ); H̄ = 0.0
        end
        
        r₀ = randn(d)
        U_current = U(θ)
        
        logu = -Inf
        if isfinite(U_current)
            hamiltonian_current = -U_current - 0.5 * dot(r₀, r₀)
            if isfinite(hamiltonian_current)
                 logu = hamiltonian_current - rand(Exponential(1.0))
            end
        end

        θ_minus, θ_plus = copy(θ), copy(θ)
        r_minus, r_plus = copy(r₀), copy(r₀)
        θ_proposal = copy(θ)
        
        j = 0; n_valid = 1; s_terminate = true
        α_tree, nα_tree = 0.0, 0

        while s_terminate && j < 10
            v = rand([-1, 1])
            
            if v == -1
                tm_sub, rm_sub, _, _, tp_cand, nv_sub, ss_sub, α_sub, nα_sub = 
                    build_tree(θ_minus, r_minus, logu, v, j, ϵ, U, ∇U)
                if ss_sub
                    θ_minus, r_minus = tm_sub, rm_sub
                end
            else # v == 1
                _, _, tp_sub, rp_sub, tp_cand, nv_sub, ss_sub, α_sub, nα_sub = 
                    build_tree(θ_plus, r_plus, logu, v, j, ϵ, U, ∇U)
                if ss_sub
                    θ_plus, r_plus = tp_sub, rp_sub
                end
            end

            if ss_sub
                if nv_sub > 0 && rand() < min(1.0, nv_sub / n_valid)
                    θ_proposal = copy(tp_cand)
                end

                n_valid += nv_sub
                α_tree += α_sub
                nα_tree += nα_sub

                s_terminate = (dot(θ_plus - θ_minus, r_minus) >= 0) && (dot(θ_plus - θ_minus, r_plus) >= 0)
            else
                s_terminate = false
            end
            
            j += 1
        end

        samples[:, m] = θ_proposal
        θ = copy(θ_proposal)

        # Dual averaging adaptation
        if m <= N_warmup
            avg_α = (nα_tree > 0) ? α_tree / nα_tree : 0.0
            if !isfinite(avg_α) avg_α = 0.0 end
            
            H̄ = (1 - 1 / (m + t₀)) * H̄ + (1 / (m + t₀)) * (δ_target - avg_α)

            log_ϵ = μ - (sqrt(m) / γ) * H̄
            if !isfinite(log_ϵ) log_ϵ = log(ϵ̄ > 0 ? ϵ̄ : 0.01) end
            ϵ = exp(log_ϵ)
            if !isfinite(ϵ) || ϵ <= 1e-9 ϵ = 1e-9 end
            if ϵ > 1e2 ϵ = 1e2 end
            
            log_ϵ̄_current = log(ϵ̄ > 0 ? ϵ̄ : 0.01)
            log_ϵ̄ = m^(-κ) * log_ϵ + (1 - m^(-κ)) * log_ϵ̄_current
            ϵ̄ = exp(log_ϵ̄)
            if !isfinite(ϵ̄) || ϵ̄ <= 0 ϵ̄ = 0.01 end
        else
            ϵ = ϵ̄
        end
    end
    
    return samples
end