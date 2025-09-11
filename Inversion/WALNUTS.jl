using LinearAlgebra, Random, StatsBase

function H(θ, p, U_func)
    return U_func(θ) + 0.5 * p' * p
end

function leapfrog(θ, p, ∇U, h)
    p_half = p - (h/2) * ∇U(θ)
    θ_new = θ + h * p_half
    p_new = p_half - (h/2) * ∇U(θ_new)
    return θ_new, p_new
end

# micro 函数 
function micro(θ, p, U_func, ∇U_func, h, δ)
    for i in 0:10 
        l = 2^i
        h_micro = h / l
        θ_curr, p_curr = θ, p
        H_init = H(θ_curr, p_curr, U_func)
        H_max, H_min = H_init, H_init

        diverged = false
        for _ in 1:l
            θ_curr, p_curr = leapfrog(θ_curr, p_curr, ∇U_func, h_micro)
            if !all(isfinite, θ_curr) || !all(isfinite, p_curr)
                diverged = true
                break
            end
            H_curr = H(θ_curr, p_curr, U_func)
            H_max = max(H_max, H_curr)
            H_min = min(H_min, H_curr)
        end
        
        if diverged continue end
        if (H_max - H_min) <= δ
            return l
        end
    end
    return 2^11 
end

# P_micro 分布 (R2P)
function P_micro_R2P(l_sample, l_bar)
    if l_sample == l_bar
        return 2/3
    elseif l_sample == 2*l_bar
        return 1/3
    else
        return 0.0
    end
end

# 扩展轨道 
function extend_orbit(start_θ, start_p, start_w, U_func, ∇U_func, h, δ, L, direction)
    ext_orbit_θ = Vector{Vector{Float64}}(undef, L)
    ext_orbit_p = Vector{Vector{Float64}}(undef, L)
    ext_weights = zeros(L)
    
    prev_θ, prev_p, prev_w = start_θ, start_p, start_w

    for i in 1:L
        # 1. 找到最优微步数 l̄
        l_bar_fwd = micro(prev_θ, prev_p, U_func, ∇U_func, h, δ)
        
        # 2. 从 P_micro 分布中采样 l
        l = rand() < 2/3 ? l_bar_fwd : 2 * l_bar_fwd
        
        # 3. 走 l 步
        h_micro = h / l
        curr_θ, curr_p = prev_θ, prev_p
        for _ in 1:l
            curr_θ, curr_p = leapfrog(curr_θ, curr_p, ∇U_func, h_micro)
        end

        # 4. 计算反向的最优微步数 l̄_rev
        l_bar_rev = micro(curr_θ, -curr_p, U_func, ∇U_func, h, δ)

        # 5. 计算权重 (Formula 16)
        prob_fwd = P_micro_R2P(l, l_bar_fwd)
        prob_rev = P_micro_R2P(l, l_bar_rev)
        
        if prob_fwd == 0.0
            curr_w = 0.0 # 不可逆的路径，权重为0
        else
            H_prev = H(prev_θ, prev_p, U_func)
            H_curr = H(curr_θ, curr_p, U_func)
            if !all(isfinite, [H_prev, H_curr])
                curr_w = 0.0
            else
                curr_w = prev_w * exp(H_prev - H_curr) * (prob_rev / prob_fwd)
            end
        end
        
        ext_orbit_θ[i], ext_orbit_p[i], ext_weights[i] = curr_θ, curr_p, curr_w
        prev_θ, prev_p, prev_w = curr_θ, curr_p, curr_w
    end
    
    if direction == -1
        # 反向扩展，动量需要翻转
        ext_orbit_p = [-p for p in ext_orbit_p]
    end
    
    return ext_orbit_θ, ext_orbit_p, ext_weights
end

function check_U_turn(θ_minus, p_minus, θ_plus, p_plus)
    p_vec = θ_plus - θ_minus
    return (p_vec' * p_minus < 0) || (p_vec' * p_plus < 0)
end

# 它递归地检查一个轨道及其所有二分产生的子轨道是否满足U-turn条件。
function check_sub_U_turn(orbit_θ, orbit_p)
    # 基线条件: 轨道长度小于2，不可能形成U-turn
    if length(orbit_θ) < 2
        return false
    end
    
    # 检查整个当前轨道是否U-turn
    is_u_turn = check_U_turn(orbit_θ[1], orbit_p[1], orbit_θ[end], orbit_p[end])
    if is_u_turn
        return true
    end
    
    # 递归地检查左右两个子轨道
    # 因为轨道长度总是2的幂，所以可以精确地二分
    mid = length(orbit_θ) ÷ 2
    
    # 分割左子轨道和右子轨道
    left_θ, left_p = orbit_θ[1:mid], orbit_p[1:mid]
    right_θ, right_p = orbit_θ[mid+1:end], orbit_p[mid+1:end]
    
    # 只要任何一个子轨道U-turn，整个检查就为真
    return check_sub_U_turn(left_θ, left_p) || check_sub_U_turn(right_θ, right_p)
end


# --- WALNUTS 主函数  ---
function Run_WALNUTS(θ_init, U_func, ∇U_func; N_iter=1000, h=0.1, δ=0.5, max_depth=10)
    N_θ = length(θ_init)
    samples = zeros(N_θ, N_iter)
    
    θ_current = copy(θ_init)

    for i in 1:N_iter
        p0 = randn(N_θ)

        # 初始化轨道和权重
        orbit_θ = [θ_current]
        orbit_p = [p0]
        weights = [exp(-H(θ_current, p0, U_func))]
        
        # 初始化有偏采样器状态
        θ_progressive_sample = θ_current
        
        # 树的构建
        for j in 0:(max_depth-1)
            direction = rand([-1, 1])
            
            O_old_θ, O_old_p, W_old = orbit_θ, orbit_p, weights
            
            start_θ, start_p = (direction == 1) ? (O_old_θ[end], O_old_p[end]) : (O_old_θ[1], -O_old_p[1])
            start_w = (direction == 1) ? W_old[end] : W_old[1]
            
            # 扩展轨道
            L = 2^j
            O_ext_θ, O_ext_p, W_ext = extend_orbit(start_θ, start_p, start_w, U_func, ∇U_func, h, δ, L, direction)
            
            # 合并轨道
            if direction == 1
                orbit_θ = [O_old_θ; O_ext_θ]
                orbit_p = [O_old_p; O_ext_p]
                weights = [W_old; W_ext]
            else
                orbit_θ = [O_ext_θ; O_old_θ]
                orbit_p = [O_ext_p; O_old_p]
                weights = [W_ext; W_old]
            end
            
            # 检查扩展出的新子树（O_ext）是否包含任何U-turn
            if check_sub_U_turn(O_ext_θ, O_ext_p)
                break
            end
            
            # 有偏渐进式采样 (Biased Progressive Sampling)
            sum_W_old = sum(W_old)
            sum_W_ext = sum(W_ext)
            if sum_W_ext > 0 && rand() < sum_W_ext / (sum_W_old + sum_W_ext)
                # 从新扩展的轨道中采样
                sample_idx = sample(1:length(W_ext), StatsBase.Weights(W_ext))
                θ_progressive_sample = O_ext_θ[sample_idx]
            end
            
            # 检查整个轨道的U-Turn
            if check_U_turn(orbit_θ[1], orbit_p[1], orbit_θ[end], orbit_p[end])
                break
            end
        end
        
        θ_current = θ_progressive_sample
        samples[:, i] = θ_current
    end
    
    return samples
end
