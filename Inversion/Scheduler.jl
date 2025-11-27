function cos_annealing(iter::IT, N_iter::IT, η_min::FT, η_max::FT) where {FT<:AbstractFloat, IT<:Int}
    return η_min + 0.5 * (η_max - η_min) * (1 + cos(pi * iter / N_iter))
end

# Warmup–Stable–Decay (WSD) Scheduler
function stable_cos_decay(iter::IT, N_iter::IT, η_min::FT, η_max::FT; N_decay = 0.5*N_iter) where {FT<:AbstractFloat, IT<:Int}
    return iter <= N_iter - N_decay  ?  η_max  :  η_min + 0.5 * (η_max - η_min) * (1 + cos(pi * (iter - (N_iter - N_decay)) / N_decay))
end

# Warmup–Stable–Decay (WSD) Scheduler
function stable_linear_decay(iter::IT, N_iter::IT, η_min::FT, η_max::FT; N_decay = 0.5*N_iter) where {FT<:AbstractFloat, IT<:Int}
    return iter <= N_iter - N_decay  ?  η_max  :  η_max - (η_max - η_min) * (iter - (N_iter - N_decay)) / N_decay
end

# Exponential Scheduler
function exponential_decay(iter::IT, N_iter::IT, η_min::FT, η_max::FT) where {FT<:AbstractFloat, IT<:Int}
    c = log(η_max / η_min) / (N_iter - 1)
    return η_min * exp(c * (N_iter - iter)) 
end

function scheduler(iter, N_iter; η_min = 0.1, η_max = 1.0, scheduler_type = "stable_cos_decay")
    if scheduler_type == "constant"
        return η_max
    elseif scheduler_type == "stable_cos_decay"
        return stable_cos_decay(iter, N_iter, η_min, η_max)
    elseif scheduler_type == "cos_annealing"
        return cos_annealing(iter, N_iter, η_min, η_max)
    elseif scheduler_type == "stable_linear_decay"
        return stable_linear_decay(iter, N_iter, η_min, η_max)
    elseif scheduler_type == "exponential_decay"
        return exponential_decay(iter, N_iter, η_min, η_max)
    end
end
