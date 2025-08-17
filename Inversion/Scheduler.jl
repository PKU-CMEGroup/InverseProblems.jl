function cos_annealing(iter::IT, N_iter::IT, η_min::FT, η_max::FT) where {FT<:AbstractFloat, IT<:Int}
    return η_min + 0.5 * (η_max - η_min) * (1 + cos(pi * iter / N_iter))
end

# Warmup–Stable–Decay (WSD) Scheduler
function stable_cos_decay(iter::IT, N_iter::IT, η_min::FT, η_max::FT; N_decay = 0.2*N_iter) where {FT<:AbstractFloat, IT<:Int}
    return iter <= N_iter - N_decay  ?  η_max  :  η_min + 0.5 * (η_max - η_min) * (1 + cos(pi * (iter - (N_iter - N_decay)) / N_decay))
end

# Warmup–Stable–Decay (WSD) Scheduler
function stable_linear_decay(iter::IT, N_iter::IT, η_min::FT, η_max::FT; N_decay = 0.2*N_iter) where {FT<:AbstractFloat, IT<:Int}
    return iter <= N_iter - N_decay  ?  η_max  :  η_max - (η_max - η_min) * (iter - (N_iter - N_decay)) / N_decay
end