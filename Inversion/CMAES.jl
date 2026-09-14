# CMA-ES wrapper using the official pycma library (https://github.com/CMA-ES/pycma).
# Requires: using PyCall; pyimport_conda("cma", "cma")

using PyCall
using LinearAlgebra

const _cma = Ref{Any}(nothing)

function _get_cma()
    if _cma[] === nothing
        _cma[] = pyimport("cma")
    end
    return _cma[]
end

function run_cmaes(
    objective::Function,      # θ → scalar (to minimize)
    x0::Vector{Float64};      # initial mean
    sigma0::Float64=1.0,      # initial step size
    max_iter::Int=500,
    popsize::Union{Int,Nothing}=nothing,
    seed::Int=42,
)
    cma = _get_cma()
    N = length(x0)

    # Build options dict for CMAEvolutionStrategy
    opts = Dict{String, Any}(
        "maxiter" => max_iter,
        "seed" => seed,
        "verbose" => -9,       # silent
    )
    if popsize !== nothing
        opts["popsize"] = popsize
    end

    # Convert x0 and sigma0 to Python floats (pycma expects scalars, not numpy arrays)
    x0_py = PyObject(Float64[x0[i] for i in 1:N])
    sigma0_py = Float64(sigma0)

    # Create CMA-ES instance
    es = cma.CMAEvolutionStrategy(x0_py, sigma0_py, opts)

    means_history = [copy(x0)]
    best_x = copy(x0)
    best_f = objective(copy(x0))
    best_history = [copy(x0)]
    best_f_history = [best_f]
    covariance_history = [sqrt(Float64(N)) * sigma0^2]

    iter = 0
    stop_dict = es.stop()
    while iter < max_iter && length(stop_dict) == 0
        # Ask for new candidate solutions (returns list of numpy arrays)
        X_py = es.ask()
        npop = length(X_py)
        # PyCall auto-converts numpy arrays to 1-indexed Julia arrays
        X = [Float64[Float64(X_py[i][j]) for j in 1:N] for i in 1:npop]

        # Population covariance norm, used by CompareFourOptimizers-style plots.
        Xmat = hcat(X...)
        Xmean = vec(mean(Xmat, dims=2))
        Z = (Xmat .- Xmean) ./ sqrt(max(npop - 1, 1))
        push!(covariance_history, norm(Z * Z'))

        # Evaluate
        fitness = [objective(x) for x in X]

        # Tell CMA-ES the results (pass back numpy arrays)
        es.tell(X_py, fitness)

        # Record current mean (auto-converted to Julia array)
        mean_arr = es.mean
        current_mean = Float64[Float64(mean_arr[j]) for j in 1:N]
        push!(means_history, current_mean)

        # Track best-so-far
        min_idx = argmin(fitness)
        if fitness[min_idx] < best_f
            best_f = fitness[min_idx]
            best_x = copy(X[min_idx])
        end
        push!(best_history, copy(best_x))
        push!(best_f_history, best_f)

        iter += 1
        stop_dict = es.stop()
    end

    return (means=means_history, best_history=best_history,
            best_x=best_x, best_f=best_f, best_f_history=best_f_history,
            covariance_history=covariance_history)
end
