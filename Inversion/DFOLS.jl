# DFO-LS wrapper using the Python dfols library (https://numericalalgorithmsgroup.github.io/dfols)
# Requires: using PyCall; install dfols in the Python used by PyCall, or in the
# project .venv as a fallback.

using PyCall
using LinearAlgebra

const _dfols = Ref{Any}(nothing)

# Project root is one level up from this file: Inversion/DFOLS.jl -> project root.
# Resolve the Python version-specific site-packages path automatically (if .venv exists).
const _DFOLS_SITE = let
    libdir = normpath(joinpath(@__DIR__, "..", ".venv", "lib"))
    if !isdir(libdir)
        nothing
    else
        pydirs = filter(d -> startswith(d, "python"), readdir(libdir))
        isempty(pydirs) ? nothing : joinpath(libdir, sort(pydirs)[end], "site-packages")
    end
end

function _get_dfols()
    if _dfols[] === nothing
        # First try the Python environment that PyCall is actually using.
        # On clusters this is usually the conda env configured for PyCall.
        try
            _dfols[] = pyimport("dfols")
        catch
            # Fallback: use the project-local .venv if it exists.
            if _DFOLS_SITE !== nothing && isdir(_DFOLS_SITE)
                py_sys = pyimport("sys")
                if !(_DFOLS_SITE in py_sys.path)
                    pushfirst!(PyVector(py_sys."path"), _DFOLS_SITE)
                end
                _dfols[] = pyimport("dfols")
            else
                rethrow()
            end
        end
    end
    return _dfols[]
end

"""
    run_dfols(residuals, x0; kwargs...)

Julia-friendly wrapper around Python DFO-LS.

`residuals(x)` should return a vector of residuals `[r1(x), ..., rm(x)]`.
The solver minimizes `sum(residuals(x).^2)` without needing derivatives.

Returns a NamedTuple with fields:
- `x`: best point found
- `resid`: residual vector at `x`
- `obj`: objective value `sum(resid.^2)`
- `jacobian`: approximate Jacobian (or `nothing`)
- `nf`: number of objective evaluations
- `flag`: exit flag
- `msg`: exit message
"""
function run_dfols(
    residuals::Function,
    x0::Vector{Float64};
    bounds::Union{Tuple,Nothing}=nothing,
    maxfun::Union{Int,Nothing}=nothing,
    rhobeg::Union{Real,Nothing}=nothing,
    rhoend::Real=1e-8,
    npt::Union{Int,Nothing}=nothing,
    objfun_has_noise::Bool=false,
    scaling_within_bounds::Bool=false,
    user_params::Union{Dict,Nothing}=nothing,
    save_history::Bool=false,
    kwargs...,
)
    dfols = _get_dfols()

    # Merge user parameters with diagnostic logging if history is requested.
    up = Dict{String,Any}()
    if user_params !== nothing
        for (k, v) in user_params
            up[String(k)] = v
        end
    end
    if save_history
        up["logging.save_diagnostic_info"] = true
        up["logging.save_xk"] = true
        up["logging.save_rk"] = true
    end

    # dfols.solve expects a Python function that returns a residual vector.
    # Convert the incoming numpy array to a Julia vector before calling user function.
    function resid_py(x)
        xj = Float64.(x)
        r = residuals(xj)
        return collect(Float64, r)
    end

    sol = dfols.solve(
        resid_py,
        x0;
        bounds=bounds,
        maxfun=maxfun,
        rhobeg=rhobeg === nothing ? nothing : Float64(rhobeg),
        rhoend=Float64(rhoend),
        npt=npt,
        objfun_has_noise=objfun_has_noise,
        scaling_within_bounds=scaling_within_bounds,
        user_params=isempty(up) ? nothing : up,
        kwargs...,
    )

    x = sol.x === nothing ? nothing : Float64.(sol.x)
    resid = sol.resid === nothing ? nothing : Float64.(sol.resid)
    jacobian = sol.jacobian === nothing ? nothing : Float64.(sol.jacobian)
    history = save_history ? _extract_dfols_history(sol) : nothing

    return (
        x=x,
        resid=resid,
        obj=Float64(sol.obj),
        jacobian=jacobian,
        nf=Int(sol.nf),
        flag=Int(sol.flag),
        msg=String(sol.msg),
        history=history,
    )
end

function _extract_dfols_history(sol)
    try
        diag = sol.diagnostic_info
        diag === nothing && return nothing

        xcol = diag["xk"]
        rcol = diag["rk"]
        fcol = diag["fk"]
        n = length(xcol)

        xhist = Vector{Vector{Float64}}(undef, n)
        rhist = Vector{Vector{Float64}}(undef, n)
        fhist = Vector{Float64}(undef, n)

        for i in 1:n
            xhist[i] = Float64.(xcol[i])
            rhist[i] = Float64.(rcol[i])
            fhist[i] = Float64(fcol[i])
        end

        return (x=xhist, resid=rhist, obj=fhist)
    catch err
        @warn "Could not extract DFO-LS diagnostic history; using final point only" err
        return nothing
    end
end
