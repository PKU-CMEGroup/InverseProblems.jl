using PyPlot
if !isdefined(@__MODULE__, :PublicationPlotStyle)
    @eval module PublicationPlotStyle

    const TITLE_SIZE = 16
    const LABEL_SIZE = 14
    const TICK_SIZE = 13
    const LEGEND_SIZE = 14

    """Apply the common paper-figure typography, including colorbars and legends."""
    function apply!(fig)
        for ax in fig.axes
            ax.title.set_fontsize(TITLE_SIZE)
            ax.xaxis.label.set_fontsize(LABEL_SIZE)
            ax.yaxis.label.set_fontsize(LABEL_SIZE)
            ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE)
            ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)
            ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
            legend = ax.get_legend()
            if legend !== nothing
                for label in legend.get_texts()
                    label.set_fontsize(LEGEND_SIZE)
                end
            end
        end
        for legend in fig.legends, label in legend.get_texts()
            label.set_fontsize(LEGEND_SIZE)
        end
        for label in fig.texts
            label.set_fontsize(LABEL_SIZE)
        end
        return fig
    end

    end
end

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    mysize = 15
    font0 = Dict(
    "font.size" => mysize,
    "axes.labelsize" => mysize,
    "xtick.labelsize" => mysize,
    "ytick.labelsize" => mysize,
    "legend.fontsize" => mysize,
    )
merge!(rcParams, font0)



function Gaussian_1d(θ_mean::FT, θθ_cov::FT, Nx::IT, θ_min=nothing, θ_max=nothing) where {FT<:AbstractFloat, IT<:Int}
    # 1d Gaussian plot
    if θ_min === nothing
        θ_min = θ_mean - 5*sqrt(θθ_cov)
    end
    if θ_max === nothing
        θ_max = θ_mean + 5*sqrt(θθ_cov)
    end
    
    θ = Array(LinRange(θ_min, θ_max, Nx))

    ρθ = similar(θ)
    
    for ix = 1:Nx  
        Δθ = θ[ix] - θ_mean
        ρθ[ix] = exp(-0.5*(Δθ/θθ_cov*Δθ)) / (sqrt(2 * pi * θθ_cov))
    end
    return θ, ρθ 
end


function Gaussian_2d(θ_mean::Array{FT,1}, θθ_cov, Nx::IT, Ny::IT, x_min=nothing, x_max=nothing, y_min=nothing, y_max=nothing) where {FT<:AbstractFloat, IT<:Int}
    # 2d Gaussian plot
    
    
    if x_min === nothing 
        x_min = θ_mean[1] - 5*sqrt(θθ_cov[1,1])
    end
    if x_max === nothing 
        x_max = θ_mean[1] + 5*sqrt(θθ_cov[1,1])
    end
    if y_min === nothing 
        y_min = θ_mean[2] - 5*sqrt(θθ_cov[2,2])
    end
    if y_max === nothing 
        y_max = θ_mean[2] + 5*sqrt(θθ_cov[2,2])
    end
    
        
    xx = Array(LinRange(x_min, x_max, Nx))
    yy = Array(LinRange(y_min, y_max, Ny))
    
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'
    Z = zeros(FT, Nx, Ny)
    
    det_θθ_cov = det(θθ_cov)

    for ix = 1:Nx
        for iy = 1:Ny
            Δxy = [xx[ix] - θ_mean[1]; yy[iy] - θ_mean[2]]
            Z[ix, iy] = exp(-0.5*(Δxy'/θθ_cov*Δxy)) / (2 * pi * sqrt(det_θθ_cov))
        end
    end
    
    return X, Y, Z
    
end

