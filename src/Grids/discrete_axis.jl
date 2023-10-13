struct DiscreteAxis{T<:AbstractFloat,I<:Integer} <: AbstractRange{T}
    origin::T
    extent::T
    step::T
    len::I
    DiscreteAxis(origin::T, extent::T, len::I) where {T,I} = new{T,I}(origin, extent, extent / len, len)
end

import Base.@pure
import Base.@propagate_inbounds

@pure Base.eltype(::DiscreteAxis{T}) where {T} = T

Base.length(ax::DiscreteAxis) = ax.len
Base.length(ax::DiscreteAxis, ::Center) = ax.len
Base.length(ax::DiscreteAxis{T,I}, ::Vertex) where {T,I} = ax.len + oneunit(I)

Base.step(ax::DiscreteAxis) = ax.step

@propagate_inbounds Base.getindex(ax::DiscreteAxis{T}, i::Integer) where {T} = ax.origin + ax.step / 2 + (i - 1) * ax.step

spacing(ax::DiscreteAxis) = ax.step
Δ(ax::DiscreteAxis) = ax.step

origin(ax::DiscreteAxis) = ax.origin
origin(ax::DiscreteAxis, ::Vertex) = ax.origin
origin(ax::DiscreteAxis, ::Center) = ax.origin + ax.step / 2

extent(ax::DiscreteAxis) = ax.extent
extent(ax::DiscreteAxis, ::Vertex) = ax.extent
extent(ax::DiscreteAxis, ::Center) = ax.extent - ax.step

coord(ax::DiscreteAxis{T}, loc::Location, i::Integer) where {T} = origin(ax, loc) + (i - 1) * ax.step
center(ax, i::Integer) = coord(ax, Center(), i)
vertex(ax, i::Integer) = coord(ax, Vertex(), i)

@inline function coords(ax::DiscreteAxis, loc::Location; halo=nothing)
    if isnothing(halo)
        halo = (0, 0)
    end

    if halo isa Integer
        halo = (halo, halo)
    end

    return _coords(ax, loc, halo)
end

@inline function _coords(ax::DiscreteAxis, loc::Location, halo::Tuple{Integer,Integer})
    start = coord(ax, loc, 1 - halo[1])
    stop = coord(ax, loc, length(ax, loc) + halo[2])
    LinRange(start, stop, length(ax, loc) + sum(halo))
end

centers(ax::DiscreteAxis; kwargs...) = coords(ax, Center(); kwargs...)
vertices(ax::DiscreteAxis; kwargs...) = coords(ax, Vertex(); kwargs...)
