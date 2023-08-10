module Grids

export Location, Center, Vertex
export DiscreteAxis, spacing, Δ, origin, extent, coord, center, vertex, coords, centers, vertices

export CartesianGrid, axis
export xcoord, ycoord, zcoord, xcenter, ycenter, zcenter, xvertex, yvertex, zvertex
export xcoords, ycoords, zcoords, xcenters, ycenters, zcenters, xvertices, yvertices, zvertices

abstract type Location end

struct Center <: Location end
struct Vertex <: Location end

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

struct CartesianGrid{N,T<:AbstractFloat,I<:Integer}
    axes::NTuple{N,DiscreteAxis{T,I}}
end

CartesianGrid(origin::NTuple{N,T}, extent::NTuple{N,T}, size::NTuple{N,I}) where {N,T,I} = CartesianGrid(DiscreteAxis.(origin, extent, size))

CartesianGrid(; origin::NTuple{N,T}, extent::NTuple{N,T}, size::NTuple{N,I}) where {N,T,I} = CartesianGrid(origin, extent, size)

@pure Base.eltype(::CartesianGrid{N,T}) where {N,T} = T
@pure Base.ndims(::CartesianGrid{N}) where {N} = N

Base.size(grid::CartesianGrid) = length.(grid.axes)

@propagate_inbounds Base.size(grid::CartesianGrid, dim::Integer) = length(grid.axes[dim])
@propagate_inbounds Base.size(grid::CartesianGrid, loc::Location, dim::Integer) = length(grid.axes[dim], loc)

Base.size(grid::CartesianGrid{N}, locs::NTuple{N,Location}) where {N} = length.(grid.axes, locs)
Base.size(grid::CartesianGrid, loc::Location) = ntuple(D -> length(grid.axes[D], loc), Val(ndims(grid)))

axis(grid::CartesianGrid, dim::Integer) = grid.axes[dim]

origin(grid::CartesianGrid) = origin.(grid.axes)
extent(grid::CartesianGrid) = extent.(grid.axes)
spacing(grid::CartesianGrid) = spacing.(grid.axes)
Δ(grid::CartesianGrid) = spacing(grid)

@propagate_inbounds origin(grid::CartesianGrid, dim::Integer) = origin(grid.axes[dim])
@propagate_inbounds extent(grid::CartesianGrid, dim::Integer) = extent(grid.axes[dim])
@propagate_inbounds spacing(grid::CartesianGrid, dim::Integer) = spacing(grid.axes[dim])
@propagate_inbounds Δ(grid::CartesianGrid, dim::Integer) = spacing(grid.axes[dim])

@propagate_inbounds coord(grid::CartesianGrid{N}, loc::Location, inds::NTuple{N}) where {N} = coord.(grid.axes, Ref(loc), inds)

@propagate_inbounds function coord(grid::CartesianGrid{N}, loc::NTuple{N,Location}, inds::NTuple{N}) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        coord(grid.axes[I], loc[I], inds[I])
    end
end

coord(grid::CartesianGrid{N}, loc, I::CartesianIndex{N}) where {N} = coord(grid, loc, Tuple(I))

@propagate_inbounds coord(grid::CartesianGrid, loc::Location, dim::Integer, i::Integer) = coord(grid.axes[dim], loc, i)
@propagate_inbounds coord(grid::CartesianGrid{N}, loc::Location, dim::Integer, inds::NTuple{N}) where {N} = coord(grid.axes[dim], loc, inds[dim])
@propagate_inbounds coord(grid::CartesianGrid{N}, locs::NTuple{N,Location}, dim::Integer, inds::NTuple{N}) where {N} = coord(grid.axes[dim], locs[dim], inds[dim])
@propagate_inbounds coord(grid::CartesianGrid{N}, locs::NTuple{N,Location}, dim::Integer, i::Integer) where {N} = coord(grid.axes[dim], locs[dim], i)

@propagate_inbounds coord(grid::CartesianGrid, loc, ::Val{D}, i) where {D} = coord(grid, loc, D, i)

center(grid::CartesianGrid{N}, inds::NTuple{N}) where {N} = center.(grid.axes, inds)
vertex(grid::CartesianGrid{N}, inds::NTuple{N}) where {N} = vertex.(grid.axes, inds)

center(grid::CartesianGrid{N}, I::CartesianIndex{N}) where {N} = center(grid, Tuple(I))
vertex(grid::CartesianGrid{N}, I::CartesianIndex{N}) where {N} = vertex(grid, Tuple(I))

@propagate_inbounds center(grid::CartesianGrid, dim::Integer, i) = coord(grid, Center(), dim, i)
@propagate_inbounds center(grid::CartesianGrid, ::Val{D}, i) where {D} = coord(grid, Center(), D, i)

@propagate_inbounds vertex(grid::CartesianGrid, dim::Integer, i) = coord(grid, Vertex(), dim, i)
@propagate_inbounds vertex(grid::CartesianGrid, ::Val{D}, i) where {D} = coord(grid, Vertex(), D, i)

xcoord(grid::CartesianGrid, loc, i) = coord(grid, loc, Val(1), i)
ycoord(grid::CartesianGrid, loc, i) = coord(grid, loc, Val(2), i)
zcoord(grid::CartesianGrid, loc, i) = coord(grid, loc, Val(3), i)

xcenter(grid::CartesianGrid, i) = center(grid, Val(1), i)
ycenter(grid::CartesianGrid, i) = center(grid, Val(2), i)
zcenter(grid::CartesianGrid, i) = center(grid, Val(3), i)

xvertex(grid::CartesianGrid, i) = vertex(grid, Val(1), i)
yvertex(grid::CartesianGrid, i) = vertex(grid, Val(2), i)
zvertex(grid::CartesianGrid, i) = vertex(grid, Val(3), i)

coords(grid::CartesianGrid, loc::Location, dim::Integer; kwargs...) = coords(grid.axes[dim], loc; kwargs...)
coords(grid::CartesianGrid, loc::Location, ::Val{D}; kwargs...) where {D} = coords(grid.axes[D], loc; kwargs...)

@inline function coords(grid::CartesianGrid{N}, locs::NTuple{N,Location}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        isnothing(halos) ? coords(grid, locs[I], Val(I)) : coords(grid, locs[I], Val(I); halo=halos[I])
    end
end

xcoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(1); kwargs...)
ycoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(2); kwargs...)
zcoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(3); kwargs...)

centers(grid::CartesianGrid, dim::Integer; kwargs...) = centers(grid.axes[dim]; kwargs...)
centers(grid::CartesianGrid, ::Val{D}; kwargs...) where {D} = centers(grid.axes[D]; kwargs...)

@inline function centers(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        isnothing(halos) ? centers(grid, Val(I)) : centers(grid, Val(I); halo=halos[I])
    end
end

xcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(1); kwargs...)
ycenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(2); kwargs...)
zcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(3); kwargs...)

vertices(grid::CartesianGrid, dim::Integer; kwargs...) = vertices(grid.axes[dim]; kwargs...)
vertices(grid::CartesianGrid, ::Val{D}; kwargs...) where {D} = vertices(grid.axes[D]; kwargs...)

@inline function vertices(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        isnothing(halos) ? vertices(grid, Val(I)) : vertices(grid, Val(I); halo=halos[I])
    end
end

xvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(1); kwargs...)
yvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(2); kwargs...)
zvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(3); kwargs...)

end