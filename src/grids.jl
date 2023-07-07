module Grids

export Center, Vertex
export CartesianGrid, origin, extent, spacing
export coord, xcoord, ycoord, zcoord
export coords, xcoords, ycoords, zcoords
export centers, xcenters, ycenters, zcenters
export vertices, xvertices, yvertices, zvertices

abstract type Location end

struct Center <: Location end
struct Vertex <: Location end

struct CartesianGrid{N,T<:AbstractFloat,I<:Integer}
    origin::NTuple{N,T}
    extent::NTuple{N,T}
    size::NTuple{N,I}
end

import Base.@pure
import Base.@propagate_inbounds

@pure Base.eltype(::CartesianGrid{N,T}) where {N,T} = T
@pure Base.ndims(::CartesianGrid{N}) where {N} = N

@inline Base.size(grid::CartesianGrid) = grid.size

@propagate_inbounds Base.size(grid::CartesianGrid, dim::Integer) = grid.size[dim]

@propagate_inbounds Base.size(grid::CartesianGrid, ::Center, dim::Integer) = grid.size[dim]

@propagate_inbounds Base.size(grid::CartesianGrid{N,T,I}, ::Vertex, dim::Integer) where {N,T,I} = grid.size[dim] + oneunit(I)

@propagate_inbounds function Base.size(grid::CartesianGrid{N}, locs::NTuple{N,Location}) where N
    ntuple(Val(N)) do I
        Base.@_inline_meta
        size(grid, locs[I], I)
    end
end

@inline origin(grid::CartesianGrid) = grid.origin
@inline extent(grid::CartesianGrid) = grid.extent
@inline spacing(grid::CartesianGrid) = extent(grid) ./ size(grid)

@propagate_inbounds origin(grid::CartesianGrid, dim) = grid.origin[dim]
@propagate_inbounds extent(grid::CartesianGrid, dim) = grid.extent[dim]
@propagate_inbounds spacing(grid::CartesianGrid, dim) = extent(grid, dim) / size(grid, dim)

@propagate_inbounds coord(grid::CartesianGrid{N}, ::Center, inds::NTuple{N}) where N = origin(grid) .+ (inds .- eltype(grid)(0.5)) ./ size(grid, dim)

@propagate_inbounds coord(grid::CartesianGrid, ::Center, dim::Integer, i) = origin(grid, dim) + (i - eltype(grid)(0.5)) / size(grid, dim)
@propagate_inbounds coord(grid::CartesianGrid, ::Vertex, dim::Integer, i) = origin(grid, dim) + (i - eltype(grid)(1)) / size(grid, dim)

@propagate_inbounds coord(grid::CartesianGrid, loc::Location, ::Val{D}, i) where {D} = coord(grid, loc, D, i)

@inline xcoord(grid::CartesianGrid, loc::Location, i) = coord(grid, loc, Val(1), i)
@inline ycoord(grid::CartesianGrid, loc::Location, i) = coord(grid, loc, Val(2), i)
@inline zcoord(grid::CartesianGrid, loc::Location, i) = coord(grid, loc, Val(3), i)

function coords(grid::CartesianGrid, loc::Location, dim; halo=nothing)
    if isnothing(halo)
        halo = (0, 0)
    end

    if halo isa Integer
        halo = (halo, halo)
    end

    _coords(grid, loc, dim, halo)
end

@inline coords(grid::CartesianGrid, loc::Location, ::Val{D}; kwargs...) where D = coords(grid, loc, D; kwargs...)

@inline function _coords(grid::CartesianGrid, loc::Location, dim, halo::Tuple{Integer,Integer})
    start = coord(grid, loc, dim, 1 - halo[1])
    stop = coord(grid, loc, dim, size(grid, loc, dim) + halo[2])
    LinRange(start, stop, size(grid, loc, dim) + sum(halo))
end

@inline xcoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(1); kwargs...)
@inline ycoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(2); kwargs...)
@inline zcoords(grid::CartesianGrid, loc::Location; kwargs...) = coords(grid, loc, Val(3); kwargs...)

@inline centers(grid::CartesianGrid, dim; kwargs...) = coords(grid, Center(), dim; kwargs...)
@inline vertices(grid::CartesianGrid, dim; kwargs...) = coords(grid, Vertex(), dim; kwargs...)

@inline function centers(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        isnothing(halos) ? centers(grid, Val(I)) : centers(grid, Val(I); halo=halos[I])
    end
end

@inline function vertices(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        isnothing(halos) ? vertices(grid, Val(I)) : vertices(grid, Val(I); halo=halos[I])
    end
end

@inline xcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(1); kwargs...)
@inline ycenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(2); kwargs...)
@inline zcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(3); kwargs...)

@inline xvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(1); kwargs...)
@inline yvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(2); kwargs...)
@inline zvertices(grid::CartesianGrid; kwargs...) = vertices(grid, Val(3); kwargs...)

end