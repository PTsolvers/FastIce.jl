module Grids

export Center, Vertex
export CartesianGrid, origin, extent, spacing
export coord, xcoord, ycoord, zcoord
export coords, xcoords, ycoords, zcoords
export centers, xcenters, ycenters, zcenters
export faces, xfaces, yfaces, zfaces

struct Center end
struct Vertex end

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

@propagate_inbounds Base.size(grid::CartesianGrid, dim) = grid.size[dim]

@inline origin(grid::CartesianGrid) = grid.origin
@inline extent(grid::CartesianGrid) = grid.extent
@inline spacing(grid::CartesianGrid) = extent(grid) ./ size(grid)

@propagate_inbounds origin(grid::CartesianGrid, dim) = grid.origin[dim]
@propagate_inbounds extent(grid::CartesianGrid, dim) = grid.extent[dim]
@propagate_inbounds spacing(grid::CartesianGrid, dim) = extent(grid, dim) / size(grid, dim)

@propagate_inbounds coord(grid::CartesianGrid, ::Type{Center}, dim::Integer, i) = origin(grid, dim) + (i - eltype(grid)(0.5)) / size(grid, dim)
@propagate_inbounds coord(grid::CartesianGrid, ::Type{Vertex}, dim::Integer, i) = origin(grid, dim) + (i - eltype(grid)(1)) / size(grid, dim)

@propagate_inbounds coord(grid::CartesianGrid, ::Type{L}, ::Val{D}, i) where {L,D} = coord(grid, L, D, i)

@inline xcoord(grid::CartesianGrid, ::Type{L}, i) where {L} = coord(grid, L, Val(1), i)
@inline ycoord(grid::CartesianGrid, ::Type{L}, i) where {L} = coord(grid, L, Val(2), i)
@inline zcoord(grid::CartesianGrid, ::Type{L}, i) where {L} = coord(grid, L, Val(3), i)

function coords(grid::CartesianGrid, ::Type{L}, dim; halo=nothing) where {L}
    if isnothing(halo)
        halo = (0, 0)
    end

    if halo isa Integer
        halo = (halo, halo)
    end

    _coords(grid, L, dim, halo)
end

@inline coords(grid::CartesianGrid, ::Type{L}, ::Val{D}; kwargs...) where {L,D} = coords(grid, L, D; kwargs...)

@inline function _coords(grid::CartesianGrid, ::Type{L}, dim, halo::Tuple{Integer,Integer}) where {L}
    start = coord(grid, L, dim, 1 - halo[1])
    stop = coord(grid, L, dim, size(grid, dim) + halo[2])
    LinRange(start, stop, size(grid, dim) + sum(halo))
end

@inline xcoords(grid::CartesianGrid, ::Type{L}; kwargs...) where {L} = coords(grid, L, Val(1); kwargs...)
@inline ycoords(grid::CartesianGrid, ::Type{L}; kwargs...) where {L} = coords(grid, L, Val(2); kwargs...)
@inline zcoords(grid::CartesianGrid, ::Type{L}; kwargs...) where {L} = coords(grid, L, Val(3); kwargs...)

@inline centers(grid::CartesianGrid, dim; kwargs...) = coords(grid, Center, dim; kwargs...)
@inline faces(grid::CartesianGrid, dim; kwargs...) = coords(grid, Vertex, dim; kwargs...)

@inline function centers(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        if isnothing(halos)
            @inbounds centers(grid, Val(I))
        else
            @inbounds centers(grid, Val(I); halo=halos[I])
        end
    end
end

@inline function faces(grid::CartesianGrid{N}; halos=nothing) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        if isnothing(halos)
            @inbounds faces(grid, Val(I))
        else
            @inbounds faces(grid, Val(I); halo=halos[I])
        end
    end
end

@inline xcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(1); kwargs...)
@inline ycenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(2); kwargs...)
@inline zcenters(grid::CartesianGrid; kwargs...) = centers(grid, Val(3); kwargs...)

@inline xfaces(grid::CartesianGrid; kwargs...) = faces(grid, Val(1); kwargs...)
@inline yfaces(grid::CartesianGrid; kwargs...) = faces(grid, Val(2); kwargs...)
@inline zfaces(grid::CartesianGrid; kwargs...) = faces(grid, Val(3); kwargs...)

end