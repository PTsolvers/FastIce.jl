"""
Rectilinear grid with uniform spacing.

# Examples

```julia-repl
julia> grid = CartesianGrid(origin=(0.0,0.0), extent=(1.0,1.0), size=(4,4))
2D 4×4 CartesianGrid{Float64}:
    x ∈ [0.0–1.0]; Δx = 0.25
    y ∈ [0.0–1.0]; Δy = 0.25
```
"""
struct CartesianGrid{N,T<:AbstractFloat,I<:Integer}
    axes::NTuple{N,DiscreteAxis{T,I}}
end

function CartesianGrid(origin::NTuple{N,T}, extent::NTuple{N,T}, size::NTuple{N,I}) where {N,T,I}
    CartesianGrid(DiscreteAxis.(origin, extent, size))
end

"""
    CartesianGrid(origin::NTuple{N,T}, extent::NTuple{N,T}, size::NTuple{N,I})

Create a Cartesian grid with a specified origin (bottom-south-west corner in 3D), spatial extent, and number of grid cells.
"""
CartesianGrid(; origin::NTuple{N,T}, extent::NTuple{N,T}, size::NTuple{N,I}) where {N,T,I} = CartesianGrid(origin, extent, size)

# overload methods from Base
@pure Base.eltype(::CartesianGrid{N,T}) where {N,T} = T
@pure Base.ndims(::CartesianGrid{N}) where {N} = N

Base.size(grid::CartesianGrid) = length.(grid.axes)

@propagate_inbounds Base.size(grid::CartesianGrid, dim::Integer) = length(grid.axes[dim])
@propagate_inbounds Base.size(grid::CartesianGrid, loc::Location, dim::Integer) = length(grid.axes[dim], loc)

Base.size(grid::CartesianGrid{N}, locs::NTuple{N,Location}) where {N} = length.(grid.axes, locs)
Base.size(grid::CartesianGrid, loc::Location) = ntuple(D -> length(grid.axes[D], loc), Val(ndims(grid)))

# pretty-printing
function Base.show(io::IO, grid::CartesianGrid{N,T}) where {N,T}
    print(io, "$(N)D $(join(size(grid), "×")) CartesianGrid{$T}:\n")
    symbols = (:x, :y, :z)
    for idim in 1:N
        l, r = origin(grid, idim), origin(grid, idim) + extent(grid, idim)
        print(io, "    $(symbols[idim]) ∈ [$(l)–$(r)]; Δ$(symbols[idim]) = $(spacing(grid, idim))\n")
    end
end

"""
    axis(grid::CartesianGrid, dim::Integer)

Return a `DiscreteAxis` corresponding to the spatial dimension `dim`.
"""
axis(grid::CartesianGrid, dim::Integer) = grid.axes[dim]

"""
    origin(grid::CartesianGrid)

Return the origin of a `CartesianGrid`. The origin corresponds to bottom-south-west corner of the grid in 3D.
"""
origin(grid::CartesianGrid) = origin.(grid.axes)

"""
    extent(grid::CartesianGrid)

Return the spatial extent of a `CartesianGrid` as a tuple.
"""
extent(grid::CartesianGrid) = extent.(grid.axes)

"""
    spacing(grid::CartesianGrid)

Return the spacing of a `CartesianGrid` as a tuple.
"""
spacing(grid::CartesianGrid) = spacing.(grid.axes)

"""
    Δ(grid::CartesianGrid)

Return the spacing of a `CartesianGrid` as a tuple. Same as `spacing`.
"""
Δ(grid::CartesianGrid) = spacing(grid)

@propagate_inbounds origin(grid::CartesianGrid, dim::Integer) = origin(grid.axes[dim])
@propagate_inbounds extent(grid::CartesianGrid, dim::Integer) = extent(grid.axes[dim])
@propagate_inbounds spacing(grid::CartesianGrid, dim::Integer) = spacing(grid.axes[dim])
@propagate_inbounds Δ(grid::CartesianGrid, dim::Integer) = spacing(grid.axes[dim])


"""
    coord(grid::CartesianGrid{N}, loc::NTuple{N,Location}, inds::NTuple{N})

Return a tuple of spatial coordinates of a grid point at location `loc` and indices `inds`.

For vertex-centered locations, first grid point is at the origin.
For cell-centered locations, first grid point at half-spacing distance from the origin.
"""
@propagate_inbounds function coord(grid::CartesianGrid{N}, loc::NTuple{N,Location}, inds::NTuple{N}) where {N}
    ntuple(Val(N)) do I
        Base.@_inline_meta
        coord(grid.axes[I], loc[I], inds[I])
    end
end

@propagate_inbounds coord(grid::CartesianGrid{N}, loc::Location, inds::NTuple{N}) where {N} = coord.(grid.axes, Ref(loc), inds)

coord(grid::CartesianGrid{N}, loc, I::CartesianIndex{N}) where {N} = coord(grid, loc, Tuple(I))

@propagate_inbounds coord(grid::CartesianGrid, loc::Location, dim::Integer, i::Integer) = coord(grid.axes[dim], loc, i)
@propagate_inbounds function coord(grid::CartesianGrid{N}, loc::Location, dim::Integer, inds::NTuple{N}) where {N}
    coord(grid.axes[dim], loc, inds[dim])
end
@propagate_inbounds function coord(grid::CartesianGrid{N}, locs::NTuple{N,Location}, dim::Integer, inds::NTuple{N}) where {N}
    coord(grid.axes[dim], locs[dim], inds[dim])
end
@propagate_inbounds function coord(grid::CartesianGrid{N}, locs::NTuple{N,Location}, dim::Integer, i::Integer) where {N}
    coord(grid.axes[dim], locs[dim], i)
end

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
