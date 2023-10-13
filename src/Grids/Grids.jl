module Grids

export Location, Center, Vertex
export DiscreteAxis, spacing, Δ, origin, extent, coord, center, vertex, coords, centers, vertices

export CartesianGrid, axis
export xcoord, ycoord, zcoord, xcenter, ycenter, zcenter, xvertex, yvertex, zvertex
export xcoords, ycoords, zcoords, xcenters, ycenters, zcenters, xvertices, yvertices, zvertices

abstract type Location end

struct Center <: Location end
struct Vertex <: Location end

include("discrete_axis.jl")
include("cartesian_grid.jl")

end
