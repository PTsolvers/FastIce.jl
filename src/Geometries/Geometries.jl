module Geometries

# export AbstractElevation, AABB, DataElevation, SyntheticElevation
# export domain, rotated_domain, rotation, generate_elevation, extents, center, dilate
export SyntheticElevation
export extents
export make_synthetic

using Chmy.Grids

include("elevation_data.jl")
include("geometry_helpers.jl")
include("synthetic_geometries.jl")

end # module
