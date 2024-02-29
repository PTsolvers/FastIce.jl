module LevelSets

export compute_level_set_from_dem!

using FastIce
using FastIce.Grids
using FastIce.Architectures
using FastIce.Fields
using KernelAbstractions
using LinearAlgebra, GeometryBasics

include("signed_distances.jl")
include("compute_level_sets.jl")

end