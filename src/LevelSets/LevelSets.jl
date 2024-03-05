module LevelSets

export compute_level_set_from_dem!
export volfrac_field, compute_volume_fractions_from_level_set!

using Chmy
using Chmy.Grids
using Chmy.Architectures
using Chmy.Fields
using KernelAbstractions
using LinearAlgebra, GeometryBasics

include("signed_distances.jl")
include("compute_level_sets.jl")
include("compute_volume_fractions.jl")

end
