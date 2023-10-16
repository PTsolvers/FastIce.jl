module BoundaryConditions

export FieldBoundaryConditions
export apply_boundary_conditions!, apply_all_boundary_conditions!

export DirichletBC, HalfCell, FullCell
export ContinuousBC, DiscreteBC
export BoundaryFunction, DiscreteBoundaryFunction, ContinuousBoundaryFunction

export HideBoundaries, hide

using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.Architectures

using KernelAbstractions
using Adapt


"""
Overload this method for a custom boundary condition type.
"""
apply_boundary_conditions!(::Val{S}, ::Val{D}, arch::Architecture, grid::CartesianGrid, bc::Nothing) where {S,D} = nothing

include("utils.jl")
include("boundary_function.jl")
include("dirichlet_bc.jl")
include("field_boundary_conditions.jl")
include("hide_boundaries.jl")

end
