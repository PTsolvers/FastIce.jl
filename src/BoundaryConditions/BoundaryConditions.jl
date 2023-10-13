module BoundaryConditions

export FieldBoundaryConditions
export apply_boundary_conditions!

export DirichletBC, HalfCell, FullCell
export ContinuousBC, DiscreteBC
export BoundaryFunction, DiscreteBoundaryFunction, ContinuousBoundaryFunction

using FastIce.Grids
using FastIce.Fields
using FastIce.Utils

using KernelAbstractions
using Adapt

include("utils.jl")
include("boundary_function.jl")
include("dirichlet_bc.jl")
include("field_boundary_conditions.jl")

end
