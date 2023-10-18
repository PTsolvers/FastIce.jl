module BoundaryConditions

export FieldBoundaryCondition, BoundaryConditionsBatch
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

abstract type FieldBoundaryCondition end

include("field_boundary_conditions.jl")
include("utils.jl")
include("boundary_function.jl")
include("dirichlet_bc.jl")
include("hide_boundaries.jl")

struct BoundaryConditionsBatch{F,BC}
    fields::F
    conditions::BC
end

@inline function apply_boundary_conditions!(::Val{S}, ::Val{D},
                                            arch::Architecture,
                                            grid::CartesianGrid,
                                            batch::BoundaryConditionsBatch) where {S,D}
    apply_boundary_conditions!(Val(S), Val(D), arch, grid, batch.fields, batch.conditions)
end

apply_boundary_conditions!(side, val, arch, grid, ::Nothing) = nothing
apply_boundary_conditions!(side, val, arch, grid, fields, ::Nothing) = nothing

end
