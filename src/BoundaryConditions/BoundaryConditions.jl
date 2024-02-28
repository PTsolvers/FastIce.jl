module BoundaryConditions

export FieldBoundaryCondition, BoundaryConditionsBatch
export apply_boundary_conditions!, apply_all_boundary_conditions!
export merge_boundary_conditions

export DirichletBC, HalfCell, FullCell
export ExtrapolateBC, ExtrapolateLogBC, ExpandBC
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
include("extrapolate_bc.jl")
include("hide_boundaries.jl")

struct BoundaryConditionsBatch{F,BC}
    fields::F
    conditions::BC
end

function merge_boundary_conditions(bc1::BoundaryConditionsBatch, bc2::BoundaryConditionsBatch)
    BoundaryConditionsBatch((bc1.fields..., bc2.fields...),
                            (bc1.conditions..., bc2.conditions...))
end

merge_boundary_conditions(bc1::BoundaryConditionsBatch, ::Nothing) = bc1

merge_boundary_conditions(::Nothing, bc2::BoundaryConditionsBatch) = bc2

@inline function apply_boundary_conditions!(::Val{S}, ::Val{D},
                                            arch::Architecture,
                                            grid::CartesianGrid,
                                            batch::BoundaryConditionsBatch; kwargs...) where {S,D}
    apply_boundary_conditions!(Val(S), Val(D), arch, grid, batch.fields, batch.conditions; kwargs...)
end

@inline function apply_boundary_conditions!(::Val{S}, ::Val{D},
                                            arch::Architecture,
                                            grid::CartesianGrid,
                                            batches::NTuple{N,BoundaryConditionsBatch}; kwargs...) where {S,D,N}
    ntuple(Val(N)) do I
        apply_boundary_conditions!(Val(S), Val(D), arch, grid, batches[I].fields, batches[I].conditions; kwargs...)
    end
end

apply_boundary_conditions!(side, val, arch, grid, ::Nothing; kwargs...) = nothing

end
