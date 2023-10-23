const FieldOrNothing = Union{FieldBoundaryCondition,Nothing}

function apply_boundary_conditions!(::Val{S}, ::Val{D},
                                    arch::Architecture,
                                    grid::CartesianGrid,
                                    fields::NTuple{N,Field},
                                    conditions::NTuple{N,FieldOrNothing}; async=true) where {S,D,N}
    _validate_fields(fields, D, S)
    sizes = ntuple(ifield -> remove_dim(Val(D), size(fields[ifield])), Val(length(fields)))
    worksize = remove_dim(Val(D), size(grid, Vertex()))
    _apply_boundary_conditions!(backend(arch), 256, worksize)(Val(S), Val(D), grid, sizes, fields, conditions)
    async || KernelAbstractions.synchronize(backend(arch))
    return
end

@kernel function _apply_boundary_conditions!(side, dim, grid, sizes, fields, conditions)
    I = @index(Global, Cartesian)
    # ntuple here unrolls the loop over fields
    ntuple(Val(length(fields))) do ifield
        Base.@_inline_meta
        if _checkindices(sizes[ifield], I)
            field = fields[ifield]
            condition = conditions[ifield]
            _apply_field_boundary_condition!(side, dim, grid, field, location(field), I, condition)
        end
    end
end

_apply_field_boundary_condition!(side, dim, grid, field, loc, I, ::Nothing) = nothing

function _validate_fields(fields::NTuple{N,Field}, dim, side) where {N}
    for f in fields
        if (location(f, Val(dim)) == Center()) && (halo(f, dim, side) < 1)
            error("to apply boundary conditions, halo width must be at least 1")
        end
    end
    return
end
