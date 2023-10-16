struct FieldBoundaryConditions{F<:Tuple,B<:Tuple}
    fields::F
    conditions::B
end

function apply_boundary_conditions!(::Val{S}, ::Val{D}, arch::Architecture, grid::CartesianGrid,
                                    bc::FieldBoundaryConditions; async=true) where {S,D}
    _validate_boundary_conditions(bc, D, S)
    sizes = ntuple(ifield -> remove_dim(Val(D), size(bc.fields[ifield])), Val(length(bc.fields)))
    worksize = remove_dim(Val(D), size(grid, Vertex()))
    # launch!(_apply_boundary_conditions! => (Val(S), Val(D), grid, bc.fields, bc.conditions); backend, worksize)
    _apply_boundary_conditions!(arch.backend, 256, worksize)(Val(S), Val(D), grid, sizes, bc.fields, bc.conditions)
    async || KernelAbstractions.synchronize(backend)
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

@inline _apply_field_boundary_condition!(side, dim, grid, f, loc, Ibc, ::Nothing) = nothing

function _validate_boundary_conditions(bc::FieldBoundaryConditions, dim, side)
    for f in bc.fields
        if halo(f, dim, side) < 1
            error("to apply boundary conditions, halo width must be at least 1")
        end
    end
    return
end
