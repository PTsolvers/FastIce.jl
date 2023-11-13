struct Traction end
struct Velocity end
struct Slip end

"""
Boundary condition for Stokes problem.
"""
struct BoundaryCondition{Kind,N,C}
    components::C
    BoundaryCondition{Kind}(components::Tuple) where {Kind} = new{Kind,length(components),typeof(components)}(components)
end

BoundaryCondition{Kind}(components...) where {Kind} = BoundaryCondition{Kind}(components)

const _COORDINATES = ((:x, 1), (:y, 2), (:z, 3))

for (dim, val) in _COORDINATES
    td = remove_dim(Val(val), _COORDINATES)
    TN = ntuple(Val(length(td))) do I
        dim2, val2 = td[I]
        val2 < val ? Symbol(:τ, dim2, dim) : Symbol(:τ, dim, dim2)
    end
    N  = Symbol(:τ, dim, dim)

    ex1 = Expr(:(=), :Pr, :(DirichletBC{HalfCell}(bc.components[$val])))
    ex2 = Expr(:(=), N, :(DirichletBC{HalfCell}(convert(eltype(bc.components[$val]), 0))))
    ex3 = ntuple(Val(length(TN))) do I
        Expr(:(=), TN[I], :(DirichletBC{FullCell}(bc.components[$(td[I][2])])))
    end

    ex_tr = Expr(:tuple, ex1, ex2, ex3...)

    ex_vel = ntuple(length(_COORDINATES)) do I
        kind = I == val ? :(FullCell) : :(HalfCell)
        Expr(:(=), Symbol(:V, _COORDINATES[I][1]), :(DirichletBC{$kind}(bc.components[$I])))
    end

    ex_vel = Expr(:tuple, ex_vel...)

    ex_slip_tr = Expr(:tuple, ex3...)

    ex_slip_vel = Expr(:(=), Symbol(:V, dim), :(DirichletBC{FullCell}(bc.components[$val])))
    ex_slip_vel = Expr(:tuple, ex_slip_vel)

    @eval begin
        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Traction})
            return $ex_tr, NamedTuple()
        end

        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Velocity})
            return NamedTuple(), $ex_vel
        end

        function extract_boundary_conditions(::Val{$val}, bc::BoundaryCondition{Slip})
            return $ex_slip_tr, $ex_slip_vel
        end
    end
end

function make_batch(::CartesianGrid{2}, bcs::NamedTuple{bnames, Tuple{}}, fields::NamedTuple) where {bnames}
    return nothing
end

function make_batch(::CartesianGrid{3}, bcs::NamedTuple{bnames, Tuple{}}, fields::NamedTuple) where {bnames}
    return nothing
end

function make_batch(::CartesianGrid{2}, bcs::NamedTuple, fields::NamedTuple)
    field_map = (Pr  = fields.Pr,
                 τxx = fields.τ.xx, τyy = fields.τ.yy, τxy = fields.τ.xy,
                 Vx  = fields.V.x, Vy  = fields.V.y)
    batch_fields = Tuple(field_map[name] for name in eachindex(bcs))
    return BoundaryConditionsBatch(batch_fields, values(bcs))
end

function make_batch(::CartesianGrid{3}, bcs::NamedTuple, fields::NamedTuple)
    field_map = (Pr  = fields.Pr,
                 τxx = fields.τ.xx, τyy = fields.τ.yy, τzz = fields.τ.zz,
                 τxy = fields.τ.xy, τxz = fields.τ.xz, τyz = fields.τ.yz,
                 Vx  = fields.V.x, Vy  = fields.V.y, Vz  = fields.V.z)
    batch_fields = Tuple(field_map[name] for name in eachindex(bcs))
    return BoundaryConditionsBatch(batch_fields, values(bcs))
end

function make_batches(grid, bcs, fields)
    ntuple(Val(ndims(grid))) do D
        ntuple(Val(2)) do S
            make_batch(grid, bcs[D][S], fields)
        end
    end
end

function make_field_boundary_conditions(grid::CartesianGrid{N}, fields, logical_boundary_conditions) where {N}
    ordering = (:x, :y, :z)

    field_bcs = ntuple(Val(N)) do dim
        left, right = logical_boundary_conditions[ordering[dim]]
        left  = extract_boundary_conditions(Val(dim), left)
        right = extract_boundary_conditions(Val(dim), right)
        Tuple(zip(left, right))
    end

    stress, velocity = zip(field_bcs...)
    stress   = make_batches(grid, stress, fields)
    velocity = make_batches(grid, velocity, fields)

    return (; stress, velocity)
end
