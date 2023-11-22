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

    ex_res_vel = ntuple(length(_COORDINATES)) do I
        kind = I == val ? :(FullCell) : :(HalfCell)
        Expr(:(=), Symbol(:V, _COORDINATES[I][1]), :(DirichletBC{$kind}(0.0)))
    end

    ex_res_vel = Expr(:tuple, ex_res_vel...)

    ex_res_slip_vel = Expr(:(=), Symbol(:V, dim), :(DirichletBC{FullCell}(0.0)))
    ex_res_slip_vel = Expr(:tuple, ex_slip_vel)

    @eval begin
        extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Traction}) = $ex_tr
        extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Velocity}) = ()
        extract_stress_bc(::Val{$val}, bc::BoundaryCondition{Slip})     = $ex_slip_tr

        extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Traction}) = ()
        extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Velocity}) = $ex_vel
        extract_velocity_bc(::Val{$val}, bc::BoundaryCondition{Slip})     = $ex_slip_vel

        extract_residuals_vel_bc(::Val{$val}, bc::BoundaryCondition{Traction}) = ()
        extract_residuals_vel_bc(::Val{$val}, bc::BoundaryCondition{Velocity}) = $ex_res_vel
        extract_residuals_vel_bc(::Val{$val}, bc::BoundaryCondition{Slip})     = $ex_res_slip_vel
    end
end

function make_batch(bcs::NamedTuple, fields::NamedTuple)
    batch_fields = Tuple(fields[name] for name in eachindex(bcs))
    return BoundaryConditionsBatch(batch_fields, values(bcs))
end

function make_stress_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields, bc) where {Kind,N}
    ordering = (:x, :y, :z)
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
                nothing
            else
                new_bc = extract_stress_bc(Val(D), bc[ordering[D]][S])
                if isempty(new_bc)
                    nothing
                else
                    make_batch(new_bc, fields)
                end
            end
        end
    end
end

function make_velocity_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields::NamedTuple{names}, bc) where {Kind,N,names}
    ordering = (:x, :y, :z)
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
                new_bc = NamedTuple{names}(ExchangeInfo(Val(S), Val(D), V) for V in fields)
                make_batch(new_bc, fields)
            else
                new_bc = extract_velocity_bc(Val(D), bc[ordering[D]][S])
                if isempty(new_bc)
                    nothing
                else
                    make_batch(new_bc, fields)
                end
            end
        end
    end
end

function make_residuals_vel_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, fields::NamedTuple{names}, bc) where {Kind,N,names}
    ordering = (:x, :y, :z)
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
                new_bc = NamedTuple{names}(ExchangeInfo(Val(S), Val(D), V) for V in fields)
                make_batch(new_bc, fields)
            else
                new_bc = extract_residuals_vel_bc(Val(D), bc[ordering[D]][S])
                if isempty(new_bc)
                    nothing
                else
                    make_batch(new_bc, fields)
                end
            end
        end
    end
end

function make_rheology_bc(arch::Architecture{Kind}, ::CartesianGrid{N}, η) where {Kind,N}
    ntuple(Val(N)) do D
        ntuple(Val(2)) do S
            if (Kind == Distributed.DistributedMPI) && has_neighbor(details(arch), D, S)
                BoundaryConditionsBatch((η,), (ExchangeInfo(Val(S), Val(D), η),))
            else
                BoundaryConditionsBatch((η,), (ExtrapolateBC(),))
            end
        end
    end
end

function make_field_boundary_conditions(arch::Architecture, grid::CartesianGrid, fields, bc)
    stress_fields   = (; Pr=fields.Pr, NamedTuple{Symbol.(:τ, keys(fields.τ))}(values(fields.τ))...)
    velocity_fields = NamedTuple{Symbol.(:V, keys(fields.V))}(values(fields.V))
    residuals_vel_fields = NamedTuple{Symbol.(:r_V, keys(fields.r_V))}(values(fields.r_V))

    return (stress   = make_stress_bc(arch, grid, stress_fields, bc),
            velocity = make_velocity_bc(arch, grid, velocity_fields, bc),
            residuals_vel = make_residuals_vel_bc(arch, grid, residuals_vel_fields, bc),
            rheology = make_rheology_bc(arch, grid, fields.η))
end
