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

# 1D boundary conditions

function BoundaryConditions.batch(side, dim::Dim{1}, grid::StructuredGrid{1}, stress::StressField, cfd_bc::BoundaryCondition{Traction,1}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[1])
    fields = (stress.P, stress.τ.xx)
    bcs    = (bc_n, bc_n)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

# 2D boundary conditions

function BoundaryConditions.batch(side, dim::Dim{1}, grid::StructuredGrid{2}, stress::StressField, cfd_bc::BoundaryCondition{Traction,2}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[1])
    bc_t   = Dirichlet(cfd_bc.components[2])
    fields = (stress.P, stress.τ.xx, stress.τ.xy)
    bcs    = (bc_n, bc_n, bc_t)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{2}, grid::StructuredGrid{2}, stress::StressField, cfd_bc::BoundaryCondition{Traction,2}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[2])
    bc_t   = Dirichlet(cfd_bc.components[1])
    fields = (stress.P, stress.τ.yy, stress.τ.xy)
    bcs    = (bc_n, bc_n, bc_t)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{D}, grid::StructuredGrid{2}, stress::StressField, cfd_bc::BoundaryCondition{Slip,2}; kwargs...) where {D}
    bc_t   = Dirichlet(cfd_bc.components[3-D])
    fields = tuple(stress.τ.xy)
    bcs    = tuple(bc_t)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

# 3D boundary conditions

function BoundaryConditions.batch(side, dim::Dim{1}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Traction,3}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[1])
    bc_t1  = Dirichlet(cfd_bc.components[2])
    bc_t2  = Dirichlet(cfd_bc.components[3])
    fields = (stress.P, stress.τ.xx, stress.τ.xy, stress.τ.xz)
    bcs    = (bc_n, bc_n, bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{2}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Traction,3}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[2])
    bc_t1  = Dirichlet(cfd_bc.components[1])
    bc_t2  = Dirichlet(cfd_bc.components[3])
    fields = (stress.P, stress.τ.yy, stress.τ.xy, stress.τ.yz)
    bcs    = (bc_n, bc_n, bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{3}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Traction,3}; kwargs...)
    bc_n   = Dirichlet(cfd_bc.components[3])
    bc_t1  = Dirichlet(cfd_bc.components[1])
    bc_t2  = Dirichlet(cfd_bc.components[2])
    fields = (stress.P, stress.τ.zz, stress.τ.xz, stress.τ.yz)
    bcs    = (bc_n, bc_n, bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{1}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Slip,3}; kwargs...)
    bc_t1  = Dirichlet(cfd_bc.components[2])
    bc_t2  = Dirichlet(cfd_bc.components[3])
    fields = (stress.τ.xy, stress.τ.xz)
    bcs    = (bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{2}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Slip,3}; kwargs...)
    bc_t1  = Dirichlet(cfd_bc.components[1])
    bc_t2  = Dirichlet(cfd_bc.components[3])
    fields = (stress.τ.xy, stress.τ.yz)
    bcs    = (bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{3}, grid::StructuredGrid{3}, stress::StressField, cfd_bc::BoundaryCondition{Slip,3}; kwargs...)
    bc_t1  = Dirichlet(cfd_bc.components[1])
    bc_t2  = Dirichlet(cfd_bc.components[2])
    fields = (stress.τ.xz, stress.τ.yz)
    bcs    = (bc_t1, bc_t2)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

# nD boundary conditions

function BoundaryConditions.batch(side, dim::Dim{D}, grid::StructuredGrid{N}, velocity::VelocityField, cfd_bc::BoundaryCondition{Velocity,N}; kwargs...) where {D,N}
    fields = Tuple(velocity.V)
    bcs    = ntuple(dim -> Dirichlet(cfd_bc.components[dim]), Val(N))
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{D}, grid::StructuredGrid{N}, velocity::VelocityField, cfd_bc::BoundaryCondition{Slip,N}; kwargs...) where {D,N}
    bc_n   = Dirichlet(cfd_bc.components[D])
    fields = tuple(velocity.V[D])
    bcs    = tuple(bc_n)
    return batch(side, dim, grid, fields, bcs; kwargs...)
end

function BoundaryConditions.batch(side, dim, grid, ::StressField, ::BoundaryCondition{Velocity}; kwargs...)
    return batch(side, dim, grid, (), (); kwargs...)
end

function BoundaryConditions.batch(side, dim, grid, ::VelocityField, ::BoundaryCondition{Traction}; kwargs...)
    return batch(side, dim, grid, (), (); kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim, grid, residual::ResidualField, ::BoundaryCondition{Traction}; kwargs...)
    return batch(side, dim, grid, (), (); kwargs...)
end

function BoundaryConditions.batch(side, dim::Dim{D}, grid, residual::ResidualField, ::BoundaryCondition; kwargs...) where {D}
    fields = tuple(residual.r_V[D])
    bcs    = tuple(Dirichlet())
    return batch(side, dim, grid, fields, bcs; kwargs...)
end
