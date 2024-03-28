module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!, compute_residuals!

using FastIce.Physics

using Chmy
using Chmy.Architectures
using Chmy.Grids
using Chmy.BoundaryConditions
using Chmy.Distributed
using Chmy.Fields
using Chmy.KernelLaunch
using Chmy.GridOperators

using KernelAbstractions

include("kernels_2d.jl")
include("kernels_3d.jl")

mutable struct IsothermalFullStokesModel{Arch,Grid,Stress,Velocity,Viscosity,Rheology,Residual,BC,Gravity,SolverParams,KL}
    arch::Arch
    grid::Grid
    stress::Stress
    velocity::Velocity
    viscosity::Viscosity
    rheology::Rheology
    gravity::Gravity
    residual::Residual
    boundary_conditions::BC
    solver_params::SolverParams
    launcher::KL
end

struct StressField{Pressure,DeviatoricStress}
    P::Pressure
    τ::DeviatoricStress
end
StressField(arch, grid::StructuredGrid) = StressField(Field(arch, grid, Center()), TensorField(arch, grid))

struct VelocityField{Velocity}
    V::Velocity
end
VelocityField(arch, grid::StructuredGrid) = VelocityField(VectorField(arch, grid))

struct ResidualField{VelocityResidual,PressureResidual}
    r_V::VelocityResidual
    r_P::PressureResidual
end
ResidualField(arch, grid::StructuredGrid) = ResidualField(VectorField(arch, grid), Field(arch, grid, Center()))

function ViscosityField(arch, grid::StructuredGrid)
    return (η=Field(arch, grid, Center()),
            η_next=Field(arch, grid, Center()))
end

include("boundary_conditions.jl")

function IsothermalFullStokesModel(; arch,
                                   grid,
                                   boundary_conditions,
                                   gravity,
                                   rheology,
                                   solver_params=(),
                                   outer_width=nothing)
    stress    = StressField(arch, grid)
    velocity  = VelocityField(arch, grid)
    viscosity = ViscosityField(arch, grid)
    residual  = ResidualField(arch, grid)

    if isnothing(outer_width)
        outer_width = ntuple(_ -> 2, Val(ndims(grid)))
    end

    launcher = Launcher(arch, grid; outer_width)

    return IsothermalFullStokesModel(arch,
                                     grid,
                                     stress,
                                     velocity,
                                     viscosity,
                                     rheology,
                                     gravity,
                                     residual,
                                     boundary_conditions,
                                     solver_params,
                                     launcher)
end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt)
    (; P, τ)      = model.stress
    (; V)         = model.velocity
    (; η, η_next) = model.viscosity
    (; Δτ)        = model.solver_params
    arch          = model.arch
    rheology      = model.rheology
    ρg            = model.gravity
    grid          = model.grid
    launch        = model.launcher

    bc1 = batch(grid, model.stress, model.boundary_conditions)
    bc2 = merge(batch(grid, model.velocity, model.boundary_conditions; exchange=Tuple(V)),
                batch(grid, η_next => Neumann(); exchange=η_next))

    launch(arch, grid, update_σ! => (P, τ, V, η, Δτ, grid); bc=bc1)
    launch(arch, grid, update_V! => (V, P, τ, η, η_next, rheology, ρg, Δτ, grid); bc=bc2)

    # swap double buffers for viscosity
    model.viscosity = NamedTuple{keys(model.viscosity)}(reverse(values(model.viscosity)))
    return
end

function compute_residuals!(model::IsothermalFullStokesModel)
    (; P, τ)     = model.stress
    (; V)        = model.velocity
    (; r_P, r_V) = model.residual
    grid         = model.grid
    ρg           = model.gravity
    launch       = model.launcher

    bc = batch(grid, model.residual, model.boundary_conditions)
    launch(model.arch, grid, compute_residuals! => (r_V, r_P, P, τ, V, ρg, grid); bc)
    return
end

end
