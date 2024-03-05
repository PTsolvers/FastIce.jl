module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!, compute_residuals!

using FastIce.Physics

using Chmy.Architectures
using Chmy.Grids
using Chmy.BoundaryConditions
using Chmy.Distributed
using Chmy.Fields
using Chmy.KernelLaunch
using Chmy.GridOperators

using KernelAbstractions

include("kernels_common.jl")
include("kernels_2d.jl")
include("kernels_3d.jl")

include("boundary_conditions.jl")

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

function StressFields(arch, grid)
    return (Pr=Field(arch, grid, Center()), τ=TensorField(arch, grid))
end

function VelocityFields(arch, grid::CartesianGrid{2})
    return VectorField(arch, grid)
end

function ResidualFields(arch, grid::CartesianGrid{2})
    (r_Pr=Field(arch, grid, Center()), r_V=VectorField(arch, grid))
end

function ViscosityFields(arch, grid::CartesianGrid)
    (η      = Field(arch, grid, Center()),
     η_next = Field(arch, grid, Center()))
end

function IsothermalFullStokesModel(; arch,
                                   grid,
                                   boundary_conditions,
                                   gravity,
                                   rheology,
                                   solver_params=(),
                                   outer_width=nothing)
    stress    = StressFields(arch, grid)
    velocity  = VelocityFields(arch, grid)
    viscosity = ViscosityFields(arch, grid)
    residual  = ResidualFields(arch, grid)

    boundary_conditions = IsothermalFullStokesBoundaryConditions(arch, grid, stress, velocity, viscosity, residual, boundary_conditions)

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
                                     hide_boundaries,
                                     solver_params,
                                     launcher)
end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt)
    (; Pr, τ)     = model.stress
    V             = model.velocity
    (; η, η_next) = model.viscosity
    (; Δτ)        = model.solver_params
    arch          = model.arch
    rheology      = model.rheology
    ρg            = model.gravity
    bc            = model.boundary_conditions
    grid          = model.grid
    launch        = model.launcher

    launch(arch, grid, update_σ! => (Pr, τ, V, η, Δτ, grid); bc=bc.stress)
    launch(arch, grid, update_V! => (V, Pr, τ, η, η_next, rheology, ρg, Δτ, grid); bc=velocity_bc)

    # swap double buffers for viscosity
    model.viscosity = NamedTuple{keys(model.viscosity)}(reverse(values(model.viscosity)))
    return
end

function compute_residuals!(model::IsothermalFullStokesModel)
    (; Pr, τ)     = model.stress
    V             = model.velocity
    (; r_Pr, r_V) = model.residual
    grid          = model.grid
    ρg            = model.gravity
    bc            = model.boundary_conditions.residual

    launch!(model.arch, grid, compute_residuals! => (r_V, r_Pr, Pr, τ, V, ρg, grid); bc)
    return
end

end
