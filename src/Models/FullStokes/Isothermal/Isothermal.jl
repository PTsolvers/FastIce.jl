module Isothermal

export BoundaryCondition, Traction, Velocity, Slip
export IsothermalFullStokesModel, advance_iteration!, advance_timestep!, compute_residuals!

using FastIce.Architectures
using FastIce.Physics
using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils
using FastIce.KernelLaunch
using FastIce.Distributed

using FastIce.GridOperators
using KernelAbstractions

include("kernels_common.jl")
include("kernels_2d.jl")
include("kernels_3d.jl")

include("boundary_conditions.jl")

mutable struct IsothermalFullStokesModel{Arch,Grid,Stress,Velocity,Viscosity,Rheology,Residual,BC,HB,Gravity,SolverParams}
    arch::Arch
    grid::Grid
    stress::Stress
    velocity::Velocity
    viscosity::Viscosity
    rheology::Rheology
    gravity::Gravity
    residual::Residual
    boundary_conditions::BC
    hide_boundaries::HB
    solver_params::SolverParams
end

function StressFields(arch, grid::CartesianGrid{2})
    (Pr=Field(arch, grid, Center(); halo=1),
     # deviatoric stress
     τ=(xx=Field(arch, grid, Center(); halo=1),
        yy=Field(arch, grid, Center(); halo=1),
        xy=Field(arch, grid, (Vertex(), Vertex()))))
end

function VelocityFields(arch, grid::CartesianGrid{2})
    (x=Field(arch, grid, (Vertex(), Center()); halo=1),
     y=Field(arch, grid, (Center(), Vertex()); halo=1))
end

function ResidualFields(arch, grid::CartesianGrid{2})
    (r_Pr=Field(arch, grid, Center()),
     r_V=(x=Field(arch, grid, (Vertex(), Center())),
          y=Field(arch, grid, (Center(), Vertex()))))
end

function StressFields(arch, grid::CartesianGrid{3})
    (Pr=Field(arch, grid, Center(); halo=1),
     # deviatoric stress
     τ=(xx=Field(arch, grid, Center(); halo=1),
        yy=Field(arch, grid, Center(); halo=1),
        zz=Field(arch, grid, Center(); halo=1),
        xy=Field(arch, grid, (Vertex(), Vertex(), Center())),
        xz=Field(arch, grid, (Vertex(), Center(), Vertex())),
        yz=Field(arch, grid, (Center(), Vertex(), Vertex()))))
end

function VelocityFields(arch, grid::CartesianGrid{3})
    (x=Field(arch, grid, (Vertex(), Center(), Center()); halo=1),
     y=Field(arch, grid, (Center(), Vertex(), Center()); halo=1),
     z=Field(arch, grid, (Center(), Center(), Vertex()); halo=1))
end

function ResidualFields(arch, grid::CartesianGrid{3})
    (r_Pr=Field(arch, grid, Center()),
     r_V=(x=Field(arch, grid, (Vertex(), Center(), Center())),
          y=Field(arch, grid, (Center(), Vertex(), Center())),
          z=Field(arch, grid, (Center(), Center(), Vertex()))))
end

function ViscosityFields(arch, grid::CartesianGrid)
    (η      = Field(arch, grid, Center(); halo=1),
     η_next = Field(arch, grid, Center(); halo=1))
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

    hide_boundaries = HideBoundaries(arch, outer_width)

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
                                     solver_params)
end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt)
    (; Pr, τ)       = model.stress
    V               = model.velocity
    (; η, η_next)   = model.viscosity
    (; Δτ)          = model.solver_params
    rheology        = model.rheology
    ρg              = model.gravity
    bc              = model.boundary_conditions
    hide_boundaries = model.hide_boundaries
    grid            = model.grid

    Δ = spacing(model.grid)

    launch!(model.arch, grid, update_σ! => (Pr, τ, V, η, Δτ, Δ, grid);
            location=Center(), expand=1, boundary_conditions=bc.stress, hide_boundaries)

    # merge boundary conditions because viscosity is double-buffered
    velocity_bc = dim_side_ntuple(Val(ndims(grid))) do D, S
        merge_boundary_conditions(bc.velocity[D][S], bc.viscosity.η_next[D][S])
    end

    launch!(model.arch, grid, update_V! => (V, Pr, τ, η, η_next, rheology, ρg, Δτ, Δ, grid);
            location=Vertex(), boundary_conditions=velocity_bc, hide_boundaries)

    # swap double buffers for viscosity
    model.viscosity = NamedTuple{keys(model.viscosity)}(reverse(values(model.viscosity)))
    bc.viscosity    = NamedTuple{keys(bc.viscosity)}(reverse(values(bc.viscosity)))
    return
end

function compute_residuals!(model::IsothermalFullStokesModel)
    (; Pr, τ) = model.stress
    V = model.velocity
    (; r_Pr, r_V) = model.residual
    ρg = model.gravity
    boundary_conditions = model.boundary_conditions.residual

    Δ = spacing(model.grid)

    launch!(model.arch, model.grid, compute_residuals! => (r_V, r_Pr, Pr, τ, V, ρg, Δ, model.grid);
            location=Vertex(), boundary_conditions)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO

    return
end

end
