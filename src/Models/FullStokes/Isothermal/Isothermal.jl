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

function StressFields(backend, grid::CartesianGrid{2})
    (Pr=Field(backend, grid, Center(); halo=1),
     # deviatoric stress
     τ=(xx=Field(backend, grid, Center(); halo=1),
        yy=Field(backend, grid, Center(); halo=1),
        xy=Field(backend, grid, (Vertex(), Vertex()))))
end

function VelocityFields(backend, grid::CartesianGrid{2})
    (x=Field(backend, grid, (Vertex(), Center()); halo=1),
     y=Field(backend, grid, (Center(), Vertex()); halo=1))
end

function ResidualFields(backend, grid::CartesianGrid{2})
    (r_Pr=Field(backend, grid, Center()),
     r_V=(x=Field(backend, grid, (Vertex(), Center())),
          y=Field(backend, grid, (Center(), Vertex()))))
end

function StressFields(backend, grid::CartesianGrid{3})
    (Pr=Field(backend, grid, Center(); halo=1),
     # deviatoric stress
     τ=(xx=Field(backend, grid, Center(); halo=1),
        yy=Field(backend, grid, Center(); halo=1),
        zz=Field(backend, grid, Center(); halo=1),
        xy=Field(backend, grid, (Vertex(), Vertex(), Center())),
        xz=Field(backend, grid, (Vertex(), Center(), Vertex())),
        yz=Field(backend, grid, (Center(), Vertex(), Vertex()))))
end

function VelocityFields(backend, grid::CartesianGrid{3})
    (x=Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
     y=Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
     z=Field(backend, grid, (Center(), Center(), Vertex()); halo=1))
end

function ResidualFields(backend, grid::CartesianGrid{3})
    (r_Pr=Field(backend, grid, Center()),
     r_V=(x=Field(backend, grid, (Vertex(), Center(), Center())),
          y=Field(backend, grid, (Center(), Vertex(), Center())),
          z=Field(backend, grid, (Center(), Center(), Vertex()))))
end

function ViscosityFields(backend, grid::CartesianGrid)
    (η      = Field(backend, grid, Center(); halo=1),
     η_next = Field(backend, grid, Center(); halo=1))
end

function IsothermalFullStokesModel(; arch,
                                   grid,
                                   boundary_conditions,
                                   gravity,
                                   rheology,
                                   solver_params=(),
                                   outer_width=nothing)
    backend = backend(arch)

    stress    = StressFields(backend, grid)
    velocity  = VelocityFields(backend, grid)
    residual  = ResidualFields(backend, grid)
    viscosity = ViscosityFields(backend, grid)

    boundary_conditions = make_field_boundary_conditions(arch, grid, fields, boundary_conditions)

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
    (; η_rel, Δτ)   = model.iter_params
    η_rh            = model.physics.rheology
    ρg              = model.gravity
    bc              = model.boundary_conditions
    hide_boundaries = model.hide_boundaries
    grid            = model.grid

    Δ = spacing(model.grid)

    launch!(model.arch, grid, update_σ! => (Pr, τ, V, η, Δτ, Δ, grid);
            location=Center(), expand=1, boundary_conditions=bc.stress, hide_boundaries)

    launch!(model.arch, grid, update_V! => (V, Pr, τ, η, η_next, ρg, Δτ, Δ, grid);
            location=Vertex(), boundary_conditions=bc.velocity, hide_boundaries)

    # rheology
    launch!(model.arch, grid, update_η! => (η, η_rh, η_rel, model.fields, grid);
            location=Center(), boundary_conditions=bc.rheology, hide_boundaries)
    return
end

function compute_residuals!(model::IsothermalFullStokesModel)
    (; Pr, τ, V, r_Pr, r_V) = model.fields
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
