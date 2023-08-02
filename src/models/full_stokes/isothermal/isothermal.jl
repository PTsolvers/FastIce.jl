module Isothermal

using FastIce.Physics

include("kernels.jl")

const DEFAULT_PHYSICS = (
    equation_of_state=default(IncompressibleIceEOS),
    thermal_properties=default(IceThermalProperties),
    rheology=default(GlensLawRheology)
)

struct IsothermalFullStokesModel{Backend,Grid,BC,Physics,Numerics,Fields} <: AbstractModel
    backend::Backend
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    numerics::Numerics
    fields::Fields
end

function IsothermalFullStokesModel(; backend, grid, boundary_conditions, phyiscs=DEFAULT_PHYSICS, numerics, fields=nothing)
    if isnothing(fields)
        fields = make_fields_mechanics(backend, grid, boundary_conditions)
    end

    return IsothermalFullStokesModel(backend, grid, boundary_conditions, phyiscs, numerics, fields)
end

fields(model::IsothermalFullStokesModel) = model.fields
grid(model::IsothermalFullStokesModel) = model.grid

function set_stress_bcs!(bcs, args...)
    for (side, bc) in bcs
        set_bc!(Val(side), bc, args...)
    end
    return
end

function set_velocity_bcs!(V, Pr, τ, Δ, bcs)
    # TODO
end

function advance_iteration!(model::IsothermalFullStokesModel, t, Δt; async = true)
    (; Pr, τ, V, η) = model.fields
    (; η_rh) = model.physics.rheology
    (; η_rel, Δτ) = model.iter_params
    Δ = spacing(model.grid)
    nx, ny, nz = size(model.grid)
    backend = model.backend

    # stress
    update_σ!(backend, 256, (nx + 2, ny + 2, nz + 2))(Pr, τ, V, η, Δτ, Δ)
    set_stress_bcs!(Pr, τ, V, Δ, model.boundary_conditions)
    # velocity
    update_V!(backend, 256, (nx + 1, ny + 1, nz + 1))(V, Pr, τ, η, Δτ, Δ)
    set_velocity_bcs!(V, Pr, τ, Δ, model.boundary_conditions)
    # rheology
    update_η!(backend, 256, (nx, ny, nz))(η, τ, η_rh, η_rel)
    extrapolate!(η)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::IsothermalFullStokesModel, t, Δt)
    # TODO
    
    return
end