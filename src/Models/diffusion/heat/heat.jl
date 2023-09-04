module Heat

export BoundaryCondition, Flux, Value
export HeatDiffusionModel, advance_iteration!, advance_timestep!

using FastIce.Grids
using FastIce.Fields
using FastIce.BoundaryConditions
using FastIce.Utils

include("kernels.jl")

include("boundary_conditions.jl")

struct HeatDiffusionModel{Backend,Grid,BC,Physics,IterParams,Fields}
    backend::Backend
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    iter_params::IterParams
    fields::Fields
end

function make_fields_diffusion(backend, grid)
    return (
        q = (
            x = Field(backend, grid, (Vertex(), Center(), Center()); halo=1),
            y = Field(backend, grid, (Center(), Vertex(), Center()); halo=1),
            z = Field(backend, grid, (Center(), Center(), Vertex()); halo=1),
        ),
    )
end

function HeatDiffusionModel(; backend, grid, boundary_conditions, physics=nothing, iter_params, fields=nothing, init_fields=nothing)
    if isnothing(fields)
        diffusion_fields = make_fields_diffusion(backend, grid)
        fields = diffusion_fields
    end

    if !isnothing(init_fields)
        fields = merge(fields, init_fields)
    end

    boundary_conditions = make_field_boundary_conditions(boundary_conditions)

    return HeatDiffusionModel(backend, grid, boundary_conditions, physics, iter_params, fields)
end

fields(model::HeatDiffusionModel) = model.fields
grid(model::HeatDiffusionModel) = model.grid

function advance_iteration!(model::HeatDiffusionModel, t, Δt; async = true)
    (; T, T_o, q) = model.fields
    (; Δτ) = model.iter_params
    λ_ρCp = model.physics.properties
    Δ = NamedTuple{(:x, :y, :z)}(spacing(model.grid))
    nx, ny, nz = size(model.grid)
    backend = model.backend

    set_bcs!(bcs) = _apply_bcs!(model.backend, model.grid, model.fields, bcs)

    # flux
    update_q!(backend, 256, (nx+1, ny+1, nz+1))(q, T, λ_ρCp, Δτ, Δ)
    set_bcs!(model.boundary_conditions.flux)
    # mass balance
    update_T!(backend, 256, (nx, ny, nz))(T, T_o, q, Δt, Δτ, Δ)
    set_bcs!(model.boundary_conditions.value)

    async || synchronize(backend)
    return
end

function advance_timestep!(model::HeatDiffusionModel, t, Δt)
    # TODO

    return
end

end