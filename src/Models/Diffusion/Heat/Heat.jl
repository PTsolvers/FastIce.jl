module Heat

export BoundaryCondition, Flux, Value
export HeatDiffusionModel, advance_iteration!, advance_timestep!

using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields
using Chmy.BoundaryConditions
using Chmy.KernelLaunch

include("kernels.jl")
include("boundary_conditions.jl")

struct HeatDiffusionModel{Arch,Grid,BC,Physics,IterParams,Temperature,HeatFlux,KL}
    arch::Arch
    grid::Grid
    boundary_conditions::BC
    physics::Physics
    iter_params::IterParams
    temperature::Temperature
    heat_flux::HeatFlux
    launcher::KL
end

function HeatDiffusionModel(; arch, grid, boundary_conditions, physics, iter_params, outer_width=nothing)
    temperature = Field(arch, grid, Center())
    heat_flux = VectorField(arch, grid)

    if isnothing(outer_width)
        outer_width = ntuple(_ -> 2, Val(ndims(grid)))
    end

    launcher = KernelLaunch(arch, grid; outer_width)

    boundary_conditions = make_field_boundary_conditions(boundary_conditions)

    return HeatDiffusionModel(arch, grid, boundary_conditions, physics, iter_params, temperature, heat_flux, launcher)
end

function advance_iteration!(model::HeatDiffusionModel, t, Δt; async=false)
    (; T, T_o, q) = model.fields
    (; Δτ) = model.iter_params
    λ_ρCp = model.physics.λ_ρCp
    arch = model.arch
    grid = model.grid
    launch = model.launcher
    boundary_conditions = model.boundary_conditions

    # flux

    launch(arch, grid, update_q! => (q, T, λ_ρCp, Δτ, grid); bc=boundary_conditions.flux)
    launch(arch, grid, update_T! => (T, T_o, q, Δt, Δτ, Δ); bc=boundary_conditions.temperature)

    async || synchronize(Architectures.get_backend(arch))
    return
end

end
