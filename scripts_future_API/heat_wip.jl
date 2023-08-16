using FastIce
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.Diffusion.Heat

using KernelAbstractions

using GLMakie
Makie.inline!(true)

backend = CPU()

# physics
λ_ρCp = 1.0
Δt    = 1.0

grid = CartesianGrid(
    origin = (-0.5, -0.5, 0.0),
    extent = ( 1.0,  1.0, 1.0),
    size   = (  32,   32,  32);
)

boundary_conditions = (
    west  = BoundaryCondition{Flux}(0.0),
    east  = BoundaryCondition{Flux}(0.0),
    south = BoundaryCondition{Value}(-0.1),
    north = BoundaryCondition{Value}(0.3),
    bot   = BoundaryCondition{Flux}(0.0),
    top   = BoundaryCondition{Flux}(0.0),
)

cfl  = 0.99/sqrt(ndims(grid))
re   = π + sqrt(π^2 + maximum(extent(grid))^2 / λ_ρCp / Δt)
θ_dτ = maximum(extent(grid)) / re / cfl / minimum(spacing(grid))
β_dτ = (re * λ_ρCp) / (cfl * minimum(spacing(grid)) * maximum(extent(grid)))

physics = (properties = λ_ρCp,)

iter_params = (
    Δτ = ( T = 1.0 / (Δt + β_dτ),
           q = 1.0 / (1.0 + θ_dτ)
    ),
)

init_fields = (
    T   = Field(backend, grid, Center(); halo=1),
    T_o = Field(backend, grid, Center(); halo=1),
)

init_gauss(x, y, z, x0, y0, z0, A, σ) = A * exp(-((x-x0)/σ)^2 - ((y-y0)/σ)^2 - ((z-z0)/σ)^2)
set!(init_fields.T, grid, init_gauss; continuous = true, parameters = (x0 = 0.0, y0 = 0.0, z0 = 0.5, A = 1.0, σ = 0.1))

model = HeatDiffusionModel(;
    backend,
    grid,
    physics,
    boundary_conditions,
    iter_params,
    init_fields
)

set!(model.fields.q.x, 0.0)
set!(model.fields.q.y, 0.0)
set!(model.fields.q.z, 0.0)
Heat.apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.flux)
Heat.apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.value)

set!(model.fields.T_o, model.fields.T)

fig = Figure(resolution=(1000,1000), fontsize=32)
ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y")
plt = heatmap!(ax, xcenters(grid), ycenters(grid), interior(model.fields.T)[:, :, size(grid,3)÷2]; colormap=:turbo)
Colorbar(fig[1,1][1,2], plt)
display(fig)

for it in 1:2000
    advance_iteration!(model, 0.0, Δt; async = false)
    if it % 100 == 0
        plt[3][] = interior(model.fields.T)[:, :, size(grid,3)÷2]
        display(fig)
    end
end

# plt[3][] = parent(model.fields.T)[2:end-1, 2:end-1, size(grid,3)÷2]