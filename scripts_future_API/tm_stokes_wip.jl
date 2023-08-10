using FastIce
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal

using KernelAbstractions

using GLMakie

# physics
ebg = 1.0

grid = CartesianGrid(
    origin = (-0.5, -0.5, 0.0),
    extent = ( 1.0,  1.0, 1.0),
    size   = (  64,   64, 64 );
)

boundary_conditions = (
    west  = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    east  = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    south = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    north = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    bot   = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
    top   = BoundaryCondition{Velocity}(0.0, 0.0, 0.0),
)

r       = 0.7
re_mech = 5π
lτ_re_m = minimum(extent(grid)) / re_mech
vdτ     = minimum(spacing(grid)) / sqrt(5.1)
θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
dτ_r    = 1.0 ./ (θ_dτ .+ 1.0)
nudτ    = vdτ * lτ_re_m

iter_params = (
    η_rel = 1e-1,
    Δτ = ( Pr = r / θ_dτ, τ = (xx = dτ_r, yy = dτ_r, zz = dτ_r, xy = dτ_r, xz = dτ_r, yz = dτ_r), V = (x = nudτ, y = nudτ, z = nudτ)),
)

physics = (rheology = GlensLawRheology(1.0, 1))

model = IsothermalFullStokesModel(;
    backend = CPU(),
    grid,
    boundary_conditions,
    iter_params,
)

set!(model.fields.Pr, 0.0)
foreach(x -> set!(x, 0.0), model.fields.τ)
Isothermal.apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.stress)

foreach(x -> set!(x, 0.0), model.fields.V)

# set!(model.fields.V.x, grid, (x, y, z, ebg) -> -x*ebg; continuous=true, parameters = (ebg, ))
# set!(model.fields.V.y, grid, (x, y, z, ebg) ->  y*ebg; continuous=true, parameters = (ebg, ))
# set!(model.fields.V.z, 0.0)
Isothermal.apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.velocity)

init_incl(x, y, z, x0, y0, z0, r, ηi, ηm) = ifelse((x-x0)^2 + (y-y0)^2 + (z-z0)^2 < r^2, ηi, ηm)
set!(model.fields.η, grid, init_incl; continuous = true, parameters = (x0 = 0.0, y0 = 0.0, z0 = 0.5, r = 0.2, ηi = 1e-1, ηm = 1.0))
extrapolate!(data(model.fields.η))


for it in 1:10
    advance_iteration!(model, 0.0, 1.0; async = false)
end

fig = Figure(resolution=(1000,1000), fontsize=32)
ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y")

plt = heatmap!(ax, xcenters(grid), ycenters(grid), interior(model.fields.Pr)[:, :, size(grid,3)÷2]; colormap=:turbo)
Colorbar(fig[1,1][1,2], plt)

plt[3][] = interior(model.fields.Pr)[:, :, size(grid,3)÷2]

yield()