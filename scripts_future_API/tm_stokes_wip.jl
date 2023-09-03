using FastIce
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal
using FastIce.Physics

using KernelAbstractions

using GLMakie

# physics
ebg = 1.0

grid = CartesianGrid(
    origin = (-0.5, -0.5, 0.0),
    extent = ( 1.0,  1.0, 1.0),
    size   = (  64,   64, 64 ),
)

psh_x(x, _, _) = -x*ebg
psh_y(_, y, _) =  y*ebg

x_bc = BoundaryFunction(psh_x; reduce_dims=false)
y_bc = BoundaryFunction(psh_y; reduce_dims=false)

boundary_conditions = (
    west   = BoundaryCondition{Velocity}(x_bc, y_bc, 0.0),
    east   = BoundaryCondition{Velocity}(x_bc, y_bc, 0.0),
    south  = BoundaryCondition{Velocity}(x_bc, y_bc, 0.0),
    north  = BoundaryCondition{Velocity}(x_bc, y_bc, 0.0),
    bottom = BoundaryCondition{Velocity}(0.0 , 0.0 , 0.0),
    top    = BoundaryCondition{Velocity}(0.0 , 0.0 , 0.0),
)

r       = 0.7
re_mech = 10π
lτ_re_m = minimum(extent(grid)) / re_mech
vdτ     = minimum(spacing(grid)) / sqrt(10.1)
θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
dτ_r    = 1.0 / (θ_dτ + 1.0)
nudτ    = vdτ * lτ_re_m

iter_params = (
    η_rel = 1e-1,
    Δτ = ( Pr = r / θ_dτ, τ = (xx = dτ_r, yy = dτ_r, zz = dτ_r, xy = dτ_r, xz = dτ_r, yz = dτ_r), V = (x = nudτ, y = nudτ, z = nudτ)),
)

backend = CPU()

physics = (rheology = GlensLawRheology(1), )
other_fields = (
    A = Field(backend, grid, Center()),
)

init_incl(x, y, z, x0, y0, z0, r, Ai, Am) = ifelse((x-x0)^2 + (y-y0)^2 + (z-z0)^2 < r^2, Ai, Am)
set!(other_fields.A, grid, init_incl; parameters = (x0 = 0.0, y0 = 0.0, z0 = 0.5, r = 0.2, Ai = 1e1, Am = 1.0))

model = IsothermalFullStokesModel(;
    backend,
    grid,
    physics,
    boundary_conditions,
    iter_params,
    other_fields
)

fig = Figure(resolution=(1000,1000), fontsize=32)
axs = (
    Pr = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Pr"),
    Vx = Axis(fig[2,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Vx"),
    Vy = Axis(fig[2,2][1,1]; aspect=DataAspect(), xlabel="x", ylabel="y", title="Vy"),
)

plt = (
    Pr = heatmap!(axs.Pr, xcenters(grid), ycenters(grid), interior(model.fields.Pr)[:, :, size(grid,3)÷2]; colormap=:turbo),
    Vx = heatmap!(axs.Vx, xvertices(grid), ycenters(grid), interior(model.fields.V.x)[:, :, size(grid,3)÷2]; colormap=:turbo),
    Vy = heatmap!(axs.Vy, xcenters(grid), yvertices(grid), interior(model.fields.V.y)[:, :, size(grid,3)÷2]; colormap=:turbo),
)
Colorbar(fig[1,1][1,2], plt.Pr)
Colorbar(fig[2,1][1,2], plt.Vx)
Colorbar(fig[2,2][1,2], plt.Vy)

set!(model.fields.Pr, 0.0)
foreach(x -> set!(x, 0.0), model.fields.τ)
Isothermal._apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.stress)

set!(model.fields.V.x, grid, psh_x)
set!(model.fields.V.y, grid, psh_y)
set!(model.fields.V.z, 0.0)
Isothermal._apply_bcs!(model.backend, model.grid, model.fields, model.boundary_conditions.velocity)

set!(model.fields.η, other_fields.A)
extrapolate!(model.fields.η)

for it in 1:10
    advance_iteration!(model, 0.0, 1.0; async = false)
    if it % 10 == 0
        plt.Pr[3][] = interior(model.fields.Pr)[:, :, size(grid,3)÷2]
        plt.Vx[3][] = interior(model.fields.V.x)[:, :, size(grid,3)÷2]
        plt.Vy[3][] = interior(model.fields.V.y)[:, :, size(grid,3)÷2]
        yield()
    end
end
