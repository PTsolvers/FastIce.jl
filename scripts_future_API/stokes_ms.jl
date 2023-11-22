using Printf

using FastIce
using FastIce.Architectures
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal
using FastIce.Physics
using FastIce.KernelLaunch

const VBC = BoundaryCondition{Velocity}
const TBC = BoundaryCondition{Traction}
const SBC = BoundaryCondition{Slip}

using KernelAbstractions

using CairoMakie

# manufactured solution for the confined Stokes flow with free-slip boundaries
# helper functions
f(ξ, η) = cos(π * ξ) * (η^2 - 1)^2 -
          cos(π * η) * (ξ^2 - 1)^2
g(ξ, η) = sin(π * η) * (ξ^2 - 1) * ξ -
          sin(π * ξ) * (η^2 - 1) * η
p(ξ, η) = cos(π * η) * (1 - 3 * ξ^2) * 2 -
          cos(π * ξ) * (1 - 3 * η^2) * 2
# velocity
vx(x, y, z) = sin(π * x) * f(y, z)
vy(x, y, z) = sin(π * y) * f(z, x)
vz(x, y, z) = sin(π * z) * f(x, y)
# diagonal deviatoric stress
τxx(x, y, z, η) = 2 * η * π * cos(π * x) * f(y, z)
τyy(x, y, z, η) = 2 * η * π * cos(π * y) * f(z, x)
τzz(x, y, z, η) = 2 * η * π * cos(π * z) * f(x, y)
# off-diagonal deviatoric stress
τxy(x, y, z, η) = 4 * η * cos(π * z) * g(x, y)
τxz(x, y, z, η) = 4 * η * cos(π * y) * g(z, x)
τyz(x, y, z, η) = 4 * η * cos(π * x) * g(y, z)
# forcing terms
ρgx(x, y, z, η) = -2 * η * sin(π * x) * (f(y, z) * π^2 - p(y, z))
ρgy(x, y, z, η) = -2 * η * sin(π * y) * (f(z, x) * π^2 - p(z, x))
ρgz(x, y, z, η) = -2 * η * sin(π * z) * (f(x, y) * π^2 - p(x, y))

function main()
    backend = CPU()
    arch = Architecture(backend)
    set_device!(arch)

    # physics
    η0 = 1.0
    A0 = 0.5

    # geometry
    grid = CartesianGrid(; origin=(-1.0, -1.0, -1.0),
                         extent=(2.0, 2.0, 2.0),
                         size=(32, 32, 32))

    free_slip = SBC(0.0, 0.0, 0.0)
    xface = (Vertex(), Center(), Center())
    yface = (Center(), Vertex(), Center())
    zface = (Center(), Center(), Vertex())

    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip),
                           z = (free_slip, free_slip))

    gravity = (x=FunctionField(ρgx, grid, xface; parameters=η0),
               y=FunctionField(ρgy, grid, yface; parameters=η0),
               z=FunctionField(ρgz, grid, zface; parameters=η0))

    # numerics
    r       = 0.7
    re_mech = 5π
    lτ_re_m = minimum(extent(grid)) / re_mech
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    iter_params = (η_rel=1e-1,
                   Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

    physics = (rheology=GlensLawRheology(1),)
    other_fields = (A=Field(backend, grid, Center()),)

    model = IsothermalFullStokesModel(;
                                      arch,
                                      grid,
                                      physics,
                                      gravity,
                                      boundary_conditions,
                                      iter_params,
                                      other_fields)

    fig = Figure(; resolution=(1200, 900), fontsize=32)
    axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pr"),
           Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vx"),
           Vy = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vy"),
           Vz = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vz"))
    plt = (Pr = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), interior(model.fields.Pr)[:, size(grid, 2)÷2, :]; colormap=:turbo),
           Vx = heatmap!(axs.Vx, xvertices(grid), zcenters(grid), interior(model.fields.V.x)[:, size(grid, 2)÷2, :]; colormap=:turbo),
           Vy = heatmap!(axs.Vy, xcenters(grid), zcenters(grid), interior(model.fields.V.y)[:, size(grid, 2)÷2, :]; colormap=:turbo),
           Vz = heatmap!(axs.Vz, xcenters(grid), zvertices(grid), interior(model.fields.V.z)[:, size(grid, 2)÷2, :]; colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Vz)

    set!(model.fields.Pr, 0.0)
    foreach(x -> set!(x, 0.0), model.fields.τ)
    foreach(x -> set!(x, 0.0), model.fields.V)

    set!(other_fields.A, A0)
    set!(model.fields.η, grid, (grid, loc, I, fields) -> physics.rheology(grid, I, fields); discrete=true, parameters=(model.fields,))

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.rheology)

    for it in 1:20maximum(size(grid))
        advance_iteration!(model, 0.0, 1.0; async=false)
        if it % maximum(size(grid)) == 0
            println("iter/nx = $(it/maximum(size(grid)))")
        end
    end

    Vx_m = Field(backend, grid, location(model.fields.V.x))
    Vy_m = Field(backend, grid, location(model.fields.V.y))
    Vz_m = Field(backend, grid, location(model.fields.V.z))

    set!(Vx_m, grid, vx)
    set!(Vy_m, grid, vy)
    set!(Vz_m, grid, vz)

    dVx = abs.(interior(Vx_m) .- interior(model.fields.V.x))
    dVy = abs.(interior(Vy_m) .- interior(model.fields.V.y))
    dVz = abs.(interior(Vz_m) .- interior(model.fields.V.z))

    plt.Pr[3][] = interior(model.fields.Pr)[:, size(grid, 2)÷2+1, :]
    plt.Vx[3][] = dVx[:, size(grid, 2)÷2+1, :]
    plt.Vy[3][] = dVy[:, size(grid, 2)÷2+1, :]
    plt.Vz[3][] = dVz[:, size(grid, 2)÷2+1, :]
    display(fig)

    @show yvertex(grid, size(grid, 2)÷2+1)

    err = (maximum(dVx) / maximum(abs.(interior(Vx_m))),
           maximum(dVy) / maximum(abs.(interior(Vy_m))),
           maximum(dVz) / maximum(abs.(interior(Vz_m))))

    @printf("err = [Vx: %1.3e, Vy: %1.3e, Vz: %1.3e]\n", err...)

    return
end

main()