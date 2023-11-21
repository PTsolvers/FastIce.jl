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

using GLMakie
Makie.inline!(true)

function main()
    backend = CPU()
    arch = Architecture(backend)
    set_device!(arch)

    # physics
    ebg = 1.0

    size_l = (64, 64, 64)

    b_width = (16, 8, 4) #(128, 32, 4)#

    grid = CartesianGrid(; origin=(-0.5, -0.5, 0.0),
                           extent=(1.0, 1.0, 1.0),
                           size=size_l)

    psh_x(x, _, _) = -x * ebg
    psh_y(_, y, _) =  y * ebg

    x_bc = BoundaryFunction(psh_x; reduce_dims=false)
    y_bc = BoundaryFunction(psh_y; reduce_dims=false)

    free_slip    = SBC(0.0, 0.0, 0.0)
    free_surface = TBC(0.0, 0.0, 0.0)

    boundary_conditions = (x = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                           y = (VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                           z = (free_slip, free_surface))

    gravity = (x=0.0, y=0.0, z=0.0)

    # numerics
    niter   = 10maximum(size(grid))
    ncheck  = maximum(size(grid))

    r       = 0.7
    re_mech = 10π
    lτ_re_m = minimum(extent(grid)) / re_mech
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 3.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    iter_params = (η_rel=1e-1,
                   Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

    physics = (rheology=GlensLawRheology(1),)
    other_fields = (A=Field(backend, grid, Center()),)

    init_incl(x, y, z, x0, y0, z0, r, Ai, Am) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, Ai, Am)
    set!(other_fields.A, grid, init_incl; parameters=(x0=0.0, y0=0.0, z0=0.5, r=0.2, Ai=1e-1, Am=1.0))

    model = IsothermalFullStokesModel(;
                                      arch,
                                      grid,
                                      physics,
                                      gravity,
                                      boundary_conditions,
                                      outer_width=b_width,
                                      iter_params,
                                      other_fields)

    fig = Figure(; resolution=(1200, 1000), fontsize=32)
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

    fill!(parent(model.fields.Pr), 0.0)
    foreach(x -> fill!(parent(x), 0.0), model.fields.τ)

    set!(model.fields.V.x, grid, psh_x)
    set!(model.fields.V.y, grid, psh_y)
    set!(model.fields.V.z, 0.0)
    set!(model.fields.η, other_fields.A)

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.rheology)

    display(fig)

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0; async=false)
        (iter % ncheck == 0) && println("iter/nx = $(iter/maximum(size(grid)))")
        if iter % ncheck == 0
            plt.Pr[3][] = interior(model.fields.Pr)[:, size(grid, 2)÷2, :]
            plt.Vx[3][] = interior(model.fields.V.x)[:, size(grid, 2)÷2, :]
            plt.Vy[3][] = interior(model.fields.V.y)[:, size(grid, 2)÷2, :]
            plt.Vz[3][] = interior(model.fields.V.z)[:, size(grid, 2)÷2, :]
            display(fig)
        end
    end

    return
end

main()
