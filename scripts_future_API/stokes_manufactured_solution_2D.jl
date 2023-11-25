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

using LinearAlgebra
using KernelAbstractions

using CairoMakie

# manufactured solution for the confined Stokes flow with free-slip boundaries
# velocity
vx(x, y) = sin(π * x) * cos(π * y)
vy(x, y) = -cos(π * x) * sin(π * y)
# deviatoric stress
τxx(x, y, η) = 2 * η * π * cos(π * x) * cos(π * y)
τyy(x, y, η) = -2 * η * π * cos(π * x) * cos(π * y)
τxy(x, y, η) = zero(x)
# forcing terms
ρgx(x, y, η) = -2 * η * π^2 * sin(π * x) * cos(π * y)
ρgy(x, y, η) = 2 * η * π^2 * cos(π * x) * sin(π * y)

@views function main()
    backend = CPU()
    arch = Architecture(backend, 2)
    set_device!(arch)

    # physics
    η0 = 1.0
    A0 = 0.5

    # geometry
    grid = CartesianGrid(; origin=(-1.0, -1.0),
                         extent=(2.0, 2.0),
                         size=(256, 256))

    free_slip = SBC(0.0, 0.0)
    xface = (Vertex(), Center())
    yface = (Center(), Vertex())

    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip))

    gravity = (x=FunctionField(ρgx, grid, xface; parameters=η0),
               y=FunctionField(ρgy, grid, yface; parameters=η0))

    # numerics
    niter  = 10maximum(size(grid))
    ncheck = maximum(size(grid))

    # PT params
    r       = 0.7
    re_mech = 8π
    lτ_re_m = minimum(extent(grid)) / re_mech
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 1.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    iter_params = (η_rel=1e-1,
                   Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, xy=dτ_r), V=(x=nudτ, y=nudτ)))

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

    set!(model.fields.Pr, 0.0)
    foreach(x -> set!(x, 0.0), model.fields.τ)
    foreach(x -> set!(x, 0.0), model.fields.V)

    set!(other_fields.A, A0)
    set!(model.fields.η, grid, (grid, loc, I, fields) -> physics.rheology(grid, I, fields); discrete=true, parameters=(model.fields,))

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.rheology)

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0)
        if iter % ncheck == 0
            compute_residuals!(model)
            err = (Pr = norm(model.fields.r_Pr, Inf),
                   Vx = norm(model.fields.r_V.x, Inf),
                   Vy = norm(model.fields.r_V.y, Inf))
            if any(.!isfinite.(values(err)))
                error("simulation failed, err = $err")
            end
            iter_nx = iter / maximum(size(grid))
            @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e]\n", iter_nx, err...)
        end
    end

    Vx_m = Field(backend, grid, location(model.fields.V.x))
    Vy_m = Field(backend, grid, location(model.fields.V.y))

    set!(Vx_m, grid, vx)
    set!(Vy_m, grid, vy)

    dVx = interior(Vx_m) .- interior(model.fields.V.x)
    dVy = interior(Vy_m) .- interior(model.fields.V.y)

    err = (norm(dVx, Inf) / norm(Vx_m, Inf),
           norm(dVy, Inf) / norm(Vy_m, Inf))

    @printf("err = [Vx: %1.3e, Vy: %1.3e]\n", err...)

    fig = Figure()
    ax  = Axis(fig[1,1][1,1]; aspect=DataAspect())
    hm  = heatmap!(ax, xvertices(grid), ycenters(grid), interior(model.fields.V.x); colormap=:turbo)
    Colorbar(fig[1,1][1,2], hm)

    display(fig)
    save("ms_2D.png", fig)

    return
end

main()
