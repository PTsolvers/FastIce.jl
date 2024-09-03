using Chmy.Architectures
using Chmy.Grids
using Chmy.Fields
using Chmy.BoundaryConditions

using FastIce, FastIce.Models.FullStokes.Isothermal, FastIce.Physics

using KernelAbstractions

using LinearAlgebra
using Printf

using CairoMakie

backend = CPU()
arch = Arch(backend)

outer_width = (4, 4, 4) #(128, 32, 4)#

grid = UniformGrid(arch;
                   origin=(-0.5, -0.5, 0.0),
                   extent=(1.0, 1.0, 1.0),
                   dims=(62, 62, 62))

FastIce.greet_fast(; bold=true, color=:blue)

const VBC = BoundaryCondition{Velocity}
const SBC = BoundaryCondition{Slip}
const TBC = BoundaryCondition{Traction}

free_slip    = SBC(0.0, 0.0, 0.0)
free_surface = TBC(0.0, 0.0, 0.0)
no_slip      = VBC(0.0, 0.0, 0.0)

boundary_conditions = (x=(free_slip, free_slip),
                       y=(free_slip, free_slip),
                       z=(no_slip, free_surface))

gravity = (x=ZeroField{Float64}(),
           y=ZeroField{Float64}(),
           z=ValueField(1.0))

# numerics
niter   = 100maximum(size(grid, Center()))
ncheck  = 2maximum(size(grid, Center()))
do_visu = true

r       = 0.7
re_mech = 2π
lτ_re_m = minimum(extent(grid, Vertex())) / re_mech
vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 1.5)
θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
dτ_r    = 1.0 / (θ_dτ + 1.0)
nudτ    = vdτ * lτ_re_m

solver_params = (Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)),)

init_incl(x, y, z, x0, y0, z0, r, ηi, ηm) = ifelse((x - x0)^2 + (y - y0)^2 + (z - z0)^2 < r^2, ηi, ηm)

η        = FunctionField(init_incl, grid, Center(); parameters=(x0=0.0, y0=0.0, z0=0.5, r=0.1, ηi=1e-1, ηm=1.0))
rheology = LinearViscousRheology(η)

model = IsothermalFullStokesModel(;
                                  arch,
                                  grid,
                                  boundary_conditions,
                                  gravity,
                                  rheology,
                                  solver_params,
                                  outer_width)

fig = Figure()
axs = (Pr = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pr"),
       Vx = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vx"),
       Vy = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vy"),
       Vz = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vz"))
plt = (Pr = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), Array(interior(model.stress.P)[:, size(grid, Center(), Val(2))÷2+1, :]); colormap=:turbo),
       Vx = heatmap!(axs.Vx, xvertices(grid), zcenters(grid), Array(interior(model.velocity.V.x)[:, size(grid, Center(), Val(2))÷2+1, :]); colormap=:turbo),
       Vy = heatmap!(axs.Vy, xcenters(grid), zcenters(grid), Array(interior(model.velocity.V.y)[:, size(grid, Vertex(), Val(2))÷2+1, :]); colormap=:turbo),
       Vz = heatmap!(axs.Vz, xcenters(grid), zvertices(grid), Array(interior(model.velocity.V.z)[:, size(grid, Center(), Val(2))÷2+1, :]); colormap=:turbo))
Colorbar(fig[1, 1][1, 2], plt.Pr)
Colorbar(fig[1, 2][1, 2], plt.Vx)
Colorbar(fig[2, 1][1, 2], plt.Vy)
Colorbar(fig[2, 2][1, 2], plt.Vz)

set!(model.stress.P, 0.0)
set!(model.stress.τ, 0.0)
set!(model.velocity.V, 0.0)

set!(model.viscosity.η, η)
set!(model.viscosity.η_next, η)

bc!(arch, grid, model.viscosity.η => Neumann())
bc!(arch, grid, model.viscosity.η_next => Neumann())

for iter in 1:niter
    advance_iteration!(model, 0.0, 1.0)
    if (iter % ncheck == 0)
        compute_residuals!(model)
        err = (Pr = norm(model.residual.r_P, Inf),
               Vx = norm(model.residual.r_V.x, Inf),
               Vy = norm(model.residual.r_V.y, Inf),
               Vz = norm(model.residual.r_V.z, Inf))
        if any(.!isfinite.(values(err)))
            error("simulation failed, err = $err")
        end
        iter_nx = iter / maximum(size(grid, Center()))
        @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e, Vz = %1.3e]\n", iter_nx, err...)
        if do_visu
            plt.Pr[3][] = interior(model.stress.P)[:, size(grid, Center(), Val(2))÷2+1, :]
            plt.Vx[3][] = interior(model.velocity.V.x)[:, size(grid, Center(), Val(2))÷2+1, :]
            plt.Vy[3][] = interior(model.velocity.V.y)[:, size(grid, Vertex(), Val(2))÷2+0, :]
            plt.Vz[3][] = interior(model.velocity.V.z)[:, size(grid, Center(), Val(2))÷2+1, :]
            # yield()
            display(fig)
        end
    end
end
