using FastIce
using FastIce.Architectures
using FastIce.Grids
using FastIce.Fields
using FastIce.Utils
using FastIce.BoundaryConditions
using FastIce.Models.FullStokes.Isothermal
using FastIce.Physics
using FastIce.KernelLaunch
using FastIce.Writers

const VBC = BoundaryCondition{Velocity}
const TBC = BoundaryCondition{Traction}
const SBC = BoundaryCondition{Slip}

using LinearAlgebra, Printf
using KernelAbstractions
# using CUDA
# using AMDGPU

using CairoMakie
# using GLMakie
# Makie.inline!(true)

@views function main(; do_visu=false, do_save=false, transient=false)
    backend = CPU()
    arch = Architecture(backend)
    set_device!(arch)

    # physics
    ebg = 2.0

    outer_width = (4, 4, 4) #(128, 32, 4)#

    grid = CartesianGrid(; origin=(-0.5, -0.5, 0.0),
                         extent=(1.0, 1.0, 1.0),
                         size=(62, 62, 62))

    FastIce.greet_fast(bold=true, color=:blue)

    psh_x(x, _, _) = -x * ebg
    psh_y(_, y, _) = y * ebg

    x_bc = BoundaryFunction(psh_x; reduce_dims=false)
    y_bc = BoundaryFunction(psh_y; reduce_dims=false)

    free_slip    = SBC(0.0, 0.0, 0.0)
    free_surface = TBC(0.0, 0.0, 0.0)

    boundary_conditions = (x=(VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                           y=(VBC(x_bc, y_bc, 0.0), VBC(x_bc, y_bc, 0.0)),
                           z=(free_slip, free_surface))
    # TODO: Add ConstantField
    ρg(x, y, z) = 0.0
    gravity = (x=FunctionField(ρg, grid, (Vertex(), Center(), Center())),
               y=FunctionField(ρg, grid, (Center(), Vertex(), Center())),
               z=FunctionField(ρg, grid, (Center(), Center(), Vertex())))

    # numerics
    niter  = 20maximum(size(grid))
    ncheck = 2maximum(size(grid))

    r       = 0.7
    re_mech = 10π
    lτ_re_m = minimum(extent(grid)) / re_mech
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
    plt = (Pr = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), Array(interior(model.stress.Pr)[:, size(grid, 2)÷2+1, :]); colormap=:turbo),
           Vx = heatmap!(axs.Vx, xvertices(grid), zcenters(grid), Array(interior(model.velocity.x)[:, size(grid, 2)÷2+1, :]); colormap=:turbo),
           Vy = heatmap!(axs.Vy, xcenters(grid), zcenters(grid), Array(interior(model.velocity.y)[:, size(grid, 2)÷2+1, :]); colormap=:turbo),
           Vz = heatmap!(axs.Vz, xcenters(grid), zvertices(grid), Array(interior(model.velocity.z)[:, size(grid, 2)÷2+1, :]); colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vx)
    Colorbar(fig[2, 1][1, 2], plt.Vy)
    Colorbar(fig[2, 2][1, 2], plt.Vz)

    fill!(parent(model.stress.Pr), 0.0)
    foreach(x -> fill!(parent(x), 0.0), model.stress.τ)

    set!(model.velocity.x, grid, psh_x)
    set!(model.velocity.y, grid, psh_y)
    set!(model.velocity.z, 0.0)
    set!(model.viscosity.η, η)
    set!(model.viscosity.η_next, η)

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.rheology)

    if do_save
        h5names = String[]
        ts = Float64[]
        isave = 0
        fields = Dict("Pr" => model.fields.Pr, "A" => model.fields.A)
        outdir = "out_visu"
        mkpath(outdir)
    end

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0)
        if (iter % ncheck == 0)
            compute_residuals!(model)
            err = (Pr = norm(model.residual.r_Pr, Inf),
                   Vx = norm(model.residual.r_V.x, Inf),
                   Vy = norm(model.residual.r_V.y, Inf),
                   Vz = norm(model.residual.r_V.z, Inf))
            if any(.!isfinite.(values(err)))
                error("simulation failed, err = $err")
            end
            iter_nx = iter / maximum(size(grid))
            @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e, Vz = %1.3e]\n", iter_nx, err...)
            if do_visu
                plt.Pr[3][] = interior(model.stress.Pr)[:, size(grid, 2)÷2+1, :]
                plt.Vx[3][] = interior(model.velocity.x)[:, size(grid, 2)÷2+1, :]
                plt.Vy[3][] = interior(model.velocity.y)[:, size(grid, 2)÷2+0, :]
                plt.Vz[3][] = interior(model.velocity.z)[:, size(grid, 2)÷2+1, :]
                # yield()
                display(fig)
            end
            if do_save && transient # saving to disk
                isave+=1
                out_h5 = @sprintf("step_%04d.h5", isave)
                @info "saving HDF5 file"
                write_h5(arch, grid, joinpath(outdir, out_h5), fields)
                push!(ts, isave)
                push!(h5names, out_h5)
            end
        end
    end

    if do_save && transient # saving to disk
        @info "saving XDMF file"
        write_xdmf(arch, grid, joinpath(outdir, "results.xdmf3"), fields, h5names, ts)
    elseif do_save && !transient # saving to disk
        out_h5 = "results.h5"
        @info "saving HDF5 file"
        write_h5(arch, grid, joinpath(outdir, out_h5), fields)
        push!(h5names, out_h5)

        @info "saving XDMF file"
        write_xdmf(arch, grid, joinpath(outdir, "results.xdmf3"), fields, h5names)
    end
    return
end

main(; do_visu=false, do_save=true, transient=false)
