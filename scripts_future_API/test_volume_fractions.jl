using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.GridOperators, Chmy.Fields, Chmy.KernelLaunch, Chmy.BoundaryConditions
using KernelAbstractions

using FastIce
using FastIce.LevelSets
using FastIce.Physics
using FastIce.Models.ImmersedBoundaryFullStokes.Isothermal

using FastIce.Writers

# using CUDA
# using GLMakie
using CairoMakie

using Printf
using LinearAlgebra

function main(backend; res)
    arch = Arch(backend)
    grid = UniformGrid(arch; origin=(-1, -1, 0), extent=(2, 2, 1), dims=res)

    grid_2D = UniformGrid(arch; origin=(-1, -1), extent=(2, 2), dims=(100, 100))

    # bed parameters
    amp    = 0.05
    ωb     = 10π / 2.0
    αx     = -tan(π / 20)
    αy     = -tan(π / 25)
    z0_bed = 0.3

    # ice parameters
    x0  = -0.1
    y0  = -0.1
    z0  = -0.1
    rad = 0.9

    bed = FunctionField(grid_2D, Vertex(); parameters=(amp, ωb, αx, αy, z0_bed)) do x, y, amp, ωb, αx, αy, z0_bed
        return z0_bed + x * αx + y * αy + amp * sin(ωb * x) * cos(ωb * y)
    end

    ice = FunctionField(grid_2D, Vertex(); parameters=(x0, y0, z0, rad)) do x, y, x0, y0, z0, rad
        return z0 + sqrt(max(rad^2 - (x - x0)^2 - (y - y0)^2, 0.0))
    end

    ψ = (ns=Field(arch, grid, Vertex()),
         na=Field(arch, grid, Vertex()))

    launch = Launcher(arch, grid)

    compute_levelset_from_dem!(arch, launch, ψ.na, ice, grid_2D, grid)
    compute_levelset_from_dem!(arch, launch, ψ.ns, bed, grid_2D, grid)

    invert_levelset!(arch, launch, ψ.ns, grid)

    free_slip = BoundaryCondition{Traction}(0.0, 0.0, 0.0)

    boundary_conditions = (x=(free_slip, free_slip),
                           y=(free_slip, free_slip),
                           z=(free_slip, free_slip))

    gravity = (x=ValueField(0.0),
               y=ValueField(0.0),
               z=ValueField(1.0))

    # numerics
    niter   = 25maximum(size(grid, Center()))
    ncheck  = 5maximum(size(grid, Center()))
    do_visu = true
    do_h5_save = true

    r       = 0.9
    re_mech = 5π
    lτ_re_m = minimum(extent(grid, Vertex())) / re_mech
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 1.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    solver_params = (Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)),)

    rheology = LinearViscousRheology(OneField{Float64}())

    model = IsothermalImmersedBoundaryFullStokesModel(;
                                                      arch,
                                                      grid,
                                                      boundary_conditions,
                                                      gravity,
                                                      rheology,
                                                      solver_params,
                                                      level_sets=ψ)

    # init
    ω_from_ψ!(arch, launch, model.field_masks.ns, ψ.ns, grid)
    ω_from_ψ!(arch, launch, model.field_masks.na, ψ.na, grid)

    set!(model.viscosity.η, rheology.η)
    set!(model.viscosity.η_next, rheology.η)

    bc!(arch, grid, model.viscosity.η => Neumann())
    bc!(arch, grid, model.viscosity.η_next => Neumann())

    nx, ny, nz = size(grid, Center())

    iy_sl = ny ÷ 2

    Vm = Field(arch, grid, Center())
    set!(Vm, grid, (g, loc, ix, iy, iz, V) -> sqrt(lerp(V.x, loc, g, ix, iy, iz)^2 +
                                                   lerp(V.y, loc, g, ix, iy, iz)^2 +
                                                   lerp(V.z, loc, g, ix, iy, iz)^2); discrete=true, parameters=(model.velocity.V,))

    fig = Figure(; size=(800, 400))
    axs = (Pr  = Axis(fig[1, 1][1, 1]; aspect=DataAspect(), ylabel="z", title="Pr"),
           Vm  = Axis(fig[1, 2][1, 1]; aspect=DataAspect(), title="|V|"),
           ωna = Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x"),
           ωns = Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x"))
    plt = (Pr  = heatmap!(axs.Pr, xcenters(grid), zcenters(grid), Array(interior(model.stress.P)[:, iy_sl, :]); colormap=:turbo),
           Vm  = heatmap!(axs.Vm, xcenters(grid), zcenters(grid), Array(interior(Vm)[:, iy_sl, :]); colormap=:turbo),
           ωna = heatmap!(axs.ωna, xcenters(grid), zcenters(grid), Array(interior(model.field_masks.na.ccc)[:, iy_sl, :]); colormap=:turbo),
           ωns = heatmap!(axs.ωns, xcenters(grid), zcenters(grid), Array(interior(model.field_masks.ns.ccc)[:, iy_sl, :]); colormap=:turbo))
    Colorbar(fig[1, 1][1, 2], plt.Pr)
    Colorbar(fig[1, 2][1, 2], plt.Vm)
    Colorbar(fig[2, 1][1, 2], plt.ωna)
    Colorbar(fig[2, 2][1, 2], plt.ωns)

    display(fig)

    if do_h5_save
        h5names = String[]
        fields = Dict("Pr" => model.stress.P, "Vm" => Vm, "wt_ns_c" => model.field_masks.ns.ccc, "wt_na_c" => model.field_masks.na.ccc)
        outdir = "out_visu"
        mkpath(outdir)
    end

    iter = 1
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
            iter_nx = iter / max(nx, ny, nz)
            @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e, Vz = %1.3e]\n", iter_nx, err...)
        end
    end
    if do_visu
        set!(Vm, grid, (g, loc, ix, iy, iz, V) -> sqrt(lerp(V.x, loc, g, ix, iy, iz)^2 +
                                                       lerp(V.y, loc, g, ix, iy, iz)^2 +
                                                       lerp(V.z, loc, g, ix, iy, iz)^2); discrete=true, parameters=(model.velocity.V,))

        plt.Pr[3][] = interior(model.stress.P)[:, iy_sl, :]
        plt.Vm[3][] = interior(Vm)[:, iy_sl, :]
        yield()
        display(fig)
    end
    if do_h5_save
        set!(Vm, grid, (g, loc, ix, iy, iz, V) -> sqrt(lerp(V.x, loc, g, ix, iy, iz)^2 +
                                                       lerp(V.y, loc, g, ix, iy, iz)^2 +
                                                       lerp(V.z, loc, g, ix, iy, iz)^2); discrete=true, parameters=(model.velocity.V,))
        out_h5 = "results.h5"
        @info "saving HDF5 file"
        write_h5(arch, grid, joinpath(outdir, out_h5), fields)
        push!(h5names, out_h5)

        @info "saving XDMF file"
        write_xdmf(arch, grid, joinpath(outdir, "results.xdmf3"), fields, h5names)
    end
    return
end

main(CPU(); res=(128, 128, 64) .- 2)
