using Chmy, Chmy.Architectures, Chmy.Grids, Chmy.GridOperators, Chmy.Fields, Chmy.KernelLaunch, Chmy.BoundaryConditions
using KernelAbstractions

using FastIce
using FastIce.Geometries
using FastIce.LevelSets
using FastIce.Physics
using FastIce.Writers

using FastIce.Models.ImmersedBoundaryFullStokes.Isothermal

using CUDA
using CairoMakie
using JLD2

using Printf
using LinearAlgebra

using Chmy.Distributed
using MPI
MPI.Init()

backend = CUDABackend()
do_visu = true
do_h5_save = false

conv(nx, tx) = tx * ((nx + tx ÷ 2 -1 ) ÷ tx)

function make_synthetic(arch::Architecture, nx, ny, lx, ly, lz, amp, ω, tanβ, el, gl)
    arch_2d = SingleDeviceArchitecture(arch)
    grid_2D = UniformGrid(arch_2d; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))

    # type = :turtle
    generate_ice(x, y, gl, lx, ly) = gl * (1.0 - ((1.7 * x / lx + 0.22)^2 + (0.5 * y / ly)^2))
    generate_bed(x, y, amp, ω, tanβ, el, lx, ly, lz) = lz * (amp * sin(ω * x / lx) * sin(ω * y /ly) + tanβ * x / lx + el + (1.5 * y / ly)^2)

    ice = FunctionField(generate_ice, grid_2D, Vertex(); parameters=(; gl, lx, ly))
    bed = FunctionField(generate_bed, grid_2D, Vertex(); parameters=(; amp, ω, tanβ, el, lx, ly, lz))

    return (; arch, bed, ice, grid_2D, lx, ly, lz)
end

function main_synthetic(backend=CPU())
    # set-up distributed
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))

    # synthetic topo
    lx, ly, lz = 5.0, 5.0, 1.0
    amp  = 0.1
    ω    = 10π
    tanβ = tan(-π/6)
    el   = 0.35
    gl   = 0.9

    nx, ny = 126, 126
    nz     = max(conv(ceil(Int, lz / lx * nx), 30), 30)
    resol  = (nx, ny, nz)

    synthetic_elevation = make_synthetic(arch, nx, ny, lx, ly, lz, amp, ω, tanβ, el, gl)

    run_simulation(synthetic_elevation..., resol...)
    return
end

function extract_dem(arch::Architecture, data_path::String)
    data  = load(data_path)
    dtype = first(keys(data))

    z_ice      = data[dtype].z_surf
    z_bed      = data[dtype].z_bed
    dm         = data[dtype].domain
    dm         = dilate(dm, 0.05)
    lx, ly, lz = extents(dm)
    nx, ny     = size(z_ice) .- 1

    z_ice .-= dm.zmin # put Z-origin at 0
    z_bed .-= dm.zmin # put Z-origin at 0

    arch_2d = SingleDeviceArchitecture(arch)
    grid_2D = UniformGrid(arch_2d; origin=(-lx/2, -ly/2), extent=(lx, ly), dims=(nx, ny))

    ice = Field(arch_2d, grid_2D, Vertex())
    bed = Field(arch_2d, grid_2D, Vertex())
    set!(ice, z_ice)
    set!(bed, z_bed)

    return (; arch, bed, ice, grid_2D, lx, ly, lz)
end

function main_vavilov(backend=CPU())
    # load dem data
    data_path = "./vavilov_dem2.jld2"

    # set-up distributed
    arch = Arch(backend, MPI.COMM_WORLD, (0, 0, 0))

    data_elevation = extract_dem(arch, data_path)

    nx, ny = 126, 126
    nz     = max(conv(ceil(Int, data_elevation.lz / data_elevation.lx * nx), 30), 60)
    resol  = (nx, ny, nz)

    run_simulation(data_elevation..., resol...)
    return
end

# function main(backend; res)
function run_simulation(arch::Architecture, bed, ice, grid_2D, lx, ly, lz, nx, ny, nz)
    topo = topology(arch)
    me   = global_rank(topo)
    # geometry
    (me == 0) && @info "size = ($nx, $ny, $nz), extent = ($lx, $ly, $lz)"
    dims_l = (nx, ny, nz)
    dims_g = dims_l .* dims(topo)
    grid   = UniformGrid(arch; origin=(-lx/2, -ly/2, 0.0), extent=(lx, ly, lz), dims=dims_g)
    launch = Launcher(arch, grid; outer_width=(16, 8, 4))
    # compute level sets
    ψ = (ns=Field(arch, grid, Vertex()),
         na=Field(arch, grid, Vertex()))

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

main_synthetic(backend)
# main_vavilov(backend)

# MPI.Finalize()
