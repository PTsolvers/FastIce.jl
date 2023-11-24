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

using LinearAlgebra, Printf
using KernelAbstractions
using AMDGPU

using FastIce.Distributed
using MPI

using CairoMakie

norm_g(A) = (sum2_l = sum(interior(A) .^ 2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
max_abs_g(A) = (max_l = maximum(abs.(interior(A))); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

@views av_xy(A) = 0.25 .* (A[1:end-1, 1:end-1, :] .+ A[2:end, 1:end-1, :] .+ A[2:end, 2:end, :] .+ A[1:end-1, 2:end, :])
@views av_xz(A) = 0.25 .* (A[1:end-1, :, 1:end-1] .+ A[2:end, :, 1:end-1, :] .+ A[2:end, :, 2:end, :] .+ A[1:end-1, :, 2:end])
@views av_yz(A) = 0.25 .* (A[:, 1:end-1, 1:end-1] .+ A[:, 2:end, 1:end-1] .+ A[:, 2:end, 2:end] .+ A[:, 1:end-1, 2:end])

function main(; do_visu=false, do_save=false)
    MPI.Init()

    backend = ROCBackend()
    dims = (2, 1, 1)
    # dims = (4, 2, 2)
    # dims = (2, 1, 1)
    topo = CartesianTopology(dims)
    arch = Architecture(backend, topo)
    set_device!(arch)

    comm = cartesian_communicator(topo)
    me = global_rank(topo) # rank

    size_l = (254, 254, 254)
    size_g = global_grid_size(topo, size_l)

    outer_width = (64, 32, 4) #(128, 32, 4)#

    grid_g = CartesianGrid(; origin=(-2.0, -1.0, 0.0),
                           extent=(4.0, 2.0, 2.0),
                           size=size_g)

    grid_l = local_grid(grid_g, topo)

    if me == 0
        printstyled("Running FastIce.jl 🧊 \n"; bold=true, color=:blue)
        printstyled(" | Global size = $(size_g) \n"; bold=true)
        printstyled(" | Global dims = $(dimensions(topo)) \n"; bold=true)
        printstyled(" | Global extend = $(extent(grid_g)) \n"; bold=true)
    end

    no_slip      = VBC(0.0, 0.0, 0.0)
    free_slip    = SBC(0.0, 0.0, 0.0)
    free_surface = TBC(0.0, 0.0, 0.0)

    boundary_conditions = (x=(free_slip, free_slip),
                           y=(free_slip, free_slip),
                           z=(no_slip, free_surface))

    ρgx(x, y, z) = 0.25
    ρgy(x, y, z) = 0.0
    ρgz(x, y, z) = 1.0
    gravity = (x=FunctionField(ρgx, grid_l, (Vertex(), Center(), Center())),
               y=FunctionField(ρgy, grid_l, (Center(), Vertex(), Center())),
               z=FunctionField(ρgz, grid_l, (Center(), Center(), Vertex())))

    # numerics
    niter  = 50maximum(size(grid_g))
    ncheck = 2maximum(size(grid_g))

    r       = 0.7
    re_mech = 4π
    lτ_re_m = minimum(extent(grid_g)) / re_mech
    vdτ     = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 1.5)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    iter_params = (η_rel=1e-1,
                   Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

    physics = (rheology=GlensLawRheology(1),)
    other_fields = (A=Field(backend, grid_l, Center()),)

    model = IsothermalFullStokesModel(;
                                      arch,
                                      grid=grid_l,
                                      physics,
                                      gravity,
                                      boundary_conditions,
                                      outer_width,
                                      iter_params,
                                      other_fields)

    (me == 0) && printstyled("Model created \n"; bold=true, color=:light_blue)

    if do_save || do_visu
        if me == 0
            Pr_g  = zeros(size(grid_g))
            τxx_g = zeros(size(grid_g))
            τyy_g = zeros(size(grid_g))
            τzz_g = zeros(size(grid_g))
            τxy_g = zeros(size(grid_g))
            τxz_g = zeros(size(grid_g))
            τyz_g = zeros(size(grid_g))
            Vx_g  = zeros(size(grid_g))
            Vy_g  = zeros(size(grid_g))
            Vz_g  = zeros(size(grid_g))
        else
            Pr_g  = nothing
            τxx_g = nothing
            τyy_g = nothing
            τzz_g = nothing
            τxy_g = nothing
            τxz_g = nothing
            τyz_g = nothing
            Vx_g  = nothing
            Vy_g  = nothing
            Vz_g  = nothing
        end
        Pr_v  = zeros(size(grid_l))
        τxx_v = zeros(size(grid_l))
        τyy_v = zeros(size(grid_l))
        τzz_v = zeros(size(grid_l))
        τxy_v = zeros(size(grid_l))
        τxz_v = zeros(size(grid_l))
        τyz_v = zeros(size(grid_l))
        Vx_v  = zeros(size(grid_l))
        Vy_v  = zeros(size(grid_l))
        Vz_v  = zeros(size(grid_l))
    end

    fill!(parent(model.fields.Pr), 0.0)
    foreach(x -> fill!(parent(x), 0.0), model.fields.τ)
    foreach(x -> fill!(parent(x), 0.0), model.fields.V)
    fill!(parent(other_fields.A), 1.0)

    set!(model.fields.η, grid_l, (grid, loc, I, fields) -> physics.rheology(grid, I, fields); discrete=true, parameters=(model.fields,))

    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.rheology)

    MPI.Barrier(comm)

    (me == 0) && printstyled("Action \n"; bold=true, color=:light_blue)

    ttot_ns = UInt64(0)
    for iter in 1:niter
        if iter == 10
            MPI.Barrier(comm)
            ttot_ns = time_ns()
        end
        advance_iteration!(model, 0.0, 1.0)
        if (iter % ncheck == 0)
            compute_residuals!(model)
            err = (Pr = max_abs_g(model.fields.r_Pr),
                   Vx = max_abs_g(model.fields.r_V.x),
                   Vy = max_abs_g(model.fields.r_V.y),
                   Vz = max_abs_g(model.fields.r_V.z))
            if (me == 0)
                any(.!isfinite.(values(err))) && error("simulation failed, err = $err")
                iter_nx = iter / maximum(size(grid_g))
                @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e, Vz = %1.3e]\n", iter_nx, err...)
            end
        end
    end
    ttot = float(time_ns() - ttot_ns)
    ttot /= (niter - 10)

    MPI.Barrier(comm)

    ttot_min = MPI.Allreduce(ttot, MPI.MIN, comm)
    ttot_max = MPI.Allreduce(ttot, MPI.MAX, comm)

    if me == 0
        Teff_min = 23 * 8 * prod(size(grid_l)) / ttot_max
        Teff_max = 23 * 8 * prod(size(grid_l)) / ttot_min
        printstyled("Performance: T_eff [min max] = $(round(Teff_min, sigdigits=4)) $(round(Teff_max, sigdigits=4)) \n"; bold=true, color=:green)
    end

    if do_save || do_visu
        copyto!(Pr_v, interior(model.fields.Pr))
        copyto!(τxx_v, interior(model.fields.τ.xx))
        copyto!(τyy_v, interior(model.fields.τ.yy))
        copyto!(τzz_v, interior(model.fields.τ.zz))
        copyto!(τxy_v, av_xy(interior(model.fields.τ.xy)))
        copyto!(τxz_v, av_xz(interior(model.fields.τ.xz)))
        copyto!(τyz_v, av_yz(interior(model.fields.τ.yz)))
        copyto!(Vx_v, avx(interior(model.fields.V.x)))
        copyto!(Vy_v, avy(interior(model.fields.V.y)))
        copyto!(Vz_v, avz(interior(model.fields.V.z)))

        KernelAbstractions.synchronize(backend)

        gather!(Pr_g, Pr_v, comm)
        gather!(τxx_g, τxx_v, comm)
        gather!(τyy_g, τyy_v, comm)
        gather!(τzz_g, τzz_v, comm)
        gather!(τxy_g, τxy_v, comm)
        gather!(τxz_g, τxz_v, comm)
        gather!(τyz_g, τyz_v, comm)
        gather!(Vx_g, Vx_v, comm)
        gather!(Vy_g, Vy_v, comm)
        gather!(Vz_g, Vz_v, comm)

        if (me == 0) && do_visu
            fig = Figure()
            axs = (Pr=Axis(fig[1, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Pr"),
            Vx=Axis(fig[1, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vx"),
            Vy=Axis(fig[2, 1][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vy"),
            Vz=Axis(fig[2, 2][1, 1]; aspect=DataAspect(), xlabel="x", ylabel="z", title="Vz"))
            plt = (Pr = heatmap!(axs.Pr, xcenters(grid_g), zcenters(grid_g), Pr_g[:, size(grid_g, 2)÷2+1, :]; colormap=:turbo),
                   Vx = heatmap!(axs.Vx, xvertices(grid_g), zcenters(grid_g), Vx_g[:, size(grid_g, 2)÷2+1, :]; colormap=:turbo),
                   Vy = heatmap!(axs.Vy, xcenters(grid_g), zcenters(grid_g), Vy_g[:, size(grid_g, 2)÷2+1, :]; colormap=:turbo),
                   Vz = heatmap!(axs.Vz, xcenters(grid_g), zvertices(grid_g), Vz_g[:, size(grid_g, 2)÷2+1, :]; colormap=:turbo))
            Colorbar(fig[1, 1][1, 2], plt.Pr)
            Colorbar(fig[1, 2][1, 2], plt.Vx)
            Colorbar(fig[2, 1][1, 2], plt.Vy)
            Colorbar(fig[2, 2][1, 2], plt.Vz)
            save("fig.png", fig)
        end

        if me == 0 && do_save
            open("data.bin", "w") do io
                write(io, Pr_g)
                write(io, τxx_g)
                write(io, τyy_g)
                write(io, τzz_g)
                write(io, τxy_g)
                write(io, τxz_g)
                write(io, τyz_g)
                write(io, Vx_g)
                write(io, Vy_g)
                write(io, Vz_g)
            end
        end
    end

    MPI.Finalize()

    return
end

main(; do_visu=false, do_save=false)
