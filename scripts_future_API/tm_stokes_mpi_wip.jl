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
using AMDGPU

using FastIce.Distributed
using MPI

@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

@views av_xy(A) = 0.25 .* (A[1:end-1, 1:end-1, :] .+ A[2:end, 1:end-1, :] .+ A[2:end, 2:end, :] .+ A[1:end-1, 2:end, :])
@views av_xz(A) = 0.25 .* (A[1:end-1, :, 1:end-1] .+ A[2:end, :, 1:end-1, :] .+ A[2:end, :, 2:end, :] .+ A[1:end-1, :, 2:end])
@views av_yz(A) = 0.25 .* (A[:, 1:end-1, 1:end-1] .+ A[:, 2:end, 1:end-1] .+ A[:, 2:end, 2:end] .+ A[:, 1:end-1, 2:end])

function main()
    MPI.Init(; threadlevel=:multiple)

    backend = ROCBackend()
    dims    = (4, 2, 2)
    # dims = (0, 0, 0)
    arch = Architecture(backend, dims, MPI.COMM_WORLD)
    set_device!(arch)

    topo = details(arch)

    size_l = (254, 254, 254)
    size_g = global_grid_size(topo, size_l)

    if global_rank(topo) == 0
        @show dimensions(topo)
        @show size_g
    end

    grid_g = CartesianGrid(; origin=(-2.0, -1.0, 0.0),
                           extent=(4.0, 2.0, 2.0),
                           size=size_g)

    grid_l = local_grid(grid_g, topo)

    no_slip      = VBC(0.0, 0.0, 0.0)
    free_slip    = SBC(0.0, 0.0, 0.0)
    free_surface = TBC(0.0, 0.0, 0.0)

    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip),
                           z = (no_slip, free_surface))

    gravity = (x=-0.25, y=0.0, z=1.0)

    # numerics
    niter  = 20maximum(size(grid_g))
    ncheck = 1maximum(size(grid_g))

    r       = 0.7
    re_mech = 5π
    lτ_re_m = minimum(extent(grid_g)) / re_mech
    vdτ     = minimum(spacing(grid_g)) / sqrt(ndims(grid_g) * 10.1)
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
                                      iter_params,
                                      other_fields)

    if global_rank(topo) == 0
        println("model created")
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

    fill!(parent(model.fields.Pr), 0.0)
    foreach(x -> fill!(parent(x), 0.0), model.fields.τ)
    foreach(x -> fill!(parent(x), 0.0), model.fields.V)
    fill!(parent(other_fields.A), 1.0)
    fill!(parent(model.fields.η), 0.5)

    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid_l, model.boundary_conditions.rheology)

    MPI.Barrier(cartesian_communicator(topo))

    if global_rank(topo) == 0
        println("action")
    end

    ttot_ns = UInt64(0)
    for iter in 1:niter
        if iter == 10
            MPI.Barrier(cartesian_communicator(topo))
            ttot_ns = time_ns()
        end
        advance_iteration!(model, 0.0, 1.0; async=false)
        if (iter % ncheck == 0) && (global_rank(topo) == 0)
            println("iter/nx = $(iter/maximum(size(grid_g)))")
        end
    end
    ttot = float(time_ns() - ttot_ns)
    ttot /= (niter - 10)

    comm = cartesian_communicator(topo)

    MPI.Barrier(comm)

    ttot = MPI.Allreduce(ttot, MPI.MIN, comm)

    if global_rank(topo) == 0
        Aeff = 23 * prod(size(grid_g)) / ttot
        println("A_eff = $Aeff")
    end

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

    if global_rank(topo) == 0
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

    MPI.Finalize()

    return
end

main()
