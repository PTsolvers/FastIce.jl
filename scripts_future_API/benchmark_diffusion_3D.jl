using FastIce.Grids
using FastIce.GridOperators
using FastIce.Fields
using FastIce.Architectures
using FastIce.BoundaryConditions
using FastIce.KernelLaunch

using KernelAbstractions
using AMDGPU

using FastIce.Distributed
using MPI

using Printf
using Plots

# @inline isin(Field) = (I <= CartesianIndex(size(grid, location(Field))))

@kernel function update_C!(C, qC, dt, Δ, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if checkbounds(Bool, C, I)
        C[I] -= dt * (∂ᶜx(qC.x, I) / Δ.x +
                      ∂ᶜy(qC.y, I) / Δ.y +
                      ∂ᶜz(qC.z, I) / Δ.z)
    end
end

@kernel function update_qC!(qC, C, dc, Δ, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if checkbounds(Bool, qC.x, I)
        qC.x[I] = -dc * ∂ᵛx(C, I) / Δ.x
    end
    @inbounds if checkbounds(Bool, qC.y, I)
        qC.y[I] = -dc * ∂ᵛy(C, I) / Δ.y
    end
    @inbounds if checkbounds(Bool, qC.z, I)
        qC.z[I] = -dc * ∂ᵛz(C, I) / Δ.z
    end
end

function compute(arch, grid, hide_boundaries, bc_q, bc_c, outer_width, qC, C, dc, dt, Δ, iters)
    tic = time_ns()
    for _ in 1:iters
        launch!(arch, grid, update_qC! => (qC, C, dc, Δ); location=Vertex(), hide_boundaries, boundary_conditions=bc_q, outer_width)
        launch!(arch, grid, update_C! => (C, qC, dt, Δ); location=Center(), hide_boundaries, boundary_conditions=bc_c, outer_width)
        synchronize(Architectures.backend(arch))
    end
    wtime = (time_ns() - tic) * 1e-9
    return wtime
end

function diffusion_3D(ka_backend=CPU(), dTyp::DataType=Float64, dims=(0, 0, 0); do_visu=true)
    # setup arch
    arch = Architecture(ka_backend, dims)
    topo = details(arch)
    set_device!(arch)
    me = global_rank(topo)
    comm = cartesian_communicator(topo)
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    dc = 1
    # numerics
    size = (1023, 1023, 1023)
    nt = 100
    iters, warmup = 20, 5
    # preprocessing
    size_g = global_grid_size(topo, size)
    global_grid = CartesianGrid(; origin=(-0.5lx, -0.5ly, -0.5lz),
                                extent=(lx, ly, lz),
                                size=size_g)
    grid = local_grid(global_grid, topo)
    Δ = NamedTuple{(:x, :y, :z)}(spacing(global_grid))
    dt = minimum(Δ)^2 / dc / ndims(grid) / 2.1
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)
    outer_width = (128, 32, 4)
    # fields
    C = Field(arch, grid, Center(), dTyp; halo=1)
    qC = (x = Field(arch, grid, (Vertex(), Center(), Center()), dTyp),
          y = Field(arch, grid, (Center(), Vertex(), Center()), dTyp),
          z = Field(arch, grid, (Center(), Center(), Vertex()), dTyp))
    # initial condition
    foreach(comp -> set!(comp, 0.0), qC)
    set!(C, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))
    # boundary conditions
    zero_flux_bc = DirichletBC{FullCell}(0.0)
    # zero flux at physical boundaries and nothing at MPI boundaries
    bc_q = ntuple(Val(3)) do D
        ntuple(Val(2)) do S
            has_neighbor(topo, D, S) ? nothing : BoundaryConditionsBatch((qC[D],), (zero_flux_bc,))
        end
    end
    # nothing at physical boundaries and communication at MPI boundaries
    bc_c = ntuple(Val(3)) do D
        ntuple(Val(2)) do S
            has_neighbor(topo, D, S) ? BoundaryConditionsBatch((C,), (ExchangeInfo(Val(S), Val(D), C),)) : nothing
        end
    end
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, bc_c)
    # # time loop
    # for it in 1:nt
    #     (me == 0) && println("it = $it")
    #     launch!(arch, grid, update_qC! => (qC, C, dc, Δ); location=Vertex(), hide_boundaries, boundary_conditions=bc_q, outer_width)
    #     launch!(arch, grid, update_C! => (C, qC, dt, Δ); location=Center(), hide_boundaries, boundary_conditions=bc_c, outer_width)
    #     synchronize(Architectures.backend(arch))
    # end

    # measure perf
    # warmup
    compute(arch, grid, hide_boundaries, bc_q, bc_c, outer_width, qC, C, dc, dt, Δ, warmup)
    # time
    MPI.Barrier(comm)
    for ex in 1:5
        (me == 0) && (sleep(2); println("Experiment = $ex"))
        MPI.Barrier(comm)
        wtime = compute(arch, grid, hide_boundaries, bc_q, bc_c, outer_width, qC, C, dc, dt, Δ, (iters - warmup))
        # report
        A_eff = 8 / 1e9 * prod(size) * sizeof(dTyp)
        wtime_it = wtime ./ (iters - warmup)
        T_eff = A_eff ./ wtime_it
        @printf("  Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (iters - warmup), wtime,
                round(T_eff, sigdigits=3), AMDGPU.device_id(AMDGPU.device()))
    end

    if do_visu
        ENV["GKSwstype"] = "nul"
        C_g = (me == 0) ? KernelAbstractions.allocate(CPU(), eltype(C), size_g) : nothing
        C_v = Array(C)
        gather!(arch, C_g, C_v)
        if me == 0
            p1 = heatmap(xcenters(global_grid), ycenters(global_grid), C_g[:, :, size_g[3]÷2];
                         aspect_ratio=1, xlims=extrema(xcenters(global_grid)), ylims=extrema(ycenters(global_grid)))
            png(p1, "C.png")
        end
    end

    return
end

ka_backend = ROCBackend()
T::DataType = Float64
dims = (0, 0, 0)

MPI.Init()
diffusion_3D(ka_backend, T, dims; do_visu=false)
MPI.Finalize()
