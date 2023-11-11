using FastIce.Grids
using FastIce.GridOperators
using FastIce.Fields
using FastIce.Architectures
using FastIce.BoundaryConditions
using FastIce.Distributed
using FastIce.KernelLaunch

using KernelAbstractions
using MPI

using Plots

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

function diffusion_3D(ka_backend=CPU())
    # setup arch
    arch = Architecture(ka_backend, (0, 0, 0))
    topo = details(arch)
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    dc = 1
    # numerics
    size_g = (32, 32, 32)
    nt = 100
    # preprocessing
    size_g = global_grid_size(topo, size_g)
    global_grid = CartesianGrid(; origin=(-0.5lx, -0.5ly, -0.5lz),
                                extent=(lx, ly, lz),
                                size=size_g)
    grid = local_grid(global_grid, topo)
    Δ = NamedTuple{(:x, :y, :z)}(spacing(global_grid))
    dt = minimum(Δ)^2 / dc / ndims(grid) / 2.1
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)
    outer_width = (4, 4, 4)
    # fields
    C = Field(arch, grid, Center(); halo=1)
    qC = (x = Field(arch, grid, (Vertex(), Center(), Center())),
          y = Field(arch, grid, (Center(), Vertex(), Center())),
          z = Field(arch, grid, (Center(), Center(), Vertex())))
    C_g = if global_rank(topo) == 0
        KernelAbstractions.allocate(Architectures.backend(arch), eltype(C), size_g)
    else
        nothing
    end
    # initial condition
    foreach(comp -> set!(comp, 0.0), qC)
    set!(C, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))
    # boundary conditions
    zero_flux_bc = DirichletBC{FullCell}(0.0)
    bc_q = NamedTuple(comp => BoundaryConditionsBatch((qC[comp],), (zero_flux_bc,)) for comp in eachindex(qC))
    # zero flux at physical boundaries and nothing at MPI boundaries
    bc_q = override_boundary_conditions(arch, ((bc_q.x, bc_q.x), (bc_q.y, bc_q.y), (bc_q.z, bc_q.z)))
    # nothing at physical boundaries and communication at MPI boundaries
    bc_c = BoundaryConditionsBatch((C,), nothing)
    bc_c = override_boundary_conditions(arch, ((bc_c, bc_c), (bc_c, bc_c), (bc_c, bc_c)); exchange=true)
    ntuple(Val(ndims(grid))) do D
        apply_boundary_conditions!(Val(1), Val(D), arch, grid, bc_c[D][1])
        apply_boundary_conditions!(Val(2), Val(D), arch, grid, bc_c[D][2])
    end
    # time loop
    for it in 1:nt
        (global_rank(topo) == 0) && println("it = $it")
        launch!(arch, grid, update_qC! => (qC, C, dc, Δ); location=Vertex(), hide_boundaries, boundary_conditions=bc_q, outer_width)
        launch!(arch, grid, update_C! => (C, qC, dt, Δ); location=Center(), hide_boundaries, boundary_conditions=bc_c, outer_width)
        synchronize(Architectures.backend(arch))
    end

    gather!(arch, C_g, C)

    if global_rank(topo) == 0
        p1 = heatmap(xcenters(global_grid), ycenters(global_grid), C_g[:, :, size_g[3]÷2];
                     aspect_ratio=1, xlims=extrema(xcenters(global_grid)), ylims=extrema(ycenters(global_grid)))
        png(p1, "C.png")
    end

    return
end

MPI.Init()
diffusion_3D()
MPI.Finalize()
