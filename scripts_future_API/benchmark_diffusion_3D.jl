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
    topology = details(arch)
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    dc = 1
    # numerics
    nx, ny, nz = 32, 32, 32
    nt = 100
    # preprocessing
    grid = CartesianGrid(; origin=(-0.5lx, -0.5ly, -0.5lz),
                         extent=(lx, ly, lz),
                         size=(nx, ny, nz))
    Δ = NamedTuple{(:x, :y, :z)}(spacing(grid))
    dt = minimum(Δ)^2 / dc / ndims(grid) / 2.1
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)
    outer_width = (4, 4, 4)
    # fields
    C = Field(arch, grid, Center(); halo=1)
    qC = (x = Field(arch, grid, (Vertex(), Center(), Center())),
          y = Field(arch, grid, (Center(), Vertex(), Center())),
          z = Field(arch, grid, (Center(), Center(), Vertex())))
    C_g = if global_rank(topology) == 0
        KernelAbstractions.allocate(Architectures.backend(arch), eltype(C), dimensions(topology) .* size(C))
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
    boundary_conditions_q = ((bc_q.x, bc_q.x), (bc_q.y, bc_q.y), (bc_q.z, bc_q.z))
    boundary_conditions_q = override_boundary_conditions(arch, boundary_conditions_q)
    # nothing at physical boundaries and communication at MPI boundaries
    bc_c = BoundaryConditionsBatch((C,), nothing)
    boundary_conditions_c = ((bc_c, bc_c), (bc_c, bc_c), (bc_c, bc_c))
    boundary_conditions_c = override_boundary_conditions(arch, boundary_conditions_c; exchange=true)
    # time loop
    for it in 1:nt
        launch!(arch, grid, update_qC! => (qC, C, dc, Δ);
                location=Vertex(), hide_boundaries, boundary_conditions=boundary_conditions_q, outer_width)
        launch!(arch, grid, update_C! => (C, qC, dt, Δ);
                location=Center(), hide_boundaries, boundary_conditions=boundary_conditions_c, outer_width)
        Architectures.synchronize(arch)
    end

    gather!(arch, C_g, C)

    if global_rank(topology) == 0
        p1 = heatmap(interior(C)[:, :, nz÷2]; aspect_ratio=1)
        png(p1, "C.png")
    end

    return
end

MPI.Init()
diffusion_3D()
MPI.Finalize()
