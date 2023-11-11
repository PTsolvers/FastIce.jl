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
                      ∂ᶜy(qC.y, I) / Δ.y)
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
end

function diffusion_2D(ka_backend=CPU())
    # setup arch
    arch = Architecture(ka_backend, (0, 0))
    topo = details(arch)
    # physics
    lx, ly = 10.0, 10.0
    dc = 1
    # numerics
    size_g = (32, 32)
    nt = 1000
    # preprocessing
    size_g = global_grid_size(topo, size_g)
    global_grid = CartesianGrid(; origin=(-0.5lx, -0.5ly),
                                extent=(lx, ly),
                                size=size_g)
    grid = local_grid(global_grid, topo)
    Δ = NamedTuple{(:x, :y)}(spacing(global_grid))
    dt = minimum(Δ)^2 / dc / ndims(grid) / 2.1
    hide_boundaries = HideBoundaries{ndims(grid)}(arch)
    outer_width = (4, 4)
    # fields
    C = Field(arch, grid, Center(); halo=1)
    qC = (x = Field(arch, grid, (Vertex(), Center()); halo=1),
          y = Field(arch, grid, (Center(), Vertex()); halo=1))
    C_g = if global_rank(topo) == 0
        KernelAbstractions.allocate(Architectures.backend(arch), eltype(C), size_g)
    else
        nothing
    end
    # initial condition
    foreach(comp -> fill!(parent(comp), 0.0), qC)
    # fill!(parent(C), me)
    set!(C, grid, (x, y) -> exp(-x^2 - y^2))
    # set!(C, me)
    # boundary conditions
    zero_flux_bc = DirichletBC{FullCell}(0.0)
    bc_q = (x = BoundaryConditionsBatch((qC.x, qC.y), (zero_flux_bc, nothing)),
            y = BoundaryConditionsBatch((qC.x, qC.y), (nothing, zero_flux_bc)))
    # zero flux at physical boundaries and nothing at MPI boundaries
    bc_q = override_boundary_conditions(arch, ((bc_q.x, bc_q.x), (bc_q.y, bc_q.y)); exchange=true)
    # nothing at physical boundaries and communication at MPI boundaries
    bc_c = BoundaryConditionsBatch((C,), nothing)
    bc_c = override_boundary_conditions(arch, ((bc_c, bc_c), (bc_c, bc_c)); exchange=true)
    for D in ndims(grid):-1:1
        apply_boundary_conditions!(Val(1), Val(D), arch, grid, bc_c[D][1])
        apply_boundary_conditions!(Val(2), Val(D), arch, grid, bc_c[D][2])
        apply_boundary_conditions!(Val(1), Val(D), arch, grid, bc_q[D][1])
        apply_boundary_conditions!(Val(2), Val(D), arch, grid, bc_q[D][2])
    end
    # time loop
    if global_rank(topo) == 0
        anim = Animation()
    end
    for it in 1:nt
        (global_rank(topo) == 0) && println("it = $it")
        launch!(arch, grid, update_qC! => (qC, C, dc, Δ); location=Vertex(), hide_boundaries, boundary_conditions=bc_q, outer_width)
        launch!(arch, grid, update_C! => (C, qC, dt, Δ); location=Center(), expand=1)
        synchronize(arch.backend)
        if it % 5 == 0
            gather!(arch, C_g, C)
            if global_rank(topo) == 0
                heatmap(C_g; aspect_ratio=1, size=(600, 600), clims=(0, 1))
                frame(anim)
            end
        end
    end
    if global_rank(topo) == 0
        gif(anim, "C.gif")
    end

    return
end

MPI.Init()
diffusion_2D()
MPI.Finalize()
