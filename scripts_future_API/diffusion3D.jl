using FastIce.Grids
using FastIce.GridOperators
using FastIce.Fields
using FastIce.Utils
using FastIce.Architectures
using FastIce.BoundaryConditions
using FastIce.KernelLaunch

using KernelAbstractions
using Plots

backend = CPU()

@kernel function update_qT!(qT, T, λ, Δ, nx, ny, nz, grid, offset=nothing)
    I = @index(Global, Cartesian)
    @inline isin(Field) = (I <= CartesianIndex(size(grid, location(Field))))
    # @inbounds if (I[1] <= nx+1 && I[2] <= ny && I[3] <= nz)
    @inbounds if isin(qT.x)
        qT.x[I] = - avᵛx(λ, I) * ∂ᵛx(T, I) / Δ.x
    end
    @inbounds if (I[1] <= nx && I[2] <= ny+1 && I[3] <= nz)
        qT.y[I] = - avᵛy(λ, I) * ∂ᵛy(T, I) / Δ.y
    end
    @inbounds if (I[1] <= nx && I[2] <= ny && I[3] <= nz+1)
        qT.z[I] = - avᵛz(λ, I) * ∂ᵛz(T, I) / Δ.z
    end
end

@kernel function update_T!(T, qT, ρ_cp, dt, Δ, offset=nothing)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, T, I)
        T[I] -= ρ_cp * dt * (∂ᶜx(qT.x, I) / Δ.x +
                             ∂ᶜy(qT.y, I) / Δ.y +
                             ∂ᶜz(qT.z, I) / Δ.z)
    end
end

function diffusion(ka_backend=CPU())
    # setup arch
    arch = Architecture(ka_backend)
    # physics
    lx, ly, lz = 10.0, 10.0, 10.0
    ρ_cp = 1.0
    λ0 = 1.0
    # numerics
    nx, ny, nz = 32, 32, 32
    nt = 100
    grid = CartesianGrid(; origin=(-0.5lx, -0.5ly, -0.5lz),
                           extent=(lx, ly, lz),
                           size=(nx, ny, nz))
    Δ = NamedTuple{(:x, :y, :z)}(spacing(grid))
    dt = minimum(Δ)^2 / (λ0 / ρ_cp) / ndims(grid) / 2.1
    # init fields
    T = Field(arch, grid, Center(); halo=1)
    λ = Field(arch, grid, Center(); halo=1)
    qT = (
        x = Field(arch, grid, (Vertex(), Center(), Center())),
        y = Field(arch, grid, (Center(), Vertex(), Center())),
        z = Field(arch, grid, (Center(), Center(), Vertex())),
    )
    # initial condition
    foreach(comp -> set!(comp, 0.0), qT)
    set!(λ, λ0)
    extrapolate!(arch, λ)
    fill!(parent(T), 0.0)
    set!(T, grid, (x, y, z) -> exp(-x^2 - y^2 - z^2))
    # boundary conditions
    no_flux_bc = DirichletBC{FullCell}(0.0)
    fixed_t0 = DirichletBC{HalfCell}(0.5)
    fixed_t1 = DirichletBC{HalfCell}(1.0)
    bc_q = NamedTuple(comp => BoundaryConditionsBatch((qT[comp],), (no_flux_bc,)) for comp in eachindex(qT))
    bc_q = ((nothing, bc_q.x), (nothing, bc_q.y), (bc_q.z, bc_q.z))
    bc_t0 = BoundaryConditionsBatch((T,), (fixed_t0,))
    bc_t1 = BoundaryConditionsBatch((T,), (fixed_t1,))
    bc_t = ((bc_t1, nothing), (bc_t0, nothing), (nothing, nothing))
    # ntuple(Val(ndims(grid))) do D
    #     apply_boundary_conditions!(Val(1), Val(D), arch, grid, bc_t[D][1])
    #     apply_boundary_conditions!(Val(2), Val(D), arch, grid, bc_t[D][2])
    # end
    # function bc_x!(_bc)
    #     apply_boundary_conditions!(Val(1), Val(1), arch, grid, _bc[1][1])
    #     apply_boundary_conditions!(Val(2), Val(1), arch, grid, _bc[1][2])
    # end

    # function bc_y!(_bc)
    #     apply_boundary_conditions!(Val(1), Val(2), arch, grid, _bc[2][1])
    #     apply_boundary_conditions!(Val(2), Val(2), arch, grid, _bc[2][2])
    # end

    # function bc_z!(_bc)
    #     apply_boundary_conditions!(Val(1), Val(3), arch, grid, _bc[3][1])
    #     apply_boundary_conditions!(Val(2), Val(3), arch, grid, _bc[3][2])
    # end
    # time loop
    for it in 1:nt
        # KA version
        # update_qT!(ka_backend, 256, (nx+1, ny+1, nz+1))(qT, T, λ, Δ)
        # bc_x!(bc_q); bc_y!(bc_q); bc_z!(bc_q)
        # update_T!(ka_backend, 256, (nx, ny, nz))(T, qT, ρ_cp, dt, Δ)
        # FastIce version
        launch!(arch, grid, update_qT! => (qT, T, λ, Δ, nx, ny, nz, grid); location=Vertex(), boundary_conditions=bc_q)
        launch!(arch, grid, update_T! => (T, qT, ρ_cp, dt, Δ); location=Center(), boundary_conditions=bc_t)
        KernelAbstractions.synchronize(backend(arch))
    end
    # visualise
    p1 = heatmap(xcenters(grid), ycenters(grid), interior(T)[:, :, nz÷2]'; aspect_ratio=1, xlims=extrema(xcenters(grid)), ylims=extrema(ycenters(grid)))
    png(p1, "T.png")

    return
end

diffusion()
