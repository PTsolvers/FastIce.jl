using Printf

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

using LinearAlgebra
using KernelAbstractions
using CUDA
CUDA.allowscalar(false)

using CairoMakie

# manufactured solution for the confined Stokes flow with free-slip boundaries
# helper functions
f(ξ, η) = cos(π * ξ) * (η^2 - 1)^2 -
          cos(π * η) * (ξ^2 - 1)^2
g(ξ, η) = sin(π * η) * (ξ^2 - 1) * ξ -
          sin(π * ξ) * (η^2 - 1) * η
p(ξ, η) = cos(π * η) * (1 - 3 * ξ^2) * 2 -
          cos(π * ξ) * (1 - 3 * η^2) * 2
# velocity
vx(x, y, z) = sin(π * x) * f(y, z)
vy(x, y, z) = sin(π * y) * f(z, x)
vz(x, y, z) = sin(π * z) * f(x, y)
# diagonal deviatoric stress
τxx(x, y, z, η) = 2 * η * π * cos(π * x) * f(y, z)
τyy(x, y, z, η) = 2 * η * π * cos(π * y) * f(z, x)
τzz(x, y, z, η) = 2 * η * π * cos(π * z) * f(x, y)
# off-diagonal deviatoric stress
τxy(x, y, z, η) = 4 * η * cos(π * z) * g(x, y)
τxz(x, y, z, η) = 4 * η * cos(π * y) * g(z, x)
τyz(x, y, z, η) = 4 * η * cos(π * x) * g(y, z)
# forcing terms
ρgx(x, y, z, η) = -2 * η * sin(π * x) * (f(y, z) * π^2 - p(y, z))
ρgy(x, y, z, η) = -2 * η * sin(π * y) * (f(z, x) * π^2 - p(z, x))
ρgz(x, y, z, η) = -2 * η * sin(π * z) * (f(x, y) * π^2 - p(x, y))

# helpers
@views avx(A) = @. 0.5 * (A[1:end-1, :, :] + A[2:end, :, :])
@views avy(A) = @. 0.5 * (A[:, 1:end-1, :] + A[:, 2:end, :])
@views avz(A) = @. 0.5 * (A[:, :, 1:end-1] + A[:, :, 2:end])
@views avxy(A) = @. 0.25 * (A[1:end-1, 1:end-1, :] + A[2:end, 1:end-1, :] + A[2:end, 2:end, :] + A[1:end-1, 2:end, :])
@views avxz(A) = @. 0.25 * (A[1:end-1, :, 1:end-1] + A[2:end, :, 1:end-1] + A[2:end, :, 2:end] + A[1:end-1, :, 2:end])
@views avyz(A) = @. 0.25 * (A[:, 1:end-1, 1:end-1] + A[:, 2:end, 1:end-1] + A[:, 2:end, 2:end] + A[:, 1:end-1, 2:end])

@views function run(dims)
    backend = CUDABackend()
    arch = Architecture(backend, 1)
    set_device!(arch)

    # outer_width = (4, 4, 4)

    # physics
    η0 = 1.0
    A0 = 0.5

    # geometry
    grid = CartesianGrid(; origin=(-1.0, -1.0, -1.0),
                         extent=(2.0, 2.0, 2.0),
                         size=dims)

    free_slip = SBC(0.0, 0.0, 0.0)
    xface = (Vertex(), Center(), Center())
    yface = (Center(), Vertex(), Center())
    zface = (Center(), Center(), Vertex())

    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip),
                           z = (free_slip, free_slip))

    gravity = (x=FunctionField(ρgx, grid, xface; parameters=η0),
               y=FunctionField(ρgy, grid, yface; parameters=η0),
               z=FunctionField(ρgz, grid, zface; parameters=η0))

    # numerics
    niter  = 10maximum(size(grid))
    ncheck = maximum(size(grid))

    # PT params
    r       = 0.7
    re_mech = 8π
    lτ_re_m = minimum(extent(grid)) / re_mech
    vdτ     = minimum(spacing(grid)) / sqrt(ndims(grid) * 1.1)
    θ_dτ    = lτ_re_m * (r + 4 / 3) / vdτ
    dτ_r    = 1.0 / (θ_dτ + 1.0)
    nudτ    = vdτ * lτ_re_m

    iter_params = (η_rel=1e-1,
                   Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, zz=dτ_r, xy=dτ_r, xz=dτ_r, yz=dτ_r), V=(x=nudτ, y=nudτ, z=nudτ)))

    physics = (rheology=GlensLawRheology(1),)
    other_fields = (A=ConstantField(A0),)

    model = IsothermalFullStokesModel(;
                                      arch,
                                      grid,
                                      physics,
                                      gravity,
                                      boundary_conditions,
                                      iter_params,
                                      other_fields)

    set!(model.fields.Pr, 0.0)
    foreach(x -> set!(x, 0.0), model.fields.τ)
    foreach(x -> set!(x, 0.0), model.fields.V)

    set!(model.fields.η, grid, (grid, loc, I, fields) -> physics.rheology(grid, I, fields); discrete=true, parameters=(model.fields,))

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.rheology)

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0)
        if iter % ncheck == 0
            compute_residuals!(model)
            err = (Pr = norm(model.fields.r_Pr, Inf),
                   Vx = norm(model.fields.r_V.x, Inf),
                   Vy = norm(model.fields.r_V.y, Inf),
                   Vz = norm(model.fields.r_V.z, Inf))
            if any(.!isfinite.(values(err)))
                error("simulation failed, err = $err")
            end
            iter_nx = iter / maximum(size(grid))
            @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e, Vz = %1.3e]\n", iter_nx, err...)
        end
    end

    V = model.fields.V
    τ = model.fields.τ

    Vm = (x=FunctionField(vx, grid, location(V.x)),
          y=FunctionField(vy, grid, location(V.y)),
          z=FunctionField(vz, grid, location(V.z)))

    τm = (xx=FunctionField(τxx, grid, location(τ.xx); parameters=η0),
          yy=FunctionField(τyy, grid, location(τ.yy); parameters=η0),
          zz=FunctionField(τzz, grid, location(τ.zz); parameters=η0),
          xy=FunctionField(τxy, grid, location(τ.xy); parameters=η0),
          xz=FunctionField(τxz, grid, location(τ.xz); parameters=η0),
          yz=FunctionField(τyz, grid, location(τ.yz); parameters=η0))

    Vm_mag = sqrt.(avx(Vm.x) .^ 2 .+ avy(Vm.y) .^ 2 .+ avz(Vm.z) .^ 2) |> CuArray
    V_mag  = sqrt.(avx(interior(V.x)) .^ 2 .+ avy(interior(V.y)) .^ 2 .+ avz(interior(V.z)) .^ 2)

    τm_mag = sqrt.(0.5 .* (τm.xx .^ 2 .+ τm.yy .^ 2 .+ τm.zz .^ 2) .+ avxy(τm.xy) .^ 2 .+ avxz(τm.xz) .^ 2 .+ avyz(τm.yz) .^ 2) |> CuArray
    τ_mag  = sqrt.(0.5 .* (interior(τ.xx) .^ 2 .+ interior(τ.yy) .^ 2 .+ interior(τ.zz) .^ 2) .+ avxy(interior(τ.xy)) .^ 2 .+ avxz(interior(τ.xz)) .^ 2 .+ avyz(interior(τ.yz)) .^ 2)

    err = (norm(Vm_mag .- V_mag, Inf) / norm(Vm_mag, Inf),
           norm(τm_mag .- τ_mag, Inf) / norm(τm_mag, Inf),
           norm(Vm_mag .- V_mag, 1) / norm(Vm_mag, 1),
           norm(τm_mag .- τ_mag, 1) / norm(τm_mag, 1),
           norm(Vm_mag .- V_mag, 2) / norm(Vm_mag, 2),
           norm(τm_mag .- τ_mag, 2) / norm(τm_mag, 2))

    @show err

    return err
end

N = 2 .^ (3:8)
errs = zeros(6, length(N))

for (iN, nx) in enumerate(N)
    @show nx
    errs[:, iN] .= run((nx, nx, nx))
end

fig = Figure(; fontsize=16)

ax = Axis(fig[1, 1];
          xscale=log2,
          yscale=log10,
          xlabel="N",
          ylabel="L-norms of error",
          xticks=LogTicks(3:8),
          yticks=LogTicks(-5:1:-1),
          title="Convergence for free slip confined flow 3D")

ylims!(ax, 1e-5, 1e-1)

N2 = N .^ (-2) / N[end]^(-2) * minimum(errs)
lines!(ax, N, N2; label=L"N^{-2}", color=:gray, linewidth=2)

labels = (L"L_\infty\text{-norm},\,v",
          L"L_\infty\text{-norm},\,\tau_\mathrm{II}",
          L"L_1\text{-norm},\,v", L"L_1\text{-norm},\,\tau_\mathrm{II}",
          L"L_2\text{-norm},\,v", L"L_2\text{-norm},\,\tau_\mathrm{II}")
colors = (:red, :red, :blue, :blue, :green, :green)
shapes = (:circle, :diamond, :rect, :star4, :utriangle, :vline)

for ierr in axes(errs, 1)
    scatter!(ax, N, errs[ierr, :];
             label       = labels[ierr],
             marker      = shapes[ierr],
             markersize  = 15,
             color       = :transparent,
             strokewidth = 1,
             strokecolor = colors[ierr])
end

axislegend(ax)

save("stokes_manufactured_solution_free_slip_3D.png", fig)
