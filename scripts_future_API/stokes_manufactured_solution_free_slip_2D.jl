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

using CairoMakie

# manufactured solution for the confined Stokes flow with free-slip boundaries
# velocity
vx(x, y) = sin(π * x) * cos(π * y)
vy(x, y) = -cos(π * x) * sin(π * y)
# deviatoric stress
τxx(x, y, η) = 2 * η * π * cos(π * x) * cos(π * y)
τyy(x, y, η) = -2 * η * π * cos(π * x) * cos(π * y)
τxy(x, y, η) = zero(x)
# forcing terms
ρgx(x, y, η) = -2 * η * π^2 * sin(π * x) * cos(π * y)
ρgy(x, y, η) = 2 * η * π^2 * cos(π * x) * sin(π * y)

# helpers
@views avx(A) = @. 0.5 * (A[1:end-1, :] + A[2:end, :])
@views avy(A) = @. 0.5 * (A[:, 1:end-1] + A[:, 2:end])
@views av4(A) = @. 0.25 * (A[1:end-1, 1:end-1] + A[2:end, 1:end-1] + A[2:end, 2:end] + A[1:end-1, 2:end])

@views function run(dims)
    backend = CPU()
    arch = Architecture(backend, 2)
    set_device!(arch)

    # physics
    η0 = 1.0
    A0 = 0.5

    # geometry
    grid = CartesianGrid(; origin=(-1.0, -1.0),
                         extent=(2.0, 2.0),
                         size=dims)

    free_slip = SBC(0.0, 0.0)
    xface = (Vertex(), Center())
    yface = (Center(), Vertex())

    boundary_conditions = (x = (free_slip, free_slip),
                           y = (free_slip, free_slip))

    gravity = (x=FunctionField(ρgx, grid, xface; parameters=η0),
               y=FunctionField(ρgy, grid, yface; parameters=η0))

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

    solver_params = (η_rel=1e-1,
                     Δτ=(Pr=r / θ_dτ, τ=(xx=dτ_r, yy=dτ_r, xy=dτ_r), V=(x=nudτ, y=nudτ)))

    rheology = LinearViscousRheology(ConstantField(1.0))

    model = IsothermalFullStokesModel(;
                                      arch,
                                      grid,
                                      boundary_conditions,
                                      gravity,
                                      rheology,
                                      solver_params)

    set!(model.stress.Pr, 0.0)
    foreach(x -> set!(x, 0.0), model.stress.τ)
    foreach(x -> set!(x, 0.0), model.velocity)

    set!(model.viscosity.η, 1.0)
    set!(model.viscosity.η_next, 1.0)

    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.stress)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.velocity)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.viscosity.η)
    KernelLaunch.apply_all_boundary_conditions!(arch, grid, model.boundary_conditions.viscosity.η_next)

    for iter in 1:niter
        advance_iteration!(model, 0.0, 1.0)
        if iter % ncheck == 0
            compute_residuals!(model)
            err = (Pr = norm(model.residual.r_Pr, Inf),
                   Vx = norm(model.residual.r_V.x, Inf),
                   Vy = norm(model.residual.r_V.y, Inf))
            if any(.!isfinite.(values(err)))
                error("simulation failed, err = $err")
            end
            iter_nx = iter / maximum(size(grid))
            @printf("  iter/nx = %.1f, err = [Pr = %1.3e, Vx = %1.3e, Vy = %1.3e]\n", iter_nx, err...)
        end
    end

    V = model.velocity
    τ = model.stress.τ

    Vm = (x=FunctionField(vx, grid, location(V.x)),
          y=FunctionField(vy, grid, location(V.y)))

    τm = (xx=FunctionField(τxx, grid, location(τ.xx); parameters=η0),
          yy=FunctionField(τyy, grid, location(τ.yy); parameters=η0),
          xy=FunctionField(τxy, grid, location(τ.xy); parameters=η0))

    Vm_mag = sqrt.(avx(Vm.x) .^ 2 .+ avy(Vm.y) .^ 2)
    V_mag  = sqrt.(avx(interior(V.x)) .^ 2 .+ avy(interior(V.y)) .^ 2)

    τm_mag = sqrt.(0.5 .* (τm.xx .^ 2 .+ τm.yy .^ 2) .+ av4(τm.xy) .^ 2)
    τ_mag  = sqrt.(0.5 .* (interior(τ.xx) .^ 2 .+ interior(τ.yy) .^ 2) + av4(interior(τ.xy)) .^ 2)

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
    errs[:, iN] .= run((nx, nx))
end

fig = Figure(; fontsize=16)

ax = Axis(fig[1, 1];
          xscale=log2,
          yscale=log10,
          xlabel="N",
          ylabel="L-norms of error",
          xticks=LogTicks(3:8),
          yticks=LogTicks(-5:1:0),
          title="Convergence for free slip confined flow 2D")

ylims!(ax, 1e-5, nothing)

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

save("stokes_manufactured_solution_free_slip_2D.png", fig)
