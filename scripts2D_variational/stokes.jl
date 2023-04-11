include("stokes_kernels.jl")

const _update_σ! = _kernel_update_σ!(get_device())
const _compute_invariants! = _kernel_compute_invariants!(get_device())
const _compute_xII_η! = _kernel_compute_xII_η!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual_P! = _kernel_compute_residual_P!(get_device())
const _compute_residual_V! = _kernel_compute_residual_V!(get_device())

function update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
    wait(_update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy; ndrange=axes(Pr)))
    return
end

function compute_invariants!(εII, τII, ε, τ, ηs, χ, mpow)
    εII_inn = inn(εII)
    τII_inn = inn(τII)
    ηs_inn = inn(ηs)
    wait(_compute_xII_η!(εII_inn, τII_inn, ε, τ, ηs_inn, χ, mpow; ndrange=axes(εII_inn)))
    bc_x_neumann!(0.0, εII)
    bc_y_neumann!(0.0, εII)
    bc_x_neumann!(0.0, τII)
    bc_y_neumann!(0.0, τII)
    bc_x_neumann!(0.0, ηs)
    bc_y_neumann!(0.0, ηs)
    return
end

function update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy)
    V_inn = (x=inn(V.x), y=inn(V.y))
    wait(_update_V!(V_inn, Pr, τ, ηs, wt, nudτ, ρg, dx, dy; ndrange=axes(Pr)))
    bc_x_neumann!(0.0, V.y)
    bc_y_neumann!(0.0, V.x)
    return
end

function compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy)
    V_inn = (x=inn(V.x), y=inn(V.y))
    e1 = _compute_residual_P!(Res, V, wt, dx, dy; ndrange=axes(Pr))
    e2 = _compute_residual_V!(Res, Pr, V_inn, τ, wt, ρg, dx, dy; ndrange=axes(Pr))
    wait.((e1, e2))
    return
end