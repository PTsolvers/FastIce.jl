include("stokes_kernels_ve.jl")

const _update_old! = _kernel_update_old!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual_P! = _kernel_compute_residual_P!(get_device())
const _compute_residual_V! = _kernel_compute_residual_V!(get_device())

function update_old!(τ_o, τ, λ)
    wait(_update_old!(τ_o, τ, λ; ndrange=axes(λ)))
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

const _increment_τ! = _kernel_increment_τ!(get_device())
const _compute_xyc! = _kernel_compute_xyc!(get_device())
const _compute_trial_τII! = _kernel_compute_trial_τII!(get_device())
const _update_τ! = _kernel_update_τ!(get_device())
const _compute_Fchk_xII_η! = _kernel_compute_Fchk_xII_η!(get_device())

function increment_τ!(Pr, ε, δτ, τ, τ_o, V, ηs, G, dt, wt, r, θ_dτ, dx, dy)
    wait(_increment_τ!(Pr, ε, δτ, τ, τ_o, V, ηs, G, dt, wt, r, θ_dτ, dx, dy; ndrange=axes(Pr)))
    return
end

function compute_xyc!(ε, δτ, τ, τ_o, ηs, G, dt, θ_dτ, wt)
    εxyc_inn = inn(ε.xyc)
    δτxyc_inn = inn(δτ.xyc)
    τxyc_inn = inn(τ.xyc)
    τ_oxyc_inn = inn(τ_o.xyc)
    ηs_inn = inn(ηs)
    wait(_compute_xyc!(εxyc_inn, δτxyc_inn, ε, τxyc_inn, τ_oxyc_inn, ηs_inn, G, dt, θ_dτ, wt; ndrange=axes(ηs_inn)))
    return
end

function compute_trial_τII!(τII, δτ, τ)
    wait(_compute_trial_τII!(τII, δτ, τ; ndrange=axes(τII)))
    return
end

function update_τ!(Pr, ε, δτ, τ, τ_o, ηs, G, dt, τII, F, λ, τ_y, sinϕ, η_reg, χλ, θ_dτ, wt)
    wait(_update_τ!(Pr, ε, δτ, τ, τ_o, ηs, G, dt, τII, F, λ, τ_y, sinϕ, η_reg, χλ, θ_dτ, wt; ndrange=axes(Pr)))
    return
end

function compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax)
    wait(_compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax; ndrange=axes(τII)))
    return
end