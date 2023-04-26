include("stokes_kernels.jl")

const _update_old! = _kernel_update_old!(get_device())
const _increment_τ! = _kernel_increment_τ!(get_device())
const _compute_xyc! = _kernel_compute_xyc!(get_device())
const _compute_trial_τII! = _kernel_compute_trial_τII!(get_device())
const _update_τ! = _kernel_update_τ!(get_device())
const _compute_εII_η! = _kernel_compute_εII_η!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual_P! = _kernel_compute_residual_P!(get_device())
const _compute_residual_V! = _kernel_compute_residual_V!(get_device())

function update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ)
    wait(_update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ; ndrange=axes(λ)))
    return
end

function increment_τ!(Pr, Pr_o, ε, ε_ve, δτ, τ, τ_o, V, η_ve, ηs, G, K, dt, wt, r, θ_dτ, dx, dy)
    wait(_increment_τ!(Pr, Pr_o, ε, ε_ve, δτ, τ, τ_o, V, η_ve, ηs, G, K, dt, wt, r, θ_dτ, dx, dy; ndrange=axes(Pr)))
    return
end

function compute_xyc!(ε, ε_ve, δτ, τ, τ_o, η_ve, ηs, G, dt, θ_dτ, wt)
    εxyc_inn = inn(ε.xyc)
    ε_vexyc_inn = inn(ε_ve.xyc)
    δτxyc_inn = inn(δτ.xyc)
    τxyc_inn = inn(τ.xyc)
    τ_oxyc_inn = inn(τ_o.xyc)
    η_ve_inn = inn(η_ve)
    ηs_inn = inn(ηs)
    wait(_compute_xyc!(εxyc_inn, ε_vexyc_inn, δτxyc_inn, ε, τxyc_inn, τ_oxyc_inn, η_ve_inn, ηs_inn, G, dt, θ_dτ, wt; ndrange=axes(ηs_inn)))
    return
end

function compute_trial_τII!(τII, δτ, τ)
    wait(_compute_trial_τII!(τII, δτ, τ; ndrange=axes(τII)))
    return
end

function update_τ!(Pr, Pr_c, ε_ve, τ, ηs, η_ve, G, K, dt, τII, τII_c, F, λ, dλdτ, dτ_λ, γλ, C, cosϕ, sinϕ, Pd, σd, σt, η_reg, θ_dτ, wt)
    wait(_update_τ!(Pr, Pr_c, ε_ve, τ, ηs, η_ve, G, K, dt, τII, τII_c, F, λ, dλdτ, dτ_λ, γλ, C, cosϕ, sinϕ, Pd, σd, σt, η_reg, θ_dτ, wt; ndrange=axes(Pr)))
    return
end

function compute_εII_η!(εII, ηs, τ, ε, wt, χ, mpow, npow, A0, ηmax)
    wait(_compute_εII_η!(εII, ηs, τ, ε, wt, χ, mpow, npow, A0, ηmax; ndrange=axes(εII)))
    return
end

function update_V!(V, Pr_c, τ, ηs, wt, nudτ, ρg, dx, dy)
    V_inn = (x=inn(V.x), y=inn(V.y))
    wait(_update_V!(V_inn, Pr_c, τ, ηs, wt, nudτ, ρg, dx, dy; ndrange=axes(Pr_c)))
    bc_x_neumann!(0.0, V.y)
    bc_y_neumann!(0.0, V.x)
    return
end

function compute_residual!(Res, Pr, Pr_o, Pr_c, V, τ, K, dt, wt, ρg, dx, dy)
    V_inn = (x=inn(V.x), y=inn(V.y))
    e1 = _compute_residual_P!(Res, Pr, Pr_o, V, K, dt, wt, dx, dy; ndrange=axes(Pr))
    e2 = _compute_residual_V!(Res, Pr_c, V_inn, τ, wt, ρg, dx, dy; ndrange=axes(Pr))
    wait.((e1, e2))
    return
end