include("stokes_kernels.jl")

# const _update_σ! = _kernel_update_σ!(get_device())
# const _compute_xII_η! = _kernel_compute_xII_η!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual_P! = _kernel_compute_residual_P!(get_device())
const _compute_residual_V! = _kernel_compute_residual_V!(get_device())

# function update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
#     wait(_update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy; ndrange=axes(Pr)))
#     return
# end

# function compute_xII_η!(εII, τII, ηs, ε, τ, wt, χ, mpow, ηmax)
#     εII_inn = inn(εII)
#     τII_inn = inn(τII)
#     ηs_inn = inn(ηs)
#     wt_inn = inn(wt.not_air.c)
#     wait(_compute_xII_η!(εII_inn, τII_inn, ηs_inn, ε, τ, wt_inn, χ, mpow, ηmax; ndrange=axes(εII_inn)))
#     bc_x_neumann!(0.0, εII)
#     bc_y_neumann!(0.0, εII)
#     bc_x_neumann!(0.0, τII)
#     bc_y_neumann!(0.0, τII)
#     bc_x_neumann!(0.0, ηs)
#     bc_y_neumann!(0.0, ηs)
#     return
# end

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
# const _compute_check_τII! = _kernel_compute_check_τII!(get_device())
const _compute_Fchk_xII_η! = _kernel_compute_Fchk_xII_η!(get_device())

function increment_τ!(Pr, ε, δτ, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
    wait(_increment_τ!(Pr, ε, δτ, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy; ndrange=axes(Pr)))
    return
end

function compute_xyc!(ε, δτ, τ, ηs, wt, dτ_r)
    εxyc_inn = inn(ε.xyc)
    δτxyc_inn = inn(δτ.xyc)
    τxyc_inn = inn(τ.xyc)
    ηs_inn = inn(ηs)
    wait(_compute_xyc!(εxyc_inn, δτxyc_inn, ε, τxyc_inn, ηs_inn, wt, dτ_r; ndrange=axes(ηs_inn)))
    return
end

function compute_trial_τII!(τII, δτ, τ)
    wait(_compute_trial_τII!(τII, δτ, τ; ndrange=axes(τII)))
    return
end

function update_τ!(Pr, ε, δτ, τ, ηs, τII, F, λ, τ_y, sinϕ, η_reg, χλ, wt, dτ_r)
    wait(_update_τ!(Pr, ε, δτ, τ, ηs, τII, F, λ, τ_y, sinϕ, η_reg, χλ, wt, dτ_r; ndrange=axes(Pr)))
    return
end

# function compute_check_τII!(τII, Fchk, Pr, τ, λ, τ_y, sinϕ, η_reg)
#     wait(_compute_check_τII!(τII, Fchk, Pr, τ, λ, τ_y, sinϕ, η_reg; ndrange=axes(τII)))
#     return
# end

function compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax)
    wait(_compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax; ndrange=axes(τII)))
    return
end