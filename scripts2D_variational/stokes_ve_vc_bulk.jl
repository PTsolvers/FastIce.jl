include("stokes_kernels_ve_vc_bulk.jl")

const _update_old! = _kernel_update_old!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual_P! = _kernel_compute_residual_P!(get_device())
const _compute_residual_V! = _kernel_compute_residual_V!(get_device())

function update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ, λv)
    wait(_update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ, λv; ndrange=axes(λ)))
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

const _increment_τ! = _kernel_increment_τ!(get_device())
# const _compute_xyc! = _kernel_compute_xyc!(get_device())
const _compute_trial_τII! = _kernel_compute_trial_τII!(get_device())
const _compute_trial_τIIv! = _kernel_compute_trial_τIIv!(get_device())
const _update_τ! = _kernel_update_τ!(get_device())
const _compute_Fchk_xII_η! = _kernel_compute_Fchk_xII_η!(get_device())

function increment_τ!(Pr, Pr_o, ε, δτ, τ, τ_o, V, ηs, G, K, dt, wt, r, θ_dτ, dx, dy)
    wait(_increment_τ!(Pr, Pr_o, ε, δτ, τ, τ_o, V, ηs, G, K, dt, wt, r, θ_dτ, dx, dy; ndrange=axes(Pr)))
    return
end

# function compute_xyc!(ε, δτ, τ, τ_o, ηs, G, dt, θ_dτ, wt)
#     εxyc_inn = inn(ε.xyc)
#     δτxyc_inn = inn(δτ.xyc)
#     τxyc_inn = inn(τ.xyc)
#     τ_oxyc_inn = inn(τ_o.xyc)
#     ηs_inn = inn(ηs)
#     wait(_compute_xyc!(εxyc_inn, δτxyc_inn, ε, τxyc_inn, τ_oxyc_inn, ηs_inn, G, dt, θ_dτ, wt; ndrange=axes(ηs_inn)))
#     return
# end

function compute_trial_τII!(τII, δτ, τ)
    τII_inn = inn(τII)
    wait(_compute_trial_τII!(τII_inn, δτ, τ; ndrange=axes(τII_inn)))
    bc_x_neumann!(0.0, τII)
    bc_y_neumann!(0.0, τII)
    return
end

function compute_trial_τIIv!(τIIv, δτ, τ)
    wait(_compute_trial_τIIv!(τIIv, δτ, τ; ndrange=axes(τIIv)))
    # bc_x_neumann!(0.0, τII)
    # bc_y_neumann!(0.0, τII)
    return
end

function update_τ!(Pr, Pr_c, ε, δτ, τ, τ_o, ηs, G, K, dt, τII, τIIv, F, Fv, λ, λv, τ_y, sinϕ, sinψ, η_reg, χλ, θ_dτ, wt)
    wait(_update_τ!(Pr, Pr_c, ε, δτ, τ, τ_o, ηs, G, K, dt, τII, τIIv, F, Fv, λ, λv, τ_y, sinϕ, sinψ, η_reg, χλ, θ_dτ, wt; ndrange=axes(Pr)))
    return
end

function compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr_c, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax)
    τII_inn = inn(τII)
    Fchk_inn = inn(Fchk)
    εII_inn = inn(εII)
    ηs_inn = inn(ηs)
    wait(_compute_Fchk_xII_η!(τII_inn, Fchk_inn, εII_inn, ηs_inn, Pr_c, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax; ndrange=axes(τII_inn)))
    bc_x_neumann!(0.0, τII)
    bc_y_neumann!(0.0, τII)
    bc_x_neumann!(0.0, ηs)
    bc_y_neumann!(0.0, ηs)
    return
end