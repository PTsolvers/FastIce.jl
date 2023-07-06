include("stokes_kernels.jl")

const _update_ηs! = _kernel_update_ηs!(get_device())
const _update_σ! = _kernel_update_σ!(get_device())
const _update_V! = _kernel_update_V!(get_device())
const _compute_residual! = _kernel_compute_residual!(get_device())

function update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ)
    wait(_update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ;ndrange=axes(ηs)))
    return
end

function update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
    wait(_update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy; ndrange=axes(Pr)))
    return
end

function update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy)
    V_inn = (x=inn(V.x), y=inn(V.y))
    wait(_update_V!(V_inn, Pr, τ, ηs, wt, nudτ, ρg, dx, dy; ndrange=axes(Pr)))
    bc_x_neumann!(0.0, V.y)
    bc_y_neumann!(0.0, V.x)
    TinyKernels.device_synchronize(FastIce.get_device())
    @. V.x[end,:] = V.x[end-1,:]*wt.not_solid.x[end-1,:]
    @. V.x[1  ,:] = V.x[2    ,:]*wt.not_solid.x[2    ,:]
    TinyKernels.device_synchronize(FastIce.get_device())
    return
end

function compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy)
    wait(_compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy; ndrange=axes(Pr)))
    return
end