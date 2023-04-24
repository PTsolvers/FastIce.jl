include("stokes_kernels.jl")

const _update_ηs! = _kernel_update_ηs!(get_device())
const _update_σ!  = _kernel_update_σ!(get_device())
const _update_V!  = _kernel_update_V!(get_device())
const _compute_residual! = _kernel_compute_residual!(get_device())

function update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ)
    wait(_update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ;ndrange=axes(ηs)))
    return
end

function update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
    wait(_update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz; ndrange=axes(Pr)))
    return
end

function update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz; bwidth)
    V_inn = (x = inn(V.x), y = inn(V.y), z = inn(V.z))
    ranges = split_ndrange(axes(Pr),bwidth)
    ie,oe  =  hide_comm(ranges) do ndrange
        _update_V!(V_inn, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz; ndrange)
    end
    wait.(oe)
    bc_x_neumann!(0.0,V.y,V.z)
    bc_y_neumann!(0.0,V.x,V.z)
    bc_z_neumann!(0.0,V.x,V.y)
    TinyKernels.device_synchronize(FastIce.get_device())
    @. V.x[end,:  ,:] = V.x[end-1,:,:]*wt.not_solid.x[end-1,:,:]
    @. V.x[1  ,:  ,:] = V.x[2    ,:,:]*wt.not_solid.x[2    ,:,:]
    @. V.y[:  ,end,:] = V.y[:,end-1,:]*wt.not_solid.y[:,end-1,:]
    @. V.y[:  ,1  ,:] = V.y[:,2    ,:]*wt.not_solid.y[:,2    ,:]
    TinyKernels.device_synchronize(FastIce.get_device())
    update_halo!(V.x,V.y,V.z)
    wait(ie)
    return
end

function compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy, dz)
    wait(_compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy, dz; ndrange=axes(Pr)))
    return
end