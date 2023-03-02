include("stokes_kernels.jl")

const _update_σ! = Kernel(_kernel_update_σ!,get_device())
const _update_V! = Kernel(_kernel_update_V!,get_device())

function update_σ!(Pr, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
    wait(_update_σ!(Pr, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz; ndrange=axes(Pr)))
    return
end

function update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz; bwidth)
    V_inn = (x = inn(V.x), y = inn(V.y), z = inn(V.z))
    ranges = split_ndrange(axes(Pr),bwidth)
    ie,oe =  hide_comm(ranges) do ndrange
        _update_V!(V_inn, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz; ndrange)
    end
    wait(oe)
    update_halo!(Vx,Vy,Vz)
    bc_x_neumann!(0.0,V.y,V.z)
    bc_y_neumann!(0.0,V.x,V.z)
    bc_z_neumann!(0.0,V.x,V.y)
    wait(ie)
    return
end