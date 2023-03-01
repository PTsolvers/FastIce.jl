include("level_set_kernels.jl")

const _init_level_set! = Kernel(_kernel_init_level_set!,get_device())
const _compute_dΨ_dt!  = Kernel(_kernel_compute_dΨ_dt! ,get_device())
const _update_Ψ!       = Kernel(_kernel_update_Ψ!      ,get_device())

function compute_level_set_from_dem!(Ψ,dem,dem_grid,Ψ_grid)
    TinyKernels.device_synchronize(get_device())
    dx,dy,dz = step.(Ψ_grid)
    cutoff   = 5max(dx,dy,dz)
    R        = LinearAlgebra.I
    wait(_init_level_set!(Ψ,dem,dem_grid,Ψ_grid,cutoff,R;ndrange=axes(Ψ)))
    return
end

