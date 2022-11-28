module CUDABackend

using CUDA
using LinearAlgebra,GeometryBasics,Printf

using ..LevelSets

export init_level_set!,solve_eikonal!

macro get_thread_idx() esc(:( begin
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    iz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    end ))
end

include("kernels.jl")

"""
    init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)

Initialise level set as a signed distance function in a narrow band around a heightmap

# Arguments
- `R` is the rotation matrix
- `cutoff` is the distance from the heightmap within which the levelset computation is accurate
"""
function init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    nthreads = (8,8,4)
    nblocks  = cld.(size(ls),nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks _init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    return
end


"""
    solve_eikonal!(ls,dldt,mask,dx,dy,dz)

Solve eikonal equation to reinitialise the level specified by the approximation `ls`
"""
function solve_eikonal!(ls,dldt,mask,dx,dy,dz;ϵtol = 1e-8)
    dt = 0.5min(dx,dy,dz)
    nthreads = (8,8,8)
    nblocks  = cld.(size(ls),nthreads)
    minsteps,maxsteps = extrema(size(ls))
    ncheck = cld(minsteps,4)
    for istep in 1:maxsteps
        CUDA.@sync @cuda threads=nthreads blocks=nblocks _update_dldt!(dldt,ls,mask,dx,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks _update_ls!(ls,dldt,dt)
        if istep % ncheck == 0
            err = maximum(abs.(dldt))
            @debug @sprintf("iteration # %d , error = %1.3e\n",istep,err)
            if err < ϵtol break end
        end
    end
    return
end

end # module CUDABackend