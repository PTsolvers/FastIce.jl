module PSBackend

using ParallelStencil
@init_parallel_stencil(CUDA,Float64,3)

using ImplicitGlobalGrid

using LinearAlgebra,GeometryBasics,Printf

using ..LevelSets

export init_level_set!,solve_eikonal!


include("ps_kernels.jl")

"""
    init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)

Initialise level set as a signed distance function in a narrow band around a heightmap

# Arguments
- `R` is the rotation matrix
- `cutoff` is the distance from the heightmap within which the levelset computation is accurate
"""
function init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    @parallel (1:size(ls,1),1:size(ls,2),1:size(ls,3)) _init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    return
end


"""
    solve_eikonal!(ls,dldt,mask,dx,dy,dz)

Solve eikonal equation to reinitialise the level specified by the approximation `ls`
"""
function solve_eikonal!(ls,dldt,mask,dx,dy,dz;ϵtol = 1e-8)
    dt = 0.5min(dx,dy,dz)
    minsteps,maxsteps = extrema(size(ls))
    ncheck = cld(minsteps,4)
    for istep in 1:maxsteps
        @parallel _update_dldt!(dldt,ls,mask,dx,dy,dz)
        @parallel _update_ls!(ls,dldt,dt)
        if istep % ncheck == 0
            err = maximum(abs.(dldt))
            @debug @sprintf("iteration # %d , error = %1.3e\n",istep,err)
            if err < ϵtol break end
        end
    end
    return
end

end # module PSBackend