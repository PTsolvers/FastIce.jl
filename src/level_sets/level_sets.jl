module LevelSets

export exact_level_set!,solve_eikonal!,rotation_matrix

using CUDA,LinearAlgebra,GeometryBasics

include("geometry.jl")
include("cuda_kernels.jl")

@inline S(x) = (x==0.0) ? 1.0 : sign(x)

@inline sign_triangle(p,a,b,c) = S(dot(p-a,cross(b-a,c-a)))

@inline function ud_triangle(p,a,b,c)
    dot2(v) = dot(v,v)
    ba  = b - a; pa = p - a
    cb  = c - b; pb = p - b
    ac  = a - c; pc = p - c
    nor = cross(ba,ac)
    return sqrt(
       (sign(dot(cross(ba,nor),pa)) +
        sign(dot(cross(cb,nor),pb)) +
        sign(dot(cross(ac,nor),pc)) < 2)
        ?
        min(
        dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
        dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb),
        dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
        :
        dot(nor,pa)*dot(nor,pa)/dot2(nor) )
end

@inline function closest_vertex_index(P,rc)
    lims = map(x->x[1:end-1],axes.(rc,1))
    Δ = step.(rc)
    O = first.(rc)
    I = @. clamp(Int(fld(P-O,Δ))+1,lims)
    return CartesianIndex(I...)
end

@inline inc(I,dim) = Base.setindex(I,I[dim]+1,dim)
@inline inc(I) = I + oneunit(I)

@inline function triangle_pair(Iv,dem,rc)
    sample_dem(I) = Point3(getindex.(rc,Tuple(I))...,dem[I])
    T_BL = Triangle(sample_dem(Iv)       ,sample_dem(inc(Iv,1)),sample_dem(inc(Iv,2)))
    T_TR = Triangle(sample_dem(inc(Iv,2)),sample_dem(inc(Iv,1))  ,sample_dem(inc(Iv)))
    return T_BL,T_TR
end

@inline function distance_to_triangle_pair(P,Iv,dem,rc)
    T_BL,T_TR = triangle_pair(Iv,dem,rc)
    ud = min(ud_triangle(P,T_BL...),ud_triangle(P,T_TR...))
    return ud,sign_triangle(P,T_BL...)
end

function sd_dem(P,cutoff,dem,rc)
    Pp = Point(P[1],P[2])
    Pp = clamp.(Pp,first.(rc),last.(rc))
    P  = Point(Pp[1],Pp[2],P[3])
    BL = closest_vertex_index(Pp.-cutoff,rc)
    TR = closest_vertex_index(Pp.+cutoff,rc)
    Ic = closest_vertex_index(Pp,rc)
    ud,sgn = distance_to_triangle_pair(P,Ic,dem,rc)
    for Iv in BL:TR
        if Iv == Ic continue end
        ud_pair,_ = distance_to_triangle_pair(P,Iv,dem,rc)
        ud = min(ud,ud_pair)
    end
    return ud,sgn
end

function rotation_matrix(α,β,γ)
    sα,cα = sincos(α)
    sβ,cβ = sincos(β)
    sγ,cγ = sincos(γ)
    return Mat3(
        cα*cβ         ,sα*cβ         ,-sβ,
        cα*sβ*sγ-sα*cγ,sα*sβ*sγ+cα*cγ,cβ*sγ,
        cα*sβ*cγ+sα*sγ,sα*sβ*cγ-cα*sγ,cβ*cγ,
    )
end


"""
    exact_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)

Initialise level set as a signed distance function in a narrow band around a heightmap

# Arguments
- `R` is the rotation matrix
- `cutoff` is the distance from the heightmap within which the levelset computation is accurate
"""
function exact_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    nthreads = (8,8,4)
    nblocks  = cld.(size(ls),nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks _exact_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
end


"""
    solve_eikonal!(ls,dldt,mask,dx,dy,dz)

Solve eikonal equation to approximate the values of the level set outside the narrow band specified by the `mask` argument
"""
function solve_eikonal!(ls,dldt,mask,dx,dy,dz)
    dt = min(dx,dy,dz)/2.0
    nthreads = (8,8,8)
    nblocks  = cld.(size(ls),nthreads)
    for _ in 1:maximum(size(ls))
        CUDA.@sync @cuda threads=nthreads blocks=nblocks _update_dldt!(dldt,ls,mask,dx,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks _update_ls!(ls,dldt,dt)
    end
    return
end

end # module LevelSets
