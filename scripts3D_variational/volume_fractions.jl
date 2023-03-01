@inline perturb(ϕ) = abs(ϕ) > 1e-20 ? ϕ : (ϕ > 0 ? 1e-20 : -1e-20)

@inline trivol(v1,v2,v3) = 0.5*abs(cross(v3-v1,v2-v1))

function volfrac(tri,ϕ::Vec3{T})::T where T
    v1,v2,v3 = tri
    if ϕ[1] < 0 && ϕ[2] < 0 && ϕ[3] < 0 # ---
        return trivol(v1,v2,v3)
    elseif ϕ[1] > 0 && ϕ[2] > 0 && ϕ[3] > 0 # +++
        return 0.0
    end
    @inline vij(i,j) = tri[j]*(ϕ[i]/(ϕ[i]-ϕ[j])) - tri[i]*(ϕ[j]/(ϕ[i]-ϕ[j]))
    v12,v13,v23 = vij(1,2),vij(1,3),vij(2,3)
    if ϕ[1] < 0
        if ϕ[2] < 0
            trivol(v1,v23,v13) + trivol(v1,v2,v23)  # --+
        else
            if ϕ[3] < 0
                trivol(v3,v12,v23) + trivol(v3,v1,v12) # -+-
            else
                trivol(v1,v12,v13) # -++
            end
        end
    else
        if ϕ[2] < 0
            if ϕ[3] < 0
                trivol(v2,v13,v12) + trivol(v2,v3,v13) # +--
            else
                trivol(v12,v2,v23) # +-+
            end
        else
            trivol(v13,v23,v3) # ++-
        end 
    end
end

function volfrac(rect::Rect2{T},ϕ::Vec4{T}) where T
    or,ws = origin(rect), widths(rect)
    v1,v2,v3,v4 = or,or+Vec(ws[1],0.0),or+ws,or+Vec(0.0,ws[2])
    ϕ1,ϕ2,ϕ3,ϕ4 = perturb.(ϕ)
    return volfrac(Vec(v1,v2,v3),Vec3{T}(ϕ1,ϕ2,ϕ3)) + 
           volfrac(Vec(v1,v3,v4),Vec3{T}(ϕ1,ϕ3,ϕ4))
end

@inline tetvol(v1,v2,v3,v4) = abs(det([v2-v1 v3-v1 v4-v1]))/6.0

function volfrac(tet,ϕ::Vec4)
    v1,v2,v3,v4 = tet
    @inline vij(i,j) = tet[j]*(ϕ[i]/(ϕ[i]-ϕ[j])) - tet[i]*(ϕ[j]/(ϕ[i]-ϕ[j]))
    nneg = count(ϕ.<0)
    if nneg == 0     # ++++
        return 0.0
    elseif nneg == 1 # -+++
        if ϕ[1] < 0
            return tetvol(v1,vij(1,2),vij(1,3),vij(1,4))
        elseif ϕ[2] < 0
            return tetvol(v2,vij(2,1),vij(2,3),vij(2,4))
        elseif ϕ[3] < 0
            return tetvol(v3,vij(3,1),vij(3,2),vij(3,4))
        else # ϕ[4] < 0
            return tetvol(v4,vij(4,1),vij(4,2),vij(4,3))
        end
    elseif nneg == 2 # --++
        if ϕ[1] < 0 && ϕ[2] < 0
            return tetvol(v1      ,v2      ,vij(1,3),vij(2,4)) +
                   tetvol(vij(2,3),v2      ,vij(1,3),vij(2,4)) +
                   tetvol(v1      ,vij(1,4),vij(1,3),vij(2,4))
        elseif ϕ[1] < 0 && ϕ[3] < 0
            return tetvol(v1      ,v3      ,vij(1,4),vij(3,2)) +
                   tetvol(vij(3,4),v3      ,vij(1,4),vij(3,2)) +
                   tetvol(v1      ,vij(1,2),vij(1,4),vij(3,2))
        elseif ϕ[1] < 0 && ϕ[4] < 0
            return tetvol(v1      ,v4      ,vij(1,2),vij(4,3)) +
                   tetvol(vij(4,2),v4      ,vij(1,2),vij(4,3)) +
                   tetvol(v1      ,vij(1,3),vij(1,2),vij(4,3))
        elseif ϕ[2] < 0 && ϕ[3] < 0
            return tetvol(v3      ,v2      ,vij(3,1),vij(2,4)) +
                   tetvol(vij(2,1),v2      ,vij(3,1),vij(2,4)) +
                   tetvol(v3      ,vij(3,4),vij(3,1),vij(2,4))
        elseif ϕ[2] < 0 && ϕ[4] < 0
            return tetvol(v4      ,v2      ,vij(4,1),vij(2,3)) +
                   tetvol(vij(2,1),v2      ,vij(4,1),vij(2,3)) +
                   tetvol(v4      ,vij(4,3),vij(4,1),vij(2,3))
        else # ϕ[3] < 0 && ϕ[4] < 0
            return tetvol(v3      ,v4      ,vij(3,1),vij(4,2)) +
                   tetvol(vij(4,1),v4      ,vij(3,1),vij(4,2)) +
                   tetvol(v3      ,vij(3,2),vij(3,1),vij(4,2))
        end
    elseif nneg == 3 # ---+
        vol_tot = tetvol(v1,v2,v3,v4)
        if ϕ[1] >= 0
            return vol_tot - tetvol(v1,vij(1,2),vij(1,3),vij(1,4))
        elseif ϕ[2] >= 0
            return vol_tot - tetvol(v2,vij(2,1),vij(2,3),vij(2,4))
        elseif ϕ[3] >= 0
            return vol_tot - tetvol(v3,vij(3,1),vij(3,2),vij(3,4))
        else # ϕ[4] >= 0
            return vol_tot - tetvol(v4,vij(4,1),vij(4,2),vij(4,3))
        end
    else # ----
        return tetvol(v1,v2,v3,v4)
    end
end

function volfrac(rect::Rect3,ϕ::Vec{8})
    or,ws = origin(rect), widths(rect)
    v000,v001,v100,v101 = or                   ,or+Vec(ws[1],0.0,0.0  ),or+Vec(0.0,ws[2],0.0  ),or+Vec(ws[1],ws[2],0.0  )
    v010,v011,v110,v111 = or+Vec(0.0,0.0,ws[3]),or+Vec(ws[1],0.0,ws[3]),or+Vec(0.0,ws[2],ws[3]),or+Vec(ws[1],ws[2],ws[3])
    ϕ = perturb.(ϕ)
    return volfrac(Vec(v000,v100,v010,v001),Vec(ϕ[1],ϕ[5],ϕ[3],ϕ[2])) + 
           volfrac(Vec(v110,v100,v010,v111),Vec(ϕ[7],ϕ[5],ϕ[3],ϕ[7])) +
           volfrac(Vec(v101,v100,v111,v001),Vec(ϕ[6],ϕ[5],ϕ[7],ϕ[2])) +
           volfrac(Vec(v011,v111,v010,v001),Vec(ϕ[4],ϕ[7],ϕ[3],ϕ[2])) +
           volfrac(Vec(v111,v100,v010,v001),Vec(ϕ[7],ϕ[5],ϕ[3],ϕ[2]))
end

include("volume_fraction_kernels.jl")

const _compute_volume_fractions_from_level_set! = Kernel(_kernel_compute_volume_fractions_from_level_set!,get_device())

function compute_volume_fractions_from_level_set!(wt,Ψ,dx,dy,dz)
    @views inn_x(A) = A[2:end-1,:,:]
    @views inn_y(A) = A[:,2:end-1,:]
    @views inn_z(A) = A[:,:,2:end-1]
    wt_inn = (;c=wt.c,x=inn_x(wt.x),y=inn_y(wt.y),z=inn_z(wt.z),xy=wt.xy,xz=wt.xz,yz=wt.yz)
    wait(_compute_volume_fractions_from_level_set!(wt_inn,Ψ,dx,dy,dz;ndrange=axes(Ψ)))
    bc_x_neumann!(0.0,wt.x)
    bc_y_neumann!(0.0,wt.y)
    bc_z_neumann!(0.0,wt.z)
    return
end