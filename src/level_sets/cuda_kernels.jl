function _exact_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    iz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if !(ix ∈ axes(ls,1)) || !(iy ∈ axes(ls,2)) || !(iz ∈ axes(ls,3)) return end
    P = R*Point3(getindex.(rc,(ix,iy,iz))...)
    ud,sgn = sd_dem(P,cutoff,dem,dem_rc)
    ls[ix,iy,iz] = ud*sgn
    mask[ix,iy,iz] = ud < cutoff
    return
end

function _update_dldt!(dldt,ls,mask,dx,dy,dz)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    iz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if !(ix ∈ axes(ls,1)[2:end-1]) || !(iy ∈ axes(ls,2)[2:end-1]) || !(iz ∈ axes(ls,3)[2:end-1]) return end
    if mask[ix,iy,iz] return end
    # eikonal solve
    dLdx   = max(ls[ix,iy,iz] - min(ls[ix-1,iy,iz],ls[ix+1,iy,iz]),0.0)/dx
    dLdy   = max(ls[ix,iy,iz] - min(ls[ix,iy-1,iz],ls[ix,iy+1,iz]),0.0)/dy
    dLdz   = max(ls[ix,iy,iz] - min(ls[ix,iy,iz-1],ls[ix,iy,iz+1]),0.0)/dz
    dldt_p = -max(S(ls[ix,iy,iz]),0.0)*(sqrt(dLdx^2+dLdy^2+dLdz^2)-1.0)
    dLdx   = min(ls[ix,iy,iz] - max(ls[ix-1,iy,iz],ls[ix+1,iy,iz]),0.0)/dx
    dLdy   = min(ls[ix,iy,iz] - max(ls[ix,iy-1,iz],ls[ix,iy+1,iz]),0.0)/dy
    dLdz   = min(ls[ix,iy,iz] - max(ls[ix,iy,iz-1],ls[ix,iy,iz+1]),0.0)/dz
    dldt_m = -min(S(ls[ix,iy,iz]),0.0)*(sqrt(dLdx^2+dLdy^2+dLdz^2)-1.0)
    dldt[ix,iy,iz] = dldt_p + dldt_m
    return
end

function _update_ls!(ls,dldt,dt)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    iz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if ix ∈ axes(ls,1)[2:end-1] && iy ∈ axes(ls,2)[2:end-1] && iz ∈ axes(ls,3)[2:end-1]
        ls[ix,iy,iz] += dt*dldt[ix,iy,iz]
    end
    # boundary conditions
    if ix == 1 && iy ∈ axes(ls,2) && iz ∈ axes(ls,3)
        ls[1  ,iy,iz] = ls[2    ,iy,iz]
        ls[end,iy,iz] = ls[end-1,iy,iz]
    end
    if ix ∈ axes(ls,1) && iy == 1 && iz ∈ axes(ls,3)
        ls[ix,1  ,iz] = ls[ix,2    ,iz]
        ls[ix,end,iz] = ls[ix,end-1,iz]
    end
    if ix ∈ axes(ls,1) && iy ∈ axes(ls,2) && iz == 1
        ls[ix,iy,1  ] = ls[ix,iy,2    ]
        ls[ix,iy,end] = ls[ix,iy,end-1]
    end
    return
end