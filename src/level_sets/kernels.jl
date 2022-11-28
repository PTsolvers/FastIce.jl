function _init_level_set!(ls,mask,dem,rc,dem_rc,cutoff,R)
    @get_thread_idx()
    if !(ix ∈ axes(ls,1)) || !(iy ∈ axes(ls,2)) || !(iz ∈ axes(ls,3)) return end
    x,y,z = rc[1][ix],rc[2][iy],rc[3][iz]
    P = R*Point3(x,y,z)
    ud,sgn = LevelSets.sd_dem(P,cutoff,dem,dem_rc)
    ls[ix,iy,iz]   = ud*sgn
    mask[ix,iy,iz] = ud < cutoff
    return
end

function _update_dldt!(dldt,ls,mask,dx,dy,dz)
    @get_thread_idx()
    if !(ix ∈ axes(ls,1)) || !(iy ∈ axes(ls,2)) || !(iz ∈ axes(ls,3)) return end
    if mask[ix,iy,iz] 
        dldt[ix,iy,iz] = 0
        return
    end
    # eikonal solve
    dLdx_m = if ix > 1;          (ls[ix  ,iy  ,iz  ]-ls[ix-1,iy  ,iz  ])/dx; else 0.0 end
    dLdy_m = if iy > 1;          (ls[ix  ,iy  ,iz  ]-ls[ix  ,iy-1,iz  ])/dy; else 0.0 end
    dLdz_m = if iz > 1;          (ls[ix  ,iy  ,iz  ]-ls[ix  ,iy  ,iz-1])/dz; else 0.0 end
    dLdx_p = if ix < size(ls,1); (ls[ix+1,iy  ,iz  ]-ls[ix  ,iy  ,iz  ])/dx; else 0.0 end
    dLdy_p = if iy < size(ls,2); (ls[ix  ,iy+1,iz  ]-ls[ix  ,iy  ,iz  ])/dy; else 0.0 end
    dLdz_p = if iz < size(ls,3); (ls[ix  ,iy  ,iz+1]-ls[ix  ,iy  ,iz  ])/dz; else 0.0 end
    dLdx2 = ls[ix,iy,iz] > 0 ? max(max(dLdx_m,0)^2,min(dLdx_p,0)^2) : max(min(dLdx_m,0)^2,max(dLdx_p,0)^2)
    dLdy2 = ls[ix,iy,iz] > 0 ? max(max(dLdy_m,0)^2,min(dLdy_p,0)^2) : max(min(dLdy_m,0)^2,max(dLdy_p,0)^2)
    dLdz2 = ls[ix,iy,iz] > 0 ? max(max(dLdz_m,0)^2,min(dLdz_p,0)^2) : max(min(dLdz_m,0)^2,max(dLdz_p,0)^2)
    dldt[ix,iy,iz] = sign(ls[ix,iy,iz])*(1 - sqrt(dLdx2 + dLdy2 + dLdz2))
    return
end

function _update_ls!(ls,dldt,dt)
    @get_thread_idx()
    if ix ∈ axes(ls,1) && iy ∈ axes(ls,2) && iz ∈ axes(ls,3)
        ls[ix,iy,iz] += dt*dldt[ix,iy,iz]
    end
    return
end