@inline lerp(a,b,t) = a*(1-t) + b*t

@tiny function _kernel_update_qT!(qT,T,wt,λ,T_atm,dx,dy,dz)
    ns,na = wt.not_solid, wt.not_air
    ix, iy, iz = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy, iz)
    @inbounds if isin(qT.x)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1,iy,iz] ≈ 0) || (ns.c[ix,iy,iz] ≈ 0)
        if !isnull && (na.x[ix,iy,iz] > 0) && (ns.x[ix,iy,iz] > 0)
            T_w = lerp(T_atm,T[ix+1,iy,iz],na.c[ix+1,iy,iz])
            T_e = lerp(T_atm,T[ix  ,iy,iz],na.c[ix  ,iy,iz])
            qT.x[ix,iy,iz] = -λ.ice*(T_w - T_e)/dx
        else
            qT.x[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(qT.y)
        # detect and eliminate null spaces
        isnull = (ns.c[ix,iy+1,iz] ≈ 0) || (ns.c[ix,iy,iz] ≈ 0)
        if !isnull && (na.y[ix,iy,iz] > 0) && (ns.y[ix,iy,iz] > 0)
            T_n = lerp(T_atm,T[ix,iy+1,iz],na.c[ix,iy+1,iz])
            T_s = lerp(T_atm,T[ix,iy  ,iz],na.c[ix,iy  ,iz])
            qT.y[ix,iy,iz] = -λ.ice*(T_n - T_s)/dy
        else
            qT.y[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(qT.z)
        # detect and eliminate null spaces
        isnull = (ns.c[ix,iy,iz+1] ≈ 0) || (ns.c[ix,iy,iz] ≈ 0)
        if !isnull && (na.z[ix,iy,iz] > 0) && (ns.z[ix,iy,iz] > 0)
            T_f = lerp(T_atm,T[ix,iy,iz+1],na.c[ix,iy,iz+1])
            T_b = lerp(T_atm,T[ix,iy,iz  ],na.c[ix,iy,iz  ])
            qT.z[ix,iy,iz] = -λ.ice*(T_f - T_b)/dz
        else
            qT.z[ix,iy,iz] = 0.0
        end
    end
end

@tiny function _kernel_update_ρU!(ρU,qT,τ,ε̇,wt,ρU_atm,dt,dx,dy,dz)
    ns,na = wt.not_solid, wt.not_air
    ix, iy, iz = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy, iz)
    @inbounds if isin(ρU)
        isnull = (na.x[ix,iy,iz] ≈ 0.0) || (na.x[ix+1,iy,iz] ≈ 0.0) ||
                 (na.y[ix,iy,iz] ≈ 0.0) || (na.y[ix,iy+1,iz] ≈ 0.0) ||
                 (na.z[ix,iy,iz] ≈ 0.0) || (na.z[ix,iy,iz+1] ≈ 0.0)
        if !isnull && (na.c[ix,iy,iz] > 0.0 && ns.c[ix,iy,iz] > 0.0)
            ∇qx = (qT.x[ix+1,iy,iz]*ns.x[ix+1,iy,iz] - qT.x[ix,iy,iz]*ns.x[ix,iy,iz])/dx
            ∇qy = (qT.y[ix,iy+1,iz]*ns.y[ix,iy+1,iz] - qT.y[ix,iy,iz]*ns.y[ix,iy,iz])/dy
            ∇qz = (qT.z[ix,iy,iz+1]*ns.z[ix,iy,iz+1] - qT.z[ix,iy,iz]*ns.z[ix,iy,iz])/dz
            ∇qT = ∇qx + ∇qy + ∇qz
            # average shear heating contribution on cell vertices
            τxyc,ε̇xyc = 0.0,0.0
            for idz = -1:-1, idy = -1:0, idx = -1:0
                ix2,iy2,iz2 = clamp(ix+idx,1,size(τ.xy,1)),clamp(iy+idy,1,size(τ.xy,2)),clamp(iz+idz,1,size(τ.xy,3))
                τxyc += τ.xy[ix2,iy2,iz2]
                ε̇xyc += ε̇.xy[ix2,iy2,iz2]
            end
            τxyc *= 0.25; ε̇xyc *= 0.25
            # average shear heating contribution on cell vertices
            τxzc,ε̇xzc = 0.0,0.0
            for idz = -1:0, idy = -1:-1, idx = -1:0
                ix2,iy2,iz2 = clamp(ix+idx,1,size(τ.xz,1)),clamp(iy+idy,1,size(τ.xz,2)),clamp(iz+idz,1,size(τ.xz,3))
                τxzc += τ.xz[ix2,iy2,iz2]
                ε̇xzc += ε̇.xz[ix2,iy2,iz2]
            end
            τxzc *= 0.25; ε̇xzc *= 0.25
            # average shear heating contribution on cell vertices
            τyzc,ε̇yzc = 0.0,0.0
            for idz = -1:0, idy = -1:0, idx = -1:-1
                ix2,iy2,iz2 = clamp(ix+idx,1,size(τ.yz,1)),clamp(iy+idy,1,size(τ.yz,2)),clamp(iz+idz,1,size(τ.yz,3))
                τyzc += τ.yz[ix2,iy2,iz2]
                ε̇yzc += ε̇.yz[ix2,iy2,iz2]
            end
            τyzc *= 0.25; ε̇yzc *= 0.25
            SH = τ.xx[ix,iy,iz]*ε̇.xx[ix,iy,iz] +
                 τ.yy[ix,iy,iz]*ε̇.yy[ix,iy,iz] +
                 τ.zz[ix,iy,iz]*ε̇.zz[ix,iy,iz] +
                 2.0*τxyc*ε̇xyc + 2.0*τxzc*ε̇xzc + 2.0*τyzc*ε̇yzc
            ρU[ix,iy,iz] += dt*(-∇qT + SH)
        else
            ρU[ix,iy,iz] = ρU_atm
        end
    end
end