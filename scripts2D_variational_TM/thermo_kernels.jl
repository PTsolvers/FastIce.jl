@inline lerp(a,b,t) = a*(1-t) + b*t

@tiny function _kernel_update_qT!(qT,T,wt,λ,T_atm,dx,dy)
    ns,na = wt.not_solid, wt.not_air
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(qT.x)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1,iy] ≈ 0) || (ns.c[ix,iy] ≈ 0)
        if !isnull && (na.x[ix,iy] > 0) && (ns.x[ix,iy] > 0)
            T_w = lerp(T_atm,T[ix+1,iy],na.c[ix+1,iy])
            T_e = lerp(T_atm,T[ix  ,iy],na.c[ix  ,iy])
            qT.x[ix,iy] = -λ.ice*(T_w - T_e)/dx
        else
            qT.x[ix,iy] = 0.0
        end
    end
    @inbounds if isin(qT.y)
        # detect and eliminate null spaces
        isnull = (ns.c[ix,iy+1] ≈ 0) || (ns.c[ix,iy] ≈ 0)
        if !isnull && (na.y[ix,iy] > 0) && (ns.y[ix,iy] > 0)
            T_n = lerp(T_atm,T[ix,iy+1],na.c[ix,iy+1])
            T_s = lerp(T_atm,T[ix,iy  ],na.c[ix,iy  ])
            qT.y[ix,iy] = -λ.ice*(T_n - T_s)/dy
        else
            qT.y[ix,iy] = 0.0
        end
    end
end

@tiny function _kernel_update_ρU!(ρU,qT,τ,ηs,wt,ρU_atm,dt,dx,dy)
    ns,na = wt.not_solid, wt.not_air
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(ρU)
        isnull = (na.x[ix,iy] ≈ 0.0) || (na.x[ix+1,iy] ≈ 0.0) ||
                 (na.y[ix,iy] ≈ 0.0) || (na.y[ix,iy+1] ≈ 0.0)
        if !isnull && (na.c[ix, iy] > 0.0 && ns.c[ix, iy] > 0.0)
            ∇qx = (qT.x[ix+1,iy]*ns.x[ix+1,iy] - qT.x[ix, iy]*ns.x[ix,iy])/dx
            ∇qy = (qT.y[ix,iy+1]*ns.y[ix,iy+1] - qT.y[ix, iy]*ns.y[ix,iy])/dy
            ∇qT = ∇qx + ∇qy

            τxyc = 0.0
            for idy = -1:0, idx = -1:0
                ix2,iy2 = clamp(ix+idx,1,size(τ.xy,1)),clamp(iy+idy,1,size(τ.xy,2))
                τxyc += τ.xy[ix2,iy2]
            end
            τxyc *= 0.25
            τII_sq = τ.xx[ix,iy]^2 + τ.yy[ix,iy]^2 + 2.0*τxyc^2
            SH = 0.5.*τII_sq/ηs[ix,iy]

            ρU[ix,iy] += dt*(-∇qT + SH)
        else
            ρU[ix,iy] = ρU_atm
        end
    end
end