@tiny function _kernel_update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ)
    ix, iy, iz = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy, iz)
    @inbounds if isin(ηs)
        ε̇xyc = 0.0
        for idz = -1:-1, idy = -1:0, idx = -1:0
            ix2,iy2,iz2 = clamp(ix+idx,1,size(ε̇.xy,1)),clamp(iy+idy,1,size(ε̇.xy,2)),clamp(iz+idz,1,size(ε̇.xy,3))
            ε̇xyc += ε̇.xy[ix2,iy2,iz2]
        end
        ε̇xyc *= 0.25
        ε̇xzc = 0.0
        for idz = -1:0, idy = -1:-1, idx = -1:0
            ix2,iy2,iz2 = clamp(ix+idx,1,size(ε̇.xz,1)),clamp(iy+idy,1,size(ε̇.xz,2)),clamp(iz+idz,1,size(ε̇.xz,3))
            ε̇xzc += ε̇.xz[ix2,iy2,iz2]
        end
        ε̇xzc *= 0.25
        ε̇yzc = 0.0
        for idz = -1:0, idy = -1:0, idx = -1:-1
            ix2,iy2,iz2 = clamp(ix+idx,1,size(ε̇.yz,1)),clamp(iy+idy,1,size(ε̇.yz,2)),clamp(iz+idz,1,size(ε̇.yz,3))
            ε̇yzc += ε̇.yz[ix2,iy2,iz2]
        end
        ε̇yzc *= 0.25
        ε̇II  = sqrt(0.5*(ε̇.xx[ix,iy,iz]^2 + ε̇.yy[ix,iy,iz]^2 + ε̇.zz[ix,iy,iz]^2) + ε̇xyc^2 + ε̇xzc^2 + ε̇yzc^2)
        ηs_t = 0.5*K*exp(-1/n*Q_R*(1/T_mlt - 1/T[ix,iy,iz]))*ε̇II^(1/n-1)
        ηs_t = 1.0/(1/ηs_t + 1/ηreg)
        ηs[ix,iy,iz] = exp(log(ηs[ix,iy,iz])*(1-χ) + log(ηs_t)*χ)
    end
end

@tiny function _kernel_update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
    ix,iy,iz = @indices
    na,ns    = wt.not_air, wt.not_solid
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    # detect and eliminate null spaces
    isnull = (na.x[ix,iy,iz] ≈ 0.0) || (na.x[ix+1,iy  ,iz  ] ≈ 0.0) ||
             (na.y[ix,iy,iz] ≈ 0.0) || (na.y[ix  ,iy+1,iz  ] ≈ 0.0) ||
             (na.z[ix,iy,iz] ≈ 0.0) || (na.z[ix  ,iy  ,iz+1] ≈ 0.0)
    if !isnull && (na.c[ix,iy,iz] > 0.0)
        ε̇.xx[ix,iy,iz] = (V.x[ix+1,iy  ,iz  ]*ns.x[ix+1,iy  ,iz  ] - V.x[ix,iy,iz]*ns.x[ix,iy,iz])/dx
        ε̇.yy[ix,iy,iz] = (V.y[ix  ,iy+1,iz  ]*ns.y[ix  ,iy+1,iz  ] - V.y[ix,iy,iz]*ns.y[ix,iy,iz])/dy
        ε̇.zz[ix,iy,iz] = (V.z[ix  ,iy  ,iz+1]*ns.z[ix  ,iy  ,iz+1] - V.z[ix,iy,iz]*ns.z[ix,iy,iz])/dz
        ∇V = ε̇.xx[ix,iy,iz] + ε̇.yy[ix,iy,iz] + ε̇.zz[ix,iy,iz]
        Pr[ix,iy,iz] -= ∇V*ηs[ix,iy,iz]*r/θ_dτ
        τ.xx[ix,iy,iz] += (-τ.xx[ix,iy,iz] + 2.0*ηs[ix,iy,iz]*(ε̇.xx[ix,iy,iz]-∇V/3.0)) * dτ_r
        τ.yy[ix,iy,iz] += (-τ.yy[ix,iy,iz] + 2.0*ηs[ix,iy,iz]*(ε̇.yy[ix,iy,iz]-∇V/3.0)) * dτ_r
        τ.zz[ix,iy,iz] += (-τ.zz[ix,iy,iz] + 2.0*ηs[ix,iy,iz]*(ε̇.zz[ix,iy,iz]-∇V/3.0)) * dτ_r
    else
        Pr[ix,iy,iz] = 0.0
        τ.xx[ix,iy,iz] = 0.0
        τ.yy[ix,iy,iz] = 0.0
        τ.zz[ix,iy,iz] = 0.0
        ε̇.xx[ix,iy,iz] = 0.0
        ε̇.yy[ix,iy,iz] = 0.0
        ε̇.zz[ix,iy,iz] = 0.0
    end
    @inbounds if isin(τ.xy)
        # detect and eliminate null spaces
        isnull = (na.x[ix+1,iy+1,iz+1] ≈ 0.0) || (na.x[ix+1,iy  ,iz+1] ≈ 0.0) ||
                 (na.y[ix+1,iy+1,iz+1] ≈ 0.0) || (na.y[ix  ,iy+1,iz+1] ≈ 0.0)
        if !isnull && (na.xy[ix,iy,iz] > 0.0)
            ε̇.xy[ix,iy,iz] =
                0.5 * (
                    (V.x[ix+1,iy+1,iz+1]*ns.x[ix+1,iy+1,iz+1] - V.x[ix+1,iy  ,iz+1]*ns.x[ix+1,iy  ,iz+1])/dy +
                    (V.y[ix+1,iy+1,iz+1]*ns.y[ix+1,iy+1,iz+1] - V.y[ix  ,iy+1,iz+1]*ns.y[ix  ,iy+1,iz+1])/dx
                )
            ηs_av = 0.25*(ηs[ix,iy,iz+1] + ηs[ix+1,iy,iz+1] + ηs[ix,iy+1,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.xy[ix,iy,iz] += (-τ.xy[ix,iy,iz] + 2.0*ηs_av*ε̇.xy[ix,iy,iz])*dτ_r
        else
            ε̇.xy[ix,iy,iz] = 0.0
            τ.xy[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(τ.xz)
        # detect and eliminate null spaces
        isnull = (na.x[ix+1,iy+1,iz+1] ≈ 0.0) || (na.x[ix+1,iy+1,iz  ] ≈ 0.0) ||
                 (na.z[ix+1,iy+1,iz+1] ≈ 0.0) || (na.z[ix  ,iy+1,iz+1] ≈ 0.0)
        if !isnull && (na.xz[ix,iy,iz] > 0.0)
            ε̇.xz[ix,iy,iz] =
                0.5 * (
                    (V.x[ix+1,iy+1,iz+1]*ns.x[ix+1,iy+1,iz+1] - V.x[ix+1,iy+1,iz  ]*ns.x[ix+1,iy+1,iz  ])/dz +
                    (V.z[ix+1,iy+1,iz+1]*ns.z[ix+1,iy+1,iz+1] - V.z[ix  ,iy+1,iz+1]*ns.z[ix  ,iy+1,iz+1])/dx
                )
            ηs_av = 0.25*(ηs[ix,iy+1,iz] + ηs[ix+1,iy+1,iz] + ηs[ix,iy+1,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.xz[ix,iy,iz] += (-τ.xz[ix,iy,iz] + 2.0*ηs_av*ε̇.xz[ix,iy,iz])*dτ_r
        else
            ε̇.xz[ix,iy,iz] = 0.0
            τ.xz[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(τ.yz)
        # detect and eliminate null spaces
        isnull = (na.y[ix+1,iy+1,iz+1] ≈ 0.0) || (na.y[ix+1,iy+1,iz  ] ≈ 0.0) ||
                 (na.z[ix+1,iy+1,iz+1] ≈ 0.0) || (na.z[ix+1,iy  ,iz+1] ≈ 0.0)
        if !isnull && (na.yz[ix,iy,iz] > 0.0)
            ε̇.yz[ix,iy,iz] =
                0.5 * (
                    (V.y[ix+1,iy+1,iz+1]*ns.y[ix+1,iy+1,iz+1] - V.y[ix+1,iy+1,iz  ]*ns.y[ix+1,iy+1,iz  ])/dz +
                    (V.z[ix+1,iy+1,iz+1]*ns.z[ix+1,iy+1,iz+1] - V.z[ix+1,iy  ,iz+1]*ns.z[ix+1,iy  ,iz+1])/dy
                )
            ηs_av = 0.25*(ηs[ix+1,iy,iz] + ηs[ix+1,iy+1,iz] + ηs[ix+1,iy,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.yz[ix,iy,iz] += (-τ.yz[ix,iy,iz] + 2.0*ηs_av*ε̇.yz[ix,iy,iz])*dτ_r
        else
            ε̇.yz[ix,iy,iz] = 0.0
            τ.yz[ix,iy,iz] = 0.0
        end
    end
    return
end

@tiny function _kernel_update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz)
    ix,iy,iz = @indices
    na,ns    = wt.not_air, wt.not_solid
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix,iy+1,iz+1] ≈ 0) ||
                 (ns.xy[ix  ,iy+1,iz  ] ≈ 0) || (ns.xy[ix,iy  ,iz  ] ≈ 0) ||
                 (ns.xz[ix  ,iy  ,iz+1] ≈ 0) || (ns.xz[ix,iy  ,iz  ] ≈ 0)
        if !isnull && (na.x[ix+1,iy+1,iz+1] > 0) && (ns.x[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_x is computed correctly
            ηs_x = max(ηs[ix,iy+1,iz+1],ηs[ix+1,iy+1,iz+1])
            ∂σxx_∂x = ((-Pr[ix+1,iy+1,iz+1]+τ.xx[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] -
                       (-Pr[ix  ,iy+1,iz+1]+τ.xx[ix  ,iy+1,iz+1])*na.c[ix  ,iy+1,iz+1])/dx
            ∂τxy_∂y = (τ.xy[ix,iy+1,iz]*na.xy[ix,iy+1,iz] - τ.xy[ix,iy,iz]*na.xy[ix,iy,iz])/dy
            ∂τxz_∂z = (τ.xz[ix,iy,iz+1]*na.xz[ix,iy,iz+1] - τ.xz[ix,iy,iz]*na.xz[ix,iy,iz])/dz
            V.x[ix,iy,iz] += (∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z - ρg.x)*nudτ/ηs_x
        else
            V.x[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(V.y)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix+1,iy,iz+1] ≈ 0) ||
                 (ns.xy[ix+1,iy  ,iz  ] ≈ 0) || (ns.xy[ix  ,iy,iz  ] ≈ 0) ||
                 (ns.yz[ix  ,iy  ,iz+1] ≈ 0) || (ns.yz[ix  ,iy,iz  ] ≈ 0)
        if !isnull && (na.y[ix+1,iy+1,iz+1] > 0) && (ns.y[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_y is computed correctly
            ηs_y = max(ηs[ix+1,iy,iz+1],ηs[ix+1,iy+1,iz+1])
            ∂σyy_∂y = ((-Pr[ix+1,iy+1,iz+1] + τ.yy[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy  ,iz+1] + τ.yy[ix+1,iy  ,iz+1])*na.c[ix+1,iy  ,iz+1])/dy
            ∂τxy_∂x = (τ.xy[ix+1,iy,iz  ]*na.xy[ix+1,iy,iz] - τ.xy[ix,iy,iz]*na.xy[ix,iy,iz])/dx
            ∂τyz_∂z = (τ.yz[ix  ,iy,iz+1]*na.yz[ix,iy,iz+1] - τ.yz[ix,iy,iz]*na.yz[ix,iy,iz])/dz
            V.y[ix,iy,iz] += (∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z - ρg.y)*nudτ/ηs_y
        else
            V.y[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(V.z)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix+1,iy+1,iz  ] ≈ 0) ||
                 (ns.xz[ix+1,iy  ,iz  ] ≈ 0) || (ns.xz[ix  ,iy  ,iz  ] ≈ 0) ||
                 (ns.yz[ix  ,iy+1,iz  ] ≈ 0) || (ns.yz[ix  ,iy  ,iz  ] ≈ 0)
        if !isnull && (na.z[ix+1,iy+1,iz+1] > 0) && (ns.z[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_z is computed correctly
            ηs_z = max(ηs[ix+1,iy+1,iz],ηs[ix+1,iy+1,iz+1])
            ∂σzz_∂z = ((-Pr[ix+1,iy+1,iz+1] + τ.zz[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy+1,iz  ] + τ.zz[ix+1,iy+1,iz  ])*na.c[ix+1,iy+1,iz  ])/dz
            ∂τxz_∂x = (τ.xz[ix+1,iy,iz]*na.xz[ix+1,iy,iz] - τ.xz[ix,iy,iz]*na.xz[ix,iy,iz])/dx
            ∂τyz_∂y = (τ.yz[ix,iy+1,iz]*na.yz[ix,iy+1,iz] - τ.yz[ix,iy,iz]*na.yz[ix,iy,iz])/dy
            V.z[ix,iy,iz] += (∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y - ρg.z)*nudτ/ηs_z
        else
            V.z[ix,iy,iz] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy, dz)
    ns,na = wt.not_solid, wt.not_air
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    if isin(Pr)
        # detect and eliminate null spaces
        isnull = (na.x[ix,iy,iz] ≈ 0.0) || (na.x[ix+1,iy  ,iz  ] ≈ 0.0) ||
                 (na.y[ix,iy,iz] ≈ 0.0) || (na.y[ix  ,iy+1,iz  ] ≈ 0.0) ||
                 (na.z[ix,iy,iz] ≈ 0.0) || (na.z[ix  ,iy  ,iz+1] ≈ 0.0)
        if !isnull && (na.c[ix,iy,iz] > 0.0)
            ε̇xx = (V.x[ix+1,iy  ,iz  ]*ns.x[ix+1,iy  ,iz  ] - V.x[ix,iy,iz]*ns.x[ix,iy,iz])/dx
            ε̇yy = (V.y[ix  ,iy+1,iz  ]*ns.y[ix  ,iy+1,iz  ] - V.y[ix,iy,iz]*ns.y[ix,iy,iz])/dy
            ε̇zz = (V.z[ix  ,iy  ,iz+1]*ns.z[ix  ,iy  ,iz+1] - V.z[ix,iy,iz]*ns.z[ix,iy,iz])/dz
            ∇V = ε̇xx + ε̇yy + ε̇zz
            Res.Pr[ix,iy,iz] = ∇V
        else
            Res.Pr[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(Res.V.x)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix,iy+1,iz+1] ≈ 0) ||
                 (ns.xy[ix  ,iy+1,iz  ] ≈ 0) || (ns.xy[ix,iy  ,iz  ] ≈ 0) ||
                 (ns.xz[ix  ,iy  ,iz+1] ≈ 0) || (ns.xz[ix,iy  ,iz  ] ≈ 0)
        if !isnull && (na.x[ix+1,iy+1,iz+1] > 0) && (ns.x[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_x is computed correctly
            ∂σxx_∂x = ((-Pr[ix+1,iy+1,iz+1]+τ.xx[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] -
                       (-Pr[ix  ,iy+1,iz+1]+τ.xx[ix  ,iy+1,iz+1])*na.c[ix  ,iy+1,iz+1])/dx
            ∂τxy_∂y = (τ.xy[ix,iy+1,iz]*na.xy[ix,iy+1,iz] - τ.xy[ix,iy,iz]*na.xy[ix,iy,iz])/dy
            ∂τxz_∂z = (τ.xz[ix,iy,iz+1]*na.xz[ix,iy,iz+1] - τ.xz[ix,iy,iz]*na.xz[ix,iy,iz])/dz
            Res.V.x[ix,iy,iz] = ∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z - ρg.x
        else
            Res.V.x[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(Res.V.y)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix+1,iy,iz+1] ≈ 0) ||
                 (ns.xy[ix+1,iy  ,iz  ] ≈ 0) || (ns.xy[ix  ,iy,iz  ] ≈ 0) ||
                 (ns.yz[ix  ,iy  ,iz+1] ≈ 0) || (ns.yz[ix  ,iy,iz  ] ≈ 0)
        if !isnull && (na.y[ix+1,iy+1,iz+1] > 0) && (ns.y[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_y is computed correctly
            ∂σyy_∂y = ((-Pr[ix+1,iy+1,iz+1] + τ.yy[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy  ,iz+1] + τ.yy[ix+1,iy  ,iz+1])*na.c[ix+1,iy  ,iz+1])/dy
            ∂τxy_∂x = (τ.xy[ix+1,iy,iz  ]*na.xy[ix+1,iy,iz] - τ.xy[ix,iy,iz]*na.xy[ix,iy,iz])/dx
            ∂τyz_∂z = (τ.yz[ix  ,iy,iz+1]*na.yz[ix,iy,iz+1] - τ.yz[ix,iy,iz]*na.yz[ix,iy,iz])/dz
            Res.V.y[ix,iy,iz] = ∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z - ρg.y
        else
            Res.V.y[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(Res.V.z)
        # detect and eliminate null spaces
        isnull = ( ns.c[ix+1,iy+1,iz+1] ≈ 0) || ( ns.c[ix+1,iy+1,iz  ] ≈ 0) ||
                 (ns.xz[ix+1,iy  ,iz  ] ≈ 0) || (ns.xz[ix  ,iy  ,iz  ] ≈ 0) ||
                 (ns.yz[ix  ,iy+1,iz  ] ≈ 0) || (ns.yz[ix  ,iy  ,iz  ] ≈ 0)
        if !isnull && (na.z[ix+1,iy+1,iz+1] > 0) && (ns.z[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_z is computed correctly
            ∂σzz_∂z = ((-Pr[ix+1,iy+1,iz+1] + τ.zz[ix+1,iy+1,iz+1])*na.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy+1,iz  ] + τ.zz[ix+1,iy+1,iz  ])*na.c[ix+1,iy+1,iz  ])/dz
            ∂τxz_∂x = (τ.xz[ix+1,iy,iz]*na.xz[ix+1,iy,iz] - τ.xz[ix,iy,iz]*na.xz[ix,iy,iz])/dx
            ∂τyz_∂y = (τ.yz[ix,iy+1,iz]*na.yz[ix,iy+1,iz] - τ.yz[ix,iy,iz]*na.yz[ix,iy,iz])/dy
            Res.V.z[ix,iy,iz] = ∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y - ρg.z
        else
            Res.V.z[ix,iy,iz] = 0.0
        end
    end
end