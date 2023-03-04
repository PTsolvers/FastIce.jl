@tiny function _kernel_update_σ!(Pr, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy, dz)
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    # detect and eliminate null spaces
    isnull = (wt.liquid.x[ix,iy,iz] ≈ 0.0) || (wt.liquid.x[ix+1,iy  ,iz  ] ≈ 0.0) ||
             (wt.liquid.y[ix,iy,iz] ≈ 0.0) || (wt.liquid.y[ix  ,iy+1,iz  ] ≈ 0.0) ||
             (wt.liquid.z[ix,iy,iz] ≈ 0.0) || (wt.liquid.z[ix  ,iy  ,iz+1] ≈ 0.0)
    if !isnull && (wt.liquid.c[ix,iy,iz] > 0.0)
        exx = (V.x[ix+1,iy  ,iz  ]*wt.fluid.x[ix+1,iy  ,iz  ] - V.x[ix,iy,iz]*wt.fluid.x[ix,iy,iz])/dx
        eyy = (V.y[ix  ,iy+1,iz  ]*wt.fluid.y[ix  ,iy+1,iz  ] - V.y[ix,iy,iz]*wt.fluid.y[ix,iy,iz])/dy
        ezz = (V.z[ix  ,iy  ,iz+1]*wt.fluid.z[ix  ,iy  ,iz+1] - V.z[ix,iy,iz]*wt.fluid.z[ix,iy,iz])/dz
        ∇V = exx + eyy + ezz
        Pr[ix,iy,iz] -= ∇V*ηs[ix,iy,iz]*r/θ_dτ
        τ.xx[ix,iy,iz] += (-τ.xx[ix,iy,iz] + 2.0*wt.liquid.c[ix,iy,iz]*ηs[ix,iy,iz]*(exx-∇V/3.0)) * dτ_r
        τ.yy[ix,iy,iz] += (-τ.yy[ix,iy,iz] + 2.0*wt.liquid.c[ix,iy,iz]*ηs[ix,iy,iz]*(eyy-∇V/3.0)) * dτ_r
        τ.zz[ix,iy,iz] += (-τ.zz[ix,iy,iz] + 2.0*wt.liquid.c[ix,iy,iz]*ηs[ix,iy,iz]*(ezz-∇V/3.0)) * dτ_r
    else
        Pr[ix,iy,iz] = 0.0
        τ.xx[ix,iy,iz] = 0.0
        τ.yy[ix,iy,iz] = 0.0
        τ.zz[ix,iy,iz] = 0.0
    end
    @inbounds if isin(τ.xy)
        # detect and eliminate null spaces
        isnull = (wt.liquid.x[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.x[ix+1,iy  ,iz+1] ≈ 0.0) ||
                 (wt.liquid.y[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.y[ix  ,iy+1,iz+1] ≈ 0.0)
        if !isnull && (wt.liquid.xy[ix,iy,iz] > 0.0)
            exy =
                0.5 * (
                    (V.x[ix+1,iy+1,iz+1]*wt.fluid.x[ix+1,iy+1,iz+1] - V.x[ix+1,iy  ,iz+1]*wt.fluid.x[ix+1,iy  ,iz+1])/dy +
                    (V.y[ix+1,iy+1,iz+1]*wt.fluid.y[ix+1,iy+1,iz+1] - V.y[ix  ,iy+1,iz+1]*wt.fluid.y[ix  ,iy+1,iz+1])/dx
                )
            ηs_av = 0.25*(ηs[ix,iy,iz+1] + ηs[ix+1,iy,iz+1] + ηs[ix,iy+1,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.xy[ix,iy,iz] += (-τ.xy[ix,iy,iz] + 2.0*wt.liquid.xy[ix,iy,iz]*ηs_av*exy)*dτ_r
        else
            τ.xy[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(τ.xz)
        # detect and eliminate null spaces
        isnull = (wt.liquid.x[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.x[ix+1,iy+1,iz  ] ≈ 0.0) ||
                 (wt.liquid.z[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.z[ix  ,iy+1,iz+1] ≈ 0.0)
        if !isnull && (wt.liquid.xz[ix,iy,iz] > 0.0)
            exz =
                0.5 * (
                    (V.x[ix+1,iy+1,iz+1]*wt.fluid.x[ix+1,iy+1,iz+1] - V.x[ix+1,iy+1,iz  ]*wt.fluid.x[ix+1,iy+1,iz  ])/dz +
                    (V.z[ix+1,iy+1,iz+1]*wt.fluid.z[ix+1,iy+1,iz+1] - V.z[ix  ,iy+1,iz+1]*wt.fluid.z[ix  ,iy+1,iz+1])/dx
                )
            ηs_av = 0.25*(ηs[ix,iy+1,iz] + ηs[ix+1,iy+1,iz] + ηs[ix,iy+1,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.xz[ix,iy,iz] += (-τ.xz[ix,iy,iz] + 2.0*wt.liquid.xz[ix,iy,iz]*ηs_av*exz)*dτ_r
        else
            τ.xz[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(τ.yz)
        # detect and eliminate null spaces
        isnull = (wt.liquid.y[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.y[ix+1,iy+1,iz  ] ≈ 0.0) ||
                 (wt.liquid.z[ix+1,iy+1,iz+1] ≈ 0.0) || (wt.liquid.z[ix+1,iy  ,iz+1] ≈ 0.0)
        if !isnull && (wt.liquid.yz[ix,iy,iz] > 0.0)
            eyz =
                0.5 * (
                    (V.y[ix+1,iy+1,iz+1]*wt.fluid.y[ix+1,iy+1,iz+1] - V.y[ix+1,iy+1,iz  ]*wt.fluid.y[ix+1,iy+1,iz  ])/dz +
                    (V.z[ix+1,iy+1,iz+1]*wt.fluid.z[ix+1,iy+1,iz+1] - V.z[ix+1,iy  ,iz+1]*wt.fluid.z[ix+1,iy  ,iz+1])/dy
                )
            ηs_av = 0.25*(ηs[ix+1,iy,iz] + ηs[ix+1,iy+1,iz] + ηs[ix+1,iy,iz+1] + ηs[ix+1,iy+1,iz+1])
            τ.yz[ix,iy,iz] += (-τ.yz[ix,iy,iz] + 2.0*wt.liquid.yz[ix,iy,iz]*ηs_av*eyz)*dτ_r
        else
            τ.yz[ix,iy,iz] = 0.0
        end
    end
    return
end

@tiny function _kernel_update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy, dz)
    ix,iy,iz = @indices
    @inline isin(A) = checkbounds(Bool,A,ix,iy,iz)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = ( wt.fluid.c[ix+1,iy+1,iz+1] ≈ 0) || ( wt.fluid.c[ix,iy+1,iz+1] ≈ 0) ||
                 (wt.fluid.xy[ix  ,iy+1,iz  ] ≈ 0) || (wt.fluid.xy[ix,iy  ,iz  ] ≈ 0) ||
                 (wt.fluid.xz[ix  ,iy  ,iz+1] ≈ 0) || (wt.fluid.xz[ix,iy  ,iz  ] ≈ 0)
        if !isnull && (wt.liquid.x[ix+1,iy+1,iz+1] > 0) && (wt.fluid.x[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_x is computed correctly
            ηs_x = max(ηs[ix,iy+1,iz+1],ηs[ix+1,iy+1,iz+1])
            ∂σxx_∂x = ((-Pr[ix+1,iy+1,iz+1]+τ.xx[ix+1,iy+1,iz+1])*wt.liquid.c[ix+1,iy+1,iz+1] -
                       (-Pr[ix  ,iy+1,iz+1]+τ.xx[ix  ,iy+1,iz+1])*wt.liquid.c[ix  ,iy+1,iz+1])/dx
            ∂τxy_∂y = (τ.xy[ix,iy+1,iz]*wt.liquid.xy[ix,iy+1,iz] - τ.xy[ix,iy,iz]*wt.liquid.xy[ix,iy,iz])/dy
            ∂τxz_∂z = (τ.xz[ix,iy,iz+1]*wt.liquid.xz[ix,iy,iz+1] - τ.xz[ix,iy,iz]*wt.liquid.xz[ix,iy,iz])/dz
            V.x[ix,iy,iz] += (∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z - ρg.x*wt.liquid.x[ix+1,iy+1,iz+1])*nudτ/ηs_x
        else
            V.x[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(V.y)
        # detect and eliminate null spaces
        isnull = ( wt.fluid.c[ix+1,iy+1,iz+1] ≈ 0) || ( wt.fluid.c[ix+1,iy,iz+1] ≈ 0) ||
                 (wt.fluid.xy[ix+1,iy  ,iz  ] ≈ 0) || (wt.fluid.xy[ix  ,iy,iz  ] ≈ 0) ||
                 (wt.fluid.yz[ix  ,iy  ,iz+1] ≈ 0) || (wt.fluid.yz[ix  ,iy,iz  ] ≈ 0)
        if !isnull && (wt.liquid.y[ix+1,iy+1,iz+1] > 0) && (wt.fluid.y[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_y is computed correctly
            ηs_y = max(ηs[ix+1,iy,iz+1],ηs[ix+1,iy+1,iz+1])
            ∂σyy_∂y = ((-Pr[ix+1,iy+1,iz+1] + τ.yy[ix+1,iy+1,iz+1])*wt.liquid.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy  ,iz+1] + τ.yy[ix+1,iy  ,iz+1])*wt.liquid.c[ix+1,iy  ,iz+1])/dy
            ∂τxy_∂x = (τ.xy[ix+1,iy,iz  ]*wt.liquid.xy[ix+1,iy,iz] - τ.xy[ix,iy,iz]*wt.liquid.xy[ix,iy,iz])/dx
            ∂τyz_∂z = (τ.yz[ix  ,iy,iz+1]*wt.liquid.yz[ix,iy,iz+1] - τ.yz[ix,iy,iz]*wt.liquid.yz[ix,iy,iz])/dz
            V.y[ix,iy,iz] += (∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z - ρg.y*wt.liquid.y[ix+1,iy+1,iz+1])*nudτ/ηs_y
        else
            V.y[ix,iy,iz] = 0.0
        end
    end
    @inbounds if isin(V.z)
        # detect and eliminate null spaces
        isnull = ( wt.fluid.c[ix+1,iy+1,iz+1] ≈ 0) || ( wt.fluid.c[ix+1,iy+1,iz  ] ≈ 0) ||
                 (wt.fluid.xy[ix+1,iy  ,iz  ] ≈ 0) || (wt.fluid.xy[ix  ,iy  ,iz  ] ≈ 0) ||
                 (wt.fluid.yz[ix  ,iy+1,iz  ] ≈ 0) || (wt.fluid.yz[ix  ,iy  ,iz  ] ≈ 0)
        if !isnull && (wt.liquid.y[ix+1,iy+1,iz+1] > 0) && (wt.fluid.y[ix+1,iy+1,iz+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_z is computed correctly
            ηs_z = max(ηs[ix+1,iy+1,iz],ηs[ix+1,iy+1,iz+1])
            ∂σzz_∂z = ((-Pr[ix+1,iy+1,iz+1] + τ.zz[ix+1,iy+1,iz+1])*wt.liquid.c[ix+1,iy+1,iz+1] - 
                       (-Pr[ix+1,iy+1,iz  ] + τ.zz[ix+1,iy+1,iz  ])*wt.liquid.c[ix+1,iy+1,iz  ])/dz
            ∂τxz_∂x = (τ.xz[ix+1,iy,iz]*wt.liquid.xz[ix+1,iy,iz] - τ.xz[ix,iy,iz]*wt.liquid.xz[ix,iy,iz])/dx
            ∂τyz_∂y = (τ.yz[ix,iy+1,iz]*wt.liquid.yz[ix,iy+1,iz] - τ.yz[ix,iy,iz]*wt.liquid.yz[ix,iy,iz])/dy
            V.z[ix,iy,iz] += (∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y - ρg.z*wt.liquid.z[ix+1,iy+1,iz+1])*nudτ/ηs_z
        else
            V.z[ix,iy,iz] = 0.0
        end
    end
    return
end
