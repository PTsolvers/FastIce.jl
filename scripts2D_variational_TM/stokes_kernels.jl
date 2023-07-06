@tiny function _kernel_update_σ!(Pr, τ, ε̇, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
    ns,na = wt.not_solid, wt.not_air
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # detect and eliminate null spaces
    isnull = (na.x[ix, iy] ≈ 0.0) || (na.x[ix+1, iy] ≈ 0.0) ||
             (na.y[ix, iy] ≈ 0.0) || (na.y[ix, iy+1] ≈ 0.0)
    if !isnull && (na.c[ix, iy] > 0.0)
        ε̇.xx[ix, iy] = (V.x[ix+1, iy] * ns.x[ix+1, iy] - V.x[ix, iy] * ns.x[ix, iy]) / dx
        ε̇.yy[ix, iy] = (V.y[ix, iy+1] * ns.y[ix, iy+1] - V.y[ix, iy] * ns.y[ix, iy]) / dy
        ∇V = ε̇.xx[ix, iy] + ε̇.yy[ix, iy]
        Pr[ix, iy] -= ∇V * ηs[ix, iy] * r / θ_dτ
        τ.xx[ix, iy] += (-τ.xx[ix, iy] + 2.0 * ηs[ix, iy] * (ε̇.xx[ix,iy] - ∇V / 3.0)) * dτ_r
        τ.yy[ix, iy] += (-τ.yy[ix, iy] + 2.0 * ηs[ix, iy] * (ε̇.yy[ix,iy] - ∇V / 3.0)) * dτ_r
    else
        Pr[ix, iy] = 0.0
        τ.xx[ix, iy] = 0.0
        τ.yy[ix, iy] = 0.0
    end
    @inbounds if isin(τ.xy)
        # detect and eliminate null spaces
        isnull = (na.x[ix+1, iy+1] ≈ 0.0) || (na.x[ix+1, iy] ≈ 0.0) ||
                 (na.y[ix+1, iy+1] ≈ 0.0) || (na.y[ix, iy+1] ≈ 0.0)
        if !isnull && (na.xy[ix, iy] > 0.0)
            ε̇.xy[ix, iy] =
                0.5 * (
                    (V.x[ix+1,iy+1]*ns.x[ix+1,iy+1] - V.x[ix+1, iy]*ns.x[ix+1, iy]) / dy +
                    (V.y[ix+1,iy+1]*ns.y[ix+1,iy+1] - V.y[ix, iy+1]*ns.y[ix, iy+1]) / dx
                )
            ηs_av = 0.25 * (ηs[ix, iy] + ηs[ix+1, iy] + ηs[ix, iy+1] + ηs[ix+1, iy+1])
            τ.xy[ix, iy] += (-τ.xy[ix, iy] + 2.0 * ηs_av * ε̇.xy[ix, iy]) * dτ_r
        else
            τ.xy[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy)
    ns,na = wt.not_solid, wt.not_air
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1, iy+1] ≈ 0) || (ns.c[ix, iy+1] ≈ 0) ||
                 (ns.xy[ix, iy+1] ≈ 0) || (ns.xy[ix, iy] ≈ 0)
        if !isnull && (ns.x[ix+1, iy+1] > 0) && (na.x[ix+1, iy+1] > 0)
            ηs_x = max(ηs[ix, iy+1], ηs[ix+1, iy+1])
            ∂σxx_∂x = ((-Pr[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * na.c[ix+1, iy+1] -
                       (-Pr[ix  , iy+1] + τ.xx[ix  , iy+1]) * na.c[ix  , iy+1]) / dx
            ∂τxy_∂y = (τ.xy[ix, iy+1] * na.xy[ix, iy+1] - τ.xy[ix, iy] * na.xy[ix, iy]) / dy
            V.x[ix, iy] += (∂σxx_∂x + ∂τxy_∂y - ρg.x) * nudτ / ηs_x
        else
            V.x[ix, iy] = 0.0
        end
    end
    @inbounds if isin(V.y)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1, iy+1] ≈ 0) || (ns.c[ix+1, iy] ≈ 0) ||
                 (ns.xy[ix+1, iy] ≈ 0) || (ns.xy[ix, iy] ≈ 0)
        if !isnull && (ns.y[ix+1, iy+1] > 0) && (na.y[ix+1, iy+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_y is computed correctly
            ηs_y = max(ηs[ix+1, iy], ηs[ix+1, iy+1])
            ∂σyy_∂y = ((-Pr[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * na.c[ix+1, iy+1] -
                       (-Pr[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * na.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * na.xy[ix+1, iy] - τ.xy[ix, iy] * na.xy[ix, iy]) / dx
            V.y[ix, iy] += (∂σyy_∂y + ∂τxy_∂x - ρg.y) * nudτ / ηs_y
        else
            V.y[ix, iy] = 0.0
        end
    end
end

@tiny function _kernel_update_ηs!(ηs,ε̇,T,wt,K,n,Q_R,T_mlt,ηreg,χ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(ηs)
        ε̇xyc = 0.0
        for idy = -1:0, idx = -1:0
            ix2,iy2 = clamp(ix+idx,1,size(ε̇.xy,1)),clamp(iy+idy,1,size(ε̇.xy,2))
            ε̇xyc += ε̇.xy[ix2,iy2]
        end
        ε̇xyc *= 0.25
        ε̇II  = sqrt(0.5*(ε̇.xx[ix,iy]^2 + ε̇.yy[ix,iy]^2) + ε̇xyc^2)
        ηs_t = 0.5*K*exp(-1/n*Q_R*(1/T_mlt - 1/T[ix,iy]))*ε̇II^(1/n-1)
        ηs_t = wt.not_air.c[ix,iy]/(1/ηs_t + 1/ηreg)
        ηs[ix,iy] = exp(log(ηs[ix,iy])*(1-χ) + log(ηs_t)*χ)
    end
end

@tiny function _kernel_compute_residual!(Res, Pr, V, τ, wt, ρg, dx, dy)
    ns,na = wt.not_solid, wt.not_air
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (na.x[ix, iy] ≈ 0.0) || (na.x[ix+1, iy] ≈ 0.0) ||
        (na.y[ix, iy] ≈ 0.0) || (na.y[ix, iy+1] ≈ 0.0)
        if !isnull && (na.c[ix, iy] > 0.0)
            exx = (V.x[ix+1, iy] * ns.x[ix+1, iy] - V.x[ix, iy] * ns.x[ix, iy]) / dx
            eyy = (V.y[ix, iy+1] * ns.y[ix, iy+1] - V.y[ix, iy] * ns.y[ix, iy]) / dy
            ∇V  = exx + eyy
            Res.Pr[ix, iy] = ∇V
        else
            Res.Pr[ix, iy] = 0.0
        end
    end
    @inbounds if isin(Res.V.x)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1, iy+1] ≈ 0) || (ns.c[ix, iy+1] ≈ 0) ||
                 (ns.xy[ix, iy+1] ≈ 0) || (ns.xy[ix, iy] ≈ 0)
        if !isnull && (na.x[ix+1, iy+1] > 0) && (ns.x[ix+1, iy+1] > 0)
            ∂σxx_∂x = ((-Pr[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * na.c[ix+1, iy+1] -
                       (-Pr[ix  , iy+1] + τ.xx[ix  , iy+1]) * na.c[ix  , iy+1]) / dx
            ∂τxy_∂y = (τ.xy[ix, iy+1] * na.xy[ix, iy+1] - τ.xy[ix, iy] * na.xy[ix, iy]) / dy
            Res.V.x[ix, iy] = ∂σxx_∂x + ∂τxy_∂y - ρg.x
        else
            Res.V.x[ix, iy] = 0.0
        end
    end
    @inbounds if isin(Res.V.y)
        # detect and eliminate null spaces
        isnull = (ns.c[ix+1, iy+1] ≈ 0) || (ns.c[ix+1, iy] ≈ 0) ||
                 (ns.xy[ix+1, iy] ≈ 0) || (ns.xy[ix, iy] ≈ 0)
        if !isnull && (na.y[ix+1, iy+1] > 0) && (ns.y[ix+1, iy+1] > 0)
            ∂σyy_∂y = ((-Pr[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * na.c[ix+1, iy+1] -
                       (-Pr[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * na.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * na.xy[ix+1, iy] - τ.xy[ix, iy] * na.xy[ix, iy]) / dx
            Res.V.y[ix, iy] = ∂σyy_∂y + ∂τxy_∂x - ρg.y
        else
            Res.V.y[ix, iy] = 0.0
        end
    end
    return
end