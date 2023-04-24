@tiny function _kernel_increment_τ!(Pr, Pr_o, ε, ε_ve, δτ, τ, τ_o, V, η_ve, ηs, G, K, dt, wt, r, θ_dτ, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
            εxx = (V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy] - V.x[ix, iy] * wt.not_solid.x[ix, iy]) / dx
            εyy = (V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1] - V.y[ix, iy] * wt.not_solid.y[ix, iy]) / dy
            ∇V  = εxx + εyy
            ε.xx[ix, iy] = εxx - ∇V / 3.0
            ε.yy[ix, iy] = εyy - ∇V / 3.0
            dPr = -∇V - (Pr[ix, iy] - Pr_o[ix, iy]) / K / dt
            # Pr[ix, iy] -= ∇V * ηs[ix, iy] * r / θ_dτ
            Pr[ix, iy] += dPr / (1.0 / (r / θ_dτ * ηs[ix, iy]) + 1.0 / K / dt)
            η_ve[ix, iy] = (1.0 / ηs[ix, iy] + 1.0 / (G * dt))^-1
            ε_ve.xx[ix, iy] = ε.xx[ix, iy] + τ_o.xx[ix, iy] / 2.0 / (G * dt)
            ε_ve.yy[ix, iy] = ε.yy[ix, iy] + τ_o.yy[ix, iy] / 2.0 / (G * dt)
            δτ.xx[ix, iy] = (-τ.xx[ix, iy] + 2.0 * η_ve[ix, iy] * ε_ve.xx[ix, iy]) * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
            δτ.yy[ix, iy] = (-τ.yy[ix, iy] + 2.0 * η_ve[ix, iy] * ε_ve.yy[ix, iy]) * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
        else
            ε.xx[ix, iy] = 0.0
            ε.yy[ix, iy] = 0.0
            Pr[ix, iy] = 0.0
            δτ.xx[ix, iy] = 0.0
            δτ.yy[ix, iy] = 0.0
            ε_ve.xx[ix, iy] = 0.0
            ε_ve.yy[ix, iy] = 0.0
        end
    end
    @inbounds if isin(ε.xy)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.xy[ix, iy] > 0.0)
            ε.xy[ix, iy] =
                0.5 * (
                    (V.x[ix+1, iy+1] * wt.not_solid.x[ix+1, iy+1] - V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy]) / dy +
                    (V.y[ix+1, iy+1] * wt.not_solid.y[ix+1, iy+1] - V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1]) / dx
                )
        else
            ε.xy[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_xyc!(εxyc, ε_vexyc, δτxyc, ε, τxyc, τ_oxyc, η_ve, ηs, G, dt, θ_dτ, wt)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(εxyc)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+2, iy+1] ≈ 0.0) ||
                 (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix+1, iy+2] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix+1, iy+1] > 0.0)
            dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
            εxyc[ix, iy] = av_xy(ε.xy)
            ε_vexyc[ix, iy] = εxyc[ix, iy] + τ_oxyc[ix, iy] / 2.0 / (G * dt)
            δτxyc[ix, iy] = (-τxyc[ix, iy] + 2.0 * η_ve[ix, iy] * ε_vexyc[ix, iy]) * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
        else
            εxyc[ix, iy]  = 0.0
            δτxyc[ix, iy] = 0.0
            ε_vexyc[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_trial_τII!(τII, δτ, τ)
    ix, iy = @indices
    @inbounds τII[ix, iy] = sqrt(0.5 * ((τ.xx[ix, iy] + δτ.xx[ix, iy])^2 + (τ.yy[ix, iy] + δτ.yy[ix, iy])^2) + (τ.xyc[ix, iy] + δτ.xyc[ix, iy])^2)
    return
end

@tiny function _kernel_update_τ!(Pr, Pr_c, ε_ve, τ, ηs, η_ve, G, K, dt, τII, τII_c, F, λ, dλdτ, C, cosϕ, sinϕ, Pd, σd, σt, η_reg, θ_dτ, wt)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
            dτ_λ = 1e-2
            # plastic business
            F[ix, iy] = sqrt(τII[ix, iy]^2 + (C[ix, iy] * cosϕ - σt * sinϕ)^2) - (C[ix, iy] * cosϕ + Pr[ix, iy] * sinϕ)
            Pr_c[ix, iy]  = 0.0
            τII_c[ix, iy] = 0.0
            if F[ix, iy] > 0.0
                c             = 0.5 * erfc(((Pr[ix, iy] - Pd) / σd))
                dQdP          = -c * sinϕ
                dQdτ          = τII[ix, iy] / sqrt(τII[ix, iy]^2 + (C[ix, iy] * cosϕ - σt * sinϕ)^2)

                Pr_c[ix, iy]  = Pr[ix, iy]  - λ[ix, iy] * K * dt * dQdP
                τII_c[ix, iy] = τII[ix, iy] - λ[ix, iy] * η_ve[ix, iy] * dQdτ
                dλdτ[ix, iy]  = 0.9 * dλdτ[ix, iy] + sqrt(τII_c[ix, iy]^2 + (C[ix, iy] * cosϕ - σt * sinϕ)^2) - (C[ix, iy] * cosϕ + Pr_c[ix, iy] * sinϕ) - λ[ix, iy] * η_reg
                λ[ix, iy]    += dλdτ[ix, iy] * dτ_λ

                εII_ve        = sqrt(0.5 * (ε_ve.xx[ix, iy]^2 + ε_ve.yy[ix, iy]^2) + ε_ve.xyc[ix, iy]^2)
                η_vep         = τII_c[ix, iy] / 2.0 / εII_ve
            else
                dλdτ[ix, iy]  = 0.0
                Pr_c[ix, iy]  = Pr[ix, iy]
                τII_c[ix, iy] = τII[ix, iy]
                η_vep         = η_ve[ix, iy]
            end
            τ.xx[ix, iy]  += (-τ.xx[ix, iy]  + 2.0 * η_vep * ε_ve.xx[ix, iy])  * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
            τ.yy[ix, iy]  += (-τ.yy[ix, iy]  + 2.0 * η_vep * ε_ve.yy[ix, iy])  * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
            τ.xyc[ix, iy] += (-τ.xyc[ix, iy] + 2.0 * η_vep * ε_ve.xyc[ix, iy]) * dτ_r * ηs[ix, iy] / η_ve[ix, iy]
        else
            τ.xx[ix, iy]  = 0.0
            τ.yy[ix, iy]  = 0.0
            τ.xyc[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_εII_η!(εII, ηs, τ, ε, wt, χ, mpow, ηmax)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(εII)
        # nonlin visc
        εII[ix, iy] = sqrt(0.5 * (ε.xx[ix, iy]^2 + ε.yy[ix, iy]^2) + ε.xyc[ix, iy]^2)
        ηs_τ = εII[ix, iy]^mpow
        ηs[ix, iy] = min((1.0 - χ) * ηs[ix, iy] + χ * ηs_τ, ηmax)# * wt.not_air.c[ix, iy]
    end
    @inbounds if isin(τ.xy)
        τ.xy[ix, iy] = av_xy(τ.xyc)
    end
    return
end

@tiny function _kernel_update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ)
    ix, iy = @indices
    τ_o.xx[ix, iy] = τ.xx[ix, iy]
    τ_o.yy[ix, iy] = τ.yy[ix, iy]
    τ_o.xyc[ix, iy] = τ.xyc[ix, iy]
    Pr[ix, iy] = Pr_c[ix, iy]
    Pr_o[ix, iy] = Pr[ix, iy]
    λ[ix, iy] = 0.0
    return
end

@tiny function _kernel_update_V!(V, Pr_c, τ, ηs, wt, nudτ, ρg, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = (wt.not_solid.c[ix+1, iy+1] ≈ 0) || (wt.not_solid.c[ix, iy+1] ≈ 0) ||
                 (wt.not_solid.xy[ix, iy+1] ≈ 0) || (wt.not_solid.xy[ix, iy] ≈ 0)
        if !isnull && (wt.not_air.x[ix+1, iy+1] > 0) && (wt.not_solid.x[ix+1, iy+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_x is computed correctly
            ηs_x = max(ηs[ix, iy+1], ηs[ix+1, iy+1])
            ∂σxx_∂x = ((-Pr_c[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr_c[ix  , iy+1] + τ.xx[ix  , iy+1]) * wt.not_air.c[ix  , iy+1]) / dx
            ∂τxy_∂y = (τ.xy[ix, iy+1] * wt.not_air.xy[ix, iy+1] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dy
            V.x[ix, iy] += (∂σxx_∂x + ∂τxy_∂y - ρg.x * wt.not_air.x[ix+1, iy+1]) * nudτ / ηs_x
        else
            V.x[ix, iy] = 0.0
        end
    end
    @inbounds if isin(V.y)
        # detect and eliminate null spaces
        isnull = (wt.not_solid.c[ix+1, iy+1] ≈ 0) || (wt.not_solid.c[ix+1, iy] ≈ 0) ||
                 (wt.not_solid.xy[ix+1, iy] ≈ 0) || (wt.not_solid.xy[ix, iy] ≈ 0)
        if !isnull && (wt.not_air.y[ix+1, iy+1] > 0) && (wt.not_solid.y[ix+1, iy+1] > 0)
            # TODO: check which cells contribute to the momentum balance to verify ηs_y is computed correctly
            ηs_y = max(ηs[ix+1, iy], ηs[ix+1, iy+1])
            ∂σyy_∂y = ((-Pr_c[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr_c[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * wt.not_air.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * wt.not_air.xy[ix+1, iy] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dx
            V.y[ix, iy] += (∂σyy_∂y + ∂τxy_∂x - ρg.y * wt.not_air.y[ix+1, iy+1]) * nudτ / ηs_y
        else
            V.y[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_residual_P!(Res, Pr, Pr_o, V, K, dt, wt, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Res.Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
        (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            exx = (V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy] - V.x[ix, iy] * wt.not_solid.x[ix, iy]) / dx
            eyy = (V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1] - V.y[ix, iy] * wt.not_solid.y[ix, iy]) / dy
            ∇V  = exx + eyy
            Res.Pr[ix, iy] = -∇V - (Pr[ix, iy] - Pr_o[ix, iy]) / K / dt
        else
            Res.Pr[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_residual_V!(Res, Pr_c, V, τ, wt, ρg, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = (wt.not_solid.c[ix+1, iy+1] ≈ 0) || (wt.not_solid.c[ix, iy+1] ≈ 0) ||
                 (wt.not_solid.xy[ix, iy+1] ≈ 0) || (wt.not_solid.xy[ix, iy] ≈ 0)
        if !isnull && (wt.not_air.x[ix+1, iy+1] > 0) && (wt.not_solid.x[ix+1, iy+1] > 0)
            ∂σxx_∂x = ((-Pr_c[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr_c[ix  , iy+1] + τ.xx[ix  , iy+1]) * wt.not_air.c[ix  , iy+1]) / dx
            ∂τxy_∂y = (τ.xy[ix, iy+1] * wt.not_air.xy[ix, iy+1] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dy
            Res.V.x[ix, iy] = ∂σxx_∂x + ∂τxy_∂y - ρg.x * wt.not_air.x[ix+1, iy+1]
        else
            Res.V.x[ix, iy] = 0.0
        end
    end
    @inbounds if isin(V.y)
        # detect and eliminate null spaces
        isnull = (wt.not_solid.c[ix+1, iy+1] ≈ 0) || (wt.not_solid.c[ix+1, iy] ≈ 0) ||
                 (wt.not_solid.xy[ix+1, iy] ≈ 0) || (wt.not_solid.xy[ix, iy] ≈ 0)
        if !isnull && (wt.not_air.y[ix+1, iy+1] > 0) && (wt.not_solid.y[ix+1, iy+1] > 0)
            ∂σyy_∂y = ((-Pr_c[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr_c[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * wt.not_air.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * wt.not_air.xy[ix+1, iy] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dx
            Res.V.y[ix, iy] = ∂σyy_∂y + ∂τxy_∂x - ρg.y * wt.not_air.y[ix+1, iy+1]
        else
            Res.V.y[ix, iy] = 0.0
        end
    end
    return
end