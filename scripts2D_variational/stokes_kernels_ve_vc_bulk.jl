@tiny function _kernel_increment_τ!(Pr, Pr_o, ε, δτ, τ, τ_o, V, ηs, G, K, dt, wt, r, θ_dτ, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
            ε.xx[ix, iy] = (V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy] - V.x[ix, iy] * wt.not_solid.x[ix, iy]) / dx
            ε.yy[ix, iy] = (V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1] - V.y[ix, iy] * wt.not_solid.y[ix, iy]) / dy
            ∇V  = ε.xx[ix, iy] + ε.yy[ix, iy]
            dPr = -∇V - (Pr[ix, iy] - Pr_o[ix, iy]) / K / dt
            # Pr[ix, iy] -= ∇V * ηs[ix, iy] * r / θ_dτ
            Pr[ix, iy] += dPr / (1.0 / (r / θ_dτ * ηs[ix, iy]) + 1.0 / K / dt)
            δτ.xx[ix, iy] = (-(τ.xx[ix, iy] - τ_o.xx[ix, iy]) * ηs[ix, iy] / (G * dt) - τ.xx[ix, iy] + 2.0 * ηs[ix, iy] * (ε.xx[ix, iy] - ∇V / 3.0)) * dτ_r
            δτ.yy[ix, iy] = (-(τ.yy[ix, iy] - τ_o.yy[ix, iy]) * ηs[ix, iy] / (G * dt) - τ.yy[ix, iy] + 2.0 * ηs[ix, iy] * (ε.yy[ix, iy] - ∇V / 3.0)) * dτ_r
        else
            ε.xx[ix, iy] = 0.0
            ε.yy[ix, iy] = 0.0
            Pr[ix, iy] = 0.0
            δτ.xx[ix, iy] = 0.0
            δτ.yy[ix, iy] = 0.0
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
            ηs_av = 0.25 * (ηs[ix, iy] + ηs[ix+1, iy] + ηs[ix, iy+1] + ηs[ix+1, iy+1])
            dτ_r_av = 1.0 / (θ_dτ + ηs_av / (G * dt) + 1.0)
            δτ.xy[ix, iy] = (-(τ.xy[ix, iy] - τ_o.xy[ix, iy]) * ηs_av / (G * dt) - τ.xy[ix, iy] + 2.0 * ηs_av * ε.xy[ix, iy]) * dτ_r_av
        else
            ε.xy[ix, iy] = 0.0
            δτ.xy[ix, iy] = 0.0
        end
    end
    return
end

# @tiny function _kernel_compute_xyc!(εxyc, δτxyc, ε, τxyc, τ_oxyc, ηs, G, dt, θ_dτ, wt)
#     ix, iy = @indices
#     @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
#     @inline isin(A) = checkbounds(Bool, A, ix, iy)
#     @inbounds if isin(εxyc)
#         # detect and eliminate null spaces
#         isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+2, iy+1] ≈ 0.0) ||
#                  (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix+1, iy+2] ≈ 0.0)
#         if !isnull && (wt.not_air.c[ix+1, iy+1] > 0.0)
#             dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
#             εxyc[ix, iy] = av_xy(ε.xy)
#             δτxyc[ix, iy] = (-(τxyc[ix, iy] - τ_oxyc[ix, iy]) * ηs[ix, iy] / (G * dt) - τxyc[ix, iy] + 2.0 * ηs[ix, iy] * εxyc[ix, iy]) * dτ_r
#         else
#             εxyc[ix, iy]  = 0.0
#             δτxyc[ix, iy] = 0.0
#         end
#     end
#     return
# end

# inn τII
@tiny function _kernel_compute_trial_τII!(τII, δτ, τ)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline inn(A) = A[ix+1, iy+1]
    @inbounds τII[ix, iy] = sqrt(0.5 * ((inn(τ.xx) + inn(δτ.xx))^2 + (inn(τ.yy) + inn(δτ.yy))^2) + 0.0*(av_xy(τ.xy) + av_xy(δτ.xy))^2)
    # @inbounds τIIv[ix, iy] = sqrt(0.5 * ((av_xy(τ.xx) + av_xy(δτ.xx))^2 + (av_xy(τ.yy) + av_xy(δτ.yy))^2) + (τ.xy[ix, iy] + av_xy[ix, iy])^2)
    return
end
@tiny function _kernel_compute_trial_τIIv!(τIIv, δτ, τ)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    # @inline inn(A) = A[ix+1, iy+1]
    # @inbounds τII[ix, iy] = sqrt(0.5 * ((inn(τ.xx) + inn(δτ.xx))^2 + (inn(τ.yy) + inn(δτ.yy))^2) + 0.0*(av_xy(τ.xy) + av_xy(δτ.xy))^2)
    @inbounds τIIv[ix, iy] = sqrt(0.5 * ((av_xy(τ.xx) + av_xy(δτ.xx))^2 + (av_xy(τ.yy) + av_xy(δτ.yy))^2) + (τ.xy[ix, iy] + δτ.xy[ix, iy])^2)
    return
end

@tiny function _kernel_update_τ!(Pr, Pr_c, ε, δτ, τ, τ_o, ηs, G, K, dt, τII, τIIv, F, Fv, λ, λv, τ_y, sinϕ, sinψ, η_reg, χλ, θ_dτ, wt)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            ∇V = ε.xx[ix, iy] + ε.yy[ix, iy]
            dτ_r = 1.0 / (θ_dτ + ηs[ix, iy] / (G * dt) + 1.0)
            # plastic business
            F[ix, iy] = τII[ix, iy] - τ_y - Pr[ix, iy] * sinϕ
            λ[ix, iy] = (1.0 - χλ) * λ[ix, iy] + χλ * (max(F[ix, iy], 0.0) / (ηs[ix, iy] * dτ_r + η_reg + K * dt * sinϕ * sinψ))
            dQdτxx = 0.5 * (τ.xx[ix, iy] + δτ.xx[ix, iy]) / τII[ix, iy]
            dQdτyy = 0.5 * (τ.yy[ix, iy] + δτ.yy[ix, iy]) / τII[ix, iy]
            Pr_c[ix, iy] = Pr[ix, iy] + K * dt * λ[ix, iy] * sinψ
            τ.xx[ix, iy] += (-(τ.xx[ix, iy] - τ_o.xx[ix, iy]) * ηs[ix, iy] / (G * dt) - τ.xx[ix, iy] + 2.0 * ηs[ix, iy] * (ε.xx[ix, iy] - ∇V / 3.0 - λ[ix, iy]  * dQdτxx)) * dτ_r
            τ.yy[ix, iy] += (-(τ.yy[ix, iy] - τ_o.yy[ix, iy]) * ηs[ix, iy] / (G * dt) - τ.yy[ix, iy] + 2.0 * ηs[ix, iy] * (ε.yy[ix, iy] - ∇V / 3.0 - λ[ix, iy]  * dQdτyy)) * dτ_r
        else
            τ.xx[ix, iy]  = 0.0
            τ.yy[ix, iy]  = 0.0
        end
    end
    @inbounds if isin(τ.xy)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.xy[ix, iy] > 0.0)
            ηs_av = 0.25 * (ηs[ix, iy] + ηs[ix+1, iy] + ηs[ix, iy+1] + ηs[ix+1, iy+1])
            dτ_r_av = 1.0 / (θ_dτ + ηs_av / (G * dt) + 1.0)
            # plastic business
            # Fv[ix, iy] = av_xy(τII) - τ_y - av_xy(Pr) * sinϕ
            Fv[ix, iy] = τIIv[ix, iy] - τ_y - av_xy(Pr) * sinϕ
            λv[ix, iy] = (1.0 - χλ) * λv[ix, iy] + χλ * (max(Fv[ix, iy], 0.0) / (ηs_av * dτ_r_av + η_reg + K * dt * sinϕ * sinψ))
            dQdτxy = (τ.xy[ix, iy] + δτ.xy[ix, iy]) / τIIv[ix, iy]
            τ.xy[ix, iy] += (-(τ.xy[ix, iy] - τ_o.xy[ix, iy]) * ηs_av / (G * dt) - τ.xy[ix, iy] + 2.0 * ηs_av * (ε.xy[ix, iy] - 0.5 * λv[ix, iy] * dQdτxy)) * dτ_r_av
        else
            τ.xy[ix, iy] = 0.0
        end
    end
    return
end

# inn τII, Fchk, εII, ηs
@tiny function _kernel_compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr_c, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline inn(A) = A[ix+1, iy+1]
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(τII)
        τII[ix, iy] = sqrt(0.5 * (inn(τ.xx)^2 + inn(τ.yy)^2) + av_xy(τ.xy)^2)
        Fchk[ix, iy] = τII[ix, iy] - τ_y - inn(Pr_c) * sinϕ - inn(λ) * η_reg
        # nonlin visc
        εII[ix, iy] = sqrt(0.5 * (inn(ε.xx)^2 + inn(ε.yy)^2) + av_xy(ε.xy)^2)
        ηs_τ = εII[ix, iy]^mpow
        ηs[ix, iy] = min((1.0 - χ) * ηs[ix, iy] + χ * ηs_τ, ηmax)# * wt.not_air.c[ix, iy]
    end
    return
end

@tiny function _kernel_update_old!(τ_o, τ, Pr_o, Pr_c, Pr, λ, λv)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        τ_o.xx[ix, iy] = τ.xx[ix, iy]
        τ_o.yy[ix, iy] = τ.yy[ix, iy]
        Pr[ix, iy] = Pr_c[ix, iy]
        Pr_o[ix, iy] = Pr[ix, iy]
        λ[ix, iy] = 0.0
    end
    @inbounds if isin(τ.xy)
        τ_o.xy[ix, iy] = τ.xy[ix, iy]
        λv[ix, iy] = 0.0
    end
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