@tiny function _kernel_increment_τ!(Pr, ε, δτ, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            ε.xx[ix, iy] = (V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy] - V.x[ix, iy] * wt.not_solid.x[ix, iy]) / dx
            ε.yy[ix, iy] = (V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1] - V.y[ix, iy] * wt.not_solid.y[ix, iy]) / dy
            ∇V = ε.xx[ix, iy] + ε.yy[ix, iy]
            Pr[ix, iy] -= ∇V * ηs[ix, iy] * r / θ_dτ
            δτ.xx[ix, iy] = (-τ.xx[ix, iy] + 2.0 * ηs[ix, iy] * (ε.xx[ix, iy] - ∇V / 3.0)) * dτ_r
            δτ.yy[ix, iy] = (-τ.yy[ix, iy] + 2.0 * ηs[ix, iy] * (ε.yy[ix, iy] - ∇V / 3.0)) * dτ_r
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
        else
            ε.xy[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_xyc!(εxyc, δτxyc, ε, τxyc, ηs, wt, dτ_r)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(εxyc)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+2, iy+1] ≈ 0.0) ||
                 (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix+1, iy+2] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix+1, iy+1] > 0.0)
            εxyc[ix, iy] = av_xy(ε.xy)
            δτxyc[ix, iy] = (-τxyc[ix, iy] + 2.0 * ηs[ix, iy] * εxyc[ix, iy]) * dτ_r
        else
            εxyc[ix, iy]  = 0.0
            δτxyc[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_trial_τII!(τII, δτ, τ)
    ix, iy = @indices
    @inbounds τII[ix, iy] = sqrt(0.5 * ((τ.xx[ix, iy] + δτ.xx[ix, iy])^2 + (τ.yy[ix, iy] + δτ.yy[ix, iy])^2) + (τ.xyc[ix, iy] + δτ.xyc[ix, iy])^2)
    return
end

@tiny function _kernel_update_τ!(Pr, ε, δτ, τ, ηs, τII, F, λ, τ_y, sinϕ, η_reg, χλ, wt, dτ_r)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Pr)
        # detect and eliminate null spaces
        isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
                 (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
        if !isnull && (wt.not_air.c[ix, iy] > 0.0)
            ∇V = ε.xx[ix, iy] + ε.yy[ix, iy]
            # plastic business
            F[ix, iy] = τII[ix, iy] - τ_y - Pr[ix, iy] * sinϕ
            λ[ix, iy] = (1.0 - χλ) * λ[ix, iy] + χλ * (max(F[ix, iy], 0.0) / (ηs[ix, iy] * dτ_r + η_reg))
            dQdτxx = 0.5 * (τ.xx[ix, iy]  + δτ.xx[ix, iy])  / τII[ix, iy]
            dQdτyy = 0.5 * (τ.yy[ix, iy]  + δτ.yy[ix, iy])  / τII[ix, iy]
            dQdτxy =       (τ.xyc[ix, iy] + δτ.xyc[ix, iy]) / τII[ix, iy]
            τ.xx[ix, iy]  += (-τ.xx[ix, iy]  + 2.0 * ηs[ix, iy] * (ε.xx[ix, iy] - ∇V / 3.0 - λ[ix, iy] * dQdτxx)) * dτ_r
            τ.yy[ix, iy]  += (-τ.yy[ix, iy]  + 2.0 * ηs[ix, iy] * (ε.yy[ix, iy] - ∇V / 3.0 - λ[ix, iy] * dQdτyy)) * dτ_r
            τ.xyc[ix, iy] += (-τ.xyc[ix, iy] + 2.0 * ηs[ix, iy] * (ε.xyc[ix, iy]     - 0.5 * λ[ix, iy] * dQdτxy)) * dτ_r
        else
            τ.xx[ix, iy]  = 0.0
            τ.yy[ix, iy]  = 0.0
            τ.xyc[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_Fchk_xII_η!(τII, Fchk, εII, ηs, Pr, τ, ε, λ, τ_y, sinϕ, η_reg, wt, χ, mpow, ηmax)
    ix, iy = @indices
    @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(τII)
        τII[ix, iy] = sqrt(0.5 * (τ.xx[ix, iy]^2 + τ.yy[ix, iy]^2) + τ.xyc[ix, iy]^2)
        Fchk[ix, iy] = τII[ix, iy] - τ_y - Pr[ix, iy] * sinϕ - λ[ix, iy] * η_reg
        # nonlin visc
        εII[ix, iy] = sqrt(0.5 * (ε.xx[ix, iy]^2 + ε.yy[ix, iy]^2) + ε.xyc[ix, iy]^2)
        ηs_τ = εII[ix, iy]^mpow
        ηs[ix, iy] = wt.not_air.c[ix, iy] * min((1.0 - χ) * ηs[ix, iy] + χ * ηs_τ, ηmax)
    end
    @inbounds if isin(τ.xy)
        τ.xy[ix, iy] = av_xy(τ.xyc)
    end
    return
end

# @tiny function _kernel_compute_xII_η!(εII, τII, ηs, ε, τ, wt, χ, mpow, ηmax)
#     ix, iy = @indices
#     @inline av_xy(A) = 0.25 * (A[ix, iy] + A[ix+1, iy] + A[ix, iy+1] + A[ix+1, iy+1])
#     @inline isin(A) = checkbounds(Bool, A, ix, iy)
#     @inbounds if isin(εII)
#         εII[ix, iy] = sqrt(0.5 * (ε.xx[ix+1, iy+1]^2 + ε.yy[ix+1, iy+1]^2) + ε.xyc[ix+1, iy+1]^2)
#         # εII[ix, iy] = sqrt(0.5 * (ε.xx[ix+1, iy+1]^2 + ε.yy[ix+1, iy+1]^2) + av_xy(ε.xy)^2)
#         # τII[ix, iy] = sqrt(0.5 * (τ.xx[ix+1, iy+1]^2 + τ.yy[ix+1, iy+1]^2) + av_xy(τ.xy)^2)
#         # nonlinear viscosity
#         ηs_τ = εII[ix, iy]^mpow
#         ηs[ix, iy] = wt[ix, iy] * min((1.0 - χ) * ηs[ix, iy] + χ * ηs_τ, ηmax)
#     end
#     return
# end

# @tiny function _kernel_update_σ!(Pr, ε, τ, V, ηs, wt, r, θ_dτ, dτ_r, dx, dy)
#     ix, iy = @indices
#     @inline isin(A) = checkbounds(Bool, A, ix, iy)
#     #=@inbounds=# if isin(Pr)
#         # detect and eliminate null spaces
#         isnull = (wt.not_air.x[ix, iy] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
#                  (wt.not_air.y[ix, iy] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
#         if !isnull && (wt.not_air.c[ix, iy] > 0.0)
#             ε.xx[ix, iy] = (V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy] - V.x[ix, iy] * wt.not_solid.x[ix, iy]) / dx
#             ε.yy[ix, iy] = (V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1] - V.y[ix, iy] * wt.not_solid.y[ix, iy]) / dy
#             ∇V = ε.xx[ix, iy] + ε.yy[ix, iy]
#             Pr[ix, iy] -= ∇V * ηs[ix, iy] * r / θ_dτ
#             τ.xx[ix, iy] += (-τ.xx[ix, iy] + 2.0 * ηs[ix, iy] * (ε.xx[ix, iy] - ∇V / 3.0)) * dτ_r
#             τ.yy[ix, iy] += (-τ.yy[ix, iy] + 2.0 * ηs[ix, iy] * (ε.yy[ix, iy] - ∇V / 3.0)) * dτ_r
#         else
#             ε.xx[ix, iy] = 0.0
#             ε.yy[ix, iy] = 0.0
#             Pr[ix, iy] = 0.0
#             τ.xx[ix, iy] = 0.0
#             τ.yy[ix, iy] = 0.0
#         end
#     end
#     #=@inbounds=# if isin(τ.xy)
#         # detect and eliminate null spaces
#         isnull = (wt.not_air.x[ix+1, iy+1] ≈ 0.0) || (wt.not_air.x[ix+1, iy] ≈ 0.0) ||
#                  (wt.not_air.y[ix+1, iy+1] ≈ 0.0) || (wt.not_air.y[ix, iy+1] ≈ 0.0)
#         if !isnull && (wt.not_air.xy[ix, iy] > 0.0)
#             ε.xy[ix, iy] =
#                 0.5 * (
#                     (V.x[ix+1, iy+1] * wt.not_solid.x[ix+1, iy+1] - V.x[ix+1, iy] * wt.not_solid.x[ix+1, iy]) / dy +
#                     (V.y[ix+1, iy+1] * wt.not_solid.y[ix+1, iy+1] - V.y[ix, iy+1] * wt.not_solid.y[ix, iy+1]) / dx
#                 )
#             ηs_av = 0.25 * (ηs[ix, iy] + ηs[ix+1, iy] + ηs[ix, iy+1] + ηs[ix+1, iy+1])
#             τ.xy[ix, iy] += (-τ.xy[ix, iy] + 2.0 * ηs_av * ε.xy[ix, iy]) * dτ_r
#         else
#             ε.xy[ix, iy] = 0.0
#             τ.xy[ix, iy] = 0.0
#         end
#     end
#     return
# end

@tiny function _kernel_update_V!(V, Pr, τ, ηs, wt, nudτ, ρg, dx, dy)
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
            ∂σxx_∂x = ((-Pr[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr[ix  , iy+1] + τ.xx[ix  , iy+1]) * wt.not_air.c[ix  , iy+1]) / dx
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
            ∂σyy_∂y = ((-Pr[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * wt.not_air.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * wt.not_air.xy[ix+1, iy] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dx
            V.y[ix, iy] += (∂σyy_∂y + ∂τxy_∂x - ρg.y * wt.not_air.y[ix+1, iy+1]) * nudτ / ηs_y
        else
            V.y[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_residual_P!(Res, V, wt, dx, dy)
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
            Res.Pr[ix, iy] = ∇V
        else
            Res.Pr[ix, iy] = 0.0
        end
    end
    return
end

@tiny function _kernel_compute_residual_V!(Res, Pr, V, τ, wt, ρg, dx, dy)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    # TODO: check which volume fraction (non-air or non-solid) really determines the null spaces
    @inbounds if isin(V.x)
        # detect and eliminate null spaces
        isnull = (wt.not_solid.c[ix+1, iy+1] ≈ 0) || (wt.not_solid.c[ix, iy+1] ≈ 0) ||
                 (wt.not_solid.xy[ix, iy+1] ≈ 0) || (wt.not_solid.xy[ix, iy] ≈ 0)
        if !isnull && (wt.not_air.x[ix+1, iy+1] > 0) && (wt.not_solid.x[ix+1, iy+1] > 0)
            ∂σxx_∂x = ((-Pr[ix+1, iy+1] + τ.xx[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr[ix  , iy+1] + τ.xx[ix  , iy+1]) * wt.not_air.c[ix  , iy+1]) / dx
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
            ∂σyy_∂y = ((-Pr[ix+1, iy+1] + τ.yy[ix+1, iy+1]) * wt.not_air.c[ix+1, iy+1] -
                       (-Pr[ix+1, iy  ] + τ.yy[ix+1, iy  ]) * wt.not_air.c[ix+1, iy  ]) / dy
            ∂τxy_∂x = (τ.xy[ix+1, iy] * wt.not_air.xy[ix+1, iy] - τ.xy[ix, iy] * wt.not_air.xy[ix, iy]) / dx
            Res.V.y[ix, iy] = ∂σyy_∂y + ∂τxy_∂x - ρg.y * wt.not_air.y[ix+1, iy+1]
        else
            Res.V.y[ix, iy] = 0.0
        end
    end
    return
end