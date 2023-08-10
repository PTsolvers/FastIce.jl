using KernelAbstractions

using FastIce.Macros

@kernel function update_η!(η, η_rh, χ, grid, fields, args...)
    ix, iy, iz = @index(Global, NTuple)
    ηt = η_rh(grid, ix, iy, iz, fields, args...)
    @inbounds @all(η) = exp(log(@all(η)) * (1 - χ) + log(ηt) * χ)
end

@kernel function update_σ!(Pr, τ, V, η, Δτ, Δ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, Pr, ix, iy, iz)
        ε̇xx = @∂_xi(V.x) / Δ.x
        ε̇yy = @∂_yi(V.y) / Δ.y
        ε̇zz = @∂_zi(V.z) / Δ.z
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # hydrostatic
        @all(Pr) -= ∇V * @inn(η) * Δτ.Pr
        # deviatoric diagonal
        @all(τ.xx) -= (@all(τ.xx) - 2.0 * @inn(η) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
        @all(τ.yy) -= (@all(τ.yy) - 2.0 * @inn(η) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
        @all(τ.zz) -= (@all(τ.zz) - 2.0 * @inn(η) * (ε̇zz - ∇V / 3.0)) * Δτ.τ.zz
    end
    @inbounds if checkbounds(Bool, τ.xy, ix, iy, iz)
        exy = (V.y[ix+1, iy, iz+1] - V.y[ix, iy, iz+1]) / Δ.x +
              (V.x[ix, iy+1, iz+1] - V.x[ix, iy, iz+1]) / Δ.y
        @all(τ.xy) -= (@all(τ.xy) - @av_xyi(η) * exy) * Δτ.τ.xy
    end
    @inbounds if checkbounds(Bool, τ.xz, ix, iy, iz)
        exz = (V.z[ix+1, iy+1, iz  ] - V.z[ix, iy+1, iz]) / Δ.x +
              (V.x[ix  , iy+1, iz+1] - V.x[ix, iy+1, iz]) / Δ.z
        @all(τ.xz) -= (@all(τ.xz) - @av_xzi(η) * exz) * Δτ.τ.xz
    end
    @inbounds if checkbounds(Bool, τ.yz, ix, iy, iz)
        eyz = (V.z[ix+1, iy+1, iz  ] - V.z[ix+1, iy, iz]) / Δ.y +
              (V.y[ix+1, iy  , iz+1] - V.y[ix+1, iy, iz]) / Δ.z
        @all(τ.yz) -= (@all(τ.yz) - @av_yzi(η) * eyz) * Δτ.τ.yz
    end
end

@kernel function update_V!(V, Pr, τ, η, Δτ, Δ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, V.x, ix, iy, iz)
        ηx = max(η[ix, iy+1, iz+1], η[ix+1, iy+1, iz+1])
        @all(V.x) += ((-@∂_xi(Pr) + @∂_xi(τ.xx)) / Δ.x + @∂_y(τ.xy) / Δ.y + @∂_z(τ.xz) / Δ.z) / ηx * Δτ.V.x
    end
    @inbounds if checkbounds(Bool, V.y, ix, iy, iz)
        ηy = max(η[ix+1, iy, iz+1], η[ix+1, iy+1, iz+1])
        @all(V.y) += ((-@∂_yi(Pr) + @∂_yi(τ.yy)) / Δ.y + @∂_x(τ.xy) / Δ.x + @∂_z(τ.yz) / Δ.z) / ηy * Δτ.V.y
    end
    @inbounds if checkbounds(Bool, V.z, ix, iy, iz)
        ηz = max(η[ix+1, iy+1, iz], η[ix+1, iy+1, iz+1])
        @all(V.z) += ((-@∂_zi(Pr) + @∂_zi(τ.zz)) / Δ.z + @∂_x(τ.xz) / Δ.x + @∂_y(τ.yz) / Δ.y) / ηz * Δτ.V.z
    end
end
