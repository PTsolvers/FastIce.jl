using KernelAbstractions

include("../common/macros.jl")

@kernel function update_η!(η, τ, η_rh, χ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds τII = sqrt(0.5 * (@inn(τ.xx)^2 + @inn(τ.yy)^2 + @inn(τ.zz)^2) + @av_xy(τ.xy)^2 + @av_xz(τ.xz)^2 + @av_yz(τ.yz)^2)
    @inbounds @inn(η) = exp(log(@inn(η)) * (1 - χ) + log(η_rh(τII)) * χ)
end

@kernel function update_σ!(Pr, τ, V, η, Δτ, Δ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, Pr, ix, iy, iz)
        ε̇xx = @∂_x(V.x) / Δ.x
        ε̇yy = @∂_y(V.y) / Δ.y
        ε̇zz = @∂_z(V.z) / Δ.z
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # hydrostatic
        @all(Pr) -= ∇V * @all(η) * Δτ.Pr
        # deviatoric diagonal
        @all(τ.xx) -= (@all(τ.xx) - 2.0 * @all(η) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
        @all(τ.yy) -= (@all(τ.yy) - 2.0 * @all(η) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
        @all(τ.zz) -= (@all(τ.zz) - 2.0 * @all(η) * (ε̇zz - ∇V / 3.0)) * Δτ.τ.zz
    end
    @inbounds if checkbounds(Bool, τ.xy, ix, iy, iz)
        @all(τ.xy) -= (@all(τ.xy) - @av_xyi(η) * (@∂_xi(V.y) / Δ.x + @∂_yi(V.x) / Δ.y)) * Δτ.τ.xy
    end
    @inbounds if checkbounds(Bool, τ.xz, ix, iy, iz)
        @all(τ.xz) -= (@all(τ.xz) - @av_xzi(η) * (@∂_xi(V.z) / Δ.x + @∂_zi(V.x) / Δ.z)) * Δτ.τ.xz
    end
    @inbounds if checkbounds(Bool, τ.yz, ix, iy, iz)
        @all(τ.yz) -= (@all(τ.yz) - @av_yzi(η) * (@∂_yi(V.z) / Δ.y + @∂_zi(V.y) / Δ.z)) * Δτ.τ.yz
    end
end

@kernel function update_V!(V, Pr, τ, η, Δτ, Δ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, V.x, ix, iy, iz)
        ηx = max(η[ix, iy+1, iz+1], η[ix+1, iy+1, iz+1])
        @inn(V.x) += ((-@∂_xi(Pr) + @∂_xi(τ.xx)) / Δ.x + @∂_y(τ.xy) / Δ.y + @∂_z(τ.xz) / Δ.z) / ηx * Δτ.V.x
    end
    @inbounds if checkbounds(Bool, V.y, ix, iy, iz)
        ηy = max(η[ix+1, iy, iz+1], η[ix+1, iy+1, iz+1])
        @inn(V.y) += ((-@∂_yi(Pr) + @∂_yi(τ.yy)) / Δ.y + @∂_x(τ.xy) / Δ.x + @∂_z(τ.yz) / Δ.z) / ηy * Δτ.V.y
    end
    @inbounds if checkbounds(Bool, V.z, ix, iy, iz)
        ηz = max(η[ix+1, iy+1, iz], η[ix+1, iy+1, iz+1])
        @inn(V.z) += ((-@∂_zi(Pr) + @∂_zi(τ.zz)) / Δ.z + @∂_x(τ.xz) / Δ.x + @∂_y(τ.yz) / Δ.y) / ηz * Δτ.V.z
    end
end
