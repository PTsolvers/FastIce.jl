module GridOperators

export IDX, IDY, IDZ
export ∂ᵛx, ∂ᵛy, ∂ᵛz, aᵛx, aᵛy, aᵛz, aᵛxy, aᵛxz, aᵛyz, mlᵛx, mlᵛy, mlᵛz
export ∂ᶜx, ∂ᶜy, ∂ᶜz, aᶜx, aᶜy, aᶜz, aᶜxy, aᶜxz, aᶜyz, mlᶜx, mlᶜy, mlᶜz

import Base.@propagate_inbounds

const IDX = CartesianIndex(1, 0, 0)
const IDY = CartesianIndex(0, 1, 0)
const IDZ = CartesianIndex(0, 0, 1)

@propagate_inbounds ∂ᶜx(fv, I) = fv[I + IDX] - fv[I]
@propagate_inbounds ∂ᶜy(fv, I) = fv[I + IDY] - fv[I]
@propagate_inbounds ∂ᶜz(fv, I) = fv[I + IDZ] - fv[I]

@propagate_inbounds ∂ᵛx(fc, I) = fc[I] - fc[I - IDX]
@propagate_inbounds ∂ᵛy(fc, I) = fc[I] - fc[I - IDY]
@propagate_inbounds ∂ᵛz(fc, I) = fc[I] - fc[I - IDZ]

@propagate_inbounds aᶜx(fv, I) = 0.5 * (fv[I] + fv[I + IDX])
@propagate_inbounds aᶜy(fv, I) = 0.5 * (fv[I] + fv[I + IDY])
@propagate_inbounds aᶜz(fv, I) = 0.5 * (fv[I] + fv[I + IDZ])

@propagate_inbounds aᵛx(fc, I) = 0.5 * (fc[I] + fc[I - IDX])
@propagate_inbounds aᵛy(fc, I) = 0.5 * (fc[I] + fc[I - IDY])
@propagate_inbounds aᵛz(fc, I) = 0.5 * (fc[I] + fc[I - IDZ])

@propagate_inbounds aᶜxy(fv, I) = 0.25 * (fv[I] + fv[I + IDX] + fv[I + IDY] + fv[I + IDX + IDY])
@propagate_inbounds aᶜxz(fv, I) = 0.25 * (fv[I] + fv[I + IDX] + fv[I + IDZ] + fv[I + IDX + IDZ])
@propagate_inbounds aᶜyz(fv, I) = 0.25 * (fv[I] + fv[I + IDY] + fv[I + IDZ] + fv[I + IDY + IDZ])

@propagate_inbounds aᵛxy(fc, I) = 0.25 * (fc[I] + fc[I - IDX] + fc[I - IDY] + fc[I - IDX - IDY])
@propagate_inbounds aᵛxz(fc, I) = 0.25 * (fc[I] + fc[I - IDX] + fc[I - IDZ] + fc[I - IDX - IDZ])
@propagate_inbounds aᵛyz(fc, I) = 0.25 * (fc[I] + fc[I - IDY] + fc[I - IDZ] + fc[I - IDY - IDZ])

@propagate_inbounds mlᶜx(fv, I) = max(fv[I], fv[I + IDX])
@propagate_inbounds mlᶜy(fv, I) = max(fv[I], fv[I + IDY])
@propagate_inbounds mlᶜz(fv, I) = max(fv[I], fv[I + IDZ])

@propagate_inbounds mlᵛx(fc, I) = max(fc[I], fc[I - IDX])
@propagate_inbounds mlᵛy(fc, I) = max(fc[I], fc[I - IDY])
@propagate_inbounds mlᵛz(fc, I) = max(fc[I], fc[I - IDZ])

end