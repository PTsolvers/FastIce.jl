module GridOperators

export IDX, IDY, IDZ
export ∂ᵛx, ∂ᵛy, ∂ᵛz, avᵛx, avᵛy, avᵛz, avᵛxy, avᵛxz, avᵛyz, maxlᵛx, maxlᵛy, maxlᵛz
export ∂ᶜx, ∂ᶜy, ∂ᶜz, avᶜx, avᶜy, avᶜz, avᶜxy, avᶜxz, avᶜyz, maxlᶜx, maxlᶜy, maxlᶜz

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

@propagate_inbounds avᶜx(fv, I) = 0.5 * (fv[I] + fv[I + IDX])
@propagate_inbounds avᶜy(fv, I) = 0.5 * (fv[I] + fv[I + IDY])
@propagate_inbounds avᶜz(fv, I) = 0.5 * (fv[I] + fv[I + IDZ])

@propagate_inbounds avᵛx(fc, I) = 0.5 * (fc[I] + fc[I - IDX])
@propagate_inbounds avᵛy(fc, I) = 0.5 * (fc[I] + fc[I - IDY])
@propagate_inbounds avᵛz(fc, I) = 0.5 * (fc[I] + fc[I - IDZ])

@propagate_inbounds avᶜxy(fv, I) = 0.25 * (fv[I] + fv[I + IDX] + fv[I + IDY] + fv[I + IDX + IDY])
@propagate_inbounds avᶜxz(fv, I) = 0.25 * (fv[I] + fv[I + IDX] + fv[I + IDZ] + fv[I + IDX + IDZ])
@propagate_inbounds avᶜyz(fv, I) = 0.25 * (fv[I] + fv[I + IDY] + fv[I + IDZ] + fv[I + IDY + IDZ])

@propagate_inbounds avᵛxy(fc, I) = 0.25 * (fc[I] + fc[I - IDX] + fc[I - IDY] + fc[I - IDX - IDY])
@propagate_inbounds avᵛxz(fc, I) = 0.25 * (fc[I] + fc[I - IDX] + fc[I - IDZ] + fc[I - IDX - IDZ])
@propagate_inbounds avᵛyz(fc, I) = 0.25 * (fc[I] + fc[I - IDY] + fc[I - IDZ] + fc[I - IDY - IDZ])

@propagate_inbounds maxlᶜx(fv, I) = max(fv[I], fv[I + IDX])
@propagate_inbounds maxlᶜy(fv, I) = max(fv[I], fv[I + IDY])
@propagate_inbounds maxlᶜz(fv, I) = max(fv[I], fv[I + IDZ])

@propagate_inbounds maxlᵛx(fc, I) = max(fc[I], fc[I - IDX])
@propagate_inbounds maxlᵛy(fc, I) = max(fc[I], fc[I - IDY])
@propagate_inbounds maxlᵛz(fc, I) = max(fc[I], fc[I - IDZ])

end