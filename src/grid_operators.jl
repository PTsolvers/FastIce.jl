module GridOperators

export ∂ᶜ, ∂ᵛ

export ∂ᵛx, ∂ᵛy, ∂ᵛz, avᵛx, avᵛy, avᵛz, avᵛxy, avᵛxz, avᵛyz, maxlᵛx, maxlᵛy, maxlᵛz
export ∂ᶜx, ∂ᶜy, ∂ᶜz, avᶜx, avᶜy, avᶜz, avᶜxy, avᶜxz, avᶜyz, maxlᶜx, maxlᶜy, maxlᶜz

import Base.@propagate_inbounds

Base.@assume_effects :foldable δ(op, I::CartesianIndex{N}, ::Val{D}) where {N,D} = ntuple(i -> i == D ? op(I[i], 1) : I[i], Val(N)) |> CartesianIndex

Base.@assume_effects :foldable function δ(op, I::CartesianIndex{N}, ::Val{D1}, ::Val{D2}) where {N,D1,D2}
    δI = ntuple(Val(N)) do i
        (i == D1 || i == D2) ? op(I[i], 1) : I[i]
    end
    return CartesianIndex(δI)
end

@propagate_inbounds ∂ᶜ(fv, I, D) = fv[δ(+, I, D)] - fv[I]
@propagate_inbounds ∂ᵛ(fc, I, D) = fc[I] - fc[δ(-, I, D)]

@propagate_inbounds avᶜ(fv, I, D) = 0.5 * (fv[δ(+, I, D)] + fv[I])
@propagate_inbounds avᵛ(fc, I, D) = 0.5 * (fc[I] + fc[δ(-, I, D)])

@propagate_inbounds avᶜ(fv, I, D1, D2) = 0.25 * (fv[I] + fv[δ(+, I, D1)] + fv[δ(+, I, D2)] + fv[δ(+, I, D1, D2)])
@propagate_inbounds avᵛ(fc, I, D1, D2) = 0.25 * (fc[I] + fc[δ(-, I, D1)] + fc[δ(-, I, D2)] + fc[δ(-, I, D1, D2)])

@propagate_inbounds maxlᶜ(fv, I, D) = max(fv[δ(+, I, D)], fv[I])
@propagate_inbounds maxlᵛ(fc, I, D) = max(fc[I], fc[δ(-, I, D)])

for (dim, val) in ((:x, 1), (:y, 2), (:z, 3))
    for loc in (:ᶜ, :ᵛ)
        ∂l    = Symbol(:∂, loc)
        avl   = Symbol(:av, loc)
        maxll = Symbol(:maxl, loc)

        ∂    = Symbol(∂l, dim)
        av   = Symbol(avl, dim)
        maxl = Symbol(maxll, dim)

        @eval begin
            @propagate_inbounds $∂(f, I)    = $∂l(f, I, Val($val))
            @propagate_inbounds $av(f, I)   = $avl(f, I, Val($val))
            @propagate_inbounds $maxl(f, I) = $maxll(f, I, Val($val))
        end
    end
end

for (dim, val1, val2) in ((:xy, 1, 2), (:xz, 1, 3), (:yz, 2, 3))
    for loc in (:ᶜ, :ᵛ)
        avl = Symbol(:av, loc)
        av  = Symbol(avl, dim)

        @eval begin
            @propagate_inbounds $av(f, I) = $avl(f, I, Val($val1), Val($val2))
        end
    end 
end

end