"""
    extents(box::AABB)

Returns the extents of the bounding box `box`.
"""
extents(box::AABB) = box.xmax - box.xmin, box.ymax - box.ymin, box.zmax - box.zmin

"AABB center"
function center(box::AABB{T}) where {T}
    half = convert(T, 0.5)
    return half * (box.xmin + box.xmax), half * (box.ymin + box.ymax), half * (box.zmin + box.zmax)
end

"Dilate AABB by extending its limits around the center by certain fraction in each dimension"
function dilate(box::AABB, fractions)
    Δx, Δy, Δz = extents(box) .* fractions
    return AABB(box.xmin - Δx, box.xmax + Δx, box.ymin - Δy, box.ymax + Δy, box.zmin - Δz, box.zmax + Δz)
end

"Filter NaNs."
filtered(X) = filter(_x -> !isnan(_x), X)

"Create AABB enclosing both box1 and box2"
function union(box1::AABB, box2::AABB)
    return AABB(min(box1.xmin, box2.xmin), max(box1.xmax, box2.xmax),
                min(box1.ymin, box2.ymin), max(box1.ymax, box2.ymax),
                min(box1.zmin, box2.zmin), max(box1.zmax, box2.zmax))
end

"Rotate field `X`, `Y`, `Z` with rotation matrix `R`."
function rotate(X, Y, Z, R)
    xrot = R[1, 1] .* X .+ R[1, 2] .* Y .+ R[1, 3] .* Z
    yrot = R[2, 1] .* X .+ R[2, 2] .* Y .+ R[2, 3] .* Z
    zrot = R[3, 1] .* X .+ R[3, 2] .* Y .+ R[3, 3] .* Z
    return xrot, yrot, zrot
end

"Rotate field `X`, `Y`, `Z` with rotation matrix `R` and return extents."
rotate_minmax(X, Y, Z, R) = rotate(collect(extrema(X)), collect(extrema(Y)), collect(extrema(filtered(Z))), R)
