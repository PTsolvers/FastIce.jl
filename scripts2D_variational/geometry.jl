using LinearAlgebra, GeometryBasics

function make_marker_chain_circle(rc, rad, hmax)
    np = ceil(Int, 2π * rad / hmax)
    return [rc + rad .* Point2(reverse(sincospi(2 * (i - 1) / np))...) for i in 1:np]
end

function signed_distance(p::Point2{T}, poly::AbstractVector{Point2{T}}) where {T}
    d = dot(p - poly[1], p - poly[1])
    s = 1.0
    j = length(poly)
    for i in eachindex(poly)
        e = poly[j] - poly[i]
        w = p - poly[i]
        b = w - e .* clamp(dot(w, e) / dot(e, e), 0.0, 1.0)
        d = min(d, dot(b, b))
        c = p[2] >= poly[i][2], p[2] < poly[j][2], e[1] * w[2] > e[2] * w[1]
        if all(c) || all(.!c)
            s = -s
        end
        j = i
    end
    return s * sqrt(d)
end