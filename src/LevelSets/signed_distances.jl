export sd_dem

@inline S(x) = x == zero(x) ? oneunit(x) : sign(x)
@inline sign_triangle(p, a, b, c) = S(dot(p - a, cross(b - a, c - a)))

@inline function ud_triangle(p, a, b, c)
    dot2(v) = dot(v, v)
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = cross(ba, ac)
    return sqrt(
        (sign(dot(cross(ba, nor), pa)) +
         sign(dot(cross(cb, nor), pb)) +
         sign(dot(cross(ac, nor), pc)) < 2)
        ?
        min(
            dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0, 1) - pa),
            dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0, 1) - pb),
            dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0, 1) - pc))
        :
        dot(nor, pa) * dot(nor, pa) / dot2(nor))
end

@inline function closest_vertex_index(P, rc)
    lims = map(x -> x[1:end-1], axes.(rc, 1))
    Δ = step.(rc)
    O = first.(rc)
    I = @. clamp(Int(fld(P - O, Δ)) + 1, lims)
    return CartesianIndex(I...)
end

@inline inc(I, dim) = Base.setindex(I, I[dim] + 1, dim)
@inline inc(I) = I + oneunit(I)

@inline function triangle_pair(Iv, dem, rc)
    @inline function sample_dem(I)
        @inbounds x, y = rc[1][I[1]], rc[2][I[2]]
        @inbounds Point3(x, y, dem[I])
    end
    T_BL = Triangle(sample_dem(Iv), sample_dem(inc(Iv, 1)), sample_dem(inc(Iv, 2)))
    T_TR = Triangle(sample_dem(inc(Iv, 2)), sample_dem(inc(Iv, 1)), sample_dem(inc(Iv)))
    return T_BL, T_TR
end

@inline function distance_to_triangle_pair(P, Iv, dem, rc)
    T_BL, T_TR = triangle_pair(Iv, dem, rc)
    ud = min(ud_triangle(P, T_BL...), ud_triangle(P, T_TR...))
    return ud, sign_triangle(P, T_BL...)
end

function sd_dem(P, cutoff, dem, rc)
    @inbounds Pp = clamp.(Point(P[1], P[2]), first.(rc), last.(rc))
    @inbounds P = Point(Pp[1], Pp[2], P[3])
    BL = closest_vertex_index(Pp .- cutoff, rc)
    TR = closest_vertex_index(Pp .+ cutoff, rc)
    Ic = closest_vertex_index(Pp, rc)
    ud, sgn = distance_to_triangle_pair(P, Ic, dem, rc)
    for Iv in BL:TR
        if Iv == Ic
            continue
        end
        ud_pair, _ = distance_to_triangle_pair(P, Iv, dem, rc)
        ud = min(ud, ud_pair)
    end
    return ud, sgn
end