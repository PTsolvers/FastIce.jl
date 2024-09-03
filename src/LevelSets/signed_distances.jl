# 2D functions

function sd_poly(p::Point2{T}, poly::AbstractVector{Point2{T}}) where {T}
    d = dot(p - poly[1], p - poly[1])
    s = 1.0
    j = length(poly)
    for i in eachindex(poly)
        e = poly[j] - poly[i]
        w = p - poly[i]
        b = w - e .* clamp(dot(w, e) / dot(e, e), zero(T), oneunit(T))
        d = min(d, dot(b, b))
        c = p[2] >= poly[i][2], p[2] < poly[j][2], e[1] * w[2] > e[2] * w[1]
        if all(c) || all(.!c)
            s = -s
        end
        j = i
    end
    return s * sqrt(d)
end

# 3D functions
@inline S(x) = x == zero(x) ? oneunit(x) : sign(x)
sign_triangle(p, a, b, c) = S(dot(p - a, cross(b - a, c - a)))

@inline function ud_triangle(p, a, b, c)
    dot2(v) = dot(v, v)
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = cross(ba, ac)
    return sqrt((sign(dot(cross(ba, nor), pa)) +
                 sign(dot(cross(cb, nor), pb)) +
                 sign(dot(cross(ac, nor), pc)) < 2)
                ?
                min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0, 1) - pa),
                    dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0, 1) - pb),
                    dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0, 1) - pc))
                :
                dot(nor, pa) * dot(nor, pa) / dot2(nor))
end

@inline function closest_vertex_index(P, grid)
    Δs = spacing(grid, Vertex(), 1, 1)
    O = Grids.origin(grid, Vertex())
    X = @. fld(P - O, Δs)
    I = @. Int(X) + 1
    I1 = 1
    I2 = size(grid, Vertex())
    return clamp.(I, I1, I2) |> CartesianIndex
end

@inline inc(I, dim) = Base.setindex(I, I[dim] + 1, dim)
@inline inc(I) = I + oneunit(I)

@inline function triangle_pair(Iv, dem, grid)
    @inline function sample_dem(I)
        @inbounds x, y = coord(grid, location(dem), I)
        @inbounds Point3(x, y, dem[I])
    end
    T_BL = Triangle(sample_dem(Iv), sample_dem(inc(Iv, 1)), sample_dem(inc(Iv, 2)))
    T_TR = Triangle(sample_dem(inc(Iv, 2)), sample_dem(inc(Iv, 1)), sample_dem(inc(Iv)))
    return T_BL, T_TR
end

@inline function distance_to_triangle_pair(P, Iv, dem, grid)
    T_BL, T_TR = triangle_pair(Iv, dem, grid)
    ud = min(ud_triangle(P, T_BL...), ud_triangle(P, T_TR...))
    return ud, sign_triangle(P, T_BL...)
end

function Base.clamp(p::NTuple{N}, grid::UniformGrid{N}) where {N}
    clamp.(p, Grids.origin(grid, Vertex()), Grids.origin(grid, Vertex()) .+ extent(grid, Vertex()))
end

function sd_dem(P, cutoff, dem, grid::UniformGrid{2})
    @inbounds Pp = clamp((P[1], P[2]), grid)
    BL = closest_vertex_index(Pp .- cutoff, grid)
    TR = closest_vertex_index(Pp .+ cutoff, grid)
    Ic = closest_vertex_index(Pp, grid)
    ud, sgn = distance_to_triangle_pair(P, Ic, dem, grid)
    for Iv in BL:TR
        if Iv == Ic
            continue
        end
        ud_pair, _ = distance_to_triangle_pair(P, Iv, dem, grid)
        ud = min(ud, ud_pair)
    end
    return ud, sgn
end
