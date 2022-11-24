# Implementation from Real-Time Collision Detection by Christer Ericson
function closest_point_on_triangle(P,T)
    A,B,C = T
    AB = B - A
    AC = C - A
    AP = P - A
    # P in vertex region outside A
    d1 = dot(AB,AP)
    d2 = dot(AC,AP)
    if d1 <= 0.0 && d2 <= 0.0 return A end
    # P in vertex region outside B
    BP = P - B
    d3 = dot(AB,BP)
    d4 = dot(AC,BP)
    if d3 >= 0.0 && d4 <= d3 return B end
    # P in edge region of AB
    vc = d1*d4 - d3*d2
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0
        v = d1/(d1 - d3)
        return A + v*AB
    end
    # P in vertex region outside C
    CP = P - C
    d5 = dot(AB,CP)
    d6 = dot(AC,CP)
    if d6 >= 0.0 && d5 <= d6 return C end
    # P in edge region of AC
    vb = d5*d2 - d1*d6
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0
        w = d2/(d2 - d6)
        return A + w*AC
    end
    # P in edge region of BC
    va = d3*d6 - d5*d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return B + w * (C - B)
    end
    # P inside triangle
    denom = 1.0/(va + vb + vc)
    v = vb*denom
    w = vc*denom
    return A + AB*v + AC*w
end