using GLMakie
using GeometryBasics
using StaticArrays
using LinearAlgebra

function dual_contour(Ψ::AbstractArray{T,3},xc,yc,zc) where T
    vertices  = Point{3,Float64}[]
    tris = TriangleFace{Int}[]
    vert_idx  = Array{Int,3}(undef,size(Ψ).-1)
    # insert vertices
    for iz in 1:size(Ψ,3)-1, iy in 1:size(Ψ,2)-1, ix in 1:size(Ψ,1)-1
        S = MArray{NTuple{3,2},T}(undef)
        for idz in 0:1,idy in 0:1,idx in 0:1
            S[idx+1,idy+1,idz+1] = Ψ[ix+idx,iy+idy,iz+idz]
        end
        change_sign = !(all(S .> 0) || all(S .< 0))
        if change_sign
            push!(vertices,Point(xc[ix],yc[iy],zc[iz]))
            vert_idx[ix,iy,iz] = length(vertices)
        end
    end
    # insert triangles
    for iz in 1:size(Ψ,3)-1, iy in 1:size(Ψ,2)-1, ix in 1:size(Ψ,1)-1
        if Ψ[ix,iy,iz]*Ψ[ix+1,iy,iz] <= 0
            if iy < 2 || iz < 2
                continue
            end
            i1 = vert_idx[ix,iy-1,iz-1]
            i2 = vert_idx[ix,iy  ,iz-1]
            i3 = vert_idx[ix,iy  ,iz  ]
            i4 = vert_idx[ix,iy-1,iz  ]
            push!(tris,TriangleFace(i1,i2,i3),TriangleFace(i1,i3,i4))
        end
        if Ψ[ix,iy,iz]*Ψ[ix,iy+1,iz] <= 0
            if ix < 2 || iz < 2
                continue
            end
            i1 = vert_idx[ix-1,iy,iz-1]
            i2 = vert_idx[ix  ,iy,iz-1]
            i3 = vert_idx[ix  ,iy,iz  ]
            i4 = vert_idx[ix-1,iy,iz  ]
            push!(tris,TriangleFace(i1,i2,i3),TriangleFace(i1,i3,i4))
        end
        if Ψ[ix,iy,iz]*Ψ[ix,iy,iz+1] <= 0
            if ix < 2 || iy < 2
                continue
            end
            i1 = vert_idx[ix-1,iy-1,iz]
            i2 = vert_idx[ix  ,iy-1,iz]
            i3 = vert_idx[ix  ,iy  ,iz]
            i4 = vert_idx[ix-1,iy  ,iz]
            push!(tris,TriangleFace(i1,i2,i3),TriangleFace(i1,i3,i4))
        end
    end
    return vertices, tris
end

# Coefficients of a cubic Hermite spline in 1D
function hspline_coeffs(p::StaticVector{2},∇p::StaticVector{2})
    return SVector(
        p[1], ∇p[1],
        3*(p[2] - p[1]) - 2*∇p[1] - ∇p[2],
        2*(p[1] - p[2]) +   ∇p[1] + ∇p[2]
    )
end

function hspline_coeffs(p::StaticMatrix{2},∇px::StaticMatrix{2},∇py::StaticMatrix{2},∇pxy::StaticMatrix{2})

end

@inline function eval_poly(a::StaticVector{4},x)
    return a[1] + (a[2] + (a[3] + a[4]*x)*x)*x
end

@inline function eval_poly(a::StaticMatrix{4},x,y)
    return eval_poly(
        SVector(eval_poly(a[:,1],x),
                eval_poly(a[:,2],x),
                eval_poly(a[:,3],x),
                eval_poly(a[:,4],x)),
        y
    )
end

function hspline_interp!(p_i,x_i,xs,ps;bcs=(nothing,nothing))
    dx = step(xs)
    for (ip,x) in enumerate(x_i)
        xdiv = (x-xs[1])/dx
        ix = clamp(floor(Int,xdiv) + 1, 1, length(xs)-1)
        t  = xdiv - (ix-1)
        p  = SVector(ps[ix], ps[ix+1])
        m1 = ix > firstindex(xs)    ? (ps[ix+1] - ps[ix-1]) / 2 : isnothing(bcs[1]) ? ps[2  ] - ps[1    ] : bcs[1] * dx
        m2 = ix < lastindex(xs) - 1 ? (ps[ix+2] - ps[ix  ]) / 2 : isnothing(bcs[2]) ? ps[end] - ps[end-1] : bcs[2] * dx
        m  = SVector(m1, m2)
        p_i[ip] = hspline_interp(p, m, t)
    end
    return
end

function test_interp()
    xs = LinRange(-π,π,11)
    qs = sin.(xs)
    x_i = LinRange(-1.1π,1.1π,101)
    q_i = similar(x_i)
    hspline_interp!(q_i,x_i,xs,qs;bcs=(-1,-1))
    fig = Figure()
    ax  = Axis(fig[1,1];aspect=DataAspect())
    lines!(ax,x_i,q_i)
    scatter!(ax,xs,qs)
    display(fig)
    return
end

test_interp()

function main()
    println("Hello world!")
    Ψ = Array{Float64}(undef,100,100,100)
    xv = LinRange(-2,2,size(Ψ,1))
    yv = LinRange(-2,2,size(Ψ,2))
    zv = LinRange(-2,2,size(Ψ,3))
    xc = 0.5.*(xv[1:end-1].+xv[2:end])
    yc = 0.5.*(yv[1:end-1].+yv[2:end])
    zc = 0.5.*(zv[1:end-1].+zv[2:end])
    for iz in axes(Ψ,3), iy in axes(Ψ,2), ix in axes(Ψ,1)
        Ψ[ix,iy,iz] = sqrt(xv[ix]^2 + yv[iy]^2 + zv[iz]^2) - 1.5
    end
    @time verts,tris = dual_contour(Ψ,xc,yc,zc)
    fig = Figure()
    ax  = Axis3(fig[1,1];aspect=:data,viewmode=:fitzoom)
    limits!(ax,extrema(xv),extrema(yv),extrema(zv))
    isosurface = GeometryBasics.Mesh(verts,tris)
    mesh!(ax,isosurface)
    # wireframe!(ax,isosurface;color=:black)
    display(fig)
    return
end

main()