# Volume fraction kernel (2D)
@kernel inbounds = true function compute_volfrac_from_level_set!(wt, Ψ, grid::UniformGrid{2}, O)
    I = @index(Global, NTuple)
    ix, iy = I + O
    dx, dy = spacing(grid)
    cell = Rect(Vec(0.0, 0.0), Vec(dx, dy))
    ω = GeometryBasics.volume(cell)
    Ψ_ax(dix, diy) = 0.5 * (Ψ[ix+dix, iy+diy] + Ψ[ix+dix+1, iy+diy])
    Ψ_ay(dix, diy) = 0.5 * (Ψ[ix+dix, iy+diy] + Ψ[ix+dix, iy+diy+1])
    Ψ_axy(dix, diy) = 0.25 * (Ψ[ix+dix, iy+diy] + Ψ[ix+dix+1, iy+diy] + Ψ[ix+dix, iy+diy+1] + Ψ[ix+dix+1, iy+diy+1])
    # cell centers
    Ψs = Vec{4}(Ψ[ix, iy], Ψ[ix+1, iy], Ψ[ix+1, iy+1], Ψ[ix, iy+1])
    wt.c[ix, iy] = volfrac(cell, Ψs) / ω
    # x faces
    Ψs = Vec{4}(Ψ_ax(0, 0), Ψ_ax(1, 0), Ψ_ax(1, 1), Ψ_ax(0, 1))
    wt.x[ix, iy] = volfrac(cell, Ψs) / ω
    # y faces
    Ψs = Vec{4}(Ψ_ay(0, 0), Ψ_ay(1, 0), Ψ_ay(1, 1), Ψ_ay(0, 1))
    wt.y[ix, iy] = volfrac(cell, Ψs) / ω
    # xy edges
    Ψs = Vec{4}(Ψ_axy(0, 0), Ψ_axy(1, 0), Ψ_axy(1, 1), Ψ_axy(0, 1))
    wt.xy[ix, iy] = volfrac(cell, Ψs) / ω
end

# Volume fraction kernel (3D)
@kernel inbounds = true function compute_volfrac_from_level_set!(wt, Ψ, grid::UniformGrid{3}, O)
    I = @index(Global, NTuple)
    ix, iy, iz = I + O
    dx, dy, dz = spacing(grid)
    cell = Rect(Vec(0.0, 0.0, 0.0), Vec(dx, dy, dz))
    ω = GeometryBasics.volume(cell)
    Ψ_ax(dix, diy, diz) = 0.5 * (Ψ[ix+dix, iy+diy, iz+diz] + Ψ[ix+dix+1, iy+diy, iz+diz])
    Ψ_ay(dix, diy, diz) = 0.5 * (Ψ[ix+dix, iy+diy, iz+diz] + Ψ[ix+dix, iy+diy+1, iz+diz])
    Ψ_az(dix, diy, diz) = 0.5 * (Ψ[ix+dix, iy+diy, iz+diz] + Ψ[ix+dix, iy+diy, iz+diz+1])
    Ψ_axy(dix, diy, diz) = 0.25 * (Ψ[ix+dix, iy+diy, iz+diz+1] + Ψ[ix+dix+1, iy+diy, iz+diz+1] + Ψ[ix+dix, iy+diy+1, iz+diz+1] + Ψ[ix+dix+1, iy+diy+1, iz+diz+1])
    Ψ_axz(dix, diy, diz) = 0.25 * (Ψ[ix+dix, iy+diy+1, iz+diz] + Ψ[ix+dix+1, iy+diy+1, iz+diz] + Ψ[ix+dix, iy+diy+1, iz+diz+1] + Ψ[ix+dix+1, iy+diy+1, iz+diz+1])
    Ψ_ayz(dix, diy, diz) = 0.25 * (Ψ[ix+dix+1, iy+diy, iz+diz] + Ψ[ix+dix+1, iy+diy+1, iz+diz] + Ψ[ix+dix+1, iy+diy, iz+diz+1] + Ψ[ix+dix+1, iy+diy+1, iz+diz+1])
    # cell centers
    Ψs = Vec{8}(Ψ[ix, iy, iz], Ψ[ix+1, iy, iz], Ψ[ix, iy+1, iz], Ψ[ix+1, iy+1, iz], Ψ[ix, iy, iz+1], Ψ[ix+1, iy, iz+1], Ψ[ix, iy+1, iz+1], Ψ[ix+1, iy+1, iz+1])
    wt.c[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # x faces
    Ψs = Vec{8}(Ψ_ax(0, 0, 0), Ψ_ax(1, 0, 0), Ψ_ax(0, 1, 0), Ψ_ax(1, 1, 0), Ψ_ax(0, 0, 1), Ψ_ax(1, 0, 1), Ψ_ax(0, 1, 1), Ψ_ax(1, 1, 1))
    wt.x[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # y faces
    Ψs = Vec{8}(Ψ_ay(0, 0, 0), Ψ_ay(1, 0, 0), Ψ_ay(0, 1, 0), Ψ_ay(1, 1, 0), Ψ_ay(0, 0, 1), Ψ_ay(1, 0, 1), Ψ_ay(0, 1, 1), Ψ_ay(1, 1, 1))
    wt.y[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # z faces
    Ψs = Vec{8}(Ψ_az(0, 0, 0), Ψ_az(1, 0, 0), Ψ_az(0, 1, 0), Ψ_az(1, 1, 0), Ψ_az(0, 0, 1), Ψ_az(1, 0, 1), Ψ_az(0, 1, 1), Ψ_az(1, 1, 1))
    wt.z[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # xy edges
    Ψs = Vec{8}(Ψ_axy(0, 0, 0), Ψ_axy(1, 0, 0), Ψ_axy(0, 1, 0), Ψ_axy(1, 1, 0), Ψ_axy(0, 0, 1), Ψ_axy(1, 0, 1), Ψ_axy(0, 1, 1), Ψ_axy(1, 1, 1))
    wt.xy[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # xz edges
    Ψs = Vec{8}(Ψ_axz(0, 0, 0), Ψ_axz(1, 0, 0), Ψ_axz(0, 1, 0), Ψ_axz(1, 1, 0), Ψ_axz(0, 0, 1), Ψ_axz(1, 0, 1), Ψ_axz(0, 1, 1), Ψ_axz(1, 1, 1))
    wt.xz[ix, iy, iz] = volfrac(cell, Ψs) / ω
    # yz edges
    Ψs = Vec{8}(Ψ_ayz(0, 0, 0), Ψ_ayz(1, 0, 0), Ψ_ayz(0, 1, 0), Ψ_ayz(1, 1, 0), Ψ_ayz(0, 0, 1), Ψ_ayz(1, 0, 1), Ψ_ayz(0, 1, 1), Ψ_ayz(1, 1, 1))
    wt.yz[ix, iy, iz] = volfrac(cell, Ψs) / ω
end
