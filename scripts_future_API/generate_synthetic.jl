using FastIce.Geometries

nx, ny     = 128, 128
lx, ly     = 5.0, 5.0
zmin, zmax = 0.0, 1.0
synth_dem  = make_synthetic(nx, ny, lx, ly, zmin, zmax, :turtle)

@views function visme(synth_dem)
    x = LinRange(synth_dem.domain.xmin, synth_dem.domain.xmax, nx + 1)
    y = LinRange(synth_dem.domain.ymin, synth_dem.domain.ymax, ny + 1)
    bed  = synth_dem.z_bed
    surf = synth_dem.z_surf

    surf2 = copy(surf)
    surf2[surf.<bed] .= NaN

    fig = Figure(; size=(1000, 400), fontsize=22)
    ax1 = Axis3(fig[1, 1]; aspect=(2, 2, 1), azimuth=-π / 8, elevation=π / 5)
    ax2 = Axis(fig[1, 2]; aspect=DataAspect())

    surface!(ax1, x, y, bed; colormap=:turbo)
    surface!(ax1, x, y, surf2; colormap=:turbo)
    plot!(ax2, x, bed[:, ceil(Int, length(y) / 2)])
    plot!(ax2, x, surf2[:, ceil(Int, length(y) / 2)])

    return display(fig)
end

visme(synth_dem)
