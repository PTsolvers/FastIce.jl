using CairoMakie

# function visme()
    nx, ny, nz = 126, 126, 126

    τxy = zeros(ny+1, nz)
    τxz = zeros(nx, nz+1)
    # τyz = zeros(nx, nz)

    x = LinRange(-2, 2, nx)
    y = LinRange(-1, 1, ny)
    z = LinRange(0, 2, nz)

    open("14.bin", "r") do io
        read!(io, τxy)
        read!(io, τxz)
    end

    fig = Figure(resolution=(1400, 900), fontsize=32)
    ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="y", ylabel="z")
    plt = heatmap!(ax, y, z, τxz; colormap=:turbo)
    Colorbar(fig[1,1][1,2], plt)

    save("shear_slices.png", fig)

#     return
# end

# visme()
