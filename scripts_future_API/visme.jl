using CairoMakie

# function visme()
    nx, ny, nz = 254*4, 254*2, 254*2

    Pr = zeros(nx, ny, nz)
    τxx = zeros(nx, ny, nz)
    τyy = zeros(nx, ny, nz)
    τzz = zeros(nx, ny, nz)
    τxy = zeros(nx, ny, nz)
    τxz = zeros(nx, ny, nz)
    τyz = zeros(nx, ny, nz)
    Vx = zeros(nx, ny, nz)
    Vy = zeros(nx, ny, nz)
    Vz = zeros(nx, ny, nz)

    x = LinRange(-2, 2, nx)
    y = LinRange(-1, 1, ny)
    z = LinRange(0, 2, nz)

    open("data.bin", "r") do io
        read!(io, Pr)
        read!(io, τxx)
        read!(io, τyy)
        read!(io, τzz)
        read!(io, τxy)
        read!(io, τxz)
        read!(io, τyz)
        read!(io, Vx)
        read!(io, Vy)
        read!(io, Vz)
    end

    fig = Figure(resolution=(1400, 900), fontsize=32)
    ax  = Axis3(fig[1,1][1,1]; aspect=:data, xlabel="x", ylabel="y", zlabel="z", title=L"v_x")
    limits!(-2, 2,-1, 1, 0, 2)
    # ax  = Axis(fig[1,1][1,1]; aspect=DataAspect(), xlabel="x", ylabel="z", title=L"v_x")
    plt = volumeslices!(ax, x, y, z, Vx; colormap=:turbo)
    # plt = heatmap!(ax, x, z, @view(Vx[:, ny÷2, :]))
    Colorbar(fig[1,1][1,2], plt)

    plt[:update_yz][](length(x))
    plt[:update_xz][](length(y)÷2)
    plt[:update_xy][](1)

    save("slices.png", fig)

#     return
# end

# visme()
