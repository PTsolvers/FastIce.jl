using GLMakie

const ω = 1π

# manufactured solution for the confined Stokes flow with free-surface boundaries
# velocity
# manufactured solution for the confined Stokes flow with free-slip boundaries
# helper functions
f(ξ, η) = cos(π * ξ) * (η^2 - 1)^2 -
          cos(π * η) * (ξ^2 - 1)^2
g(ξ, η) = sin(π * η) * (ξ^2 - 1) * ξ -
          sin(π * ξ) * (η^2 - 1) * η
p(ξ, η) = cos(π * η) * (1 - 3 * ξ^2) * 2 -
          cos(π * ξ) * (1 - 3 * η^2) * 2
# velocity
vx(x, y, z) = sin(π * x) * f(y, z)
vy(x, y, z) = sin(π * y) * f(z, x)
vz(x, y, z) = sin(π * z) * f(x, y)
# diagonal deviatoric stress
τxx(x, y, z, η) = 2 * η * π * cos(π * x) * f(y, z)
τyy(x, y, z, η) = 2 * η * π * cos(π * y) * f(z, x)
τzz(x, y, z, η) = 2 * η * π * cos(π * z) * f(x, y)
# off-diagonal deviatoric stress
τxy(x, y, z, η) = 4 * η * cos(π * z) * g(x, y)
τxz(x, y, z, η) = 4 * η * cos(π * y) * g(z, x)
τyz(x, y, z, η) = 4 * η * cos(π * x) * g(y, z)
# forcing terms
ρgx(x, y, z, η) = -2 * η * sin(π * x) * (f(y, z) * π^2 - p(y, z))
ρgy(x, y, z, η) = -2 * η * sin(π * y) * (f(z, x) * π^2 - p(z, x))
ρgz(x, y, z, η) = -2 * η * sin(π * z) * (f(x, y) * π^2 - p(x, y))

function visme()
    xs = LinRange(-1, 1, 201)
    ys = LinRange(-1, 1, 201)
    zs = LinRange(-1, 1, 201)

    Vxm = [vx(x, y, z) for x in xs, y in ys, z in zs]
    Vym = [vy(x, y, z) for x in xs, y in ys, z in zs]
    Vzm = [vz(x, y, z) for x in xs, y in ys, z in zs]

    Vmag = sqrt.(Vxm .^ 2 .+ Vym .^ 2 .+ Vzm .^ 2)
    Vmag_max = maximum(Vmag)
    Vxm .*= 0.2 / Vmag_max
    Vym .*= 0.2 / Vmag_max
    Vzm .*= 0.2 / Vmag_max

    η0 = 1.0

    τxxm = [τxx(x, y, z, η0) for x in xs, y in ys, z in zs]
    τyym = [τyy(x, y, z, η0) for x in xs, y in ys, z in zs]
    τzzm = [τzz(x, y, z, η0) for x in xs, y in ys, z in zs]
    τxym = [τxy(x, y, z, η0) for x in xs, y in ys, z in zs]
    τxzm = [τxz(x, y, z, η0) for x in xs, y in ys, z in zs]
    τyzm = [τyz(x, y, z, η0) for x in xs, y in ys, z in zs]

    τII = sqrt.(0.5 .* (τxxm .^ 2 .+ τyym .^ 2 .+ τzzm .^ 2) .+ τxym .^ 2 .+ τxzm .^ 2 .+ τyzm .^ 2)

    fig = Figure(; size=(500, 450))
    ax  = Axis3(fig[1, 1][1, 1]; aspect=:data, xlabel="x", ylabel="y", zlabel="", title="free slip 3D")
    limits!(ax, -1, 1, -1, 1, -1, 1)
    # hm = heatmap!(ax, xs, ys, τII; colormap=:roma)
    vl = volume!(ax, xs, ys, zs, τII; algorithm=:absorption, colormap=:roma)
    Colorbar(fig[1, 1][1, 2], vl)
    st = 20
    pt = vec([Point3(x, y, z) for x in xs[1:st:end], y in ys[1:st:end], z in zs[1:st:end]])
    dr = Vec3.(vec(Vxm[1:st:end, 1:st:end, 1:st:end]),
               vec(Vym[1:st:end, 1:st:end, 1:st:end]),
               vec(Vzm[1:st:end, 1:st:end, 1:st:end]))
    # arrows!(ax, pt, dr; color=:grey, arrowsize=0.05)

    display(fig)

    # save("free_surface_sol.png", fig)
    # save("free_slip_sol.png", fig)
    save("free_slip_sol_3D.png", fig)
    return
end

visme()
