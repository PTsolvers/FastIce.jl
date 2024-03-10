function generate_turtle(nx, ny, lx, ly, zmin, zmax)
    @info "expects lx, ly, lz = 5.0, 5.0, 1.0"
    # Numerics
    amp  = 0.1
    ω    = 10π
    tanβ = tan(-π/6)
    el   = 0.35
    gl   = 0.9
    x    = LinRange(-lx / 2, lx / 2, nx + 1)
    y    = LinRange(-lx / 2, ly / 2, ny + 1)
    # Functions
    generate_bed(x, y) = amp * sin(ω * x) * sin(ω * y) + tanβ * x + el + (1.5 * y)^2
    generate_surf(x, y) = 1.0 - ((1.7 * x + 0.22)^2 + (0.5 * y)^2)
    # Generate
    bed  = [(zmax - zmin) * generate_bed(x / lx, y / ly) for x in x, y in y]
    surf = [gl * generate_surf(x / lx, y / ly) for x in x, y in y]
    return bed, surf
end

function generate_dome(nx, ny, lx, ly)
    # Numerics
    h_0 = 0.8
    x   = LinRange(-lx / 2, lx / 2, nx + 1)
    y   = LinRange(-lx / 2, ly / 2, ny + 1)
    # Functions
    generate_bed(x, y)  = 0.0
    generate_surf(x, y) = h_0 - (x^2 + y^2)
    # Compute
    bed  = [generate_bed(_x, _y) for _x in x, _y in y]
    surf = [generate_surf(_x, _y) for _x in x, _y in y]
    surf[surf.<bed] .= 0.0
    return bed, surf
end

function generate_valley(nx, ny, lx, ly, ox, oy, Δz, rgl, ogx, ogy, ogz)
    x   = LinRange(ox, ox + lx, nx + 1)
    y   = LinRange(ox, oy + ly, ny + 1)
    bed = zeros(nx + 1, ny + 1)
    for iy in axes(bed, 2), ix in axes(bed, 1)
        ωx = 2π * 10 * (x[ix] - ox) / lx
        ωy = 2π * 10 * (y[iy] - oy) / lx
        bed[ix, iy] = 0.025 * rgl * sin(ωx) * cos(ωy)
    end
    bed .= bed .- minimum(bed) .+ Δz
    surf = zeros(nx + 1, ny + 1)
    for iy in axes(surf, 2), ix in axes(surf, 1)
        δx = x[ix] - ogx
        δy = y[iy] - ogy
        surf[ix, iy] = sqrt(max(rgl^2 - δx^2 - δy^2, 0.0)) + ogz
    end
    return bed, surf
end

"""
    make_synthetic(nx, ny, lx, ly, zmin, zmax, type::Symbol)

Create a synthetic elevation model based on the specified type.

# Arguments
- `nx`: Number of grid points in the x-direction.
- `ny`: Number of grid points in the y-direction.
- `lx`: Length of the domain in the x-direction.
- `ly`: Length of the domain in the y-direction.
- `zmin`: Minimum elevation.
- `zmax`: Maximum elevation.
- `type`: Type of synthetic model to generate. Options are `:dome`, `:turtle`, and `:valley`.

# Notes
- For `:valley`, the function is not yet fully implemented and will issue a warning.

"""
function make_synthetic(nx, ny, lx, ly, zmin, zmax, type::Symbol)
    if type == :dome
        bed, surf = generate_dome(nx, ny, lx, ly)
    elseif type == :turtle
        bed, surf = generate_turtle(nx, ny, lx, ly, zmin, zmax)
    elseif type == :valley
        @warn "not yet doing the right thing"
        lx, ly, lz = 5.0, 2.5, 1.0
        ox, oy = -lx / 2, -ly / 2
        Δz = 0.1lz
        rgl = 4lz
        ogx, ogy, ogz = 0.0lx, 0.0ly, -3.2lz
        bed, surf = generate_valley(nx, ny, lx, ly, ox, oy, Δz, rgl, ogx, ogy, ogz)
    else
        error("Synthetic dem type not defined")
    end
    return SyntheticElevation(lx, ly, zmin, zmax, bed, surf)
end
