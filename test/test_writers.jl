include("common.jl")

using Chmy.Architectures
using Chmy.Fields
using Chmy.Grids
using FastIce.Writers

using HDF5
using LightXML

XML_ref = """
<?xml version="1.0" encoding="utf-8"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid GridType="Collection" CollectionType="Temporal">
      <Grid GridType="Uniform">
        <Topology TopologyType="3DCoRectMesh" Dimensions="7 6 5"/>
        <Time Value="0.0"/>
        <Geometry GeometryType="ORIGIN_DXDYDZ">
          <DataItem Format="XML" NumberType="Float" Dimensions="3 ">0.0 -0.5 -0.4</DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.19999999999999998 0.22000000000000003 0.25</DataItem>
        </Geometry>
        <Attribute Name="Fa" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="6 5 4">t:/Fa</DataItem>
        </Attribute>
        <Attribute Name="Fb" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="6 5 4">t:/Fb</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

grid = CartesianGrid(; origin=(-0.4, -0.5, 0.0),
                     extent=(1.0, 1.1, 1.2),
                     size=(4, 5, 6))

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        arch = Arch(backend)

        Fa = Field(backend, grid, Center())
        Fb = Field(backend, grid, Center())

        fill!(parent(Fa), 1.0)
        fill!(parent(Fb), 2.0)

        fields = Dict("Fa" => Fa, "Fb" => Fb)

        @testset "writers" begin
            @testset "write_dset" begin
                fname = "test.h5"
                isfile(fname) && run(`rm $fname`)
                I = CartesianIndices(size(grid))
                h5open(fname, "w") do io
                    FastIce.Writers.write_dset(io, fields, size(grid), I.indices)
                end
                @test all(Array(interior(Fa)) .== h5read(fname, "Fa"))
                @test all(Array(interior(Fb)) .== h5read(fname, "Fb"))
                isfile(fname) && run(`rm $fname`)
            end
            @testset "write_h5" begin
                fname = "test2.h5"
                isfile(fname) && run(`rm $fname`)
                write_h5(arch, grid, fname, fields)
                @test all(Array(interior(Fa)) .== h5read(fname, "Fa"))
                @test all(Array(interior(Fb)) .== h5read(fname, "Fb"))
                isfile(fname) && run(`rm $fname`)
            end
            @testset "write_xdmf3" begin
                fname = "test.xdmf3"
                isfile(fname) && run(`rm $fname`)
                write_xdmf(arch, grid, fname, fields, "test.h5")
                @test XML_ref == string(parse_file(fname))
                isfile(fname) && run(`rm $fname`)
            end
        end
    end
end
