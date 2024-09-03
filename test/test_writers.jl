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
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.09999999999999999 -0.39 -0.275</DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.19999999999999998 0.22000000000000003 0.25</DataItem>
        </Geometry>
        <Attribute Name="Fa" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="6 5 4">test.h5:/Fa</DataItem>
        </Attribute>
        <Attribute Name="Fb" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="6 5 4">test.h5:/Fb</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

h5_fname = "test.h5"
xdmf_fname = "test.xdmf3"

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        arch = Arch(backend)
        grid = UniformGrid(arch; origin=(-0.4, -0.5, 0.0), extent=(1.0, 1.1, 1.2), dims=(4, 5, 6))

        Fa = Field(backend, grid, Center())
        Fb = Field(backend, grid, Center())

        fill!(parent(Fa), 1.0)
        fill!(parent(Fb), 2.0)

        fields = Dict("Fa" => Fa, "Fb" => Fb)

        @testset "writers" begin
            @testset "write_dset" begin
                isfile(h5_fname) && run(`rm $h5_fname`)
                I = CartesianIndices(size(grid, Center()))
                h5open(h5_fname, "w") do io
                    FastIce.Writers.write_dset(io, fields, size(grid, Center()), I.indices)
                end
                @test all(Array(interior(Fa)) .== h5read(h5_fname, "Fa"))
                @test all(Array(interior(Fb)) .== h5read(h5_fname, "Fb"))
                isfile(h5_fname) && run(`rm $h5_fname`)
            end
            @testset "write_h5" begin
                isfile(h5_fname) && run(`rm $h5_fname`)
                write_h5(arch, grid, h5_fname, fields)
                @test all(Array(interior(Fa)) .== h5read(h5_fname, "Fa"))
                @test all(Array(interior(Fb)) .== h5read(h5_fname, "Fb"))
                isfile(h5_fname) && run(`rm $h5_fname`)
            end
            @testset "write_xdmf3" begin
                isfile(xdmf_fname) && run(`rm $xdmf_fname`)
                write_xdmf(arch, grid, xdmf_fname, fields, (h5_fname,))
                @test XML_ref == string(parse_file(xdmf_fname))
                isfile(xdmf_fname) && run(`rm $xdmf_fname`)
            end
        end
    end
end
