include("common.jl")

using Chmy.Architectures
using Chmy.Distributed
using Chmy.Fields
using Chmy.Grids

using FastIce.Writers

using MPI
using HDF5
using LightXML

MPI.Init()

backends = [CPU()] # until we have testing environment setup for GPU-aware MPI, run only on CPU

XML_ref = """
<?xml version="1.0" encoding="utf-8"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid GridType="Collection" CollectionType="Temporal">
      <Grid GridType="Uniform">
        <Topology TopologyType="3DCoRectMesh" Dimensions="13 11 9"/>
        <Time Value="0.0"/>
        <Geometry GeometryType="ORIGIN_DXDYDZ">
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.049999999999999996 -0.445 -0.3375</DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.09999999999999999 0.11000000000000001 0.125</DataItem>
        </Geometry>
        <Attribute Name="Fa" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="12 10 8">test_d.h5:/Fa</DataItem>
        </Attribute>
        <Attribute Name="Fb" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="12 10 8">test_d.h5:/Fb</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""

dims_g = (0, 0, 0)
size_l = (4, 5, 6)

h5_fname = "test_d.h5"
xdmf_fname = "test_d.xdmf3"

for backend in backends

    arch = Arch(backend, MPI.COMM_WORLD, dims_g)
    topo = topology(arch)
    me = global_rank(topo) # rank
    mpi_dims = dims(topo)
    comm = cart_comm(topo)

    size_g = size_l .* dims(topo)
    grid = UniformGrid(arch; origin=(-0.4, -0.5, 0.0), extent=(1.0, 1.1, 1.2), dims=size_g)

    Fa_g = (me == 0) ? zeros(Float64, size_g) : nothing
    Fb_g = (me == 0) ? zeros(Float64, size_g) : nothing

    @testset "$(basename(@__FILE__)) (backend: $backend)" begin

        HDF5.has_parallel() || (@warn("HDF5 has no parallel support. Skipping $(basename(@__FILE__)) (backend: $backend)."); return)

        Fa_l = Field(backend, grid, Center())
        Fb_l = Field(backend, grid, Center())

        fill!(parent(Fa_l), 1.0 * me)
        fill!(parent(Fb_l), 2.0 * me)

        fields = Dict("Fa" => Fa_l, "Fb" => Fb_l)

        @testset "Distributed writers 3D" begin
            @testset "write h5" begin
                (me == 0) && (isfile(h5_fname) && run(`rm $h5_fname`))
                write_h5(arch, grid, h5_fname, fields)
                MPI.Barrier(comm)

                gather!(arch, Fa_g, Fa_l)
                gather!(arch, Fb_g, Fb_l)
                MPI.Barrier(comm)

                if me == 0
                    @test all(Fa_g .== h5read(h5_fname, "Fa"))
                    @test all(Fb_g .== h5read(h5_fname, "Fb"))
                    isfile(h5_fname) && run(`rm $h5_fname`)
                end
            end
            MPI.Barrier(comm)

            @testset "write/read xdmf3" begin
                if me == 0
                    isfile(xdmf_fname) && run(`rm $xdmf_fname`)
                    write_xdmf(arch, grid, xdmf_fname, fields, (h5_fname, ))

                    @test XML_ref == string(parse_file(xdmf_fname))
                    isfile(xdmf_fname) && run(`rm $xdmf_fname`)
                end
            end
        end
    end
end

MPI.Finalize()
