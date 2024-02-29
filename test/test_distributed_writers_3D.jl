include("common.jl")

using FastIce.Architectures
using FastIce.Distributed
using FastIce.Fields
using FastIce.Grids
using FastIce.Writers

using MPI
using HDF5
using LightXML

MPI.Init()

backends = [CPU()] # until we have testing environment setup for GPU-aware MPI, run only on CPU

dims = (0, 0, 0)
topo = CartesianTopology(dims)
mpi_dims = dimensions(topo)
me = global_rank(topo) # rank
comm = cartesian_communicator(topo)

(me == 0) && (XML_ref = """
<?xml version="1.0" encoding="utf-8"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid GridType="Collection" CollectionType="Temporal">
      <Grid GridType="Uniform">
        <Topology TopologyType="3DCoRectMesh" Dimensions="13 11 9"/>
        <Time Value="0.0"/>
        <Geometry GeometryType="ORIGIN_DXDYDZ">
          <DataItem Format="XML" NumberType="Float" Dimensions="3 ">0.0 -0.5 -0.4</DataItem>
          <DataItem Format="XML" NumberType="Float" Dimensions="3">0.09999999999999999 0.11000000000000001 0.125</DataItem>
        </Geometry>
        <Attribute Name="Fa" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="12 10 8">t:/Fa</DataItem>
        </Attribute>
        <Attribute Name="Fb" Center="Cell">
          <DataItem Format="HDF" NumberType="Float" Precision="8" Dimensions="12 10 8">t:/Fb</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
""")

size_l = (4, 5, 6)
size_g = global_grid_size(topo, size_l)

Fa_g = (me == 0) ? zeros(Float64, mpi_dims .* size_l) : nothing
Fb_g = (me == 0) ? zeros(Float64, mpi_dims .* size_l) : nothing

grid_g = CartesianGrid(; origin=(-0.4, -0.5, 0.0),
                       extent=(1.0, 1.1, 1.2),
                       size=size_g)

grid_l = local_grid(grid_g, topo)

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin

        HDF5.has_parallel() || (@warn("HDF5 has no parallel support. Skipping $(basename(@__FILE__)) (backend: $backend)."); return)

        arch = Architecture(backend, topo)
        set_device!(arch)

        Fa_l = Field(backend, grid_l, Center())
        Fb_l = Field(backend, grid_l, Center())

        fill!(parent(Fa_l), 1.0 * me)
        fill!(parent(Fb_l), 2.0 * me)

        fields = Dict("Fa" => Fa_l, "Fb" => Fb_l)

        @testset "Distributed writers 3D" begin
            @testset "write h5" begin
                fname = "test_d.h5"
                (me == 0) && (isfile(fname) && run(`rm $fname`))
                write_h5(arch, grid_g, fname, fields)
            end
            @testset "read h5" begin
                fname = "test_d.h5"
                Fa_v = zeros(size(grid_l))
                Fb_v = zeros(size(grid_l))
                copyto!(Fa_v, interior(Fa_l))
                copyto!(Fb_v, interior(Fb_l))
                KernelAbstractions.synchronize(backend)
                gather!(Fa_g, Fa_v, comm)
                gather!(Fb_g, Fb_v, comm)
                if me == 0
                    @test all(Fa_g .== h5read(fname, "Fa"))
                    @test all(Fb_g .== h5read(fname, "Fb"))
                    isfile(fname) && run(`rm $fname`)
                end
            end
            @testset "write xdmf3" begin
                if me == 0
                    fname = "test_d.xdmf3"
                    isfile(fname) && run(`rm $fname`)
                    write_xdmf(arch, grid_g, fname, fields, "test_d.h5")
                end
            end
            @testset "read xdmf3" begin
                if me == 0
                    fname = "test_d.xdmf3"
                    @test XML_ref == string(parse_file(fname))
                    isfile(fname) && run(`rm $fname`)
                end
            end
        end
    end
end

MPI.Finalize()
