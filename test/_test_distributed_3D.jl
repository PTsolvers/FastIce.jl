include("common.jl")

using MPI
using FastIce.Distributed

MPI.Init()

nprocs = MPI.Comm_size(MPI.COMM_WORLD)

backends = [CPU()] # until we have testing environment setup for GPU-aware MPI, run only on CPU

for backend in backends
    @testset "$(basename(@__FILE__)) (backend: $backend)" begin
        mpi_dims = parse.(Int, (get(ENV, "DIMX", "1"),
                                get(ENV, "DIMY", "1"),
                                get(ENV, "DIMZ", "1")))
        dims = (0, 0, 0)
        topo = CartesianTopology(dims)
        local_size = (4, 5, 6)
        @testset "Topology" begin
            @test dimensions(topo) == mpi_dims
            @test length(neighbors(topo)) == 3
            @test node_size(topo) == nprocs
            if global_rank(topo) == 0
                @test neighbor(topo, 1, 1) == MPI.PROC_NULL
                @test neighbor(topo, 2, 1) == MPI.PROC_NULL
                @test neighbor(topo, 3, 1) == MPI.PROC_NULL
                @test has_neighbor(topo, 1, 1) == false
                @test has_neighbor(topo, 2, 1) == false
                @test has_neighbor(topo, 3, 1) == false
                if mpi_dims[2] > 1 && mpi_dims[3] > 1
                    @test neighbor(topo, 1, 2) == mpi_dims[2] * mpi_dims[3]
                    @test neighbor(topo, 2, 2) == mpi_dims[3]
                    @test has_neighbor(topo, 1, 2) == true
                    @test has_neighbor(topo, 2, 2) == true
                else
                    @test neighbor(topo, 1, 2) == MPI.PROC_NULL
                    @test neighbor(topo, 2, 2) == MPI.PROC_NULL
                    @test has_neighbor(topo, 1, 2) == false
                    @test has_neighbor(topo, 2, 2) == false
                end
            end
            @test global_grid_size(topo, local_size) == mpi_dims .* local_size
        end
        @testset "gather!" begin
            src = fill(global_rank(topo) + 1, local_size)
            dst = (global_rank(topo) == 0) ? zeros(Int, mpi_dims .* local_size) : nothing
            gather!(dst, src, cartesian_communicator(topo))
            ranks_mat = permutedims(reshape(1:global_size(topo), dimensions(topo)), reverse(1:3))
            if global_rank(topo) == 0
                @test dst == repeat(ranks_mat; inner=size(src))
            end
        end
    end
end

MPI.Finalize()
