include("common.jl")

using MPI
using FastIce.Distributed

MPI.Init()

nprocs = MPI.Comm_size(MPI.COMM_WORLD)

@testset "Distributed" begin
    backend = CPU() # until we have testing environment setup for GPU-aware MPI, run only on CPU
    @testset "2D" begin
        mpi_dims = parse.(Int, (get(ENV, "DIMX", "1"),
                                get(ENV, "DIMY", "1")))
        dims = (0, 0)
        topo = CartesianTopology(dims)
        local_size = (4, 5)
        @testset "Topology" begin
            @test dimensions(topo) == mpi_dims
            @test length(neighbors(topo)) == 2
            @test node_size(topo) == nprocs
            if global_rank(topo) == 0
                @test neighbor(topo, 1, 1) == MPI.PROC_NULL
                @test neighbor(topo, 2, 1) == MPI.PROC_NULL
                @test neighbor(topo, 1, 2) == mpi_dims[2]
                @test has_heighbor(topo, 1, 1) == false
                @test has_heighbor(topo, 2, 1) == false
                @test has_heighbor(topo, 1, 2) == true
            end
            @test global_grid_size(topo, local_size) == mpi_dims .* local_size
        end
        @testset "gather!" begin
            src = fill!(me + 1, global_rank(topo))
            dst = (global_rank(topo) == 0) ? zeros(Int, mpi_dims .* local_size) : nothing
            gather!(dst, src, cartesian_communicator(topo))
            @test dst == repeat(reshape(1:global_size(topo), dimensions(topo))'; inner=size(src))
        end
    end
    @testset "3D" begin
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
                @test neighbor(topo, 1, 2) == mpi_dims[2] * mpi_dims[3]
                @test neighbor(topo, 2, 2) == mpi_dims[3]
                @test has_heighbor(topo, 1, 1) == false
                @test has_heighbor(topo, 2, 1) == false
                @test has_heighbor(topo, 3, 1) == false
                @test has_heighbor(topo, 1, 2) == true
            end
            @test global_grid_size(topo, local_size) == mpi_dims .* local_size
        end
        @testset "gather!" begin
            src = fill!(me + 1, global_rank(topo))
            dst = (global_rank(topo) == 0) ? zeros(Int, mpi_dims .* local_size) : nothing
            gather!(dst, src, cartesian_communicator(topo))
            ranks_mat = permutedims(reshape(1:global_size(topo), dimensions(topo)), reverse(1:3))
            @test dst == repeat(ranks_mat; inner=size(src))
        end
    end
end

MPI.Finalize()
