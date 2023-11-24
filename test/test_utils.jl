include("common.jl")

using FastIce.Utils

@testset "pipelines" begin
    @testset "pre" begin
        a = 0
        pipe = Pipeline(; pre=() -> a += 1)
        put!(() -> nothing, pipe)
        wait(pipe)
        @test a == 1
        close(pipe)
    end

    @testset "post" begin
        a = 0
        pipe = Pipeline(; post=() -> a += 2)
        put!(pipe) do
            a -= 1
        end
        wait(pipe)
        close(pipe)
        @test a == 1
    end

    @testset "do work" begin
        a = 0
        pipe = Pipeline()
        put!(pipe) do
            a += 1
        end
        wait(pipe)
        close(pipe)
        @test a == 1
    end

    @testset "not running" begin
        pipe = Pipeline()
        close(pipe)
        @test_throws ErrorException wait(pipe)
    end
end
