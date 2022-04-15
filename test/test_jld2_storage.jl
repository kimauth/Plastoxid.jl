using JLD2
@testset "jld2 storage" begin
    jld2_path = "test_jld2_storage_temp"
    nsteps_per_file = 3
    max_nfiles = 1000

    # dummy data
    t = rand()
    split_level = 0
    vars = rand(10)
    history_vars = [rand(3) for i = 1:6]

    ## Last file not fully "filled"
    jld2_storage = JLD2Storage(jld2_path, nsteps_per_file, max_nfiles)
    for i = 1:10
        Plastoxid.save_step!(jld2_storage, i, t, split_level, vars, history_vars)
    end
    Plastoxid.close(jld2_storage)

    files = readdir(jld2_path)
    @test length(files) == 4
    @test all(files .== [string("file_000", i, ".jld2") for i=1:4])
    jldopen(joinpath(jld2_path, first(files))) do f
        @test all(keys(f) .== [string("step_", i) for i=1:nsteps_per_file])
    end
    jldopen(joinpath(jld2_path, files[end])) do f
        @test keys(f) == [string("step_", 10)]
    end

    # delete files for this test
    rm(jld2_path; recursive=true)

    ## Last file fully "filled"
    nsteps_per_file = 5
    jld2_storage = JLD2Storage(jld2_path, nsteps_per_file, max_nfiles)
    for i = 1:10
        Plastoxid.save_step!(jld2_storage, i, t, split_level, vars, history_vars)
    end
    Plastoxid.close(jld2_storage)

    @test length(readdir(jld2_path)) == 2

    # delete files
    rm(jld2_path; recursive=true)
end