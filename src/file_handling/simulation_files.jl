struct SimulationFiles{F}
    # input
    mesh_file::String
    crystal_orientations_file::Union{Nothing, String}
    # output files
    simulation_setup_file::String # file name
    convergence_file::String # file name
    logging_file::String # file name
    # streams
    convergence::IOStream
    outstreams::IOStream
    # sophisticated jld2 result files
    jld2_storage::JLD2Storage{F}

    # restore original streams, perhaps doesn't belong here 
    orig_stdout
    orig_stderr
end

function SimulationFiles(
    mesh_file,
    crystal_orientations_file,
    simulation_setup_file,
    convergence_file,
    outstreams_file,
    jld2_storage,
)

    # make sure input files exist
    isfile(mesh_file) || error("Mesh file $mesh_file does not exist.")
    if !isnothing(crystal_orientations_file)
        isfile(crystal_orientations_file) || error("Crystal orientations file $crystal_orientations_file does not exist.")
    end

    # make sure nothing will be overwritten by new results
    result_files = [simulation_setup_file, convergence_file, outstreams_file]
    for file in result_files
        check_path(file)
    end

    # open streams
    convergence = open(convergence_file, "w")
    outstreams = open(outstreams_file, "w")

    files = SimulationFiles(
        mesh_file,
        crystal_orientations_file,
        simulation_setup_file,
        convergence_file,
        outstreams_file,
        convergence,
        outstreams,
        jld2_storage,
        stdout,
        stderr,
    )

    return files
end

"""
    check_path(path::String)

Make sure `path` is free for result writing. If `path` exists, it is renamed by 
`Dr.Watson.recursively_clear_path`. Ensures that `dirname(path)` exists.
"""
function check_path(path::String)
    if isfile(path) 
        DrWatson.recursively_clear_path(dirname(path))
    elseif isdir(path)
        DrWatson.recursively_clear_path(path)
    end
    !ispath(dirname(path)) && mkpath(dirname(path))
    return path
end

function Base.close(files::SimulationFiles; compress_jld2=true)
    # close open files
    close(files.convergence)
    close(files.outstreams)
    close(files.jld2_storage)
    # compress + tar jld2 files
    compress_jld2 && tar_compress(files.jld2_storage)
    return files
end

# do not close the file that captures stderr when simulation throws an error
# (want to capture the error in files.outstreams)
function close_on_error(files::SimulationFiles; compress_jld2=true)
    # close open files
    close(files.convergence)
    close(files.jld2_storage)
    # compress + tar jld2 files
    compress_jld2 && tar_compress(files.jld2_storage)
    return files
end

# in case Julia is still running after the simulation threw an error
function close_outstreams(files::SimulationFiles)
    redirect_stdio(;stdout=files.orig_stdout, stderr=files.orig_stderr)
    close(files.outstreams)
    return files.outstreams
end