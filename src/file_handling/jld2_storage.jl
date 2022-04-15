
"""
Result storage specification:

For each time step, the following data is stored:
- time point `t`
- field values `vars`
- material history variables `history_vars`
- the number of times the initial time step has been split in half `split_level`; `split_level=0` means the step wasn't split, `split_level=1` means the step was halfed once etc.

Consequently, time derivative independent data can be reconstructed from the results of a
time step alone, time derivative dependent data can be reconstructed from the results of the
current + the previous time step.

In order to reduce the number of result files (motivated by the file limit of the cluster),
time steps are grouped together for storage. The results are stored in the directory
`jld2_path`, files are called `file_0001.jld2`, `file_0002.jld2` etc.; the 0-padding depends
on the maximum possible number of time steps including splitting.

Within each jld2 file, each time step is represented by a group called `step_n` (where n is 
the time step index). The steps in each file can easily be retrieved by `keys(file)`.
The only top-level keys a file has are the time step keys (`step_n`).

The number of time steps in each file is determined by `nsteps_per_file`. For efficient
post-processing, the number of files should be within a few times the number of threads used
for post-processing.
"""

mutable struct JLD2Storage{F}
    jld2_path::String
    nsteps_per_file::Int
    formatter_function::F # with how many zeros to pad file names
    # mutable states
    nsteps_in_current_file::Int
    current_file_idx::Int
    current_file::JLD2.JLDFile{JLD2.MmapIO}
    is_open::Bool
end

function JLD2Storage(jld2_path::AbstractString, nsteps_per_file::Int, max_nfiles::Int)
    check_path(jld2_path)
    mkdir(jld2_path)
    formatter_function = generate_formatter("%0$(ndigits(max_nfiles))d")
    nsteps_in_current_file = 1
    current_file_idx = 1
    filename = joinpath(jld2_path, string("file_", formatter_function(1), ".jld2"))
    current_file = jldopen(filename, "w")
    is_open = true

    jld2_storage = JLD2Storage(
        jld2_path,
        nsteps_per_file,
        formatter_function,
        nsteps_in_current_file,
        current_file_idx,
        current_file,
        is_open,
    )
    return jld2_storage
end

function step!(jld2_storage::JLD2Storage)

    jld2_storage.nsteps_in_current_file += 1

    if jld2_storage.nsteps_in_current_file > jld2_storage.nsteps_per_file
        jld2_storage.nsteps_in_current_file = 1
        jld2_storage.current_file_idx += 1
        close(jld2_storage.current_file)
        jld2_storage.is_open = false
        number_str = jld2_storage.formatter_function(jld2_storage.current_file_idx)
        filename = joinpath(jld2_storage.jld2_path, string("file_", number_str, ".jld2"))
        jld2_storage.current_file = jldopen(filename, "w")
        jld2_storage.is_open = true
    end
    return jld2_storage
end

function Base.close(jld2_storage::JLD2Storage)
    empty_file = isempty(jld2_storage.current_file) # check if file has any data
    close(jld2_storage.current_file)
    jld2_storage.is_open = false
    empty_file && rm(jld2_storage.current_file.path) # delete file if it has no data
    return jld2_storage.jld2_path
    # TODO: tar + compress jld2 folder, delete uncompressed files?
end

function save_step!(jld2_storage, step, t, split_level, vars, history_vars)
    file = jld2_storage.current_file

    group = JLD2.Group(file, string("step_", step))
    group["t"] = t
    group["split_level"] = split_level
    group["vars"] = vars
    group["history_vars"] = history_vars

    step!(jld2_storage)
end

function tar_compress(jld2_storage::JLD2Storage)
    jld2_path = jld2_storage.jld2_path

    tar_name = string(basename(jld2_path), ".tar.zst")
    tar_path = joinpath(dirname(jld2_path), tar_name)
    tar_file = open(tar_path, write=true)
    tar = ZstdCompressorStream(tar_file)
    Tar.create(jld2_path, tar)
    close(tar)
    rm(jld2_path; recursive = true) # remove jld2 folder
end
