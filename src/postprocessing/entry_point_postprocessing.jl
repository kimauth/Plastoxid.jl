
function get_last_step(result_file)
    last_step = 0
    jldopen(result_file) do f
        last_step_str = last(keys(f))
        m = match(r"\d+", last_step_str)
        last_step = parse(Int, m.match)
    end
    return last_step
end

function previous_data(filenames)
    outdata = Tuple[]

    for (i, filename) in enumerate(filenames)
        if i==1
            jldopen(filename) do f
                t_old = f[first(keys(f))]["t"]
                old_states = f[first(keys(f))]["history_vars"]
                push!(outdata, (t_old, old_states))
            end
        else
            jldopen(filenames[i-1]) do f
                t_old = f[last(keys(f))]["t"]
                old_states = f[last(keys(f))]["history_vars"]
                push!(outdata, (t_old, old_states))
            end
        end
    end
    return outdata
end

function Plastoxid.postprocess!(
    func::Function,
    results,
    # arguments
    result_file,
    fesets,
    dh;
    steps=nothing, 
)

    decompress = false
    if isfile(result_file) # unpack result file
        decompress = true
        tar_gz = open(result_file)
        tar = ZstdDecompressorStream(tar_gz)
        dir = Tar.extract(tar)
        close(tar)
    elseif isdir(result_file)
        dir = result_file
    else
        error("$result_file is neither a file nor a directory.")
    end

    files = readdir(dir; join=true)

    #######################################################################################
    last_stepids = get_last_step.(files)
    # find the step-range that each file covers
    if isnothing(steps)
        timesteps = 0:last(last_stepids)
    else
        timesteps = steps
    end
    file_step_ranges = [intersect(timesteps, (get(last_stepids, i-1, -1)+1):last_stepids[i]) for i in eachindex(last_stepids)]
    prev_data = previous_data(files)

    file_data = tuple.(files, file_step_ranges, prev_data)

    ch_data = Channel{typeof(file_data[1])}(length(file_data))
    for fd in file_data
        put!(ch_data, fd)
    end
    #######################################################################################

    # allocate buffers for each thread
    ch_buffers = Channel(Threads.nthreads())
    for _ = 1:Threads.nthreads()
        qp_data = [Plastoxid.allocate_matrices(feset) for feset in fesets]
        states_per_feset = [similar.(feset.states.material_states) for feset in fesets]
        old_states_per_feset = [similar.(feset.states.material_states) for feset in fesets]
        put!(ch_buffers, (qp_data, states_per_feset, old_states_per_feset))
    end

    # resize result array to right number of steps if needed
    nsteps = last(last_stepids)
    if length(results) != nsteps + 1
        resize!(results, nsteps + 1)
    end

    for _ in files
        qp_data, states_per_feset, old_states_per_feset = take!(ch_buffers)
        file, step_range, prev_data = take!(ch_data)

        jldopen(file) do f
            for stepid in step_range
                #############################
                # read data
                group = f[string("step_", stepid)]
                t = group["t"]
                vars = group["vars"]
                states = group["history_vars"]
                if stepid-1 < first(step_range) # previous step not in same file
                    t_old, old_states = prev_data
                else 
                    old_group = f[string("step_", stepid-1)]
                    t_old = old_group["t"]
                    old_states = old_group["history_vars"]
                end
                Δt = t - t_old
                ###############################
                # now postprocess
                Plastoxid.postprocess!(
                    qp_data,
                    states_per_feset,
                    old_states_per_feset,
                    fesets,
                    vars,
                    Δt,
                    states,
                    old_states,
                    dh,
                )
                results[stepid+1] = func(vars, qp_data, fesets, dh, t)
            end
        end
        put!(ch_buffers, (qp_data, states_per_feset, old_states_per_feset))
    end             
    if decompress
        rm(dir; recursive=true)
    end
    return results
end    

# convenience wrapper, but uses Vector{Any}
function Plastoxid.postprocess!(
    func::Function,
    # arguments
    result_file,
    fesets,
    dh;
    steps=nothing, 
) 
    results = []
    postprocess!(func, results, result_file, fesets, dh; steps)
end