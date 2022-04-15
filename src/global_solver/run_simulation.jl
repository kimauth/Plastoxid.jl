function run_simulation(
    vars::AbstractMatrix{Float64},
    fe_sets::Vector{<:FESet},
    assembler,
    dh::MixedDofHandler, 
    ch::ConstraintHandler,
    solver::Newton,
    guess::AbstractGuess,
    nbcs::Tuple,
    time_iterator,
    local_solver_options,
    files::SimulationFiles,
)
    # write output to dedicated file
    redirect_stdio(;stdout=files.outstreams, stderr=files.outstreams)

    # store input parameters in jld2 file
    jldsave(files.simulation_setup_file; vars, fe_sets, assembler, dh, ch, solver, guess, nbcs, time_iterator, local_solver_options, files)

    # store initial values as time step 0
    save_step!(files.jld2_storage, 0, 0.0, 0, vars[:,end], get_history_variables(fe_sets))

    time_points = CircularBuffer{eltype(time_iterator)}(3)
    push!(time_points, zero(eltype(time_iterator)))

    for (step_id, t) in enumerate(time_iterator)
        push!(time_points, t)
        # no guess based on previous 2 steps possible
        initial_guess = step_id < 3 ? NoGuess() : guess
        try
            do_timestep!(
                vars,
                fe_sets,
                assembler,
                dh,
                ch,
                solver,
                initial_guess,
                nbcs,
                time_points,
                step_id,
                local_solver_options,
                files.convergence,
                files.jld2_storage,
            )
        catch e
            # save data before interrupting
            close_on_error(files; compress_jld2=true)
            rethrow(e)
        end
    end

    redirect_stdio(;stdout=files.orig_stdout, stderr=files.orig_stderr)
    close(files; compress_jld2=true)
    return files
end