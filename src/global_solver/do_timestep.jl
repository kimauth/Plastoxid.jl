"""
    do_timestep!(vars, fe_sets, assembler, dh, ch, solver, guess, nbcs, time_points, step_idx, local_solver_options, jld2_path)

Solve a timestep of the simulation and store the result in a jld2 file.

The following steps are performed:
1. Take an initial guess for the step based on `guess`.
2. Update Dirichlet and Neumann boundary conditions for the current time point `t = last(time_points)`.
3. Solve the step by `solver`.
4. Write converged results to jld2 file. For a specification of the result data layout see [ref].
"""
function do_timestep!(
    vars::AbstractMatrix{Float64},
    fe_sets::Vector{<:FESet},
    assembler,
    dh::MixedDofHandler, 
    ch::ConstraintHandler,
    solver::Newton,
    guess::AbstractGuess,
    nbcs::Tuple,
    time_points::CircularBuffer{Float64},
    step_idx::Int,
    local_solver_options,
    convergence_stream::IOStream,
    jld2_storage::JLD2Storage,
    )
    # time
    t = last(time_points)
    Δt = time_points[end] - time_points[end-1]

    initial_guess!(vars, guess, time_points)

    # boundary conditions
    update!(ch, t) # update Dirichlet bcs for potential time step
    @views apply!(vars[:, end], ch) # apply prescribed values to u
    for nbc in nbcs
        update!(nbc, t) # update Neumann bcs for potential time step_idx
    end

    println(convergence_stream, "")
    println(convergence_stream, "_______________________________________________________")
    @printf convergence_stream "time step %i: t_start = %.3g, Δt = %.3g\n" step_idx time_points[end-1] Δt
    println(convergence_stream, "_______________________________________________________")

    iterate!(vars, solver, fe_sets, assembler, dh, ch, nbcs, local_solver_options, Δt)
    
    print_residuals(convergence_stream, solver.residuals) # convergence print-outs

    # store results in jld2 file
    save_step!(jld2_storage, step_idx, t, 0, vars[:,end], get_history_variables(fe_sets))

    return vars
end
