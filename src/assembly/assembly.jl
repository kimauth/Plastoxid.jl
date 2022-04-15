
function assembly!(
    A::Union{Ferrite.AssemblerSparsityPattern{Float64, Int64}, Vector{Ferrite.AssemblerSparsityPattern{Float64, Int64}},},
    fe_sets::Vector{<:FESet},
    dh::MixedDofHandler,
    u::AbstractMatrix{Float64},
    neumann_bcs::Tuple = (),
    options=Dict{Symbol, Any}(),
    Δt::Union{Nothing, Float64}=nothing,
)
#allow c (and u?) to be c::Vector{Vector{Float64}} and check that they do not have more than 2 entries
# --> can store c and cold
    for fe_set in fe_sets
        assemble_feset!(A, fe_set, dh, u, neumann_bcs, options, Δt)
    end
    return nothing
end

# serial assembly
function assemble_feset!(
    A::Ferrite.AssemblerSparsityPattern{Float64, Int64},
    feset::FESet{PT, CD},
    dh::MixedDofHandler,
    u::AbstractMatrix{Float64},
    neumann_bcs::Tuple,
    options,
    Δt::Union{Nothing, Float64},
    ) where {PT<:AbstractProblem, CD<:AbstractCellBuffer}

    cb = first(feset.cellbuffers)

    for color in feset.cellset
        for i = 1:length(color) # loop over cells of color
            (;globalid, localid) = color[i]

            reinit!(cb, dh, feset.material_mapping, globalid)

            cell_states = feset.states.material_states[localid]
            cell_states_temp = feset.states.material_states_temp[localid]

            assemble_cell!(cb, feset.problem, u[cb.dofs, :], cell_states, cell_states_temp, neumann_bcs, options, Δt)
            assemble!(A, cb.dofs, cb.ke, cb.re)
        end
    end
    return nothing
end

# threaded assembly
function assemble_feset!(
    A::Vector{Ferrite.AssemblerSparsityPattern{Float64, Int64}},
    feset::FESet{PT, CD},
    dh::MixedDofHandler,
    u::AbstractMatrix{Float64},
    neumann_bcs::Tuple,
    options,
    Δt::Union{Nothing, Float64},
    ) where {PT<:AbstractProblem, CD<:AbstractCellBuffer}

    for color in feset.cellset
        Threads.@threads for i = 1:length(color) # loop over cells of color
            thread_id = Threads.threadid()
            cb = feset.cellbuffers[thread_id]
            assembler = A[thread_id]

            (;globalid, localid) = color[i]

            reinit!(cb, dh, feset.material_mapping, globalid)

            @views cell_states = feset.states.material_states[localid]
            @views cell_states_temp = feset.states.material_states_temp[localid]

            assemble_cell!(cb, feset.problem, u[cb.dofs, :], cell_states, cell_states_temp, neumann_bcs, options, Δt)
            assemble!(assembler, cb.dofs, cb.ke, cb.re)
        end
    end
    return nothing
end