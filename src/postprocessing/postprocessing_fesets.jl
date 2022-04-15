
function allocate_matrices(feset::Plastoxid.FESet{PT}) where {M, D, PT<:Plastoxid.AbstractProblem{M, D}}

    ncells = getncells(feset)
    nqp = getnquadpoints(first(feset.cellbuffers))
    material_types = pp_types(M, D)
    cell_types = pp_types(PT)
    k = (keys(material_types)..., keys(cell_types)...)
    t = (material_types..., cell_types...)
    matrices = NamedTuple(key=>Matrix{type}(undef, nqp, ncells) for (key, type) in zip(k, t))

    return matrices
end

# all fesets
function postprocess!(
    qp_values,
    all_states, # doesn't use state containers in fesets for being thread-safe
    all_old_states, # doesn't use state containers in fesets for being thread-safe
    fesets::AbstractVector{FESet},
    vars::AbstractVector,
    Δt::Float64,
    states::AbstractVector{<:AbstractVector{<:AbstractMaterialState}},
    old_states::AbstractVector{<:AbstractVector{<:AbstractMaterialState}},
    dh::Ferrite.AbstractDofHandler,
)
    for (i, feset) in enumerate(fesets)
        set_history_variables!(feset, all_states[i], states)
        set_history_variables!(feset, all_old_states[i], old_states)
        postprocess!(
            qp_values[i],
            feset,
            vars,
            Δt,
            all_states[i],
            all_old_states[i],
            dh)
    end
    return qp_values
end

# one feset
function postprocess!(
    qp_values,
    feset::FESet{PT},
    vars::AbstractVector,
    Δt::Float64,
    states::AbstractVector{<:AbstractVector{MS}},
    old_states::AbstractVector{<:AbstractVector{MS}},
    dh::Ferrite.AbstractDofHandler,
) where {PT, MS<:AbstractMaterialState}

    # loop over cells in feset
    for color in feset.cellset, (;globalid, localid) in color
        cell_states = states[localid]
        old_cell_states = old_states[localid]
        # dofs = celldofs(dh, globalid)
        cb = feset.cellbuffers[Threads.threadid()]
        reinit!(cb, dh, feset.material_mapping, globalid)
        @views cell_vars = vars[cb.dofs]
        postprocess!(qp_values, localid, cb, feset.problem, cell_vars, Δt, cell_states, old_cell_states)
    end
end
