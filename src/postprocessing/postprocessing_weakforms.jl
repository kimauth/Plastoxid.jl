
# weak form postprocessing
function postprocess!(
    qp_values,
    cellid::Int,
    cb::CellBuffer,
    problem::MechanicalEquilibrium{M,D},
    ue::AbstractVector{Float64},
    Δt::Union{Nothing,Float64},
    states::Vector{MS},
    old_states = nothing
) where {M,D,MS}

    (;cv, xe, material_id) = cb

    material = problem.materials[material_id[]]

    Ferrite.reinit!(cv, xe)

    for qp in 1:getnquadpoints(cv)
        # element postprocessed quantities
        qp_values.X[qp, cellid] = spatial_coordinate(cv, qp, xe)
        qp_values.u[qp, cellid] = function_value(cv, qp, ue)

        # material postprocessed quantities
        ε = function_symmetric_gradient(cv, qp, ue)
        ε_3D = MaterialModels.increase_dim(ε)
        postprocess_material!(qp_values, cellid, qp, ε_3D, material, states[qp], Δt)
    end
    return qp_values
end

@inline pp_types(::Type{MechanicalEquilibrium{M,D}}) where {M,dim, D<:AbstractDim{dim}} = (
    X = Vec{dim, Float64},
    u = Vec{dim, Float64},
)

# weak form postprocessing
function postprocess!(
    qp_values,
    cellid,
    cb::CellBuffer,
    problem::Interface{M,D},
    ue::AbstractVector{Float64},
    Δt::Union{Nothing,Float64},
    states::Vector{MS},
    old_states = nothing
) where {M,D,MS}

    (;cv, xe, material_id) = cb

    material = problem.materials[material_id[]]

    Ferrite.reinit!(cv, xe)

    for qp in 1:getnquadpoints(cv)
        R = getR(cv, qp)

        # compute separations
        Δ_global = function_value(cv, qp, ue)

        # material postprocessed quantities
        Δ_local = Δ_global ⋅ R
        postprocess_material!(qp_values, cellid, qp, Δ_local, material, states[qp], Δt)
        T_local = qp_values.T[cellid, qp] # retrieve, still need this

        # element postprocessed quantities
        X = spatial_coordinate(cv, qp, xe)
        u = function_midplane_value(cv, qp, ue)
        T_global = R ⋅ T_local
        qp_values.X[qp, cellid] = X
        qp_values.u[qp, cellid] = u
        qp_values.Δ_global[qp, cellid] = Δ_global
        qp_values.T_global[qp, cellid] = T_global
    end
    return qp_values
end

@inline pp_types(::Type{Interface{M,D}}) where {M,dim, D<:AbstractDim{dim}} = (
    X = Vec{dim, Float64},
    u = Vec{dim, Float64},
    Δ_global = Vec{dim, Float64},
    T_global = Vec{dim, Float64},
)

# weak form postprocessing
function postprocess!(
    qp_values,
    cellid,
    cb::MixedCellBuffer,
    problem::InterfaceFicksLaw,
    vars::AbstractVector{Float64},
    Δt::Union{Nothing,Float64},
    states::AbstractVector{MS},
    old_states::AbstractVector{MS},
) where {MS}

    (;cv_u, cv_c, xe, material_id) = cb

    material = problem.materials[material_id[]]

    # material parameters
    (; D, Vₒ₂) = material # questionable if they should be stored under these names
    r = material.R
    ϑ = material.T
    h = problem.shell_thickness

    Ferrite.reinit!(cv_u, xe)
    Ferrite.reinit!(cv_c, xe)

    dofs_u = problem.dof_ranges[:u]
    dofs_c = problem.dof_ranges[:c]

    @views begin
        ce = vars[dofs_c]
        ue = vars[dofs_u]
    end

    # flux computations
    Δₘₐₓ_e = Vector{typeof(first(states).Δₘₐₓ)}(undef, length(cv_u.N_qp_dict)) # buffer this
    for (i, qp) in cv_u.N_qp_dict
        Δₘₐₓ_e[i] = states[qp].Δₘₐₓ
    end
    cᵍᵇₘₐₓ_e = Vector{typeof(first(states).cᵍᵇₘₐₓ)}(undef, length(cv_u.N_qp_dict)) # buffer this
    for (i, qp) in cv_c.N_qp_dict
        cᵍᵇₘₐₓ_e[i] = states[qp].cᵍᵇₘₐₓ
    end

    for qp in 1:getnquadpoints(cv_u)
        R = getR(cv_u, qp)

        # displacement jump
        Δ_global = function_value(cv_u, qp, ue)
        Δ_local = Δ_global ⋅ R
        ∇Δ_local = R' ⋅ function_gradient(cv_u, qp, ue) ⋅ R

        # concentration on mid-plane
        cᵐⁱᵈ = function_midplane_value(cv_c, qp, ce)
        cᵍᵇ = h * cᵐⁱᵈ
        ∇ᵍᵇcᵍᵇ = h * function_midplane_gradient(cv_c, qp, ce) ⋅ R

        # spatial gradients of history variables
        ∇ᵍᵇcᵍᵇₘₐₓ = function_inplane_gradient(cv_c, qp, cᵍᵇₘₐₓ_e) ⋅ R
        ∇ᵍᵇΔₘₐₓ = function_inplane_gradient(cv_u, qp, Δₘₐₓ_e) ⋅ R
        
        # derivatives for flux computation
        ∂T̂ₙ∂cᵍᵇₘₐₓ = gradient(cᵍᵇₘₐₓ->T̂ₙ(material, Δ_local, states[qp].Δₘₐₓ, cᵍᵇₘₐₓ), states[qp].cᵍᵇₘₐₓ)
        ∂T̂ₙ∂Δₘₐₓ = gradient(Δₘₐₓ->T̂ₙ(material, Δ_local, Δₘₐₓ, states[qp].cᵍᵇₘₐₓ), states[qp].Δₘₐₓ)
        ∂T̂ₙ∂Δ = gradient(Δ->T̂ₙ(material, Δ, states[qp].Δₘₐₓ, states[qp].cᵍᵇₘₐₓ), Δ_local)

        # quantities in the element routine are in local coordinates,
        # but for post-processing purposes, we export global quantities from the element
        ∇ᵍᵇT̂ₙ = ∂T̂ₙ∂cᵍᵇₘₐₓ * ∇ᵍᵇcᵍᵇₘₐₓ + ∂T̂ₙ∂Δₘₐₓ ⋅ ∇ᵍᵇΔₘₐₓ + ∂T̂ₙ∂Δ ⋅ ∇Δ_local
        ∇ᵍᵇT̂ₙ_global = R ⋅ ∇ᵍᵇT̂ₙ
        jᵍᵇ_chem_global = R ⋅ (-D * ∇ᵍᵇcᵍᵇ)
        cᵍᵇ_pos = 1/2 * (abs(cᵍᵇ) + cᵍᵇ)
        jᵍᵇ_mech_global = R ⋅ (D * Vₒ₂ / (ϑ * r) * cᵍᵇ_pos * ∇ᵍᵇT̂ₙ)
        jᵍᵇ_global = jᵍᵇ_chem_global + jᵍᵇ_mech_global

        # material postprocessed quantities
        postprocess_material!(qp_values, cellid, qp, Δ_local, cᵍᵇ, material, states[qp], old_states[qp], Δt)
        T_local = qp_values.T[qp, cellid] # retrieve, still need this
        T_global = R ⋅ T_local

        # element postprocessed quantities
        X = spatial_coordinate(cv_u, qp, xe)
        u = function_midplane_value(cv_u, qp, ue)
        qp_values.X[qp, cellid] = X
        qp_values.u[qp, cellid] = u
        qp_values.Δ_global[qp, cellid] = Δ_global
        qp_values.T_global[qp, cellid] = T_global
        qp_values.jᵍᵇ_global[qp, cellid] = jᵍᵇ_global
        qp_values.jᵍᵇ_mech_global[qp, cellid] = jᵍᵇ_mech_global
        qp_values.jᵍᵇ_chem_global[qp, cellid] = jᵍᵇ_chem_global
        qp_values.∇ᵍᵇT̂ₙ_global[qp, cellid] = ∇ᵍᵇT̂ₙ_global
    end

    return qp_values
end

@inline pp_types(::Type{InterfaceFicksLaw{M,D}}) where {M,dim, D<:AbstractDim{dim}} = (
    X = Vec{dim, Float64},
    u = Vec{dim, Float64},
    Δ_global = Vec{dim, Float64},
    T_global = Vec{dim, Float64},
    jᵍᵇ_global = Vec{dim, Float64}, 
    jᵍᵇ_mech_global = Vec{dim, Float64},
    jᵍᵇ_chem_global = Vec{dim, Float64},
    ∇ᵍᵇT̂ₙ_global = Vec{dim, Float64},
)
