# continuum elements
# Assemble the residual vector and the tangent stiffness matrix for a continuum element
# according to
# ℝᵢ = ∫ ∇Nᵢˢʸᵐ : σ dV
# 𝕂ᵢⱼ = ∫ ∇Nᵢˢʸᵐ : σ : ∇Nⱼˢʸᵐ dV .
# Tested only for 2D, should work in 3D
function assemble_cell!(
    cell_data::CD,
    problem::MechanicalEquilibrium{M, D},
    ue::AbstractMatrix{Float64},
    states::Vector{MS},
    states_temp::Vector{MS},
    neumann_bcs::Tuple=(),
    options=Dict{Symbol, Any}(),
    Δt::Union{Nothing, Float64}=1.0,
) where {M, MS, D, CD}

    (;re, ke, cv, fv, xe, global_cellid, material_id, nfaces, material_cache) = cell_data

    dim_type = problem.dim_type
    material = problem.materials[material_id[]]

    Ferrite.reinit!(cv, xe)

    n_basefuncs = getnbasefunctions(cv)

    @views Δue = ue[:,2] - ue[:,1]

    for q_point in 1:getnquadpoints(cv)
        # For each integration point, compute stress and material stiffness
        Δε = function_symmetric_gradient(cv, q_point, Δue) # strain increment
        
        σ, dσdε, states_temp[q_point] =
            material_response(dim_type, material, Δε, states[q_point], Δt; cache = material_cache, options = options)

        dΩ = getdetJdV(cv, q_point) * get_outofdim_measurement(problem)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_symmetric_gradient(cv, q_point, i)
            re[i] += ∇Nᵢ ⊡ σ * dΩ # internal force vector
            #stiffness matrix
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_symmetric_gradient(cv, q_point, j)
                ke[i,j] += ∇Nᵢ ⊡ dσdε ⊡ ∇Nⱼ * dΩ
            end
        end
    end

    # Neumann boundary conditions
    for nbc in neumann_bcs
        t = get_time(nbc)
        for face_id in 1:nfaces[]
            if FaceIndex(global_cellid[], face_id) ∈ nbc.faceset
                reinit!(fv, xe, face_id)
                for qp in 1:getnquadpoints(fv)
                    x = spatial_coordinate(fv, qp, xe)
                    tᵖ = nbc.f(x, t)
                    dΓ = getdetJdV(fv, qp) * get_outofdim_measurement(problem)
                    for i in 1:n_basefuncs
                        Nᵢ = shape_value(fv, qp, i)
                        re[i] -= Nᵢ ⋅ tᵖ * dΓ
                    end
                end
            end
        end
    end

    return nothing
end

# cohesive element
# Assemble the residual vector and the tangent stiffness matrix for a continuum element
# according to
# ℝᵢ = ∫ N ⋅ R ⋅ Tˡᵒᶜ dS
# 𝕂ᵢⱼ = ∫ N ⋅ R ⋅ ∂Tˡᵒᶜ∂Δ ⋅ Rᵀ ⋅ Nᵀ dV .
# Tested only for 2D, not working in 3D
function assemble_cell!(
    cell_data::CD,
    problem::Interface{M, D},
    ue::Matrix{Float64},
    states::Vector{MS},
    states_temp::Vector{MS},
    neumann_bcs::Tuple=(),
    options=Dict{Symbol, Any}(),
    Δt::Union{Nothing, Float64}=1.0,
) where {M, MS, D, CD}

    (;re, ke, cv, xe, material_id) = cell_data

    dim_type = problem.dim_type
    material = problem.materials[material_id[]]

    # update cellvalues for current cell
    reinit!(cv, xe)

    nbase_funcs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)

        #Rotation matrix
        R = getR(cv, qp)
        #jacobian determinant for usage of isoparametric coordinates
        dΓ = getdetJdA(cv, qp)*get_outofdim_measurement(problem)

        #compute separations
        @views Δ_global = function_value(cv, qp, ue[:, 2])
        Δ_local = Δ_global ⋅ R

        #constitutive_driver computes traction and its derivative
        T_local, dTdΔ_local, states_temp[qp] =
            constitutive_driver(material, Δ_local, states[qp])

        #rotate back to global coordinates
        T = T_local ⋅ R'
        dTdΔ = R ⋅ dTdΔ_local ⋅ R'

        #compute element force vector and element stiffness matrix
        for i in 1:nbase_funcs
            Nⁱ = shape_value(cv, qp, i)
            re[i] += (T ⋅ Nⁱ) * dΓ
            for j in 1:nbase_funcs
                Nʲ = shape_value(cv, qp, j)
                ke[i,j] += Nⁱ ⋅ dTdΔ ⋅ Nʲ * dΓ
            end
        end

    end

    return nothing
end

# Diffusion
function assemble_cell!(
    cell_data::CD,
    problem::FicksLaw{M, Dim},
    vars::Matrix{Float64},
    states::Vector{MS},
    states_temp::Vector{MS},
    neumann_bcs::Tuple=(),
    options=Dict{Symbol, Any}(),
    Δt::Union{Nothing, Float64}=1.0,
) where {M, MS, Dim, CD}

    # check that ce has two entries
    # size(var, 2) == 2 || error("ces must have 2 entries (ce_old and ce).")
    ce_old = vars[:, 1]
    ce = vars[:, 2]

    # new 1.7 syntax
    (;re, ke, cv, fv, xe, global_cellid, material_id, nfaces) = cell_data

    material = problem.materials[material_id[]]
    (; D) = material

    # update cellvalues for current cell
    reinit!(cv, xe)

    n_basefuncs = getnbasefunctions(cv)

    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)*get_outofdim_measurement(problem)

        c = function_value(cv, q_point, ce)
        c_old = function_value(cv, q_point, ce_old)
        ∇c = function_gradient(cv, q_point, ce)

        for i in 1:n_basefuncs
            Nⁱ = shape_value(cv, q_point, i)
            ∇Nⁱ = shape_gradient(cv, q_point, i)
            re[i] += ((c-c_old)*Nⁱ + Δt*D*∇c ⋅ ∇Nⁱ)*dΩ
            #stiffness matrix
            for j in 1:n_basefuncs
                Nʲ = shape_value(cv, q_point, j)
                ∇Nʲ = shape_gradient(cv, q_point, j)
                ke[i,j]+= (Nʲ*Nⁱ + Δt*D*∇Nʲ ⋅ ∇Nⁱ)*dΩ
            end
        end
    end

    # Neumann boundary conditions
    for nbc in neumann_bcs
        t = get_time(nbc)
        for face_id in 1:nfaces
            if FaceIndex(global_cellid, face_id) ∈ nbc.faceset
                reinit!(fv, xe, face_id)
                for qp in 1:getnquadpoints(fv)
                    x = spatial_coordinate(fv, qp, xe)
                    jᵖ = nbc.f(x, t)
                    dΓ = getdetJdV(fv, qp) * get_outofdim_measurement(problem)
                    for i in 1:n_basefuncs
                        Nᵢ = shape_value(fv, q_point, i)
                        re[i] += Nᵢ * jᵖ * dΓ
                    end
                end
            end
        end
    end
    return nothing
end
