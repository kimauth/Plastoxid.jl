function assemble_cell!(
    cell_data::CD,
    problem::InterfaceFicksLaw{Mat, Dim},
    vars::Matrix{Float64},
    states::Vector{MS},
    states_temp::Vector{MS},
    neumann_bcs::Tuple=(),
    options=Dict{Symbol, Any}(),
    Δt::Union{Nothing, Float64}=1.0,
) where {Mat, MS, Dim, CD}

    (;re, ke, cv_u, cv_c, fv_c, xe, global_cellid, material_id, nfaces, material_cache) = cell_data

    dim_type = problem.dim_type
    h = problem.shell_thickness
    material = problem.materials[material_id[]]

    # material parameters
    (; D, Vₒ₂) = material # questionable if they should be stored under these names
    r = material.R
    ϑ = material.T
    M = D*Vₒ₂/(r*ϑ)

    Ferrite.reinit!(cv_u, xe)
    Ferrite.reinit!(cv_c, xe)

    dofs_u = problem.dof_ranges[:u]
    dofs_c = problem.dof_ranges[:c]

    @views begin
        ce = vars[dofs_c, 2]
        ce_old = vars[dofs_c, 1]
        ue = vars[dofs_u, 2]
    end

    # compute "nodal" values of history variables
    Δₘₐₓ_e = Vector{typeof(first(states).Δₘₐₓ)}(undef, length(cv_u.N_qp_dict)) # buffer this
    for (i, qp) in cv_u.N_qp_dict
        R = getR(cv_u, qp)
        Δ_local = function_value(cv_u, qp, ue) ⋅ R
        Δₘₐₓ_e[i] = max_jump(Δ_local, states[qp].Δₘₐₓ)
    end
    cᵍᵇₘₐₓ_e = Vector{typeof(first(states).cᵍᵇₘₐₓ)}(undef, length(cv_u.N_qp_dict)) # buffer this
    for (i, qp) in cv_c.N_qp_dict
        cᵍᵇ = function_value(cv_c, qp, ce) * h
        cᵍᵇₘₐₓ_e[i] = max_concentration(cᵍᵇ, states[qp].cᵍᵇₘₐₓ)
    end

    re_u = view(re, dofs_u)
    ke_uu = view(ke, dofs_u, dofs_u)
    ke_uc = view(ke, dofs_u, dofs_c)

    re_c = view(re, dofs_c)
    ke_cc = view(ke, dofs_c, dofs_c)
    ke_cu = view(ke, dofs_c, dofs_u)

    # cv_u and cv_c are required to share their quadrature points
    for qp in 1:getnquadpoints(cv_u)
        R = getR(cv_u, qp)
        dA = getdetJdA(cv_u, qp) * get_outofdim_measurement(problem)

        # displacement jump
        Δ_local = function_value(cv_u, qp, ue) ⋅ R
        ∇Δ_local = R' ⋅ function_gradient(cv_u, qp, ue) ⋅ R

        # concentration on mid-plane
        cᵐⁱᵈ = function_midplane_value(cv_c, qp, ce)
        cᵍᵇ = h * cᵐⁱᵈ
        ∇ᵍᵇcᵍᵇ = h * function_midplane_gradient(cv_c, qp, ce) ⋅ R
        cᵐⁱᵈ_old = function_midplane_value(cv_c, qp, ce_old)
        cᵍᵇ_old = h * cᵐⁱᵈ_old
        # concentration jump
        Δc = function_value(cv_c, qp, ce)
        Δc_old = function_value(cv_c, qp, ce_old)
        ∇ᵍᵇΔc = function_gradient(cv_c, qp, ce) ⋅ R

        # spatial gradients of history variables
        ∇ᵍᵇcᵍᵇₘₐₓ = function_inplane_gradient(cv_c, qp, cᵍᵇₘₐₓ_e) ⋅ R
        ∇ᵍᵇΔₘₐₓ = function_inplane_gradient(cv_u, qp, Δₘₐₓ_e) ⋅ R

        T, dTdΔ, dTdcᵍᵇ, states_temp[qp], extras = material_response(material, Δ_local, cᵍᵇ, states[qp], Δt; cache = material_cache, options = options)
        (;
        # 1st derivatives
        ∂T̂ₙ∂cᵍᵇₘₐₓ,
        ∂T̂ₙ∂Δₘₐₓ,
        ∂T̂ₙ∂Δ,
        # 2nd derivatives
        ∂²T̂ₙ∂cᵍᵇₘₐₓ²,
        ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δₘₐₓ,
        ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δ,
        ∂²T̂ₙ∂Δₘₐₓ²,
        ∂²T̂ₙ∂Δₘₐₓ∂Δ,
        ∂²T̂ₙ∂Δ²,
        # history variable derivatives
        ∂Δₘₐₓ∂Δ,
        ∂cᵍᵇₘₐₓ∂cᵍᵇ,
        ) = extras

        # chemical contribution

        # flux computation
        ∇ᵍᵇT̂ₙ = ∂T̂ₙ∂cᵍᵇₘₐₓ * ∇ᵍᵇcᵍᵇₘₐₓ + ∂T̂ₙ∂Δₘₐₓ ⋅ ∇ᵍᵇΔₘₐₓ + ∂T̂ₙ∂Δ ⋅ ∇Δ_local
        # just a trial 
        cᵍᵇ_pos = 1/2*(abs(cᵍᵇ) + cᵍᵇ)
        jᵍᵇ = -D * ∇ᵍᵇcᵍᵇ + D * Vₒ₂ / (ϑ * r) * cᵍᵇ_pos * ∇ᵍᵇT̂ₙ
        Δj = -D * ∇ᵍᵇΔc + D * Vₒ₂ / (ϑ * r) * Δc * ∇ᵍᵇT̂ₙ

        for i in 1:getnbasefunctions(cv_c) # TODO: the loops are organized the wrong way around :(
            Nᶜᵢᵐⁱᵈ = shape_midplane_value(cv_c, qp, i)
            ∇ᵍᵇNᶜᵢᵐⁱᵈ = shape_midplane_gradient(cv_c, qp, i) ⋅ R
            Nᶜᵢ = shape_value(cv_c, qp, i)
            ∇ᵍᵇNᶜᵢ = shape_gradient(cv_c, qp, i) ⋅ R
            re_c[i] += ( (cᵍᵇ - cᵍᵇ_old) * Nᶜᵢᵐⁱᵈ - Δt * jᵍᵇ ⋅ ∇ᵍᵇNᶜᵢᵐⁱᵈ +
                       h/12 * (Δc - Δc_old) * Nᶜᵢ + Δt / h * D * Δc * Nᶜᵢ + Δt*h/12 * Δj ⋅ ∇ᵍᵇNᶜᵢ ) * dA
            for j in 1:getnbasefunctions(cv_c)
                Nᶜⱼ = shape_value(cv_c, qp, j)
                ∇ᵍᵇNᶜⱼ = shape_gradient(cv_c, qp, j) ⋅ R

                Nᶜⱼᵐⁱᵈ = shape_midplane_value(cv_c, qp, j)
                ∂cᵍᵇₘₐₓ∂c̲ⱼ = ∂cᵍᵇₘₐₓ∂cᵍᵇ * h * Nᶜⱼᵐⁱᵈ
                ∂²T̂ₙ∂cᵍᵇₘₐₓ∂c̲ⱼ = ∂²T̂ₙ∂cᵍᵇₘₐₓ² * ∂cᵍᵇₘₐₓ∂c̲ⱼ
                ∂²T̂ₙ∂Δₘₐₓ∂c̲ⱼ = ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δₘₐₓ * ∂cᵍᵇₘₐₓ∂c̲ⱼ
                ∂²T̂ₙ∂Δ∂c̲ⱼ = ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δ * ∂cᵍᵇₘₐₓ∂c̲ⱼ

                ∇ᵍᵇNᶜⱼᵐⁱᵈ = shape_midplane_gradient(cv_c, qp, j) ⋅ R
                ∂∇ᵍᵇcᵍᵇₘₐₓ∂c̲ⱼ = ∂cᵍᵇₘₐₓ∂cᵍᵇ * h * ∇ᵍᵇNᶜⱼᵐⁱᵈ

                ∂∇ᵍᵇT̂ₙ∂c̲ⱼ = ∂²T̂ₙ∂cᵍᵇₘₐₓ∂c̲ⱼ * ∇ᵍᵇcᵍᵇₘₐₓ + ∂T̂ₙ∂cᵍᵇₘₐₓ * ∂∇ᵍᵇcᵍᵇₘₐₓ∂c̲ⱼ +
                            ∂²T̂ₙ∂Δₘₐₓ∂c̲ⱼ ⋅ ∇ᵍᵇΔₘₐₓ +
                            ∂²T̂ₙ∂Δ∂c̲ⱼ ⋅ ∇Δ_local

                ∂jᵍᵇ∂c̲ⱼ = h * (-D * ∇ᵍᵇNᶜⱼᵐⁱᵈ + M * heaviside(cᵐⁱᵈ) * (Nᶜⱼᵐⁱᵈ * ∇ᵍᵇT̂ₙ + cᵐⁱᵈ * ∂∇ᵍᵇT̂ₙ∂c̲ⱼ))
                ∂Δj∂c̲ⱼ = -D * ∇ᵍᵇNᶜⱼ + M * (Nᶜⱼ * ∇ᵍᵇT̂ₙ + Δc * ∂∇ᵍᵇT̂ₙ∂c̲ⱼ)

                ke_cc[i,j] += h * Nᶜⱼᵐⁱᵈ * Nᶜᵢᵐⁱᵈ * dA -
                             Δt * ∂jᵍᵇ∂c̲ⱼ ⋅ ∇ᵍᵇNᶜᵢᵐⁱᵈ * dA +
                             h/12. * Nᶜⱼ * Nᶜᵢ * dA +
                             Δt/h * D * Nᶜⱼ * Nᶜᵢ * dA +
                             Δt*h/12. * ∂Δj∂c̲ⱼ ⋅ ∇ᵍᵇNᶜᵢ * dA
            end

            for j in 1:getnbasefunctions(cv_u)
                Nᵘⱼ = shape_value(cv_u, qp, j)
                dΔdu̲ⱼ = Nᵘⱼ ⋅ R

                #!!!
                ∇ᵍᵇNᵘⱼ = shape_gradient(cv_u, qp, j) ⋅ R
                d∇Δdu̲ⱼ = ∇ᵍᵇNᵘⱼ ⋅ R

                ∂²T̂ₙ∂cᵍᵇₘₐₓ∂u̲ⱼ = (∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δₘₐₓ ⋅ ∂Δₘₐₓ∂Δ + ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δ) ⋅ dΔdu̲ⱼ
                ∂²T̂ₙ∂Δₘₐₓ∂u̲ⱼ = (∂²T̂ₙ∂Δₘₐₓ² ⋅ ∂Δₘₐₓ∂Δ + ∂²T̂ₙ∂Δₘₐₓ∂Δ) ⋅ dΔdu̲ⱼ
                ∂²T̂ₙ∂Δ∂u̲ⱼ = (∂²T̂ₙ∂Δₘₐₓ∂Δ ⋅ ∂Δₘₐₓ∂Δ + ∂²T̂ₙ∂Δ²) ⋅ dΔdu̲ⱼ

                ∂∇ᵍᵇT̂ₙ∂u̲ⱼ = ∂²T̂ₙ∂cᵍᵇₘₐₓ∂u̲ⱼ * ∇ᵍᵇcᵍᵇₘₐₓ +
                           ∂²T̂ₙ∂Δₘₐₓ∂u̲ⱼ ⋅ ∇ᵍᵇΔₘₐₓ + ∂T̂ₙ∂Δₘₐₓ ⋅ ∂Δₘₐₓ∂Δ ⋅ d∇Δdu̲ⱼ +
                           ∂²T̂ₙ∂Δ∂u̲ⱼ ⋅ ∇Δ_local + ∂T̂ₙ∂Δ ⋅ d∇Δdu̲ⱼ

                ∂jᵍᵇ∂u̲ⱼ = heaviside(cᵐⁱᵈ) * h * M * cᵐⁱᵈ * ∂∇ᵍᵇT̂ₙ∂u̲ⱼ
                ∂Δj∂u̲ⱼ = h * M * Δc * ∂∇ᵍᵇT̂ₙ∂u̲ⱼ

                ke_cu[i,j] += -Δt * ∂jᵍᵇ∂u̲ⱼ ⋅ ∇ᵍᵇNᶜᵢᵐⁱᵈ * dA + Δt*h/12. * ∂Δj∂u̲ⱼ ⋅ ∇ᵍᵇNᶜᵢ * dA
            end
        end

        # mechanical contribution
        for i in 1:getnbasefunctions(cv_u)
            Nᵘᵢ = shape_value(cv_u, qp, i)
            dδΔdu̲ᵢ = Nᵘᵢ ⋅ R
            re_u[i] += T ⋅  dδΔdu̲ᵢ * dA
            for j in 1:getnbasefunctions(cv_u)
                Nᵘⱼ = shape_value(cv_u, qp, j)
                dΔdu̲ⱼ = Nᵘⱼ ⋅ R
                dTdu̲ⱼ = dTdΔ ⋅ dΔdu̲ⱼ
                ke_uu[i,j] += dTdu̲ⱼ ⋅ dδΔdu̲ᵢ * dA
            end
            for j in 1:getnbasefunctions(cv_c)
                Nᶜⱼ = shape_value(cv_c, qp, j)
                dcᵍᵇdc̲ⱼ = h * Nᶜⱼ
                dTdc̲ⱼ = dTdcᵍᵇ * dcᵍᵇdc̲ⱼ
                ke_uc[i,j] += dTdc̲ⱼ ⋅ dδΔdu̲ᵢ * dA
            end
        end
    end

    # Neumann boundary conditions
    for nbc in neumann_bcs
        if nbc.field == :c
            t = get_time(nbc)
            for face_id in 1:nfaces[]
                if FaceIndex(global_cellid[], face_id) ∈ nbc.faceset
                    reinit!(fv_c, xe, face_id)
                    for qp in 1:getnquadpoints(fv_c)
                        x = spatial_coordinate(fv_c, qp, xe)
                        jᵖ = nbc.f(x, t)
                        dΓ = getdetJdV(fv_c, qp) * get_outofdim_measurement(problem)
                        for i in 1:n_basefuncs
                            Nᵢ = shape_midplane_value(fv_c, q_point, i)
                            re[i] += Nᵢ * jᵖ * dΓ
                        end
                    end
                end
            end
        end
    end
    
    return nothing
end

heaviside(x) = x > zero(x) ? x : zero(x)