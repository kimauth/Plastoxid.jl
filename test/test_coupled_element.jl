function test_re_ke(re_aligned, re_rotated, ke_aligned, ke_rotated, R)
    for i=1:4
        idx_u = (i*2-1):2i
        # re_u
        @test Vec((re_rotated[(i*2-1):2i]...)) ⋅ R ≈ Vec((re_aligned[(i*2-1):2i]...))
        # re_c
        @test re_rotated[8+i] ≈ re_aligned[8+i]
        # ke_uu
        @test R' ⋅ Tensor{2,2}(ke_rotated[idx_u, idx_u]) ⋅ R ≈ Tensor{2,2}(ke_aligned[idx_u, idx_u])
        # ke_uc
        @test Vec{2}(ke_rotated[idx_u, 8+i]) ⋅ R ≈ ke_aligned[idx_u, 8+i]
        # ke_cu
        @test isapprox(Vec{2}(ke_rotated[8+i, idx_u]) ⋅ R, ke_aligned[8+i, idx_u]; atol=1e-16)
        # ke_cc
        @test ke_rotated[8+i, 8+i] ≈ ke_aligned[8+i, 8+i]
    end
end

@testset "Coupled cohesive element" begin
    # aligned system
    xe_aligned = [Vec((0.0, 0.0)), Vec((5.0, 0.0)), Vec((0.0, 0.0)),  Vec((5.0, 0.0))]
    cells = Ferrite.AbstractCell[]
    push!(cells, CohesiveQuadrilateral((1,2,3,4)))
    grid_aligned = Grid(cells, Node.(xe_aligned))
    # rotated system
    xe_rotated = [Vec((0.0, 0.0)), Vec((4.0, 3.0)), Vec((0.0, 0.0)),  Vec((4.0, 3.0))]
    grid_rotated = Grid(cells, Node.(xe_rotated))

    # construct material
    D = 1e-10 # mm²/s 
    cᵍᵇ_char = 0.1e-15 # mol/mm²
    cᵍᵇ₀ = 0.0
    Vₒ₂ = 7.93 # mm³/mol
    δₙ = 1.5e-3 # mm
    δₜ = 1.5e-3 # mm
    σₘₐₓ = τₘₐₓ = 2e3
    Φₙ = σₘₐₓ * δₙ / (1. - (1. - exp(-1.0)) * Plastoxid.ℋᵣ(δₙ, δₙ))# N/mm
    Φₜ = τₘₐₓ * δₜ * exp(0.5) # N/mm
    M = 9.8e-3 # mm²/N 
    dₒ₂ₘₐₓ = 0.8
    R = 8.314e3 # N mm / (mol K)
    T = 700. + 273.15 # K
    material = Plastoxid.CoupledKolluri(D, cᵍᵇ_char, cᵍᵇ₀, Vₒ₂, δₙ, δₜ, Φₙ, Φₜ, M, dₒ₂ₘₐₓ, R, T)

    # fields for Problem
    materials = [material]
    dim_type = MaterialModels.Dim{2}()
    t = 2.0
    shell_thickness = 1e-4

    # interpolations
    ip_base_f = Lagrange{1, RefCube, 1}()
    ip = JumpInterpolation(ip_base_f)
    ip_base_geo = Lagrange{1, RefCube, 1}()
    ip_geo = MidPlaneInterpolation(ip_base_geo)
    qr = QuadratureRule{1,RefCube}(:lobatto, 3)
    qr_face = QuadratureRule{0,RefCube}(1)

    # manual set-up here, matches interpolations above
    dof_ranges = Dict(:u => 1:8, :c => 9:12)
    problem = InterfaceFicksLaw(materials, shell_thickness, dim_type, t, dof_ranges)

    # CellBuffer
    cb = cellbuffer(problem, ip, ip, ip_geo, qr, qr_face)

    # Material states
    states = [initial_material_state(material) for qp in 1:getnquadpoints(cb.cv_u)]
    states_temp = deepcopy(states)

    # Dofs
    field_u = Field(:u, ip, 2)
    field_c = Field(:c, ip, 1)
    fh = FieldHandler([field_u, field_c], Set((1,)))

    dh_aligned = MixedDofHandler(grid_aligned)
    push!(dh_aligned, fh)
    close!(dh_aligned)

    dh_rotated = MixedDofHandler(grid_rotated)
    push!(dh_rotated, fh)
    close!(dh_rotated)

    # variables
    vars = zeros(12,2)

    ## no load leads to no residuals
    reinit!(cb, dh_aligned, Dict(1=>1), 1)
    assemble_cell!(cb, problem, vars, states, states_temp, (), Dict{Symbol, Any}(), 1.0)
    @test cb.re == zeros(12)

    # hit maximum normal traction
    vars = zeros(12,2)
    ue = [0.0, -δₙ/2, 0.0, -δₙ/2, 0.0, δₙ/2, 0.0, δₙ/2]
    vars[1:8, 2] = ue
    reinit!(cb, dh_aligned, Dict(1=>1), 1)
    assemble_cell!(cb, problem, vars, states, states_temp, (), Dict{Symbol, Any}(), 1.0)
    @test cb.re[1:4] ≈ -cb.re[5:8]
    @test cb.re[2] ≈ -σₘₐₓ*t*norm(xe_aligned[2]-xe_aligned[1]) / 2
    re_mode1 = copy(cb.re) # store for next test

    # adding oxygen lowers traction response
    vars = zeros(12,2)
    ue = [0.0, -δₙ/2, 0.0, -δₙ/2, 0.0, δₙ/2, 0.0, δₙ/2]
    vars[1:8, 2] = ue
    ce = [0., cᵍᵇ_char/shell_thickness, 0., cᵍᵇ_char/shell_thickness]
    vars[9:12, 2] = ce
    reinit!(cb, dh_aligned, Dict(1=>1), 1)
    assemble_cell!(cb, problem, vars, states, states_temp, (), Dict{Symbol, Any}(), 1.0)
    @test cb.re[1:4] ≈ -cb.re[5:8]
    @test all(cb.re[[6,8]] .<  re_mode1[[6,8]])

    ## Evaluate an element that is aligned with the coordinate system 
    # and one that is rotated. The results should be the same when rotated back to the same system.
    function test_rotations(cb, dh_aligned, dh_rotated, problem, states, states_temp, ue_aligned, ue_rotated, ce)
        vars = zeros(12,2)
        # assemble aligned cell
        vars[1:8, 2] = ue_aligned
        vars[9:12, 2] = ce
        reinit!(cb, dh_aligned, Dict(1=>1), 1)
        assemble_cell!(cb, problem, vars, states, states_temp, (), Dict{Symbol, Any}(), 1.0)
        re_aligned = copy(cb.re)
        ke_aligned = copy(cb.ke)

        # assemble rotated cell
        fill!(vars, 0.0)
        vars[1:8, 2] = ue_rotated
        vars[9:12, 2] = ce
        reinit!(cb, dh_rotated, Dict(1=>1), 1)
        assemble_cell!(cb, problem, vars, states, states_temp, (), Dict{Symbol, Any}(), 1.0)
        re_rotated = copy(cb.re)
        ke_rotated = copy(cb.ke)

        test_re_ke(re_aligned, re_rotated, ke_aligned, ke_rotated, R)
    end

    # get rotation matrix
    reinit!(cb.cv_u, xe_rotated)
    R = cb.cv_u.R[1]

    # test that rotations within element make sense 
    ue_rotated = [1.5, 0., 0., -0.5, 0., 2.0, 0., 0.5] * 1e-3 # still in global coordinates
    ue_aligned = reinterpret(Float64, [Vec((ue_rotated[(i*2-1):2i]...)) ⋅ R for i=1:4])
    ce = [0., cᵍᵇ_char/shell_thickness, 0., cᵍᵇ_char/shell_thickness]

    # 3 loading combination: only u, only c, u & c
    ues_rotated = [ue_rotated, zeros(8), ue_rotated]
    ues_aligned = [ue_aligned, zeros(8), ue_aligned]
    ces = [zeros(4), ce, ce]

    for case = 1:3
        test_rotations(cb, dh_aligned, dh_rotated, problem, states, states_temp, ues_aligned[case], ues_rotated[case], ces[case])
    end
end


# # Benchmarking
# @btime reinit!($cb, $dh, $(Dict(1=>1)), $1)
# @btime assemble_cell!($cb, $problem, $vars, $states, $states_temp, $(), $(Dict{Symbol, Any}()), $1.0)

# # display the loading situation
# using Plots
# coords_aligned = reshape(reinterpret(Float64, xe_aligned), (2,4))
# coords_rotated = reshape(reinterpret(Float64, xe_rotated), (2,4))
# disp_aligned = reshape(ue_aligned, (2,4))
# disp_rotated = reshape(ue_rotated, (2,4))
# data_aligned = coords_aligned + disp_aligned*200
# data_rotated = coords_rotated + disp_rotated*200
# node_order = [1,2,4,3,1]
# p = plot(coords_aligned[1,:], coords_aligned[2,:], linecolor=:green, label = "aligned", legend = :topleft)
# scatter!(data_aligned[1,:], data_aligned[2,:], markercolor=:green, label = false)
# plot!(coords_rotated[1,:], coords_rotated[2,:], linecolor=:goldenrod, label = "rotated")
# scatter!(data_rotated[1,:], data_rotated[2,:], markercolor=:goldenrod, label = false)
# for i=1:4
#     node_idxs = node_order[i:i+1]
#     plot!(data_aligned[1,node_idxs], data_aligned[2,node_idxs], line=:dash, linecolor=:green, label=false)
#     plot!(data_rotated[1,node_idxs], data_rotated[2,node_idxs], line=:dash, linecolor=:goldenrod, label=false)
# end
# p