@testset "continuum elements" begin
    # truss example
    dim = 1
    dim_type = UniaxialStress()
    A = 15.0

    L = 3.5
    grid = generate_grid(Line, (1,), Vec((0.0)), Vec((L,)))

    ip = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(2)
    qr_face = QuadratureRule{dim-1, RefCube}(2)

    dh = MixedDofHandler(grid)
    push!(dh, :u, 1, ip)
    close!(dh)

    material = LinearElastic(E=200e3, ν=0.3)

    problem = MechanicalEquilibrium([material], dim_type, A)

    cb = cellbuffer(problem, ip, ip, qr, qr_face)

    material_mapping = Plastoxid.preprocess_material_mapping([Set(1:getncells(grid))])
    reinit!(cb, dh, material_mapping, 1)

    states = [initial_material_state(material) for i=1:2]
    states_temp = deepcopy(states)

    # Dirichlet bcs
    ue = zeros(ndofs_per_cell(dh, 1), 2)
    ue[2,2] = 0.1

    assemble_cell!(cb, problem, ue, states, states_temp)

    @test cb.ke ≈ material.E*A/L * [1.0 -1.0; -1.0 1.0]
    @test cb.re ≈ cb.ke * ue[:,2]

    # Neumann bcs
    ue = zeros(ndofs_per_cell(dh, 1), 2)
    cb.re .= 0.0
    cb.ke .= 0.0
    nbc = Plastoxid.Neumann((x,t)->2e4, getfaceset(grid, "right"), :u, 0.0)
    assemble_cell!(cb, problem, ue, states, states_temp, (nbc,))

    @test cb.ke ≈ material.E*A/L * [1.0 -1.0; -1.0 1.0]
    @test cb.re[1] ≈ 0.0
    @test cb.re[2] ≈ -A*2e4
end
