@testset "do timestep" begin
    dim = 1
    dim_type = UniaxialStress()
    A = 1.0

    grid = generate_grid(Line, (2,))

    ip = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(2)
    qr_face = QuadratureRule{dim-1, RefCube}(2)

    dh = MixedDofHandler(grid)
    push!(dh, :u, 1, ip)
    close!(dh)

    material = Plastic(E=200e3, ν=0.3, σ_y=200., H=50., r=0.5, κ_∞=13., α_∞=13.)

    problem = MechanicalEquilibrium([material], dim_type, A)

    cb = cellbuffer(problem, ip, ip, qr, qr_face)

    material_mapping = Plastoxid.preprocess_material_mapping([Set(1:getncells(grid))])

    feset = Plastoxid.FESet(Set(1:getncells(grid)), grid, problem, cb, material_mapping, 1)

    vars = zeros(ndofs(dh), 3)

    bc_left = Dirichlet(:u, getfaceset(grid, "left"), (x,t)->0.0)
    bc_right = Dirichlet(:u, getfaceset(grid, "right"), (x,t)->0.01)
    ch = ConstraintHandler(dh)
    add!(ch, bc_left)
    add!(ch, bc_right)
    close!(ch)

    nbcs = ()

    K = create_sparsity_pattern(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)


    tol_scaling = AbsoluteTolerance(1e-8)
    tol = GlobalTolerance(tol_scaling)
    solver = Plastoxid.Newton(20, 5, tol)

    initial_guess = NoGuess()

    time_points = Plastoxid.CircularBuffer{Float64}(3)
    append!(time_points, range(0.0, 2.0; length=3))

    local_solver_options = Dict{Symbol, Any}()

    jld2_path = "test_do_timestep_jld2_temp"
    jld2_storage = JLD2Storage(jld2_path, 1, 1)

    convergence_path = "test_do_timestep_convergence_temp"
    convergence_stream = open(convergence_path, "w")

    Plastoxid.do_timestep!(vars, [feset], assembler, dh, ch, solver, initial_guess, nbcs,
        time_points, 1, local_solver_options, convergence_stream, jld2_storage)

    close(convergence_stream)
    close(jld2_storage)

    rm(convergence_path; recursive = true) # remove temp directory
    rm(jld2_path; recursive = true) # remove temp directory

    @test feset.states.material_states_temp == feset.states.material_states
    @test vars[:,2] == vars[:,3] != vars[:,1]
end
