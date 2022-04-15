@testset "run simulation" begin
    dim = 1
    dim_type = UniaxialStrain()
    A = 1.0

    grid = generate_grid(Line, (10,))

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

    f1(x) = x
    t_scale1 = 2.0
    nsteps1 = 5
    lti = BasicTimeIterator_linear(f1, t_scale1, nsteps1)

    bc_left = Dirichlet(:u, getfaceset(grid, "left"), (x,t)->0.0)
    bc_right = Dirichlet(:u, getfaceset(grid, "right"), (x,t)->0.002*t/t_scale1)
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

    initial_guess = TimeBasedGuess()

    local_solver_options = Dict{Symbol, Any}()

    jld2_path = "test_run_simulation_temp"
    mkdir(jld2_path) # temp directory for writing results

    run_simulation(
        vars,
        [feset],
        assembler,
        dh,
        ch,
        solver,
        initial_guess,
        nbcs,
        lti,
        local_solver_options,
        jld2_path,
    )

    rm(jld2_path; recursive = true) # remove temp directory

end