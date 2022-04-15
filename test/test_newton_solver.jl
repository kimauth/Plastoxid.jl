@testset "Newton solver" begin
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

    K = create_sparsity_pattern(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)


    tol_scaling = AbsoluteTolerance(1e-8)
    tol = GlobalTolerance(tol_scaling)
    solver = Plastoxid.Newton(20, 5, tol)

    update!(ch)
    @views apply!(vars[:,3], ch)

    @test iterate!(vars, solver, [feset], assembler, dh, ch)
    convergence_rates = Plastoxid.get_convergence_rates(solver.residuals)
    @test convergence_rates[end] > 1.9

    @test vars[2,end] ≈ 1/2*(vars[1,end]+vars[3,end])
end