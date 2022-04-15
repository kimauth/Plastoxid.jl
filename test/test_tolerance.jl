@testset "tolerance scaling" begin
    r = collect(1.0:10.0)

    tol_scaling = AbsoluteTolerance(1.0)
    @test Plastoxid.scaled_tolerance(tol_scaling, r) == 1.0

    tol_scaling = ReactionScaling(1.0, 0.1)
    @test Plastoxid.scaled_tolerance(tol_scaling, r) == norm(r)*tol_scaling.base_tol
    @test Plastoxid.scaled_tolerance(tol_scaling, zeros(10)) == tol_scaling.min_tol
end

@testset "tolerance" begin
    # reuse simple tolerance scaling for tolerance tests
    tol_scaling = AbsoluteTolerance(1.0)

    # GlobalTolerance
    t = GlobalTolerance(tol_scaling)
    update_tolerance!(t, rand(10))
    @test t.tol == 1.0
    @test check_convergence([10., 0.1], t)
    @test !check_convergence([10., 10.], t)

    # FieldTolerance
    # assume dofs 1 & 2 belonged to a field and dofs 3 & 4 belonged to another field
    fielddofs = Dict(:u=>[1,2], :v=>[3,4])
    t = FieldTolerance(Dict(:u=>tol_scaling, :v=>tol_scaling), fielddofs)
    f = ones(4)*0.1
    update_tolerance!(t, f)
    residuals = Dict(key=>[10., 0.1] for key in keys(fielddofs))
    @test check_convergence(residuals, t)
    residuals = Dict(key=>[10., 10.] for key in keys(fielddofs))
    @test !check_convergence(residuals, t)
end