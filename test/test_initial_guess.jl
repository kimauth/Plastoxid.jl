@testset "intitial guess" begin
    vars = Matrix{Float64}(undef, 5, 3)
    @views fill!(vars[:, 1], 1.0)
    @views fill!(vars[:, 2], 2.0)

    time_points = [0.0, 1.0, 2.5]

    guess = NoGuess()
    initial_guess!(vars, guess, time_points)
    @test vars[:, end] ≈ vars[:, end-1]

    time_guess = TimeBasedGuess()
    vars_time = copy(vars)
    initial_guess!(vars_time, time_guess, time_points)
    @test vars_time[:, end] ≈ vars[:,2] + (vars[:,2] - vars[:,1]) * (time_points[end] - time_points[end-1])*(time_points[end-1]-time_points[end-2])

    function_guess = FunctionBasedGuess(identity)
    vars_function = copy(vars)
    initial_guess!(vars_function, function_guess, time_points)
    @test vars_time ≈ vars_function
end