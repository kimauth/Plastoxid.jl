## BasicTimeIterator
# Linear loading
f1(x) = x
t_scale1 = 2.0
nsteps1 = 5

lti = BasicTimeIterator_linear(f1, t_scale1, nsteps1)
@test all(collect(lti) .≈ range(0.0, t_scale1; length=nsteps1+1)[2:end])

# simple cyclic loading
f2 = triangular_wave
t_scale2 = 1.0
nsteps2 = 8
cti = BasicTimeIterator_cyclic(f2, t_scale2, nsteps2, nsteps2)
@test all(collect(cti) .≈ range(0.0, t_scale2; length=nsteps2+1)[2:end])

## CyclicHoldTimeIterator
# time iterator parameters 
f_cyclic = triangular_wave
nsteps_per_cycle = 8
base_period = 2π

f_hold1(x) = x
nsteps_per_hold1 = 3
sample_bounds1 = (0.0, 3.0)

cycle_period = 1.0
hold_time1 = π / 2
hold_time2 = π / 2

mean_load = 5.0
load_amplitude = 2.0

f_load = sin

# construct loading curve and compute parameters for initial loading
hw(t) = cyclic_hold_loading(f_load, t, cycle_period, hold_time1, hold_time2; mean_load, load_amplitude)
time_init, y0 = initial_loading_params(hw)
@test y0 == mean_load
@test hw(cycle_period/4) ≈ mean_load + load_amplitude

# construct complete loading function
loading(t) = full_loading(hw, t, time_init, y0)
@test loading(time_init+cycle_period/4) ≈ mean_load + load_amplitude

# simplified cases for testing
# no cycling, only loading
ti = CyclicHoldTimeIterator(
    ; 
    ncycles = 0,
    f_cyclic,
    nsteps_per_cycle,
    f_hold1,
    nsteps_per_hold1,
    sample_bounds1,
    cycle_period,
    hold_time1,
    hold_time2,
    time_init
)

@test all(collect(ti) .≈ range(0.0, time_init; length = 5)[2:end])

# no intial loading, no hold times
ti = CyclicHoldTimeIterator(
    ; 
    ncycles = 1,
    f_cyclic,
    nsteps_per_cycle,
    f_hold1,
    nsteps_per_hold1 = 0,
    sample_bounds1,
    cycle_period,
    hold_time1 = 0.0,
    hold_time2 = 0.0,
    time_init = 0.0,
)

@test collect(ti) == collect(cti)

# interrupt time iterator after nsteps
ti_nsteps = CyclicHoldTimeIterator(
    ; 
    ncycles = 1,
    f_cyclic,
    nsteps_per_cycle,
    f_hold1,
    nsteps_per_hold1 = 0,
    sample_bounds1,
    cycle_period,
    hold_time1 = 0.0,
    hold_time2 = 0.0,
    time_init = 0.0,
    maxnsteps = 3,
)
@test length(ti_nsteps) == 3
@test collect(ti_nsteps) == collect(ti)[1:3]

# # show results
# using Plots
# plot(loading, range(0.0, 15.0; length = 201))
# scatter!(loading, collect(ti))
# plot!(x->mean_load)