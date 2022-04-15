struct BasicTimeIterator{F}
    f::F
    t_scale::Float64
    nsteps::Int
    Δx::Float64
end

Base.length(bti::BasicTimeIterator) = bti.nsteps
Base.eltype(::Type{<:BasicTimeIterator}) = Float64

mutable struct BasicTimeState
    step::Int
    x::Float64
    t::Float64
end

start_state(::BasicTimeIterator) = BasicTimeState(1, 0.0, 0.0)

function update_time_state!(state, ti)
    x_new = state.x + ti.Δx
    state.t += abs(ti.f(x_new) - ti.f(state.x)) * ti.t_scale
    state.step += 1
    state.x = x_new
    return state
end

function Base.iterate(ti::BasicTimeIterator, state::BasicTimeState=start_state(ti))
    if state.step > ti.nsteps
        return nothing
    else
        update_time_state!(state, ti)
        return state.t, state
    end
end


"""
    BasicTimeIterator_cyclic(f, t_scale, nsteps_per_cycle, nsteps[; base_period=2π])

Return a `BasicTimeIterator` that is based on a cyclic function `f` with period `base_period`.

`f` is normalized such that iterating over a single cycle returns time values between
0 and t_scale.

# Arguments
- `f`: A cyclic function. Values for the time are sampled out of this function in regular intervals.
- `t_scale`: Period of the output.
- `nsteps_per_cycle`: Number of steps sampled from each cycle of `f`. Must be divadable by 4.
- `nsteps`: Total number of steps that the iterator takes. 
"""
function BasicTimeIterator_cyclic(f, t_scale, nsteps_per_cycle, nsteps; base_period=2π)
    abs(f(0.0) - f(base_period)) < 1e-12 || error("Function f is not period in $base_period.")
    nsteps_per_cycle % 4 == 0 || error("The total number of steps per period must be divadable by 4.")
    # find accumulated y-range over period (usually 4.0)
    y_accum = 0.0
    for i = 1:4
        y_accum += abs(f(i*base_period/4)- f((i-1)*base_period/4))
    end
    Δx = base_period / nsteps_per_cycle
    return BasicTimeIterator(x->f(x)/y_accum, t_scale, nsteps, Δx)
end

"""
    BasicTimeIterator_linear(f, t_scale, nsteps[; sample_bounds=(0.0, 1.0)])

Return a `BasicTimeIterator` that is based on the values of function `f(x)` in the interval
`x ∈ sample_bounds`.

`f` is normalized such that the time iterator returns time values between 0 and t_scale.

# Arguments
- `f`: A function. Values for the time are sampled out of this function in regular intervals.
- `t_scale`: Maximum time of the output.
- `nsteps`: Total number of steps that the iterator takes. 
"""
function BasicTimeIterator_linear(f, t_scale, nsteps; sample_bounds=(0.0, 1.0))
    x_stretch = sample_bounds[2] - sample_bounds[1]
    y_stretch = f(sample_bounds[2]) - f(sample_bounds[1])
    f_out(x) = f(x*x_stretch) / y_stretch
    Δx = 1.0 / nsteps
    return BasicTimeIterator(f_out, t_scale, nsteps, Δx)
end

## cyclic loading + hold time, time sampling
struct CyclicHoldTimeIterator{TI0, TI1, TI2, TI3}
    # linear BasicTimeIterator for initial loading
    initial_lti::TI0 
    # cyclic BasicTimeIterator
    cti::TI1 
    # linear BasicTimeIterators
    lti1::TI2
    lti2::TI3
    # total number of cycles
    ncycles::Int # total number of cycles with hold time
    maxnsteps::Union{Nothing, Int} # stop after this many steps
end

nsteps_all_cycles(ti::CyclicHoldTimeIterator) = ti.initial_lti.nsteps + (ti.cti.nsteps*4 + ti.lti1.nsteps + ti.lti2.nsteps)*ti.ncycles
Base.length(ti::CyclicHoldTimeIterator) = isnothing(ti.maxnsteps) ? nsteps_all_cycles(ti) : ti.maxnsteps
Base.eltype(::Type{<:CyclicHoldTimeIterator}) = Float64

"""
    CyclicHoldTimeIterator(; kwargs...)

Return a `CyclicHoldTimeIterator`. The iterator returns the time points for loading to a
mean load followed by cycling around the mean load with a holdtime wave. Time samples for 
cyclic loading/unloading can be taken from a cyclic function, time samples for the hold periods
can be taken from any function. The time steps for the initial loading are chosen as close
as possible to the first load step in the holdtime wave.

# Keyword Arguments
- `ncycles`: Number of full cycles
- `f_cyclic = triangular_wave`: Cyclic function. Values for the loading/unloading period are
sampled from here in regular intervals.
- `base_period = 2π`: Period of `f_cyclic`
- `nsteps_per_cycle`: Number of steps per cycle of `f_cyclic`. Must be divadable by 4.
- `f_hold1 = x->x`: Function to sample the time steps for the first hold period.
- `sample_bounds1`: x-values between which `f_hold1` is sampled
- `nsteps_per_hold1`: Number of steps during the first hold period.
- `f_hold2 = f_hold1`: Function to sample the time steps for the second hold period.
- `sample_bounds2 = sample_bounds2`: x-values between which `f_hold2` is sampled
- `nsteps_per_hold2 = nsteps_per_hold1`: Number of steps during the second hold period.
The following keyword arguments reflect how long the different parts of the loading should take.
These values must align with the employed loading function! See: [`cyclic_hold_loading`](@ref)
- `cycle_period`: The total time that a cycle of `f_cyclic` should take.
- `hold_time1`: The total time the first hold period takes.
- `hold_time2 = hold_time1`: The total time the second hold period takes.
- `time_init = 0.0`: The time that it initially takes to load from zero to the mean load. 
"""
function CyclicHoldTimeIterator(;
    ncycles,
    maxnsteps = nothing,
    # cyclic base function
    f_cyclic = triangular_wave,
    base_period = 2π,
    nsteps_per_cycle,
    # hold period 1 (upper)
    f_hold1 = x->x,
    sample_bounds1 = (0.0, 1.0),
    nsteps_per_hold1,
    # hold period 2 (lower)
    f_hold2 = f_hold1,
    sample_bounds2 = sample_bounds1,
    nsteps_per_hold2 = nsteps_per_hold1,

    # times to stretch the sections by
    cycle_period,
    hold_time1,
    hold_time2 = t_scale_hold1,
    time_init = 0.0,
)
    cti = BasicTimeIterator_cyclic(f_cyclic, cycle_period, nsteps_per_cycle, Int(nsteps_per_cycle/4); base_period)
    lti1 = BasicTimeIterator_linear(f_hold1, hold_time1, nsteps_per_hold1; sample_bounds=sample_bounds1)
    lti2 = BasicTimeIterator_linear(f_hold2, hold_time2, nsteps_per_hold2; sample_bounds=sample_bounds2)

    dt1_cyclic = cti.f(cti.Δx) * cti.t_scale
    nsteps_init = time_init ≈ 0.0 ? 0 : ceil(Int, time_init/dt1_cyclic)
    initial_lti = BasicTimeIterator_linear(x->x, time_init, nsteps_init)
    return CyclicHoldTimeIterator(initial_lti, cti, lti1, lti2, ncycles, maxnsteps)
end

mutable struct CyclicHoldTimeState
    section::Int
    cycle::Int
    cyclic_state::BasicTimeState
    hold_state::BasicTimeState
    total_nsteps::Int
end

start_state(ti::CyclicHoldTimeIterator) = CyclicHoldTimeState(0, 0, start_state(ti.cti), start_state(ti.lti1), 1)

function get_base_iterator(ti::CyclicHoldTimeIterator, state::CyclicHoldTimeState)
    if state.section == 0
        bti = ti.initial_lti
        base_state = state.hold_state
    elseif state.section ∈ (1,3,4,6)
        bti = ti.cti
        base_state = state.cyclic_state
    else
        bti = state.section == 2 ? ti.lti1 : (state.section == 5 ? ti.lti2 : error("Section exceeds limits, section = $(state.section)."))
        base_state = state.hold_state
    end
    return bti, base_state
end

function Base.iterate(ti::CyclicHoldTimeIterator, state::CyclicHoldTimeState=start_state(ti))
    bti, base_state = get_base_iterator(ti, state)
    # update_time_state!(base_state, bti)
    # update section + cycle
    new_bti, new_base_state = bti, base_state
    while new_base_state.step > new_bti.nsteps
        # we're done with this iterator, so reset steps
        base_state.step = 1
        if state.section == 0 # initial loading is over
            state.cycle = 1
            state.section = 1
        elseif state.section < 6
            state.section += 1
        else
            state.section = 1
            state.cycle += 1
        end
        new_bti, new_base_state = get_base_iterator(ti, state)
        # carry time over to next state
        new_base_state.t = base_state.t
        # always reset linear x to zero for new section
        state.hold_state.x = 0.0
    end
    if (!isnothing(ti.maxnsteps) && state.total_nsteps > ti.maxnsteps) || state.cycle > ti.ncycles
        return nothing
    else
        update_time_state!(new_base_state, new_bti)
        state.total_nsteps += 1
        return new_base_state.t, state
    end
end

## TODO: belong in wave_functions.jl
"""
    cyclic_hold_loading(f, t, cycle_period, hold1, hold2[; base_period=2π, mean_load=0.0, load_amplitude=1.0])

Return values of 
"""
function cyclic_hold_loading(f, t, cycle_period, hold1, hold2; base_period=2π, mean_load=0.0, load_amplitude=1.0)

    f_wave(t) = holdtime_wave(f, t, cycle_period, hold1, hold2, base_period)
    y = f_wave(t)*load_amplitude + mean_load

    return y
end

# find values for constructing the CyclicHoldTimeIterator
function initial_loading_params(f)
    y0 = f(0.0)
    dydt = ForwardDiff.derivative(f, 0.0)
    dt = y0/dydt

    return dt, y0
end

function full_loading(f, t, dt_initial, y0_initial)
    if t <= dt_initial
        # could replace this with more complicated initial loading
        return y0_initial * t / dt_initial
    else
        return f(t-dt_initial)
    end
end



