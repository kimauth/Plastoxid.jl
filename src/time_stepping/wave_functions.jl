function triangular_wave(x::T) where T
    p = 2π # period
    y = 4.0/p * (x - p/2*floor(2x/p + 0.5))*(-1.0)^floor(2*x/p + 0.5)
    return y
end

function holdtime_wave(f, t, period=2π, t_hold_top=1.0, t_hold_bottom=t_hold_top, base_period=2π)
    T = period
    cycle_length = T + t_hold_top + t_hold_bottom
    t̄ = t - floor(t/cycle_length) * cycle_length # how much we are into the cycle

    if t̄ < T/4
        return f(t̄/T*base_period)
    elseif t̄ < T/4 + t_hold_top
        return f(base_period/4)
    elseif t̄ < 3T/4 + t_hold_top
        return f((t̄ - t_hold_top)/T*base_period)
    elseif t̄ < 3T/4 + t_hold_top + t_hold_bottom
        return f(3/4*base_period)
    elseif t̄ <= cycle_length
        return f((t̄ - t_hold_top - t_hold_bottom)/T*base_period)
    else
        error("Time exceeds cycle length")
    end
end