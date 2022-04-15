# triangular wave
T = 2π
@test triangular_wave(0.0) == 0.0
@test triangular_wave(T/4) == 1.0
@test triangular_wave(T/2) == 0.0
@test triangular_wave(3T/4) == -1.0
# linear behavior inbetween characteristic points
@test triangular_wave(T/8) ≈ 0.5
@test triangular_wave(T/2+T/8) ≈ -0.5

##############################################################
# holdtime wave
T = 2π
t_hold = π/2
hw(t) = holdtime_wave(triangular_wave, t, T, t_hold)
@test hw(0.0) == 0.0
@test hw(T/4) == triangular_wave(T/4)
@test hw(T/4 + t_hold) == triangular_wave(T/4)
@test hw(T/2 + t_hold) == triangular_wave(T/2)
@test hw(3T/4 + t_hold) == triangular_wave(3T/4)
@test hw(3T/4 + 2t_hold) == triangular_wave(3T/4)
@test hw(T + 2t_hold) == triangular_wave(T)

# holdtime wave with streched cycle_period
T = 10.
t_hold = 2.
hw(t) = holdtime_wave(triangular_wave, t, T, t_hold)
@test hw(0.0) == 0.0
@test hw(T/4) == triangular_wave(π/2)
@test hw(T/4 + t_hold) == triangular_wave(π/2)
@test hw(T/2 + t_hold) == triangular_wave(π)
@test hw(3T/4 + t_hold) == triangular_wave(3π/2)
@test hw(3T/4 + 2t_hold) == triangular_wave(3π/2)
@test hw(T + 2t_hold) == triangular_wave(2π)