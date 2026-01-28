# Case name
case =  "lake_at_rest_immersed_bump"

# Gravity
const grav = 9.81

# Smoothing kernel function 
                    #skf = 1, cubic spline kernel by W4 - Spline (Monaghan 1985)
                    #    = 2, Gauss kernel   (Gingold and Monaghan 1981) 
                    #    = 3, Quintic kernel (Morris 1997)
const skf = 1

# maximum timestep allowed
const dt_max = 10.0

# Courant number
const CFL = 0.4

# Print interval (seconds)
const print_step = 10.0

# End of the simulation (seconds)
const t_end = 100.0

# Smoothing length coefficient: hsml=dx*coef
const coef = 1.0

