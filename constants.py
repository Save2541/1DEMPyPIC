import math

# PHYSICAL CONSTANTS (MKS UNITS)
c = 3.00E8  # speed of light
mu = 1.26E-6  # vacuum permeability
epsilon = 1 / mu / c ** 2  # 8.85E-12  # vacuum permittivity
me_real = 9.11E-31  # electron mass
mp_real = 1.67E-27  # proton mass
qe_real = 1.60E-19  # elementary charge
kb = 1.38E-23  # Boltzmann constant
sqrt_mu_over_epsilon = math.sqrt(mu / epsilon)  # square root mu naught over epsilon naught

# SET SMOOTHING WEIGHT
w_smooth = 0.5
