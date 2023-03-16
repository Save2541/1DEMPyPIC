# TURN ELECTROMAGNETIC ON OR OFF
is_electromagnetic = True

# SIMULATION GRID SPECIFICATIONS
ng = 1024 #8192  # number of grids (please use powers of 2 e.g. 4, 8, 1024)
nt = 2048 #33554432 #131072  # number of time steps to run (please use powers of 2 e.g. 4, 8, 1024)
dx = 1  # grid size in terms of the Debye length (dx <= 1 for accurate result)
dt = 1  # time step in terms of dx / c (MUST BE 1 FOR EM)

# EXTERNAL MAGNETIC FIELD SPECIFICATIONS (VECTOR ON X-Z PLANE)
b0 = 5E-8  # 10 * 0.01E-5  # magnetic field strength (T)
theta = 17.5  # magnetic field angle (deg) (e.g., 0 is along +z-axis, 90 is along +x-axis)

# EXTERNAL ELECTRIC FIELD SPECIFICATIONS (Y-DIRECTION)
e_ext = 0  # electric field strength (V/m)

# PLASMA CONFIGURATION (CHOOSE FROM EXISTING PRESETS OR MAKE YOUR OWN CONFIGURATIONS IN plasma_cauldron.py)
preset = 3

# PLOT SPECIFICATIONS
n_sample = ng * 10  # number of particles to plot per specie
nt_sample = 1024 #2048  # ng * 32  # how many time steps to store (use powers of 2)

# OUTPUT SPECIFICATIONS
output_names = ["x", "v", "ex", "ey", "ez", "by", "bz", "rho", "jy", "jz"]