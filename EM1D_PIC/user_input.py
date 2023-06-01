# TURN ELECTROMAGNETIC ON OR OFF
is_electromagnetic = True

# SIMULATION GRID SPECIFICATIONS
ng = 1024 #8192  # number of grids (please use powers of 2 e.g. 4, 8, 1024)
nt = 16384 #1048576 #536870912 # number of time steps to run (please use powers of 2 e.g. 4, 8, 1024)
dx = 1 #0.503891 # grid size in terms of the Debye length (dx <= 1 for accurate result)
dt = 1  # time step in terms of dx * Debye length / c (MUST BE 1 FOR EM)

# EXTERNAL MAGNETIC FIELD SPECIFICATIONS (VECTOR ON X-Z PLANE)
b0 = 0 #5E-8  # 10 * 0.01E-5  # magnetic field strength (T)
theta = 0 #90  # magnetic field angle (deg) (e.g., 0 is along +z-axis, 90 is along +x-axis)

# EXTERNAL ELECTRIC FIELD SPECIFICATIONS (Y-DIRECTION)
e_ext = 0  # electric field strength (V/m)

# PLASMA CONFIGURATION (CHOOSE FROM EXISTING PRESETS OR MAKE YOUR OWN CONFIGURATIONS IN plasma_cauldron.py)
preset = 5

# PLOT SPECIFICATIONS
n_sample = 2 #ng * 10  # number of particles to plot per specie
nt_sample = 16384 #2048  # ng * 32  # how many time steps to store (use powers of 2)

# OUTPUT SPECIFICATIONS
output_names = ["x", "v", "ex", "ey", "ez", "by", "bz", "rho", "jy", "jz", "den"]
