import io
import math
import time
from datetime import timedelta
import numpy
import scipy.fft
import zarr

# GET STARTING TIME
start_time = time.monotonic()

# PHYSICAL CONSTANTS (MKS UNITS)
epsilon = 8.85E-12  # vacuum permittivity
mu = 1.26E-6  # vacuum permeability
c = math.sqrt(1 / mu / epsilon)  # speed of light
me_real = 9.11E-31  # electron mass
mi_real = 100 * me_real  # 1.67E-27  # ion mass (proton)
qe_real = 1.60E-19  # (-) electron charge
qi_real = qe_real  # ion (proton) charge
qm = 1.76E11  # electron charge-to-mass ratio (q/me)
qmi = qi_real / mi_real  # proton charge-to-mass ratio (q/mi)
kb = 1.38E-23  # Boltzmann constant

# PLASMA SPECIFICATIONS
n0 = 1E8  # plasma electron density (1/m^3)
n0i = n0  # plasma ion density (1/m^3)
te = 1E4  # electron temperature (K)
ti = te  # ion temperature (K)

# EXTERNAL MAGNETIC FIELD SPECIFICATIONS (VECTOR ON X-Z PLANE)
b0 = 0.01E-5  # magnetic field strength (T)
theta = math.pi / 2  # magnetic field angle (rad) (e.g., 0 is along +z-axis, Pi/2 is along +x-axis)
sin_theta = math.sin(theta)
cos_theta = math.cos(theta)

# EXTERNAL ELECTRIC FIELD SPECIFICATIONS (Y-DIRECTION)
e_ext = 0  # electric field strength (V/m)

# DERIVED QUANTITIES
bz0 = b0 * cos_theta  # magnetic field along z (T)
bx0 = b0 * sin_theta  # magnetic field along x (T)
wp = math.sqrt(n0 * qm * qe_real / epsilon)  # electron plasma frequency (Hz)
wpi = math.sqrt(n0i * qmi * qi_real / epsilon)  # ion plasma frequency (Hz)
wc = qm * b0  # electron cyclotron frequency (Hz)
wci = qmi * b0  # ion cyclotron frequency (Hz)
kte = kb * te  # electron energy kT (J)
kti = kb * ti  # ion energy kT (J)
v_th = math.sqrt(2 * kte / me_real)  # electron thermal velocity (m/s)
vi_th = math.sqrt(2 * kti / mi_real)  # ion thermal velocity (m/s)
lambda_d = v_th / math.sqrt(2) / wp  # Debye length (m)
rho_mass = n0 * me_real + n0i * mi_real  # mass density

# SIMULATION SPECIFICATIONS
ng = 4096  # number of grids (please use powers of 2 e.g. 4, 8, 1024)
nt = 8 * 16384  # number of time steps to run
ne = ng * 10  # ng * 100  # number of PIC electrons
ni = ne  # number of PIC ions
np = ne + ni  # number of PIC particles

# CREATE LOG (TEXT FILE)
if ni > 0:
    name = "ni{:.0e}_ti{:.0e}_b0{:.0e}_theta{}".format(n0i, ti, b0, int(theta * 180 / math.pi))  # name string
else:
    name = "ne{:.0e}_te{:.0e}_b0{:.0e}_theta{}".format(n0, te, b0, int(theta * 180 / math.pi))  # name string

log = open("{}.txt".format(name), "w")
print("---------------------------PHYSICAL CONSTANTS---------------------------", file=log)
print("Vacuum permittivity                              = {:.2e} F m^-1".format(epsilon), file=log)
print("Vacuum permeability                              = {:.2e} H m^-1".format(mu), file=log)
print("Speed of light                                   = {:.2e} m s^-1".format(c), file=log)
print("Electron charge-to-mass ratio                    = -{:.2e} C/kg".format(qm), file=log)
if ni > 0:
    print("Ion charge-to-mass ratio                         = {:.2e} C/kg".format(qmi), file=log)
print("Boltzmann constant                               = {:.2e} J K^-1".format(kb), file=log)
print("---------------------------INITIAL CONDITIONS---------------------------", file=log)
print("Plasma electron density                          = {:.2e} m^-3".format(n0), file=log)
print("Electron temperature                             = {:.2e} K".format(te), file=log)
if ni > 0:
    print("Plasma ion density                               = {:.2e} m^-3".format(n0i), file=log)
    print("Ion temperature                                  = {:.2e} K".format(ti), file=log)
print("x-Magnetic field                                 = {:.2e} T".format(bx0), file=log)
print("z-Magnetic field                                 = {:.2e} T".format(bz0), file=log)
print("y-Electric field                                 = {:.2e} V/m".format(e_ext), file=log)
print("---------------------------DERIVED QUANTITIES---------------------------", file=log)
print("Electron plasma frequency                        = {:.2e} Hz".format(wp), file=log)
print("Electron cyclotron frequency                     = {:.2e} Hz".format(wc), file=log)
print("Electron energy (kT)                             = {:.2e} J".format(kte), file=log)
print("Electron thermal velocity                        = {:.2e} m/s".format(v_th), file=log)
print("Debye length                                     = {:.2e} m".format(lambda_d), file=log)
if ni > 0:
    print("Ion plasma frequency                             = {:.2e} Hz".format(wpi), file=log)
    print("Ion cyclotron frequency                          = {:.2e} Hz".format(wci), file=log)
    print("Ion energy (kT)                                  = {:.2e} J".format(kti), file=log)
    print("Ion thermal velocity                             = {:.2e} m/s".format(vi_th), file=log)

# DEFINITION OF UNIT QUANTITIES
scu = c ** 2 * epsilon * lambda_d / qm  # simulation unit charge (C)
slu = lambda_d  # simulation unit length (m)
stu = lambda_d / c  # simulation unit time (s)
smu = scu / qm  # simulation unit mass (kg)

# SCALED PHYSICAL CONSTANTS
epsilon = 1  # scaled vacuum permittivity
c = 1  # scaled speed of light
mu = 1 / c ** 2 / epsilon  # scaled vacuum permeability
qm = 1  # scaled electron charge-to-mass ratio (q/me)
qmi = qmi * smu / scu  # scaled ion charge-to-mass ratio (q/mi)

# SCALED DERIVED QUANTITIES
wp = wp * stu  # scaled electron plasma frequency
wpi = wpi * stu  # scaled ion plasma frequency
wc = wc * stu  # scaled electron cyclotron frequency
wci = wci * stu  # scaled ion cyclotron frequency
kte = kte * stu ** 2 / smu / slu ** 2  # scaled electron energy kT (J)
kti = kti * stu ** 2 / smu / slu ** 2  # scaled ion energy kT (J)
v_th = v_th * stu / slu  # scaled electron thermal velocity
vi_th = vi_th * stu / slu  # scaled ion thermal velocity
lambda_d = lambda_d / slu  # scaled Debye length (m)

# SCALED EXTERNAL MAGNETIC FIELD
b0 = wc / qm  # field strength
bz0 = b0 * cos_theta  # field component along z
bx0 = b0 * sin_theta  # field component along x

# SCALED EXTERNAL ELECTRIC FIELD
e_ext = e_ext * stu ** 2 * scu / smu / slu  # field strength

# print text to log
print("---------------------------SIMULATION UNITS---------------------------", file=log)
print("simulation unit charge                           = {:.2e} C".format(scu), file=log)
print("simulation unit length                           = {:.2e} m".format(slu), file=log)
print("simulation unit time                             = {:.2e} s".format(stu), file=log)
print("simulation unit mass                             = {:.2e} kg".format(smu), file=log)
print("---------------------------PHYSICAL CONSTANTS IN SIMULATION UNITS---------------------------", file=log)
print("Vacuum permittivity                              = {:.2e}".format(epsilon), file=log)
print("Vacuum permeability                              = {:.2e}".format(mu), file=log)
print("Speed of light                                   = {:.2e}".format(c), file=log)
print("Electron charge-to-mass ratio                    = -{:.2e}".format(qm), file=log)
if ni > 0:
    print("Ion charge-to-mass ratio                         = {:.2e}".format(qmi), file=log)
print("---------------------------DERIVED QUANTITIES IN SIMULATION UNITS---------------------------", file=log)
print("Electron energy (kT)                             = {:.2e}".format(kte), file=log)
print("Electron plasma frequency                        = {:.2e}".format(wp), file=log)
print("Electron cyclotron frequency                     = {:.2e}".format(wc), file=log)
print("Electron thermal velocity                        = {:.2e}".format(v_th), file=log)
print("Debye length                                     = {:.2e}".format(lambda_d), file=log)
if ni > 0:
    print("Ion energy (kT)                                  = {:.2e}".format(kti), file=log)
    print("Ion plasma frequency                             = {:.2e}".format(wpi), file=log)
    print("Ion cyclotron frequency                          = {:.2e}".format(wci), file=log)
    print("Ion thermal velocity                             = {:.2e}".format(vi_th), file=log)
print("x-Magnetic field                                 = {:.2e}".format(bx0), file=log)
print("z-Magnetic field                                 = {:.2e}".format(bz0), file=log)
print("y-Electric field                                 = {:.2e}".format(e_ext), file=log)

# GRID SIZES
dx = lambda_d  # spatial grid size
length = ng * dx  # length of the system
dt = dx / c  # duration of time step

# PROPERTIES OF PIC PARTICLES
qe = wp ** 2 * length / ne * epsilon / qm  # charge per PIC particle
me = qe / qm
if ni > 0:
    qi = wpi ** 2 * length / ni * epsilon / qmi
    mi = qi / qmi

# print text to log
print("---------------------------SIMULATION GRID PROPERTIES---------------------------", file=log)
print("number of time steps                             = {}".format(nt), file=log)
print("number of spatial grids                          = {}".format(ng), file=log)
print("duration of one time step (in simulation units)  = {:.2e}".format(dt), file=log)
print("duration of one time step (in mks)               = {:.2e} s".format(dt * stu), file=log)
print("length of one spatial grid (in simulation units) = {:.2e}".format(dx), file=log)
print("length of one spatial grid (in mks)              = {:.2e} m".format(dx * slu), file=log)
print("simulation length (in simulation units)          = {:.2e}".format(length), file=log)
print("simulation length (in mks)                       = {:.2e} m".format(length * slu), file=log)
print("simulation time (in simulation units)            = {:.2e}".format(dt * nt), file=log)
print("simulation time (in mks)                         = {:.2e} s".format(dt * nt * stu), file=log)
print("---------------------------SIMULATION PARTICLE PROPERTIES---------------------------", file=log)
print("number of simulation particles                   = {}".format(np), file=log)
print("number of simulation electrons                   = {}".format(ne), file=log)
print("number of simulation ions                        = {}".format(ni), file=log)
if ne > 0:
    print("simulation electron charge (in simulation units) = -{:.2e}".format(qe), file=log)
    print("simulation electron charge (in mks)              = -{:.2e} C".format(qe * scu), file=log)
    print("simulation electron mass (in simulation units)   = {:.2e}".format(me), file=log)
    print("simulation electron mass (in mks)                = {:.2e} kg".format(me * smu), file=log)
if ni > 0:
    print("simulation ion charge (in simulation units)      = {:.2e}".format(qi), file=log)
    print("simulation ion charge (in mks)                   = {:.2e} C".format(qi * scu), file=log)
    print("simulation ion mass (in simulation units)        = {:.2e}".format(mi), file=log)
    print("simulation ion mass (in mks)                     = {:.2e} kg".format(mi * smu), file=log)

# check whether v_th < c or not
assert v_th < c, "ELECTRON THERMAL VELOCITY GREATER THAN THE SPEED OF LIGHT!"
assert vi_th < c, "ION THERMAL VELOCITY GREATER THAN THE SPEED OF LIGHT!"

# PLOT SPECIFICATIONS
n_sample = min(ne, ni, ng)  # number of particles to plot per specie
nt_sample = ng * 16  # how many time steps to store (use powers of 2)
dt_sample = dt * nt / nt_sample  # duration per sample


# DEFINE CLASS TO STORE GRID DATA
class GridPointList:
    def __init__(self, x, rho=numpy.zeros(ng), jy_old=numpy.zeros(ng), jy=numpy.zeros(ng), jz_old=numpy.zeros(ng),
                 jz=numpy.zeros(ng), ex=numpy.zeros(ng), ey=numpy.zeros(ng), ez=numpy.zeros(ng),
                 by=numpy.zeros(ng),
                 bz=numpy.zeros(ng) + bz0,
                 phi=numpy.zeros(ng),
                 f_right_old=numpy.zeros(ng) + (epsilon * e_ext + math.sqrt(epsilon / mu) * bz0) / 2,
                 f_left_old=numpy.zeros(ng) + (epsilon * e_ext - math.sqrt(epsilon / mu) * bz0) / 2,
                 f_right=numpy.zeros(ng) + (epsilon * e_ext + math.sqrt(epsilon / mu) * bz0) / 2,
                 f_left=numpy.zeros(ng) + (epsilon * e_ext - math.sqrt(epsilon / mu) * bz0) / 2,
                 g_right_old=numpy.zeros(ng) + 0.0,
                 g_left_old=numpy.zeros(ng) + 0.0,
                 g_right=numpy.zeros(ng) + 0.0, g_left=numpy.zeros(ng) + 0.0):
        self.x = x  # position of the grid point
        self.n = x / length * ng  # grid number
        self.rho = rho  # charge density
        self.jy_old = jy_old  # previous current density y
        self.jy = jy  # current current density y
        self.jz_old = jz_old  # previous current density z
        self.jz = jz  # current current density z
        self.ex = ex  # electric field x
        self.ey = ey  # electric field y
        self.ez = ez  # electric field z
        self.by = by  # magnetic field y
        self.bz = bz  # magnetic field z (or direction b0 if theta != 0)
        self.phi = phi  # electric potential
        self.f_right_old = f_right_old  # right going field quantity F, previous
        self.f_left_old = f_left_old  # left going field quantity F, previous
        self.f_right = f_right  # right going field quantity F, current
        self.f_left = f_left  # left going field quantity F, current
        self.g_right_old = g_right_old  # right going field quantity G, previous
        self.g_left_old = g_left_old  # left going field quantity G, previous
        self.g_right = g_right  # right going field quantity G, current
        self.g_left = g_left  # left going field quantity G, current

    # Field quantities: F_right/left = 1/2*[Ey (+/-) Bz], G_right/left = 1/2*[Ez (-/+) By]

    # PRINT INFO
    def print(self, t):
        # PICK EXAMPLE GRID TO PRINT
        index = 1
        print(
            "Time = {}. Grid: {}. rho = {:.2e}. j = ({:.2e}, {:.2e}). e = ({:.2e}, {:.2e}, {:.2e}). b = ({:.2e}, "
            "{:.2e}, {:.2e}). (f_left, f_right) = ({:.2e}, {:.2e}). (g_left, g_right) = ({:.2e}, {:.2e}).".format(
                t, int(self.n[index]), self.rho[index], self.jy[index], self.jz[index], self.ex[index], self.ey[index],
                self.ez[index], bx0, self.by[index], self.bz[index],
                self.f_left[index], self.f_right[index], self.g_left[index], self.g_right[index]))


# DEFINE CLASS TO STORE PARTICLE DATA
class ParticleList:
    def __init__(self, x, vy=numpy.zeros(ne), vxp=numpy.zeros(ne), vb0=numpy.zeros(ne),
                 x_old=numpy.zeros(ne), vx_old=numpy.zeros(ne), vy_old=numpy.zeros(ne),
                 vxp_old=numpy.zeros(ne), vb0_old=numpy.zeros(ne), left_grid=numpy.zeros(ne),
                 left_grid_old=numpy.zeros(ne)):
        self.x = x  # position at current time
        self.vx = vxp * cos_theta + vb0 * sin_theta  # vx at current time
        self.vx_old = vx_old  # vx from the previous time step
        self.vy = vy  # vy at current time
        self.vb0 = vb0  # vb0 at current time (parallel to b0)
        self.vxp = vxp  # vx' at current time (perpendicular to b0)
        self.x_old = x_old  # position from the previous time step
        self.vy_old = vy_old  # vy from the previous time step
        self.vxp_old = vxp_old  # vxp from the previous time step
        self.vb0_old = vb0_old  # vb0 from the previous time step
        self.v = numpy.sqrt(vxp ** 2 + vy ** 2 + vb0 ** 2)
        self.v_old = numpy.sqrt(vxp_old ** 2 + vy_old ** 2 + vb0_old ** 2)
        self.left_grid = left_grid  # nearest left grid
        self.left_grid_old = left_grid_old  # old nearest left grid

    # FUNCTION TO UPDATE THE NEAREST LEFT GRIDS OF THE PARTICLES
    def update_nearest_grid(self):
        floored = numpy.floor(self.x / dx).astype(int)
        self.left_grid_old = self.left_grid
        self.left_grid = floored

    # FUNCTION TO OUTPUT THE NEAREST LEFT AND RIGHT GRIDS OF THE PARTICLES
    def nearest_grids(self):
        left = self.left_grid
        right = left + 1
        right[right == ng] = 0
        return left, right

    # FUNCTION TO OUTPUT THE NEAREST OLD LEFT AND RIGHT GRIDS OF THE PARTICLES
    def nearest_grids_old(self):
        left_old = self.left_grid_old
        right_old = left_old + 1
        right_old[right_old == ng] = 0
        return left_old, right_old


# DEFINE CLASS TO STORE PIC ELECTRON DATA
class ElectronList(ParticleList):
    def __init__(self, x, vy=numpy.zeros(ne), vxp=numpy.zeros(ne), vb0=numpy.zeros(ne),
                 x_old=numpy.zeros(ne), vx_old=numpy.zeros(ne), vy_old=numpy.zeros(ne),
                 vxp_old=numpy.zeros(ne),
                 vb0_old=numpy.zeros(ne), left_grid=numpy.zeros(ne), left_grid_old=numpy.zeros(ne)):
        super().__init__(x, vy, vxp, vb0, x_old, vx_old, vy_old, vxp_old, vb0_old, left_grid, left_grid_old)
        self.q = -qe  # PIC electron charge
        self.m = me  # PIC electron mass
        self.qm = -qm  # electron charge-to-mass ratio
        self.type = "electron"  # PIC electron type


# DEFINE CLASS TO STORE PIC ION DATA
class IonList(ParticleList):
    def __init__(self, x, vy=numpy.zeros(ni), vxp=numpy.zeros(ni), vb0=numpy.zeros(ni),
                 x_old=numpy.zeros(ni), vx_old=numpy.zeros(ni), vy_old=numpy.zeros(ni),
                 vxp_old=numpy.zeros(ni),
                 vb0_old=numpy.zeros(ni), left_grid=numpy.zeros(ni), left_grid_old=numpy.zeros(ni)):
        super().__init__(x, vy, vxp, vb0, x_old, vx_old, vy_old, vxp_old, vb0_old, left_grid, left_grid_old)
        self.q = qi  # PIC ion charge
        self.m = mi  # PIC ion mass
        self.qm = qmi  # ion charge-to-mass ratio
        self.type = "ion"  # PIC ion type


# GENERATE GRID POINTS (FROM LIST OF POSITIONS)
grids = GridPointList(numpy.arange(0, length, dx))

# CONSTRUCT A RANDOM NUMBER GENERATOR
rng = numpy.random.default_rng()

# GENERATE PARTICLES (FROM LIST OF POSITIONS AND VELOCITIES)
if ni == 0:
    species = [ElectronList(rng.uniform(0, length, size=ne),  # uniform position distribution
                            # gaussian velocity distribution
                            vxp=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vy=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vb0=rng.normal(0.0, v_th / math.sqrt(2), size=ne))]
else:
    species = [ElectronList(rng.uniform(0, length, size=ne),  # uniform position distribution
                            # gaussian velocity distribution
                            vxp=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vy=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vb0=rng.normal(0.0, v_th / math.sqrt(2), size=ne)),
               IonList(rng.uniform(0, length, size=ni),  # uniform position distribution
                       # gaussian velocity distribution
                       vxp=rng.normal(0.0, vi_th / math.sqrt(2), size=ni),
                       vy=rng.normal(0.0, vi_th / math.sqrt(2), size=ni),
                       vb0=rng.normal(0.0, vi_th / math.sqrt(2), size=ni))]


def initialization():
    # INITIAL WEIGHTING (x to rho, hat function)

    for specie in species:  # loop for each specie
        # UPDATE NEAREST GRIDS
        specie.update_nearest_grid()
        # GET NEAREST GRIDS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids()
        # GET POSITIONS
        x_left_grid = grids.x[nearest_left_grid]  # get the positions of the nearest left grids
        xi = specie.x  # get particle positions
        coeff = specie.q / dx / dx  # calculate a coefficient to be used
        d_rho_right = coeff * (xi - x_left_grid)  # calculate densities to be assigned to the nearest right grids
        d_rho_left = coeff * dx - d_rho_right  # calculate densities to be assigned to the nearest left grids
        # ADD DENSITIES TO CORRESPONDING GRIDS
        grids.rho = grids.rho + numpy.bincount(nearest_left_grid, weights=d_rho_left, minlength=ng) + numpy.bincount(
            nearest_right_grid, weights=d_rho_right, minlength=ng)

    # INITIAL FIELD SOLVER (Find Ex)
    rho_n_list = dx * scipy.fft.rfft(grids.rho)  # fourier transform, rho(x) to rho(k)
    phi_n_list = numpy.concatenate(
        ([0], rho_n_list[1:] * ksqi_list / epsilon))  # calculate phi(k) from rho(k)
    phi_list = 1 / dx * scipy.fft.irfft(phi_n_list)  # inverse fourier transform, phi(k) to phi(x)
    # CALCULATE E(x) FROM phi(x)
    grids.ex = (numpy.roll(phi_list, 1) - numpy.roll(phi_list, -1)) / 2 / dx

    # INITIAL PARTICLE MOVER (MOVE BACK V FROM t = 0 to t = - dt/2, hat function)
    for specie in species:
        coefficient = specie.qm * (-dt / 2)  # calculate a coefficient to be used
        d_theta = - b0 * coefficient  # calculate rotation angle due to magnetic field
        cos_d_theta = numpy.cos(d_theta)  # cosine of d_theta
        sin_d_theta = numpy.sin(d_theta)  # sine of d_theta
        # GET NEAREST GRIDS
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids()
        # GET POSITIONS
        xi = specie.x  # get particle positions
        x_right_grid = grids.x[nearest_left_grid_point] + dx  # get the positions of nearest right grids
        # CALCULATE SELF-CONSISTENT ELECTRIC FIELD
        e_sc = (x_right_grid - xi) / dx * grids.ex[nearest_left_grid_point] + (
                xi - grids.x[nearest_left_grid_point]) / dx * grids.ex[nearest_right_grid_point]
        # CALCULATE THE THREE VELOCITY COMPONENTS
        specie.vxp_old = (cos_d_theta * specie.vxp - sin_d_theta * specie.vy) + coefficient * e_sc * cos_theta
        specie.vy_old = sin_d_theta * specie.vxp + cos_d_theta * specie.vy + coefficient * e_ext
        specie.vb0_old = specie.vb0 + coefficient * e_sc * sin_theta

    # STORE INITIAL DATA FOR PHASE SPACE PLOT
    for i in range(len(species)):
        x_data[i][0] = species[i].x[plot_particles_id]
        v_data[i][0] = species[i].vx[plot_particles_id]

    # FIRST STEP IN TIME
    move_particles_init()
    weigh_to_grid()
    solve_field()


def move_particles_init():
    """Particle mover for the first time step"""
    for specie in species:  # loop for each specie
        specie.x_old = specie.x  # store previous positions
        q_by_m = specie.qm  # get the charge-per-mass ratio
        # GET NEAREST GRIDS
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids()
        # GET POSITIONS
        xi = specie.x  # get particle positions
        x_left_grid = grids.x[nearest_left_grid_point]  # get nearest left grid positions

        # WEIGHT FOR INTERPOLATION
        x_right_weight = (xi - x_left_grid) / dx

        # INTERPOLATION FUNCTION
        def interpolate(value):
            value_left = value[nearest_left_grid_point]
            new_value = value_left + x_right_weight * (value[nearest_right_grid_point] - value_left)
            return new_value

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
        bz = interpolate(grids.bz)
        ex = interpolate(grids.ex)
        ey = interpolate(grids.ey)
        ez = interpolate(grids.ez)

        # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
        bb0 = bz * cos_theta + bx0 * sin_theta
        # CALCULATE ROTATION ANGLE
        d_theta = - q_by_m * bb0 * (dt / 2)
        sin_d_theta = numpy.sin(d_theta)
        cos_d_theta = numpy.cos(d_theta)
        # CALCULATE HALF ACCELERATION IN xp AND y DIRECTION
        half_acceleration_xp = (q_by_m * (ex * cos_theta - ez * sin_theta) * (dt / 2))
        half_acceleration_y = (q_by_m * ey * (dt / 2))
        # CALCULATE FULL ACCELERATION IN b0 DIRECTION
        full_acceleration_b0 = (q_by_m * (ex * sin_theta + ez * cos_theta) * dt)
        # ADD HALF ACCELERATIONS TO CORRESPONDING VELOCITIES
        vxp_1 = specie.vxp_old + half_acceleration_xp
        vy_1 = specie.vy_old + half_acceleration_y
        # APPLY ROTATION AND HALF ACCELERATIONS
        specie.vxp = (cos_d_theta * vxp_1 - sin_d_theta * vy_1) + half_acceleration_xp
        specie.vy = (sin_d_theta * vxp_1 + cos_d_theta * vy_1) + half_acceleration_y
        # APPLY FULL ACCELERATION FOR vb0
        specie.vb0 = specie.vb0_old + full_acceleration_b0
        # MOVE X
        specie.x = numpy.fmod(
            (specie.vxp * cos_theta + specie.vb0 * sin_theta) * dt + specie.x_old, length)
        # MAKE SURE THAT 0 < X < LENGTH
        specie.x[specie.x < 0] += length
        # UPDATE NEAREST GRIDS
        specie.update_nearest_grid()


# INTERPOLATION FUNCTION
def interpolate(value, nearest_left_grid_point, nearest_right_grid_point, x_right_weight):
    # print("starting interpolation")
    value_left = value[nearest_left_grid_point]
    new_value = value_left + x_right_weight * (value[nearest_right_grid_point] - value_left)
    return new_value


def move_particles():  # option="init" for initialization, otherwise, option="normal"
    """Particle mover"""
    for specie in species:  # loop for each specie
        specie.x_old = specie.x  # store previous positions
        # STORE PREVIOUS VELOCITIES
        specie.vy_old = specie.vy
        specie.vxp_old = specie.vxp
        specie.vb0_old = specie.vb0
        q_by_m = specie.qm  # get the charge-per-mass ratio
        # GET NEAREST GRIDS
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids()
        # GET POSITIONS
        xi = specie.x  # get particle positions
        x_left_grid = grids.x[nearest_left_grid_point]  # get nearest left grid positions

        # WEIGHT FOR INTERPOLATION
        x_right_weight = (xi - x_left_grid) / dx

        # INTERPOLATION FUNCTION
        # def interpolate(value):
        #    value_left = value[nearest_left_grid_point]
        #    new_value = value_left + x_right_weight * (value[nearest_right_grid_point] - value_left)
        #    return new_value

        args_interpolate = (nearest_left_grid_point, nearest_right_grid_point, x_right_weight)

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES
        bz = interpolate(grids.bz, *args_interpolate)
        ex = interpolate(grids.ex, *args_interpolate)
        ey = interpolate(grids.ey, *args_interpolate)
        ez = interpolate(grids.ez, *args_interpolate)

        # MULTIPROCESS = False
        # if not MULTIPROCESS:
        #     bz = interpolate(grids.bz, *args_interpolate)
        #     ex = interpolate(grids.ex, *args_interpolate)
        #     ey = interpolate(grids.ey, *args_interpolate)
        #     ez = interpolate(grids.ez, *args_interpolate)
        # else:
        #     import multiprocessing as mp
        #     with mp.Pool(processes=4) as pool:
        #         bz_task = pool.apply_async(interpolate, args=(grids.bz, *args_interpolate))
        #         ex_task = pool.apply_async(interpolate, args=(grids.ex, *args_interpolate))
        #         ey_task = pool.apply_async(interpolate, args=(grids.ey, *args_interpolate))
        #         ez_task = pool.apply_async(interpolate, args=(grids.ez, *args_interpolate))
        #         bz = bz_task.get()
        #         ex = ex_task.get()
        #         ey = ey_task.get()
        #         ez = ez_task.get()

        # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
        bb0 = bz * cos_theta + bx0 * sin_theta
        # CALCULATE ROTATION ANGLE
        d_theta = (- q_by_m * dt / 2) * bb0
        sin_d_theta = numpy.sin(d_theta)
        cos_d_theta = numpy.cos(d_theta)
        # CALCULATE HALF ACCELERATION IN xp AND y DIRECTION
        half_acceleration_xp = (q_by_m * dt / 2) * (ex * cos_theta - ez * sin_theta)
        half_acceleration_y = (q_by_m * dt / 2) * ey
        # CALCULATE FULL ACCELERATION IN b0 DIRECTION
        full_acceleration_b0 = (q_by_m * dt) * (ex * sin_theta + ez * cos_theta)
        # ADD HALF ACCELERATIONS TO CORRESPONDING VELOCITIES
        vxp_1 = specie.vxp_old + half_acceleration_xp
        vy_1 = specie.vy_old + half_acceleration_y
        # APPLY ROTATION AND HALF ACCELERATIONS
        specie.vxp = (cos_d_theta * vxp_1 - sin_d_theta * vy_1) + half_acceleration_xp
        specie.vy = (sin_d_theta * vxp_1 + cos_d_theta * vy_1) + half_acceleration_y
        # APPLY FULL ACCELERATION FOR vb0
        specie.vb0 = specie.vb0_old + full_acceleration_b0
        # MOVE X
        specie.x = numpy.fmod(
            (specie.vxp * cos_theta + specie.vb0 * sin_theta) * dt + specie.x_old, length)
        # MAKE SURE THAT 0 < X < LENGTH
        specie.x[specie.x < 0] += length
        # UPDATE NEAREST GRIDS
        specie.update_nearest_grid()


# UPDATE GRID VALUES BASED ON PARTICLE VALUES
def weigh_to_grid():  # (x,v) to (rho, j)
    """UPDATE GRID VALUES BASED ON PARTICLE VALUES"""

    # STORE OLD FIELD QUANTITIES
    grids.f_right_old = grids.f_right
    grids.f_left_old = grids.f_left
    grids.g_right_old = grids.g_right
    grids.g_left_old = grids.g_left

    # REINITIALIZE CURRENT DENSITIES
    grids.jy_old = numpy.zeros(ng)
    grids.jy = numpy.zeros(ng)
    grids.jz_old = numpy.zeros(ng)
    grids.jz = numpy.zeros(ng)

    # REINITIALIZE RHO
    grids.rho = numpy.zeros(ng)

    for specie in species:  # loop for each specie
        # GET PREVIOUS NEAREST GRIDS AND GRID POSITIONS
        old_nearest_left_grid, old_nearest_right_grid = specie.nearest_grids_old()
        old_x_left_grid = grids.x[old_nearest_left_grid]
        # GET CURRENT NEAREST GRIDS AND CORRESPONDING GRID POSITIONS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids()
        x_left_grid = grids.x[nearest_left_grid]
        # GET PARTICLE CHARGE
        qc = specie.q
        # GET PARTICLE POSITIONS
        xi = specie.x
        # GET PARTICLE POSITIONS FROM THE PREVIOUS TIME STEP
        xi_old = specie.x_old

        # WEIGHING FUNCTIONS

        def weigh_old(values, destination):
            """weigh a value to grids before particle movement"""
            d_right = values * (xi_old - old_x_left_grid)
            d_left = values * dx - d_right
            weighted = destination + numpy.bincount(old_nearest_left_grid, weights=d_left,
                                                    minlength=ng) + numpy.bincount(old_nearest_right_grid,
                                                                                   weights=d_right, minlength=ng)
            return weighted

        def weigh_current(values, destination):
            """weigh a value to grids after particle movement"""
            d_right = values * (xi - x_left_grid)
            d_left = values * dx - d_right
            weighted = destination + numpy.bincount(nearest_left_grid, weights=d_left, minlength=ng) + numpy.bincount(
                nearest_right_grid, weights=d_right, minlength=ng)
            return weighted

        # WEIGH Jy

        value = qc / dx * specie.vy / dx
        grids.jy_old = weigh_old(value, grids.jy_old)
        grids.jy = weigh_current(value, grids.jy)

        # WEIGH Jz

        value = qc / dx * (specie.vb0 * cos_theta - specie.vxp * sin_theta) / dx
        grids.jz_old = weigh_old(value, grids.jz_old)
        grids.jz = weigh_current(value, grids.jz)

        # WEIGH RHO

        value = qc / dx / dx
        grids.rho = weigh_current(value, grids.rho)


# 1/K^2 VALUES TO BE USED IN Ex FIELD SOLVER
ksqi_list = 1 / (numpy.arange(1, ng // 2 + 1) * 2 * math.pi / length * numpy.sinc(
    numpy.arange(1, ng // 2 + 1) * dx / length)) ** 2


def solve_field_x():
    """Find Ex using Poisson's Equation, similar to INITIAL FIELD SOLVER"""

    # FOURIER TRANSFORM, FROM rho(x) TO rho(k)
    rho_n_list = dx * scipy.fft.rfft(grids.rho)
    # POISSON'S EQUATION, FROM rho(k) to phi(k)
    phi_n_list = numpy.concatenate(
        ([0], rho_n_list[1:] * ksqi_list / epsilon))
    # INVERSE FOURIER TRANSFORM, phi(k) TO phi(x)
    phi_list = 1 / dx * scipy.fft.irfft(phi_n_list)
    # SOLVE FOR Ex FROM phi(x)
    grids.ex = (numpy.roll(phi_list, 1) - numpy.roll(phi_list, -1)) / 2 / dx


def solve_field():  # Ey and Bz (also Ez and By if B0 not on z axis)
    """Find Ey, Bz, Ez, and By using Maxwell's Equation"""

    # SOLVE RIGHT-GOING FIELD QUANTITY F = 1/2*(Ey + Bz)
    grids.f_right = numpy.roll(grids.f_right_old, 1) - (dt / 4) * (
            numpy.roll(grids.jy_old, 1) + grids.jy)
    # SOLVE LEFT-GOING FIELD QUANTITY F = 1/2*(Ey - Bz)
    grids.f_left = numpy.roll(grids.f_left_old, -1) - (dt / 4) * (
            numpy.roll(grids.jy_old, -1) + grids.jy)
    # SOLVE Ey
    grids.ey = (grids.f_right + grids.f_left) / epsilon
    # SOLVE Bz
    grids.bz = (grids.f_right - grids.f_left) * math.sqrt(mu / epsilon)
    # SOLVE RIGHT-GOING FIELD QUANTITY G = 1/2*(Ez - By)
    grids.g_right = numpy.roll(grids.g_right_old, 1) - (dt / 4) * (
            numpy.roll(grids.jz_old, 1) + grids.jz)
    # SOLVE LEFT-GOING FIELD QUANTITY G = 1/2*(Ez + By)
    grids.g_left = numpy.roll(grids.g_left_old, -1) - (dt / 4) * (
            numpy.roll(grids.jz_old, -1) + grids.jz)
    # SOLVE Ez
    grids.ez = (grids.g_right + grids.g_left) / epsilon
    # SOLVE By
    grids.by = (grids.g_left - grids.g_right) * math.sqrt(mu / epsilon)


# MAIN PROGRAM
def main():
    """MAIN PROGRAM"""

    # INITIALIZE ARRAYS TO STORE DATA
    ex_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ex VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
    ey_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ey VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
    ez_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ez VALUES FOR SPECTRUM PLOTTING (2D ARRAY)

    def main_loop(time_step):
        """MAIN PROGRAM TO BE LOOPED"""

        # MAIN PROGRAM
        move_particles()  # move particles
        weigh_to_grid()  # weight to grid
        solve_field_x()  # solve Ex from rho (Poisson's eqn)
        solve_field()  # solve other field quantities (Maxwell's eqn)

        # PRINT REPORT EVERY 100 FRAMES
        if time_step % 100 == 0:
            grids.print(time_step)

        # STORE DATA PERIODICALLY
        index = time_step * nt_sample / nt
        if index.is_integer():
            # STORE DATA FOR PHASE SPACE PLOT (ELECTRONS ONLY)
            index = int(index)
            for i in range(len(species)):
                x_data[i][index + 1] = species[i].x[plot_particles_id]
                v_data[i][index + 1] = species[i].vx[plot_particles_id]
            # STORE THE E FIELD FOR PLOTTING
            ex_list[index] = grids.ex
            ey_list[index] = grids.ey
            ez_list[index] = grids.ez

    # RUN THE MAIN LOOP FOR A NUMBER OF TIME STEPS
    for count in range(nt):
        main_loop(count)

    # OUTPUT RUNTIME
    end_time = time.monotonic()
    print("Duration = ", timedelta(seconds=end_time - start_time))
    print("---------------------------RUNTIME---------------------------", file=log)
    print("Duration                        = {}".format(timedelta(seconds=end_time - start_time)), file=log)

    # CLOSE LOG FILE
    log.close()

    # SAVE ARRAYS TO FILE
    zarr.save('{}.zip'.format(name),
              kte=kte, kti=kti, mi=mi_real, b0=b0, rho=rho_mass,
              ex=ex_list, ey=ey_list, ez=ez_list, x=x_data, vx=v_data, ng=ng, nt=nt_sample, smu=smu, scu=scu,
              slu=slu, stu=stu, c=c, dx=dx, dt=dt_sample, wp=wp, wpi=wpi, wc=wc, wci=wci, theta=theta, v_th=v_th,
              vi_th=vi_th)


#  INDICES OF PARTICLES TO BE PLOTTED IN THE ANIMATION
if ni == 0:
    plot_particles_id = rng.choice(ne, n_sample)
else:
    plot_particles_id = rng.choice(min(ne, ni), n_sample)

#  CREATE ARRAYS FOR STORING PHASE SPACE DATA FOR ANIMATION
x_data = numpy.zeros(shape=(len(species), nt_sample + 1, n_sample))
v_data = numpy.zeros(shape=(len(species), nt_sample + 1, n_sample))

# INITIALIZE GRIDS
initialization()

# RUN MAIN PROGRAM
# main()

# RUN MAIN PROGRAM WITH PROFILER
if __name__ == '__main__':
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats()

    with open('outstats.txt', 'w+') as f:
        f.write(s.getvalue())
