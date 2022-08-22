import math
import scipy.fft
import numpy
import time
import zarr
from datetime import timedelta
import initialization

# GET STARTING TIME
start_time = time.monotonic()

# physical constants
epsilon = 8.85E-12  # vacuum permittivity
mu = 1.26E-6  # vacuum permeability
c = 3.00E8  # speed of light
me_real = 9.11E-31  # electron mass
mi_real = 1.67E-27  # ion mass (proton)
qe_real = 1.60E-19  # (-) electron charge
qi_real = qe_real  # ion (proton) charge
qm = 1.76E11  # electron charge-to-mass ratio (q/me)
qmi = qi_real / mi_real  # proton charge-to-mass ratio (q/mi)
kb = 1.38E-23  # Boltzmann constant

# specifications for plasma
n0 = 1E8  # plasma electron density (1/m^3)
n0i = 0  # plasma ion density (1/m^3)
te = 1E4  # electron temperature (K)
ti = 1E4  # ion temperature (K)

# specifications for external, uniform magnetic field (in x-z plane)
b0 = 0.01E-5  # 1E-6  # 0.01E-5  # field strength
theta = 0  # math.pi / 2  # field angle in radians (e.g., 0 is along +z-axis, Pi/2 is along +x-axis)
sin_theta = math.sin(theta)  # sin(theta)
cos_theta = math.cos(theta)  # cos(theta)

# specifications for external, uniform electric field (along y-axis)
e_ext = 0  # field strength

# derived quantities
bz0 = b0 * cos_theta  # field component along z
bx0 = b0 * sin_theta
wp = math.sqrt(n0 * qm * qe_real / epsilon)  # electron plasma frequency (Hz)
wpi = math.sqrt(n0i * qmi * qi_real / epsilon)  # electron plasma frequency (Hz)
wc = qm * b0  # electron cyclotron frequency (Hz)
wci = qmi * b0  # ion cyclotron frequency (Hz)
kte = kb * te  # electron energy kT (J)
kti = kb * ti  # ion energy kT (J)
v_th = math.sqrt(2 * kte / me_real)  # electron thermal velocity
vi_th = math.sqrt(2 * kti / mi_real)  # ion thermal velocity
lambda_d = v_th / math.sqrt(2) / wp  # Debye length (m)

# simulation specifications
ng = 4096  # 16384  # number of grids (please use powers of 2 e.g. 4, 8, 1024)
nt = 16384  # 1024 #16384 #65536 #1024  # number of time steps to run
ne = ng * 100  # number of PIC electrons
ni = 0  # number of PIC ions
np = ne + ni  # number of PIC particles

# CREATE LOG TEXT FILE
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

# define unit quantities
scu = c ** 2 * epsilon * lambda_d / qm  # simulation unit charge in Coulombs (C)
slu = lambda_d  # simulation unit length in meters (m)
stu = lambda_d / c  # simulation unit time in seconds (s)
smu = scu / qm  # simulation unit mass in kilograms (kg)

# scale physical constants
epsilon = 1  # scaled vacuum permittivity
c = 1  # scaled speed of light
mu = 1 / c ** 2 / epsilon  # scaled vacuum permeability
qm = 1  # scaled electron charge-to-mass ratio (q/me)
qmi = qmi * smu / scu  # scaled ion charge-to-mass ratio (q/mi)

# scale derived quantities
wp = wp * stu  # scaled electron plasma frequency
wpi = wpi * stu  # scaled ion plasma frequency
wc = wc * stu  # scaled electron cyclotron frequency
wci = wci * stu  # scaled ion cyclotron frequency
kte = kte * stu ** 2 / smu / slu ** 2  # scaled electron energy kT (J)
kti = kti * stu ** 2 / smu / slu ** 2  # scaled ion energy kT (J)
v_th = v_th * stu / slu  # scaled electron thermal velocity
vi_th = vi_th * stu / slu  # scaled ion thermal velocity
lambda_d = lambda_d / slu  # scaled Debye length (m)

# scale external, uniform magnetic field (in x-z plane)
b0 = wc / qm  # field strength
bz0 = b0 * cos_theta  # field component along z
bx0 = b0 * sin_theta  # field component along x

# scale external, uniform electric field (along y-axis)
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

# Grid sizes
dx = lambda_d  # grid size
length = ng * dx  # length of the system
dt = dx / c  # time step

# PIC particle properties
qe = wp ** 2 * length / ne * epsilon / qm  # charge per PIC particle
me = qe / qm  # PIC electron mass
mi = qe / qmi  # 100 * me  # PIC ion mass

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
print("simulation electron charge (in simulation units) = -{:.2e}".format(qe), file=log)
print("simulation electron charge (in mks)              = -{:.2e} C".format(qe * scu), file=log)
print("simulation electron mass (in simulation units)   = {:.2e}".format(me), file=log)
print("simulation electron mass (in mks)                = {:.2e} kg".format(me * smu), file=log)
if ni > 0:
    print("simulation ion charge (in simulation units)      = {:.2e}".format(qe), file=log)
    print("simulation ion charge (in mks)                   = {:.2e} C".format(qe * scu), file=log)
    print("simulation ion mass (in simulation units)        = {:.2e}".format(mi), file=log)
    print("simulation ion mass (in mks)                     = {:.2e} kg".format(mi * smu), file=log)

# check whether v_th < c or not
assert v_th < c, "THERMAL VELOCITY GREATER THAN THE SPEED OF LIGHT!"

# PHASE SPACE PLOT SPECIFICATIONS
n_sample = ng  # number of particles to plot
assert n_sample <= np, "NOT ENOUGH PARTICLES TO PLOT PHASE SPACE!"


# DEFINE CLASS TO STORE GRID DATA
class GridPointList:
    def __init__(self, x, rho=numpy.zeros(ng), jy_old=numpy.zeros(ng), jy=numpy.zeros(ng), jz_old=numpy.zeros(ng),
                 jz=numpy.zeros(ng), ex=numpy.zeros(ng), ey=numpy.zeros(ng), ez=numpy.zeros(ng),
                 bx=numpy.zeros(ng) + bx0, by=numpy.zeros(ng),
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
        self.bx = bx  # magnetic field x, not used at all
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
                self.ez[index], self.bx[index], self.by[index], self.bz[index],
                self.f_left[index], self.f_right[index], self.g_left[index], self.g_right[index]))


# DEFINE CLASS TO STORE PARTICLE DATA
class ParticleList:
    def __init__(self, x, vy=numpy.zeros(ne), vxp=numpy.zeros(ne), vb0=numpy.zeros(ne),
                 x_old=numpy.zeros(ne), vx_old=numpy.zeros(ne), vy_old=numpy.zeros(ne),
                 vxp_old=numpy.zeros(ne),
                 vb0_old=numpy.zeros(ne)):  # Two interchangeable system of coordinates: (x, y, z) and (xp, y, b0)
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

    # FUNCTION TO FIND THE NEAREST LEFT AND RIGHT GRIDS OF THE PARTICLES
    def nearest_grids(self,
                      option="new"):  # option="new": current particle positions, "old": particle positions from the
        # previous time step
        if option == "old":
            position = self.x_old
        else:
            position = self.x
        floored = numpy.floor(position / dx).astype(int)
        ceiled = floored + 1
        ceiled[ceiled == ng] = 0  # Grid ng corresponds to Grid 0 due to periodicity
        return floored, ceiled  # return [nearest left grids], [nearest right grids]


# DEFINE CLASS TO STORE PIC ELECTRON DATA
class ElectronList(ParticleList):
    def __init__(self, x, vy=numpy.zeros(ne), vxp=numpy.zeros(ne), vb0=numpy.zeros(ne),
                 x_old=numpy.zeros(ne), vx_old=numpy.zeros(ne), vy_old=numpy.zeros(ne),
                 vxp_old=numpy.zeros(ne),
                 vb0_old=numpy.zeros(ne)):
        super().__init__(x, vy, vxp, vb0, x_old, vx_old, vy_old, vxp_old, vb0_old)
        self.q = -qe  # PIC electron charge
        self.m = me  # PIC electron mass
        self.qm = -qm  # electron charge-to-mass ratio
        self.type = "electron"  # PIC electron type


# DEFINE CLASS TO STORE PIC ION DATA
class IonList(ParticleList):
    def __init__(self, x, vy=numpy.zeros(ni), vxp=numpy.zeros(ni), vb0=numpy.zeros(ni),
                 x_old=numpy.zeros(ni), vx_old=numpy.zeros(ni), vy_old=numpy.zeros(ni),
                 vxp_old=numpy.zeros(ni),
                 vb0_old=numpy.zeros(ni)):
        super().__init__(x, vy, vxp, vb0, x_old, vx_old, vy_old, vxp_old, vb0_old)
        self.q = qe  # PIC ion charge
        self.m = mi  # PIC ion mass
        self.qm = qe / mi  # ion charge-to-mass ratio
        self.type = "ion"  # PIC ion type


# GENERATE GRID POINTS (FROM LIST OF POSITIONS)
grids = GridPointList(numpy.arange(0, length, dx))

# CONSTRUCT A RANDOM NUMBER GENERATOR
rng = numpy.random.default_rng()

# GENERATE PARTICLES (FROM LIST OF POSITIONS AND VELOCITIES)
if ni > 0:  # with ions
    species = [ElectronList(rng.uniform(0, length, size=ne),  # uniform position distribution
                            # gaussian velocity distribution
                            vxp=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vy=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vb0=rng.normal(0.0, v_th / math.sqrt(2), size=ne)),
               IonList(rng.uniform(0, length, size=ni),  # uniform position distribution
                       # gaussian velocity distribution
                       vxp=rng.normal(0.0, v_th / math.sqrt(2), size=ni),
                       vy=rng.normal(0.0, v_th / math.sqrt(2), size=ni),
                       vb0=rng.normal(0.0, v_th / math.sqrt(2), size=ni))]
else:  # no ions
    species = [ElectronList(rng.uniform(0, length, size=ne),  # uniform position distribution
                            # gaussian velocity distribution
                            vxp=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vy=rng.normal(0.0, v_th / math.sqrt(2), size=ne),
                            vb0=rng.normal(0.0, v_th / math.sqrt(2), size=ne))]

def move_particles(option="normal"):  # option="init" for initialization, otherwise, option="normal"
    for specie in species:  # loop for each specie
        specie.x_old = specie.x  # store previous positions
        if option == "normal":  # for non-initial runs
            # STORE PREVIOUS VELOCITIES
            specie.vy_old = specie.vy
            specie.vxp_old = specie.vxp
            specie.vb0_old = specie.vb0
        q_by_m = specie.qm  # get the charge-per-mass ratio
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids()  # get nearest left and right grids
        xi = specie.x  # get particle positions
        x_left_grid = grids.x[nearest_left_grid_point]  # get nearest left grid positions

        # WEIGHT FOR INTERPOLATION
        x_right_weight = (xi - x_left_grid) / dx

        # INTERPOLATION FUNCTION
        def interpolate(value):
            value_left = value[nearest_left_grid_point]
            d_value = value[nearest_right_grid_point] - value_left
            new_value = value_left + x_right_weight * d_value
            return new_value

        # INTERPOLATE FIELD QUANTITIES FROM GRIDS TO PARTICLES (MAKE A FUNCTION?)
        bx = interpolate(grids.bx)
        bz = interpolate(grids.bz)
        ex = interpolate(grids.ex)
        ey = interpolate(grids.ey)
        ez = interpolate(grids.ez)

        # PROJECT MAGNETIC FIELD TO THE b0 DIRECTION (b0 direction = (sin(theta), cos(theta)))
        bb0 = bz * cos_theta + bx * sin_theta
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
        # assert numpy.all((specie.x < length) & (specie.x >= 0)), "FMOD DID NOT WORK!!!!!"


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

    for specie in species:  # loop for each specie
        # GET PREVIOUS NEAREST GRIDS AND GRID POSITIONS
        old_nearest_left_grid, old_nearest_right_grid = specie.nearest_grids("old")
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
            numpy.add.at(destination, old_nearest_left_grid, d_left)
            numpy.add.at(destination, old_nearest_right_grid, d_right)

        def weigh_current(values, destination):
            """weigh a value to grids after particle movement"""
            d_right = values * (xi - x_left_grid)
            d_left = values * dx - d_right
            numpy.add.at(destination, nearest_left_grid, d_left)
            numpy.add.at(destination, nearest_right_grid, d_right)

        # WEIGH Jy

        value = qc / dx * specie.vy / dx
        weigh_old(value, grids.jy_old)
        weigh_current(value, grids.jy)

        # WEIGH Jz

        value = qc / dx * (specie.vb0 * cos_theta - specie.vxp * sin_theta) / dx
        weigh_old(value, grids.jz_old)
        weigh_current(value, grids.jz)

        # SUBTRACT rho FROM OLD GRIDS AND ADD TO NEW GRIDS

        value = qc / dx / dx
        weigh_old(-value, grids.rho)
        weigh_current(value, grids.rho)


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
    ex_list = numpy.zeros(shape=(nt, ng))  # STORE Ex VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
    ey_list = numpy.zeros(shape=(nt, ng))  # STORE Ey VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
    ez_list = numpy.zeros(shape=(nt, ng))  # STORE Ez VALUES FOR SPECTRUM PLOTTING (2D ARRAY)

    def main_loop(time_step):
        """MAIN PROGRAM TO BE LOOPED"""

        # STORE DATA FOR PHASE SPACE PLOT (ELECTRONS ONLY)
        x_data[time_step + 1] = species[0].x[plot_particles_id]
        v_data[time_step + 1] = species[0].vx[plot_particles_id]

        # MAIN PROGRAM
        move_particles()  # move particles
        weigh_to_grid()  # weight to grid
        solve_field_x()  # solve Ex from rho (Poisson's eqn)
        solve_field()  # solve other field quantities (Maxwell's eqn)

        # PRINT REPORT EVERY 100 FRAMES
        if time_step % 100 == 0:
            grids.print(time_step)

        # STORE THE E FIELD FOR PLOTTING
        ex_list[time_step] = grids.ex
        ey_list[time_step] = grids.ey
        ez_list[time_step] = grids.ez

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
    zarr.save('{}.zip'.format(name), ex=ex_list, ey=ey_list, ez=ez_list, x=x_data, vx=v_data, ng=ng, nt=nt, slu=slu,
              stu=stu, c=c, dx=dx, dt=dt, wp=wp, wpi=wpi, wc=wc, wci=wci, theta=theta, v_th=v_th, vi_th=vi_th)


#  INDICES OF PARTICLES TO BE PLOTTED IN THE ANIMATION
plot_particles_id = rng.choice(ne, n_sample)

#  CREATE ARRAYS FOR STORING PHASE SPACE DATA FOR ANIMATION
x_data = numpy.zeros(shape=(nt + 1, n_sample))
v_data = numpy.zeros(shape=(nt + 1, n_sample))

# RUN MAIN PROGRAM
main()
