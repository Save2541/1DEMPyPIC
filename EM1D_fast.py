import io
import math
import time
from datetime import timedelta
import numpy
import scipy.fft
import zarr
import particle_list
import grid_list
import field_solver
import particle_mover
import digital_filtering
import particles_to_grids

# TURN ELECTROMAGNETIC ON OR OFF
is_electromagnetic = True

# GET STARTING TIME
start_time = time.monotonic()

# GET TIME STRING FOR FILE NAME
timestr = time.strftime("%Y%m%d-%H%M%S")

# PHYSICAL CONSTANTS (MKS UNITS)
c = 3.00E8  # speed of light
mu = 1.26E-6  # vacuum permeability
epsilon = 1 / mu / c ** 2  # 8.85E-12  # vacuum permittivity
me_real = 9.11E-31  # electron mass
mi_real = 100 * me_real  # 1.67E-27 # ion mass (proton)
qe_real = 1.60E-19  # (-) electron charge
qi_real = qe_real  # ion (proton) charge
qm = 1.76E11  # electron charge-to-mass ratio (q/me)
qmi = qi_real / mi_real  # proton charge-to-mass ratio (q/mi)
kb = 1.38E-23  # Boltzmann constant

# PLASMA SPECIFICATIONS
n0 = 1E8  # plasma electron density (1/m^3)
n0i = n0  # plasma ion density (1/m^3)
te = 1E4  # electron temperature (K)
ti = 1E4  # 1E4  # ion temperature (K)

# EXTERNAL MAGNETIC FIELD SPECIFICATIONS (VECTOR ON X-Z PLANE)
b0 = 1E-4  # 10 * 0.01E-5  # magnetic field strength (T)
theta = 0 # math.pi / 2  # magnetic field angle (rad) (e.g., 0 is along +z-axis, Pi/2 is along +x-axis)
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
sqrt_mu_over_epsilon = math.sqrt(mu / epsilon)  # square root mu naught over epsilon naught

# SIMULATION SPECIFICATIONS
ng = 256  # 4096 # number of grids (please use powers of 2 e.g. 4, 8, 1024)
nt = 16384  # 64 * 16384  # number of time steps to run
ne = ng * 10  # ng * 100  # number of PIC electrons
ni = ne  # number of PIC ions
np = ne + ni  # number of PIC particles

# INITIAL DENSITY WAVES (LONGITUDINAL)
d_nw_e = 0  # number of waves (electrons)
d_amplitude_e = 0  # amplitude (electrons) (must be between 0 and 1)
d_nw_i = 0  # number of waves (ions)
d_amplitude_i = 0  # amplitude (ions) (must be between 0 and 1)

init_d_wv = {  # initialize a wave in each component: [number of waves per system length, amplitude]
    "electron": [d_nw_e, d_amplitude_e],
    "ion": [d_nw_i, d_amplitude_i]
}
print(init_d_wv)

# INITIAL VELOCITY WAVES (TRANSVERSE)
nw_e = 0  # number of waves (electrons)
amplitude_e = 0  # amplitude (electrons)
nw_i = 0  # 130  # number of waves (ions)
amplitude_i = 0  # vi_th * 100  # amplitude (ions)

init_v_wv = {  # initialize a wave in each component: [number of waves per system length, amplitude]
    "electron": {
        "vxp": [nw_e, amplitude_e],
        "vy": [nw_e, amplitude_e],
        "vb0": [nw_e, amplitude_e]
    },
    "ion": {
        "vxp": [nw_i, amplitude_i],
        "vy": [nw_i, amplitude_i],
        "vb0": [nw_i, amplitude_i]
    }
}

# CREATE LOG (TEXT FILE)
if ni > 0:
    name = "{}_ni{:.0e}_ti{:.0e}_te{:.0e}_b0{:.0e}_theta{}".format(timestr, n0i, ti, te, b0,
                                                                   int(theta * 180 / math.pi), nt)  # name string
else:
    name = "{}_ne{:.0e}_te{:.0e}_b0{:.0e}_theta{}".format(timestr, n0, te, b0, int(theta * 180 / math.pi), nt)

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
print("---------------------------INITIAL DENSITY WAVES---------------------------", file=log)
print("(Component: [number of waves, amplitude (probability)])", file=log)
print("Electrons:", file=log)
print(init_d_wv["electron"], file=log)
if ni > 0:
    print("Ions:", file=log)
    print(init_d_wv["ion"], file=log)
print("---------------------------INITIAL VELOCITY WAVES---------------------------", file=log)
print("(Component: [number of waves, amplitude (m/s)])", file=log)
print("Electrons:", file=log)
print(init_v_wv["electron"], file=log)
if ni > 0:
    print("Ions:", file=log)
    print(init_v_wv["ion"], file=log)
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

# SCALED VELOCITY WAVE AMPLITUDES & RENAMED DICTIONARY KEYS
for specie in init_v_wv:
    for v in init_v_wv[specie]:
        init_v_wv[specie][v][1] = init_v_wv[specie][v][1] * stu / slu

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
print("---------------------------INITIAL VELOCITY WAVES IN SIMULATION UNITS---------------------------", file=log)
print("(Component: [number of waves, amplitude (m/s)])", file=log)
print("Electrons:", file=log)
print(init_v_wv["electron"], file=log)
if ni > 0:
    print("Ions:", file=log)
    print(init_v_wv["ion"], file=log)
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
dx = lambda_d / 2  # spatial grid size
length = ng * dx  # length of the system
dt = dx / c  # 73 # duration of time step

print("wp = ", wp, ". wc = ", wc, ".")
if wp * dt > 0.3:
    print("WARNING: TIME STEP TOO LARGE! TO GET AN ACCURATE RESULT, USE dt <= ", 0.3 / wp / dx * c, " * dx / c.")
if wc * dt >= 0.35:
    print("WARNING: MAGNETIC FIELD IS TOO STRONG! TO GET AN ACCURATE RESULT, USE B_0 < ", wc / qm * smu / scu / stu,
          " T.")

# PROPERTIES OF PIC PARTICLES
qe = wp ** 2 * length / ne * epsilon / qm  # charge per PIC particle
me = qe / qm
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
n_sample = min(ne, ng)  # number of particles to plot per specie
nt_sample = 16384  # ng * 32  # how many time steps to store (use powers of 2)
assert nt_sample <= nt, "NOT ENOUGH TIME STEPS TO STORE!"
dt_sample = dt * nt / nt_sample  # duration per sample

# ARGUMENTS TO BE SENT TO GRID GENERATORS
args_grid = (ng, epsilon, mu, bz0, e_ext)

# GENERATE GRID POINTS (FROM LIST OF POSITIONS)
grids = grid_list.GridPointList(numpy.arange(0, length, dx), *args_grid)

# CONSTRUCT A RANDOM NUMBER GENERATOR
rng = numpy.random.default_rng()

# DEFAULT DENSITY DISTRIBUTIONS (UNIFORM)
x_list = {  # list of electron positions and ion positions
    "electron": rng.uniform(0, length, size=ne),
    "ion": rng.uniform(0, length, size=ni)
}

# INITIAL DENSITY WAVES
xx = numpy.linspace(0, length, int(1E8))  # evenly spaced choices of x
for specie_key in x_list:
    nw, amplitude = init_d_wv[specie_key]
    prob_list = 1 + amplitude * numpy.sin(
        2 * math.pi * nw / length * xx)  # probability of particles to be in each grid cell
    prob_list = prob_list / sum(prob_list)  # normalized probability distribution
    number = len(x_list[specie_key])  # number of particles
    x_list[specie_key] = numpy.random.choice(xx, number, p=prob_list)

# DEFAULT VELOCITY DISTRIBUTIONS (GAUSSIAN)
v_list = {
    "electron": {
        "vxp": rng.normal(0.0, v_th / math.sqrt(2), size=ne),
        "vy": rng.normal(0.0, v_th / math.sqrt(2), size=ne),
        "vb0": rng.normal(0.0, v_th / math.sqrt(2), size=ne)
    },
    "ion": {
        "vxp": rng.normal(0.0, vi_th / math.sqrt(2), size=ni),
        "vy": rng.normal(0.0, vi_th / math.sqrt(2), size=ni),
        "vb0": rng.normal(0.0, vi_th / math.sqrt(2), size=ni)
    }
}

# INITIAL VELOCITY WAVES
for specie_key in v_list:
    for v_key in v_list[specie_key]:
        nw, amplitude = init_v_wv[specie_key][v_key]
        v_list[specie_key][v_key] = v_list[specie_key][v_key] + amplitude * numpy.sin(
            2 * math.pi * nw / length * x_list[specie_key])

# ARGUMENTS TO BE SENT TO PARTICLE GENERATORS
args_e_list = (ne, qe, me, qm)
args_i_list = (ni, qi, mi, qmi)

# GENERATE PARTICLES (FROM LIST OF POSITIONS AND VELOCITIES)
if ni == 0:
    species = [particle_list.ElectronList(x_list["electron"],  # uniform position distribution
                                          *args_e_list,
                                          # gaussian velocity distribution
                                          vxp=v_list["electron"]["vxp"],
                                          vy=v_list["electron"]["vy"],
                                          vb0=v_list["electron"]["vb0"])]
else:
    species = [particle_list.ElectronList(x_list["electron"],  # uniform position distribution
                                          *args_e_list,
                                          # gaussian velocity distribution
                                          vxp=v_list["electron"]["vxp"],
                                          vy=v_list["electron"]["vy"],
                                          vb0=v_list["electron"]["vb0"]),
               particle_list.IonList(x_list["ion"],  # uniform position distribution
                                     *args_i_list,
                                     # gaussian velocity distribution
                                     vxp=v_list["ion"]["vxp"],
                                     vy=v_list["ion"]["vy"],
                                     vb0=v_list["ion"]["vb0"])]

# GET SAMPLE FREQUENCIES
sample_k = 2 * math.pi * scipy.fft.rfftfreq(ng, dx)

# SET SMOOTHING WEIGHT
w_smooth = 0.5


def initialize():
    # INITIAL WEIGHTING (x to rho, hat function)

    for specie in species:  # loop for each specie
        # UPDATE NEAREST GRIDS
        specie.update_nearest_grid(dx)
        # GET NEAREST GRIDS
        nearest_left_grid, nearest_right_grid = specie.nearest_grids(ng)
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
    rho_n_list = digital_filtering.smooth(rho_n_list, sample_k, dx, w_smooth)
    phi_n_list = numpy.concatenate(
        ([0], rho_n_list[1:] * ksqi_over_epsilon))  # calculate phi(k) from rho(k)
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
        nearest_left_grid_point, nearest_right_grid_point = specie.nearest_grids(ng)
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
        v_data[i][0] = species[i].vx(sin_theta, cos_theta)[plot_particles_id]

    # FIRST STEP IN TIME
    particle_mover.move_particles_init(species, grids, dx, dt, ng, length, bx0, sin_theta, cos_theta, bz0, e_ext, is_electromagnetic)
    particles_to_grids.weigh_to_grid(grids, species, ng, dx, sin_theta, cos_theta)
    if is_electromagnetic:
        field_solver.solve_field(grids, dt, epsilon, sqrt_mu_over_epsilon)

    # STORE THE E FIELD FOR PLOTTING
    ex_list[0] = grids.ex
    ey_list[0] = grids.ey
    ez_list[0] = grids.ez
    # STORE THE B FIELD FOR PLOTTING
    by_list[0] = grids.by
    bz_list[0] = grids.bz
    # STORE RHO FOR PLOTTING
    rho_list[0] = grids.rho


# 1/K^2/epsilon VALUES TO BE USED IN Ex FIELD SOLVER
ksqi_over_epsilon = 1 / (numpy.arange(1, ng // 2 + 1) * 2 * math.pi / length * numpy.sinc(
    numpy.arange(1, ng // 2 + 1) * dx / length)) ** 2 / epsilon


# MAIN PROGRAM
def main():
    """MAIN PROGRAM"""

    def update_to_move_particles_es(*args):
        args_es = args + (bz0, e_ext)
        particle_mover.move_particles_es(*args_es)

    if is_electromagnetic:
        move_particles = particle_mover.move_particles_em
    else:
        move_particles = update_to_move_particles_es

    def main_loop(time_step):
        """MAIN PROGRAM TO BE LOOPED"""

        # MAIN PROGRAM
        move_particles(species, grids, dx, dt, ng, length, bx0, sin_theta, cos_theta)  # move particles
        particles_to_grids.weigh_to_grid(grids, species, ng, dx, sin_theta, cos_theta)  # weight to grid
        field_solver.solve_field_x(grids, dx, ksqi_over_epsilon, sample_k,
                                   w_smooth)  # solve Ex from rho (Poisson's eqn)
        if is_electromagnetic:
            field_solver.solve_field(grids, dt, epsilon,
                                     sqrt_mu_over_epsilon)  # solve other field quantities (Maxwell's eqn)

        # PRINT REPORT EVERY 100 FRAMES
        if time_step % 100 == 1:
            grids.print(time_step, bx0)

        # STORE DATA PERIODICALLY
        index = time_step * nt_sample / nt
        if index.is_integer():
            # STORE DATA FOR PHASE SPACE PLOT (ELECTRONS ONLY)
            index = int(index)
            for i in range(len(species)):
                x_data[i][index] = species[i].x[plot_particles_id]
                v_data[i][index] = species[i].vx(sin_theta, cos_theta)[plot_particles_id]
            # STORE THE E FIELD FOR PLOTTING
            ex_list[index] = grids.ex
            ey_list[index] = grids.ey
            ez_list[index] = grids.ez
            # STORE THE B FIELD FOR PLOTTING
            by_list[index] = grids.by
            bz_list[index] = grids.bz
            # STORE RHO FOR PLOTTING
            rho_list[index] = grids.rho
            if index == nt_sample - 1:
                return True

        return False

    # RUN THE MAIN LOOP FOR A NUMBER OF TIME STEPS
    for count in range(1, nt):
        if main_loop(count):
            break

    # OUTPUT RUNTIME
    end_time = time.monotonic()
    print("Duration = ", timedelta(seconds=end_time - start_time))
    print("---------------------------RUNTIME---------------------------", file=log)
    print("Duration                        = {}".format(timedelta(seconds=end_time - start_time)), file=log)

    # STORE K-VALUES OF THE INITIAL DENSITY WAVES
    init_d_k_list = []
    for i in init_d_wv:
        init_d_k_list.append(init_d_wv[i][0] * 2 * math.pi / length / slu)

    # STORE K-VALUES OF THE INITIAL VELOCITY WAVES
    init_k_list = []
    for i in init_v_wv:
        k_temp = []
        for j in init_v_wv[i]:
            k_temp.append(init_v_wv[i][j][0] * 2 * math.pi / length / slu)
        init_k_list.append(k_temp)

    # CLOSE LOG FILE
    log.close()
    # SAVE ARRAYS TO FILE
    zarr.save('{}.zip'.format(name),
              kte=kte, kti=kti, mi=mi_real, b0=b0, rho=rho_mass,
              ex=ex_list, ey=ey_list, ez=ez_list, by=by_list, bz=bz_list, rho_list=rho_list, x=x_data, vx=v_data,
              ng=ng, nt=nt_sample,
              smu=smu, scu=scu,
              slu=slu, stu=stu, c=c, dx=dx, dt=dt_sample, wp=wp, wpi=wpi, wc=wc, wci=wci, theta=theta, v_th=v_th,
              vi_th=vi_th, init_d_k=init_d_k_list, init_k=init_k_list, grid_x=grids.x)


#  INDICES OF PARTICLES TO BE PLOTTED IN THE ANIMATION
if ni == 0:
    plot_particles_id = rng.choice(ne, n_sample)
else:
    plot_particles_id = rng.choice(min(ne, ni), n_sample)

#  CREATE ARRAYS FOR STORING PHASE SPACE DATA FOR ANIMATION
x_data = numpy.zeros(shape=(len(species), nt_sample, n_sample))
v_data = numpy.zeros(shape=(len(species), nt_sample, n_sample))

# INITIALIZE ARRAYS TO STORE DATA
ex_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ex VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
ey_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ey VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
ez_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Ez VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
by_list = numpy.zeros(shape=(nt_sample, ng))  # STORE By VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
bz_list = numpy.zeros(shape=(nt_sample, ng))  # STORE Bz VALUES FOR SPECTRUM PLOTTING (2D ARRAY)
rho_list = numpy.zeros(shape=(nt_sample, ng))  # STORE RHO VALUES FOR SPECTRUM PLOTTING (2D ARRAY)

# INITIALIZE GRIDS
initialize()

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
