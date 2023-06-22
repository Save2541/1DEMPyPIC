from . import field_solver
from . import particle_mover
from . import particles_to_grids
from . import qol
from . import user_input


def first_step(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, bz0, e_ext, epsilon, sqrt_mu_over_epsilon,
               comm, is_electromagnetic=user_input.is_electromagnetic):
    """
    First step in time
    :param species: particle list
    :param grids: grid list
    :param dx: grid size
    :param dt: time step
    :param length: spatial length of simulation
    :param bx0: external magnetic field in the x direction
    :param sin_theta: sine of b0 angle
    :param cos_theta: cosing of b0 angle
    :param bz0: external magnetic field in the z direction
    :param e_ext: external electric field in the y direction
    :param epsilon: epsilon naught
    :param sqrt_mu_over_epsilon: sqrt(mu_0/epsilon_0)
    :param comm: mpi comm
    :param is_electromagnetic: flag for EM code
    :return: none
    """
    particle_mover.move_particles_init(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, bz0, e_ext)
    particles_to_grids.weigh_to_grid(grids, species, dx, sin_theta, cos_theta, comm)
    if is_electromagnetic:
        field_solver.solve_field(grids, dt, epsilon, sqrt_mu_over_epsilon)


def initialize(species, grids, almanac, sample_k, ksqi_over_epsilon, output, plot_particles_id, comm):
    """
    Initialize grid values and particle values
    :param species: particle list
    :param grids: grid list
    :param almanac: dictionary of useful numbers
    :param sample_k: list of sample frequencies k to be sent to the smoothing function
    :param ksqi_over_epsilon: list of values of k-squared-inverse divided by epsilon naught
    :param output: output list
    :param plot_particles_id: indices of selected particles to be plotted in phase space plot
    :param comm: mpi comm
    :return: none
    """
    # GET VALUES FROM THE ALMANAC
    (dx, dt, b0, bx0, bz0, e_ext, sin_theta, cos_theta, length, epsilon, sqrt_mu_over_epsilon) = qol.read_almanac(
        almanac, "dx", "dt", "b0", "bx0", "bz0", "e_ext", "sin_theta", "cos_theta", "length", "epsilon",
        "sqrt_mu_over_epsilon")

    # INITIAL WEIGHTING (x to rho, hat function)
    particles_to_grids.init_weigh_to_grid(species, grids, dx, comm)

    # INITIAL FIELD SOLVER (Find Ex)
    field_solver.solve_field_x(grids, dx, ksqi_over_epsilon, sample_k)

    # INITIAL PARTICLE MOVER (MOVE BACK V FROM t = 0 to t = - dt/2, hat function)
    particle_mover.move_back_v(species, grids, dx, dt, b0, e_ext, sin_theta, cos_theta)

    # OUTPUT INITIAL DATA FOR PHASE SPACE PLOT
    output.update_xv_output(0, species, plot_particles_id, sin_theta, cos_theta)

    # FIRST STEP IN TIME
    first_step(species, grids, dx, dt, length, bx0, sin_theta, cos_theta, bz0, e_ext, epsilon, sqrt_mu_over_epsilon,
               comm)

    # OUTPUT FIELD QUANTITIES FOR PLOTTING
    output.update_output(0, grids)
