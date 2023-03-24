from . import field_solver
from . import particle_mover
from . import particles_to_grids
from . import qol
from . import user_input


def main(species, grids, almanac, ksqi_over_epsilon, sample_k, output, plot_particles_id, comm, rank, basket,
         nt=user_input.nt, nt_sample=user_input.nt_sample, is_electromagnetic=user_input.is_electromagnetic):
    """
    Loop through time steps
    :param species: particle list
    :param grids: grid list
    :param almanac: dictionary of useful numbers
    :param ksqi_over_epsilon: list of values of k-squared-inverse divided by epsilon naught
    :param sample_k: list of sample frequencies k to be sent to the smoothing function
    :param output: output list
    :param plot_particles_id: indices of selected particles to be plotted in phase space plot
    :param comm: mpi comm
    :param rank: processor rank
    :param basket: reusable array for gathering
    :param nt: number of time steps
    :param nt_sample: number of time steps to be recorded
    :param is_electromagnetic: flag for EM code
    :return: none
    """

    # GET VALUES FROM THE ALMANAC
    (dx, dt, b0, bx0, bz0, e_ext, sin_theta, cos_theta, length, epsilon, sqrt_mu_over_epsilon) = qol.read_almanac(
        almanac, "dx", "dt", "b0", "bx0", "bz0", "e_ext", "sin_theta", "cos_theta", "length", "epsilon",
        "sqrt_mu_over_epsilon")

    def update_to_move_particles_es(*args):
        """
        Change function to ES counterpart when not running EM code
        :param args: arguments originally meant for EM code
        :return: none
        """
        args_es = args + (bz0, e_ext)
        particle_mover.move_particles_es(*args_es)

    if is_electromagnetic:
        move_particles = particle_mover.move_particles_em
    else:
        move_particles = update_to_move_particles_es

    def main_loop(time_step):
        """
        Function to be looped
        :param time_step: current time step
        :return: True if last loop, False if not last loop
        """

        # MAIN PROGRAM
        move_particles(species, grids, dx, dt, length, bx0, sin_theta, cos_theta)  # move particles
        particles_to_grids.weigh_to_grid(grids, species, dx, sin_theta, cos_theta, comm, basket)  # weight to grid
        field_solver.solve_field_x(grids, dx, ksqi_over_epsilon, sample_k)  # solve Ex from rho (Poisson's eqn)
        if is_electromagnetic:
            field_solver.solve_field(grids, dt, epsilon,
                                     sqrt_mu_over_epsilon)  # solve other field quantities (Maxwell's eqn)

        # STORE DATA PERIODICALLY
        index = time_step * nt_sample / nt
        if index.is_integer():
            # STORE DATA FOR PHASE SPACE PLOT (ELECTRONS ONLY)
            index = int(index)
            if rank == 0:
                grids.print(time_step, bx0)
                output.update_xv_output(index, species, plot_particles_id, sin_theta, cos_theta)
                # STORE FIELD DATA FOR PLOTTING
                output.update_output(index, grids)
            if index == nt_sample - 1:
                return True

        return False

    # RUN THE MAIN LOOP FOR A NUMBER OF TIME STEPS
    for count in range(1, nt):
        if main_loop(count):
            break
