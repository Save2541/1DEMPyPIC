import io
import time

import numpy

from EM1D_PIC import grid_generator
from EM1D_PIC import initializer
from EM1D_PIC import input_compiler
from EM1D_PIC import main
from EM1D_PIC import multiprocessor
from EM1D_PIC import output_list
from EM1D_PIC import particle_distributor
from EM1D_PIC import particle_generator
from EM1D_PIC import plasma_cauldron
from EM1D_PIC import qol
from EM1D_PIC import scribe
from EM1D_PIC import specie_list
from EM1D_PIC import unit_scale
from EM1D_PIC import user_input
from EM1D_PIC import zipper


def get_user_input(preset, n_sample, output_names):
    """
    Set default value to user input.
    :param preset: plasma preset selected by the user
    :param n_sample: number of particles whose info will be stored at this end
    :param output_names: list of requested outputs
    :return: preset, n_sample, output_names
    """
    if preset is None:
        preset = user_input.preset
    if n_sample is None:
        n_sample = user_input.n_sample
    if output_names is None:
        output_names = user_input.output_names
    return preset, n_sample, output_names


def run(preset=None, n_sample=None, output_names=None):
    """
    Run the program
    :param preset: plasma preset selected by the user
    :param n_sample: number of particles whose info will be stored at this end
    :param output_names: list of requested outputs
    :return: none
    """
    # SETUP MULTIPROCESSING
    comm, size, rank = multiprocessor.setup_mpi()

    # GET USER INPUT
    preset, n_sample, output_names = get_user_input(preset, n_sample, output_names)

    # CONVERT TO n_sample PER PROCESSOR
    n_sample = n_sample // size

    # GENERATE PLASMA
    specie_names = plasma_cauldron.generate_plasma(preset)

    # CREATE A SPECIE LIST
    sp_list = specie_list.SpecieList(len(specie_names))

    # COMPILE INPUT AND DERIVE QUANTITIES
    almanac = input_compiler.compile_input(sp_list, specie_names, size, rank)

    # PERFORM SANITY CHECK (PROGRAM WILL BREAK IF USER INPUT IS UNREASONABLE)
    if rank == 0:
        qol.sanity_check(sp_list, almanac, n_sample)

    # CREATE LOG (TEXT FILE)
        print("Logging...")
        scribe.create_log(sp_list, almanac)

    # SCALE QUANTITIES TO SIMULATION UNITS
    unit_scale.scale_quantities(sp_list, almanac)

    # RECORD SCALE QUANTITIES TO LOG
    if rank == 0:
        scribe.add_to_log(sp_list, almanac)

    # GENERATE GRID POINTS (FROM LIST OF POSITIONS)
    grids = grid_generator.generate_grids(almanac, sp_list.n_sp)

    # CONSTRUCT A RANDOM NUMBER GENERATOR
    rng = numpy.random.default_rng()

    # ARGUMENTS TO BE SENT TO DISTRIBUTOR
    dist_args = (sp_list, almanac, rng)

    # GET PARTICLE POSITION DICTIONARY
    x_list = particle_distributor.distribute_positions(*dist_args)

    # GET PARTICLE VELOCITY DICTIONARY
    v_list = particle_distributor.distribute_velocities(x_list, *dist_args)

    # ARGUMENTS TO BE SENT TO PARTICLE GENERATORS
    args_list = numpy.transpose(numpy.vstack((sp_list.np, sp_list.charge, sp_list.mass, sp_list.qm)))

    # GENERATE PARTICLES (FROM LIST OF POSITIONS AND VELOCITIES)
    species = particle_generator.generate_particles(sp_list, x_list, v_list, args_list)

    # GET SAMPLE FREQUENCIES
    sample_k = qol.get_sample_frequencies(almanac["dx"])

    # GET 1/K^2/epsilon VALUES TO BE USED IN Ex FIELD SOLVER
    ksqi_over_epsilon = qol.get_ksqi_over_epsilon(almanac)

    #  INDICES OF PARTICLES TO BE PLOTTED IN THE ANIMATION
    plot_particles_id = rng.choice(numpy.amin(sp_list.np), n_sample)

    # INITIALIZE ARRAYS TO STORE OUTPUT DATA
    args = (sp_list.n_sp, len(sp_list.out_sp), n_sample)
    if rank == 0:
        output = output_list.OutputList(output_names, *args)

    else:
        new_output_names = []
        if "x" in output_names:
            new_output_names.append("x")
        if "v" in output_names:
            new_output_names.append("v")
        output = output_list.OutputList(new_output_names, *args)

    # GET STARTING TIME
    start_time = time.monotonic()

    # INITIALIZE GRIDS
    initializer.initialize(species, grids, almanac, sample_k, ksqi_over_epsilon, output, plot_particles_id, comm)

    # RUN MAIN PROGRAM WITH PROFILER
    if __name__ == '__main__':
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        main.main(species, grids, almanac, ksqi_over_epsilon, sample_k, output, plot_particles_id, comm, rank)
        profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
        stats.print_stats()

        if rank == 0:
            with open('outstats.txt', 'w+') as f:
                f.write(s.getvalue())

    # GATHER XV OUTPUT
    if rank == 0:
        print("GATHERING PARTICLES...")
    x_data, v_data = multiprocessor.gather_xv(output, comm, size, rank)

    if rank == 0:
        # OUTPUT RUNTIME AND FINISH LOG
        file_name = almanac["file name"]
        scribe.finish_log(start_time, file_name)

        if hasattr(output, "x"):
            output.x = x_data
        if hasattr(output, "v"):
            output.v = v_data

        # SAVE DATA TO ZIP FILE
        print("SAVING DATA...")
        zipper.save_to_zip(sp_list, almanac, output, grids)


run()
