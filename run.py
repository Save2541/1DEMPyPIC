import io
import time

import numpy

import output_list
import particle_distributor
import plasma_cauldron
import particle_generator
import grid_generator
import scribe
import unit_scale
import qol
import user_input
import input_compiler
import specie_list
import initializer
import main
import zipper


def run(preset=user_input.preset, n_sample=user_input.n_sample, output_names=user_input.output_names):
    """
    Run the program.
    :param preset: plasma preset selected by the user
    :param n_sample: number of particles whose info will be stored at this end
    :param output_names: list of requested outputs
    :return: none
    """

    # GENERATE PLASMA
    specie_names = plasma_cauldron.generate_plasma(preset)

    # CREATE A SPECIE LIST
    sp_list = specie_list.SpecieList(len(specie_names))

    # COMPILE INPUT AND DERIVE QUANTITIES
    almanac = input_compiler.compile_input(sp_list, specie_names)

    # PERFORM SANITY CHECK (PROGRAM WILL BREAK IF USER INPUT IS UNREASONABLE)
    qol.sanity_check(sp_list, almanac)

    # CREATE LOG (TEXT FILE)
    scribe.create_log(sp_list, almanac)

    # SCALE QUANTITIES TO SIMULATION UNITS
    unit_scale.scale_quantities(sp_list, almanac)

    # RECORD SCALE QUANTITIES TO LOG
    scribe.add_to_log(sp_list, almanac)

    # GENERATE GRID POINTS (FROM LIST OF POSITIONS)
    grids = grid_generator.generate_grids(almanac)

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
    output = output_list.OutputList(output_names, sp_list.n_sp)

    # GET STARTING TIME
    start_time = time.monotonic()

    # INITIALIZE GRIDS
    initializer.initialize(species, grids, almanac, sample_k, ksqi_over_epsilon, output, plot_particles_id)

    # RUN MAIN PROGRAM WITH PROFILER
    if __name__ == '__main__':
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        main.main(species, grids, almanac, ksqi_over_epsilon, sample_k, output, plot_particles_id)
        profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
        stats.print_stats()

        with open('outstats.txt', 'w+') as f:
            f.write(s.getvalue())

    # OUTPUT RUNTIME AND FINISH LOG
    file_name = almanac["file name"]
    scribe.finish_log(start_time, file_name)

    # SAVE DATA TO ZIP FILE
    zipper.save_to_zip(sp_list, almanac, output, grids)


run()
