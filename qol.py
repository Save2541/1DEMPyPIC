import constants
import math
import numpy
import scipy.fft
import user_input


def electronvolt_to_kelvin(ev):
    """
    Function to convert energy (eV) to temperature (K)
    :param ev: energy in electron volts
    :return: temperature in Kelvins
    """
    return ev * constants.qe_real / constants.kb


def realistic_electrons():
    """
    Function to set the mass and charge of a specie to be those of electrons.
    :return: dictionary describing the mass and charge of the specie
    """
    return {"mass": constants.me_real, "charge": -constants.qe_real}


def realistic_protons():
    """
    Function to set the mass and charge of a specie to be those of protons.
    :return: dictionary describing the mass and charge of the specie
    """
    return {"mass": constants.mp_real, "charge": constants.qe_real}


def realistic_ions(mass, charge):
    """
    Function to set the mass and charge of a specie to be those of some specific ions
    :param mass: mass number
    :param charge: charge number
    :return: dictionary describing the mass and charge of the specie
    """
    return {"mass": mass * constants.mp_real, "charge": charge * constants.qe_real}


def no_initial_density_wave():
    """
    Function to set no initial density wave
    :return: dictionary describing the initial density wave of the specie
    """
    return {"initial density wave": {
        "number of waves": 0,
        "amplitude": 0
    }}


def no_initial_velocity_wave():
    """
    Function to set no initial velocity wave in any direction
    :return: dictionary describing the initial velocity wave of the specie
    """
    return {"initial velocity wave": {
        "vxp": {
            "number of waves": 0,
            "amplitude": 0
        },
        "vy": {
            "number of waves": 0,
            "amplitude": 0
        },
        "vb0": {
            "number of waves": 0,
            "amplitude": 0
        }
    }}


def no_initial_wave():
    """
    Function to set no initial density wave AND no initial velocity wave in any direction
    :return: dictionary describing the initial density wave and the initial velocity wave of the specie
    """
    return {**no_initial_density_wave(), **no_initial_velocity_wave()}


def number_of_waves_to_wave_number(nw, length):
    """
    Function to convert number of waves to wave number (k)
    :param nw: number of waves
    :param length: simulation length
    :return: wave number (k)
    """
    return 2 * math.pi * nw / length


def sanity_check(sp_list, almanac):
    """
    Check if setup is reasonable.
    :param sp_list: list of specie parameters
    :param almanac: dictionary of general parameters
    :return: None
    """
    assert numpy.all(sp_list.drift_velocity < constants.c), "DRIFT VELOCITY GREATER THAN THE SPEED OF LIGHT!"
    assert numpy.all(sp_list.vth < constants.c), "THERMAL VELOCITY GREATER THAN THE SPEED OF LIGHT!"
    minimum = numpy.amin(sp_list.np)
    assert user_input.n_sample <= minimum, "NOT ENOUGH PARTICLES TO STORE! n_sample CANNOT BE GREATER THAN {}!".format(
        minimum)
    assert user_input.nt_sample <= user_input.nt, "NOT ENOUGH TIME STEPS TO STORE! nt_sample CANNOT BE GREATER THAN {}!".format(
        user_input.nt)
    if user_input.is_electromagnetic:
        assert constants.c * almanac["dt"] == almanac["dx"], "WRONG TIME STEP FOR EM CODE! USE dt = 1 * dx / c."
    if sp_list.wp[0] * almanac["dt"] > 0.3:
        print("WARNING: TIME STEP TOO LARGE! TO GET AN ACCURATE RESULT, USE dt <= ",
              0.3 / sp_list.wp[0] / almanac["dx"] * constants.c, "* dx / c.")
    if sp_list.wp[0] * almanac["dt"] >= 0.35:
        print("WARNING: MAGNETIC FIELD IS TOO STRONG! TO GET AN ACCURATE RESULT, USE B_0 < ",
              sp_list.wc[0] / abs(sp_list.qm[0]), " T.")
    if almanac["dx"] > almanac["lambda_d"]:
        print("WARNING: GRID SIZE IS BIGGER THAN THE DEBYE LENGTH!")


def read_almanac(almanac, *args):
    """
    Read values from almanac
    :param almanac: dictionary of useful quantities
    :param args: names of entries to be read
    :return: requested values
    """
    output = ()
    for key in args:
        output += (almanac[key],)
    return output


def get_sample_frequencies(dx, ng=user_input.ng):
    """
    Get sample frequencies
    :param dx: grid size
    :param ng: number of grid cells
    :return: list of sample frequencies
    """
    return 2 * math.pi * scipy.fft.rfftfreq(ng, dx)


def get_ksqi_over_epsilon(almanac):
    """
    Get 1/K^2/epsilon values to be used in Ex solver
    :param almanac: dictionary of useful quantities
    :return: list of 1/K^2/epsilon values
    """
    (length, dx, epsilon) = read_almanac(almanac, "length", "dx", "epsilon")
    return 1 / (numpy.arange(1, user_input.ng // 2 + 1) * 2 * math.pi / length * numpy.sinc(
        numpy.arange(1, user_input.ng // 2 + 1) * dx / length)) ** 2 / epsilon
