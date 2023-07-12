import math

from . import constants
from . import qol
from . import user_input


def scale_quantities(sp_list, almanac):
    """
    Scale mks quantities to simulation units
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    """
    # DEFINITION OF UNIT QUANTITIES
    almanac["scu"] = constants.c ** 2 * constants.epsilon * almanac["lambda_d"] / abs(
        sp_list.qm[0])  # simulation unit charge (C)
    almanac["slu"] = almanac["lambda_d"]  # simulation unit length (m)
    almanac["stu"] = almanac["lambda_d"] / constants.c  # simulation unit time (s)
    almanac["smu"] = almanac["scu"] / abs(sp_list.qm[0])  # simulation unit mass (kg)
    stu_over_slu = almanac["stu"] / almanac["slu"]
    smu_over_scu = almanac["smu"] / almanac["scu"]

    # SCALED PHYSICAL CONSTANTS
    almanac["epsilon"] = 1  # scaled vacuum permittivity
    almanac["c"] = 1  # scaled speed of light
    almanac["mu"] = 1 / (almanac["c"] ** 2 * almanac["epsilon"])  # scaled vacuum permeability
    almanac["sqrt_mu_over_epsilon"] = math.sqrt(almanac["mu"] / almanac["epsilon"])

    # SCALED VELOCITY WAVE AMPLITUDES
    for specie in sp_list.init_v_wv:
        v_wv_all = sp_list.init_v_wv[specie]
        for component in v_wv_all:
            v_wv_all[component]["amplitude"] *= stu_over_slu

    # SCALED PLASMA PARAMETERS
    sp_list.qm *= smu_over_scu  # scaled charge-to-mass ratios (q/m)
    sp_list.wp *= almanac["stu"]  # scaled plasma frequencies
    sp_list.wc *= almanac["stu"]  # scaled cyclotron frequencies
    sp_list.kt *= stu_over_slu ** 2 / almanac["smu"]  # scaled thermal energies kT's
    sp_list.drift_velocity *= stu_over_slu  # scaled drift velocities
    sp_list.vth *= stu_over_slu  # scaled thermal velocities
    almanac["lambda_d"] /= almanac["slu"]  # scaled Debye length

    # SCALED EXTERNAL MAGNETIC FIELD
    almanac["b0"] = sp_list.wc[0] / abs(sp_list.qm[0])  # field strength
    almanac["bz0"] = almanac["b0"] * almanac["cos_theta"]  # field component along z
    almanac["bx0"] = almanac["b0"] * almanac["sin_theta"]  # field component along x

    # SCALED EXTERNAL ELECTRIC FIELD
    almanac["e_ext"] = user_input.e_ext * stu_over_slu * almanac["stu"] / smu_over_scu  # field strength

    # SCALED GRID SIZES
    almanac["dx"] /= almanac["slu"]  # spatial grid size
    almanac["length"] = user_input.ng * almanac["dx"]  # length of the system
    almanac["dt"] /= almanac["stu"]  # duration of time step
    almanac["dt_sample"] = almanac["dt"] * user_input.nt / user_input.nt_sample  # duration per sample

    # PROPERTIES OF PIC PARTICLES
    sp_list.charge = sp_list.wp ** 2 * almanac["length"] / sp_list.np_all * almanac[
        "epsilon"] / sp_list.qm  # charge per PIC particle
    sp_list.mass = sp_list.charge / sp_list.qm  # mass per PIC particle


def convert_to_joules(value, almanac):
    """
    Convert energy from simulation units to joules
    :param value: energy in simulation units
    :param almanac: dictionary of useful numbers
    :return: energy in joules
    """
    (smu, slu, stu) = qol.read_almanac(almanac, "smu", "slu", "stu")
    return value * smu * slu ** 2 / stu ** 2


def convert_to_radians_per_second(value, almanac):
    """
    Convert frequency from simulation units to radians per second
    :param value: frequency in simulation units
    :param almanac: dictionary of useful numbers
    :return: frequency in radians per second
    """
    return value / almanac["stu"]


def convert_to_meters_per_second(value, almanac):
    """
    Convert speed from simulation units to meters per second
    :param value: speed in simulation units
    :param almanac: dictionary of useful numbers
    :return: speed in meters per second
    """
    (slu, stu) = qol.read_almanac(almanac, "slu", "stu")
    return value * slu / stu


def convert_to_tesla(value, almanac):
    """
    Convert magnetic field from simulation units to tesla.
    :param value: magnetic field in simulation units
    :param almanac: dictionary of useful numbers
    :return: magnetic field in tesla
    """
    (smu, scu, stu) = qol.read_almanac(almanac, "smu", "scu", "stu")
    return value * smu / (scu * stu)


def convert_to_meters(value, almanac):
    """
    Convert length from simulation units to meters
    :param value: length in simulation units
    :param almanac: dictionary of useful numbers
    :return: length in meters
    """
    return value * almanac["slu"]


def convert_to_seconds(value, almanac):
    """
    Convert time from simulation units to seconds
    :param value: time in simulation units
    :param almanac: dictionary of useful numbers
    :return: time in seconds
    """
    return value * almanac["stu"]


def convert_to_volts_per_meter(value, almanac):
    """
    Convert electric field from simulation units to volts per meter
    :param value: electric field in simulation units
    :param almanac: dictionary of useful numbers
    :return: electric field in volts per meter
    """
    (slu, smu, scu, stu) = qol.read_almanac(almanac, "slu", "smu", "scu", "stu")
    return value * slu * smu / (stu ** 2 * scu)


def convert_to_coulombs_per_cubic_meter(value, almanac):
    """
    Convert charge density from simulation units to coulombs per cubic meters
    :param value: charge density in simulation units
    :param almanac: dictionary of useful numbers
    :return: charge density in coulombs per cubic meter
    """
    (slu, scu) = qol.read_almanac(almanac, "slu", "scu")
    return value * scu / slu ** 3


def convert_to_particles_per_cubic_meter(value, almanac):
    """
    Convert number density from simulation units to particles per cubic meters
    :param value: number density in simulation units
    :param almanac: dictionary of useful numbers
    :return: number density in particles per cubic meter
    """
    return value / almanac["slu"] ** 3


def convert_to_amperes_per_square_meter(value, almanac):
    """
    Convert current density from simulation units to amperes per square meters
    :param value: current density in simulation units
    :param almanac: dictionary of useful numbers
    :return: current density in amperes per square meter
    """
    (slu, scu, stu) = qol.read_almanac(almanac, "slu", "scu", "stu")
    return value * scu / (stu * slu ** 2)
