import os
import time
from datetime import timedelta

from . import constants
from . import user_input


def create_log(sp_list, almanac):
    """
    Create a log text file
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    :return: none
    """
    # GET TIME STRING FOR FILE NAME
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if user_input.is_electromagnetic:
        mode = "EM"
    else:
        mode = "ES"
    almanac["file name"] = "{}_{}".format(mode, time_str)
    path = "log/{}.txt".format(almanac["file name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = open(path, "w")
    print("---------------------------PHYSICAL CONSTANTS---------------------------", file=log)
    print("Vacuum permittivity (epsilon_0)                  = {:.2e} F m^-1".format(constants.epsilon), file=log)
    print("Vacuum permeability (mu_0)                       = {:.2e} H m^-1".format(constants.mu), file=log)
    print("Speed of light (c)                               = {:.2e} m s^-1".format(constants.c), file=log)
    print("Electron mass (m_e)                              = {:.2e} kg".format(constants.me_real), file=log)
    print("Proton mass (m_p)                                = {:.2e} kg".format(constants.mp_real), file=log)
    print("Elementary charge (e)                            = {:.2e} C".format(constants.qe_real), file=log)
    print("Boltzmann constant (k_B)                         = {:.2e} J K^-1".format(constants.kb), file=log)
    print("---------------------------EXTERNAL FIELDS---------------------------", file=log)
    print("x-Magnetic field                                 = {:.2e} T".format(almanac["bx0"]), file=log)
    print("z-Magnetic field                                 = {:.2e} T".format(almanac["bz0"]), file=log)
    print("y-Electric field                                 = {:.2e} V m^-1".format(user_input.e_ext), file=log)
    print("---------------------------PLASMA SPECIES---------------------------", file=log)
    for i in range(sp_list.n_sp):
        print("Specie {}: {}".format(i, sp_list.name[i]), file=log)
        print("\tmass (m)                                       = {:.2e} kg".format(sp_list.mass[i]), file=log)
        print("\tcharge (q)                                     = {:.2e} C".format(sp_list.charge[i]), file=log)
        print("\tnumber density (n_0)                           = {:.2e} particles m^-3".format(sp_list.density[i]),
              file=log)
        print("\ttemperature (T)                                = {:.2e} K".format(sp_list.temperature[i]), file=log)
        print("\tdrift velocity (u_0)                           = {:.2e} m s^-1".format(sp_list.drift_velocity[i]),
              file=log)
        print("\tnumber of simulated particles (np)             = {} particles".format(sp_list.np_all[i]), file=log)
        print("\tplasma frequency (omega_p)                     = {:.2e} rad s^-1".format(sp_list.wp[i]), file=log)
        print("\tcyclotron frequency (omega_c)                  = {:.2e} rad s^-1".format(sp_list.wc[i]), file=log)
        print("\tthermal energy (kT)                            = {:.2e} J".format(sp_list.kt[i]), file=log)
        print("\tthermal velocity (v_th)                        = {:.2e} m s^-1".format(sp_list.vth[i]), file=log)
    print("\nDebye length (lambda_D)                            = {:.2e} m".format(almanac["lambda_d"]), file=log)
    print("---------------------------INITIAL DENSITY WAVES---------------------------", file=log)
    for i in range(sp_list.n_sp):
        specie_name = sp_list.name[i]
        print("Specie {}: {}".format(i, specie_name), file=log)
        d_wv = sp_list.init_d_wv[specie_name]
        nw = d_wv["number of waves"]
        print("\tnumber of waves                                = {} waves".format(nw), file=log)
        print("\tamplitude (probability)                        = {}".format(d_wv["amplitude"]), file=log)
        print("\twave number (k)                                = {} rad m^-1".format(d_wv["wave number (k)"]),
              file=log)
    print("---------------------------INITIAL VELOCITY WAVES---------------------------", file=log)
    for i in range(sp_list.n_sp):
        specie_name = sp_list.name[i]
        print("Specie {}: {}".format(i, specie_name), file=log)
        v_wv_all = sp_list.init_v_wv[specie_name]
        for component in ["vxp", "vy", "vb0"]:
            print("\tcomponent: {}".format(component), file=log)
            v_wv = v_wv_all[component]
            nw = v_wv["number of waves"]
            print("\t\tnumber of waves                              = {} waves".format(v_wv["number of waves"]),
                  file=log)
            print("\t\tamplitude                                    = {} m s^-1".format(v_wv["amplitude"]), file=log)
            print("\t\twave number (k)                              = {} rad m^-1".format(v_wv["wave number (k)"]),
                  file=log)
    print("---------------------------SIMULATION GRID PROPERTIES---------------------------", file=log)
    print("number of time steps (nt)                        = {}".format(user_input.nt), file=log)
    print("number of spatial grids (ng)                     = {}".format(user_input.ng), file=log)
    print("duration of one time step (dt)                   = {:.2e} s".format(almanac["dt"]), file=log)
    print("length of one spatial grid (dx)                  = {:.2e} m".format(almanac["dx"]), file=log)
    print("simulation length (L)                            = {:.2e} m".format(almanac["length"]), file=log)
    print("simulation time                                  = {:.2e} s".format(almanac["dt"] * user_input.nt), file=log)
    # CLOSE LOG FILE
    log.close()


def add_to_log(sp_list, almanac, nt=user_input.nt):
    """
    Add to log text file
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful quantities
    :param nt: number of time steps
    :return: none
    """
    log = open("log/{}.txt".format(almanac["file name"]), "a")
    print("---------------------------SIMULATION UNITS---------------------------", file=log)
    print("simulation unit charge                           = {:.2e} C".format(almanac["scu"]), file=log)
    print("simulation unit length                           = {:.2e} m".format(almanac["slu"]), file=log)
    print("simulation unit time                             = {:.2e} s".format(almanac["stu"]), file=log)
    print("simulation unit mass                             = {:.2e} kg".format(almanac["smu"]), file=log)
    print("---------------------------PHYSICAL CONSTANTS (SIMULATION UNITS)---------------------------", file=log)
    print("Vacuum permittivity (epsilon_0)                  = {:.2e}".format(almanac["epsilon"]), file=log)
    print("Vacuum permeability (mu_0)                       = {:.2e}".format(almanac["mu"]), file=log)
    print("Speed of light (c)                               = {:.2e}".format(almanac["c"]), file=log)
    print("---------------------------EXTERNAL FIELDS (SIMULATION UNITS)---------------------------", file=log)
    print("x-Magnetic field                                 = {:.2e}".format(almanac["bx0"]), file=log)
    print("z-Magnetic field                                 = {:.2e}".format(almanac["bz0"]), file=log)
    print("y-Electric field                                 = {:.2e}".format(almanac["e_ext"]), file=log)
    print("---------------------------SIMULATION GRID PROPERTIES (SIMULATION UNITS)---------------------------",
          file=log)
    print("duration of one time step (dt)                   = {:.2e}".format(almanac["dt"]), file=log)
    print("length of one spatial grid (dx)                  = {:.2e}".format(almanac["dx"]), file=log)
    print("simulation length (L)                            = {:.2e}".format(almanac["length"]), file=log)
    print("simulation time                                  = {:.2e}".format(almanac["dt"] * nt), file=log)
    print("---------------------------PIC PARTICLE PROPERTIES (SIMULATION UNITS)---------------------------", file=log)
    for i in range(sp_list.n_sp):
        print("Specie {}: {}".format(i, sp_list.name[i]), file=log)
        print("\tmass (m)                                       = {:.2e}".format(sp_list.mass[i]), file=log)
        print("\tcharge (q)                                     = {:.2e}".format(sp_list.charge[i]), file=log)
        print("\tdrift velocity (u_0)                           = {:.2e}".format(sp_list.drift_velocity[i]), file=log)
        print("\tnumber of particles (np)                       = {}".format(sp_list.np_all[i]), file=log)
        print("\tplasma frequency (omega_p)                     = {:.2e}".format(sp_list.wp[i]), file=log)
        print("\tcyclotron frequency (omega_c)                  = {:.2e}".format(sp_list.wc[i]), file=log)
        print("\tthermal energy (kT)                            = {:.2e}".format(sp_list.kt[i]), file=log)
        print("\tthermal velocity (v_th)                        = {:.2e}".format(sp_list.vth[i]), file=log)
    print("\nDebye length (lambda_D)                            = {:.2e}".format(almanac["lambda_d"]), file=log)
    # CLOSE LOG FILE
    log.close()


def finish_log(start_time, file_name):
    """
    Finish writing log.
    :param start_time: the time when the simulation started
    :param file_name: name of the log text file
    :return: none
    """
    end_time = time.monotonic()
    log = open("log/{}.txt".format(file_name), "a")
    print("Duration = ", timedelta(seconds=end_time - start_time))
    print("---------------------------RUNTIME---------------------------", file=log)
    print("Duration                        = {}".format(timedelta(seconds=end_time - start_time)), file=log)
    # CLOSE LOG FILE
    log.close()
    # PRINT FILE NAME
    print("File name = ", file_name)
