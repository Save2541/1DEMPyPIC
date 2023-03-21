from . import constants
from . import qol


def generate_plasma(preset):
    """
    Function to generate plasma with specific configurations.
    :param preset: number key
    :return: dictionary of plasma species
    """
    if preset == 1:
        specie_names = {
            "electron_r": {
                **qol.realistic_electrons(),
                "number density": 1E8,
                "temperature": 1E4,
                "drift velocity": 120E4,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            },
            "electron_l": {
                **qol.realistic_electrons(),
                "number density": 1E8,
                "temperature": 1E4,
                "drift velocity": -120E4,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            }
        }
    elif preset == 2:
        specie_names = {
            "electron": {
                **qol.realistic_electrons(),
                "number density": 1E8,
                "temperature": 1E4,
                "drift velocity": 0,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            },
            "proton": {
                "mass": 100 * constants.me_real,
                "charge": constants.qe_real,
                "number density": 1E8,
                "temperature": 1E3,
                "drift velocity": 0,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            }
        }
    elif preset == 3:
        specie_names = {
            "electron": {
                **qol.realistic_electrons(),
                "number density": 1E8,
                "temperature": qol.electronvolt_to_kelvin(1),
                "drift velocity": 251450, #5.9E4,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            },
            "proton": {
                ##**qol.realistic_protons(),
                "mass": constants.me_real * 100,
                "charge": constants.qe_real,
                "number density": 0.5E8,
                "temperature": qol.electronvolt_to_kelvin(100),
                "drift velocity": 502900, #1.18E5,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            },
            "oxygen ion": {
                ##**qol.realistic_ions(16, 1),
                "mass": constants.me_real * 100 * 16,
                "charge": constants.qe_real,
                "number density": 0.5E8,
                "temperature": qol.electronvolt_to_kelvin(0.4),
                "drift velocity": 0,
                "number of simulated particles per grid cell": 10,
                **qol.no_initial_wave()
            }
        }
    else:
        assert False, "INVALID SCENARIO SELECTION!"
    return specie_names
