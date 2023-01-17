import numpy


class SpecieList:
    def __init__(self, n_sp, name=None, mass=None, real_mass=None, charge=None, real_charge=None, density=None,
                 temperature=None,
                 drift_velocity=None, np=None, init_d_wv=None, init_v_wv=None, qm=None, wp=None, wc=None, kt=None,
                 vth=None):
        """
        Store specie data
        :param n_sp: number of species
        :param name: list of specie names
        :param mass: list of specie masses
        :param real_mass: list of real specie masses
        :param charge: list of specie charges
        :param charge: list of real specie charges
        :param density: list of specie densities
        :param temperature: list of specie temperatures
        :param drift_velocity: list of specie drift velocities
        :param np: list of specie number of particles
        :param init_d_wv: dictionary of specie initial density waves
        :param init_v_wv: dictionary of specie initial velocity waves
        :param qm: list of specie charge-to-mass ratios
        :param wp: list of specie plasma frequencies
        :param wc: list of specie cyclotron frequencies
        :param kt: list of specie thermal energies
        :param vth: list of specie thermal velocities
        """
        self.n_sp = n_sp
        self.name = name
        self.mass = mass
        self.real_mass = real_mass
        self.charge = charge
        self.real_charge = real_charge
        self.density = density
        self.temperature = temperature
        self.drift_velocity = drift_velocity
        self.np = np
        self.init_d_wv = init_d_wv
        self.init_v_wv = init_v_wv
        self.qm = qm
        self.wp = wp
        self.wc = wc
        self.kt = kt
        self.vth = vth
        # SET INITIAL VALUES
        if name is None:
            self.name = []
        if np is None:
            self.np = numpy.zeros(n_sp, dtype=int)
        if init_d_wv is None:
            self.init_d_wv = {}
        if init_v_wv is None:
            self.init_v_wv = {}
        val_dict = {"mass": mass, "real_mass": real_mass, "charge": charge, "real_charge": real_charge,
                    "density": density, "temperature": temperature, "drift_velocity": drift_velocity, "qm": qm,
                    "wp": wp, "wc": wc, "kt": kt, "vth": vth}
        for key in val_dict.keys():
            if val_dict[key] is None:
                setattr(self, key, numpy.zeros(n_sp))

    def get_info(self, *args):
        """
        Get info from the list and store them in variables
        :param args: attribute names
        :return: requested attributes
        """
        output = ()
        for attr in args:
            output += (getattr(self, attr),)
        return output
