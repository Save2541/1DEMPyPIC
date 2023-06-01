import numpy

from . import user_input


class GridPointList:
    def __init__(self, x, almanac, n_sp, rho=None, jy_left=None, jy_right=None, jz_left=None, jz_right=None,
                 ex=None, ey=None, ez=None, by=None, bz=None, f_right=None, f_left=None, g_right=None, g_left=None,
                 den=None):
        """
        Store grid data
        :param x: position of the grid point
        :param almanac: dictionary of useful quantities
        :param n_sp: number of species
        :param rho: charge density
        :param jy_left: left going current density Jy
        :param jy_right: right going current density Jy
        :param jz_left: left going current density Jz
        :param jz_right: right going current density Jz
        :param ex: electric field x
        :param ey: electric field y
        :param ez: electric field z
        :param by: magnetic field y
        :param bz: magnetic field z
        :param f_right: right going field quantity F
        :param f_left: left going field quantity F
        :param g_right: right going field quantity G
        :param g_left: left going field quantity G
        :param den: number density of each specie
        Definition of field quantities: F_right/left = 1/2*[Ey (+/-) Bz], G_right/left = 1/2*[Ez (-/+) By]
        """
        self.x = x  # [x1, x2, x3, ...]
        self.rho = rho
        self.jy_left = jy_left
        self.jy_right = jy_right
        self.jz_left = jz_left
        self.jz_right = jz_right
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.by = by
        self.bz = bz
        self.f_right = f_right
        self.f_left = f_left
        self.g_right = g_right
        self.g_left = g_left
        self.den = den

        bz0_contribution = almanac["bz0"] / (2 * almanac["sqrt_mu_over_epsilon"])
        e_ext_contribution = almanac["epsilon"] * almanac["e_ext"] / 2
        val_dict = dict(rho=rho, jy_left=jy_left, jy_right=jy_right, jz_left=jz_left, jz_right=jz_right, ex=ex, ez=ez,
                        by=by, g_right=g_right, g_left=g_left)
        # SET INITIAL VALUES
        for key in val_dict.keys():
            if val_dict[key] is None:
                setattr(self, key, numpy.zeros(user_input.ng))
        if ey is None:
            self.ey = numpy.zeros(user_input.ng) + almanac["e_ext"]
        if bz is None:
            self.bz = numpy.zeros(user_input.ng) + almanac["bz0"]
        if f_right is None:
            self.f_right = numpy.zeros(user_input.ng) + e_ext_contribution + bz0_contribution
        if f_left is None:
            self.f_left = numpy.zeros(user_input.ng) + e_ext_contribution - bz0_contribution
        if den is None:
            self.den = numpy.zeros(shape=(n_sp, user_input.ng))

    def jy(self, index):
        """
        Return Jy
        """
        return 0.5 * (self.jy_left[index]+self.jy_right[index])

    def jz(self, index):
        """
        Return Jz
        """
        return 0.5 * (self.jz_left[index]+self.jz_right[index])

    def print(self, t, bx0, index=1):
        """
        Print info
        :param t: time step at printing
        :param index: index of grid to print info
        :param bx0: magnetic field in the x-direction (always constant)
        :return: info at the grid
        """
        print(
            "Time = {}. Grid: {}. rho = {:.2e}. j = ({:.2e}, {:.2e}). e = ({:.2e}, {:.2e}, {:.2e}). b = ({:.2e}, "
            "{:.2e}, {:.2e}). (f_left, f_right) = ({:.2e}, {:.2e}). (g_left, g_right) = ({:.2e}, {:.2e}).".format(
                t, index, self.rho[index], self.jy(index), self.jz(index), self.ex[index], self.ey[index],
                self.ez[index], bx0, self.by[index], self.bz[index],
                self.f_left[index], self.f_right[index], self.g_left[index], self.g_right[index]))


def generate_grids(almanac, n_sp):
    """
    Generate grids
    :param almanac: dictionary of useful numbers
    :param n_sp: number of species
    :return: grid list
    """
    return GridPointList(numpy.arange(0, almanac["length"], almanac["dx"]), almanac, n_sp)
