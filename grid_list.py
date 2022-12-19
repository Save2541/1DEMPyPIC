import numpy
import math


class GridPointList:
    def __init__(self, x, ng, epsilon, mu, bz0, e_ext, rho=None, jy_old=None, jy=None, jz_old=None, jz=None,
                 ex=None, ey=None, ez=None, by=None, bz=None,
                 f_right_old=None, f_left_old=None, f_right=None, f_left=None,
                 g_right_old=None, g_left_old=None, g_right=None, g_left=None):
        """
        Store grid data
        :param x: position of the grid point
        :param ng: number of grids
        :param epsilon: permittivity of free space (epsilon_0)
        :param mu: vacuum permeability (mu_naught)
        :param bz0: initial magnetic field (b0)
        :param e_ext: initial electric field (y)
        :param rho: charge density
        :param jy_old: current density y, previous
        :param jy: current density y, current
        :param jz_old: current density z, previous
        :param jz: current density z, current
        :param ex: electric field x
        :param ey: electric field y
        :param ez: electric field z
        :param by: magnetic field y
        :param bz: magnetic field z
        :param f_right_old: right going field quantity F, previous
        :param f_left_old: left going field quantity F, previous
        :param f_right: right going field quantity F, current
        :param f_left: left going field quantity F, current
        :param g_right_old: right going field quantity G, previous
        :param g_left_old: left going field quantity G, previous
        :param g_right: right going field quantity G, current
        :param g_left: left going field quantity G, current
        Definition of field quantities: F_right/left = 1/2*[Ey (+/-) Bz], G_right/left = 1/2*[Ez (-/+) By]
        """
        self.x = x
        self.rho = rho
        self.jy_old = jy_old
        self.jy = jy
        self.jz_old = jz_old
        self.jz = jz
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.by = by
        self.bz = bz
        self.f_right_old = f_right_old
        self.f_left_old = f_left_old
        self.f_right = f_right
        self.f_left = f_left
        self.g_right_old = g_right_old
        self.g_left_old = g_left_old
        self.g_right = g_right
        self.g_left = g_left

        bz0_contribution = math.sqrt(epsilon / mu) * bz0 / 2
        e_ext_contribution = epsilon * e_ext / 2
        val_dict = dict(rho=rho, jy_old=jy_old, jy=jy, jz_old=jz_old, jz=jz, ex=ex, ey=ey, ez=ez, by=by, bz=bz,
                        f_right_old=f_right_old, f_left_old=f_left_old, f_right=f_right, f_left=f_left,
                        g_right_old=g_right_old, g_left_old=g_left_old, g_right=g_right, g_left=g_left)
        # SET INITIAL VALUES
        for key in val_dict.keys():
            if val_dict[key] is None:
                if key == "ey":
                    val = numpy.zeros(ng) + e_ext
                elif key == "bz":
                    val = numpy.zeros(ng) + bz0
                elif key == "f_right_old" or key == "f_right":
                    val = numpy.zeros(ng) + e_ext_contribution + bz0_contribution
                elif key == "f_left_old" or key == "f_left":
                    val = numpy.zeros(ng) + e_ext_contribution - bz0_contribution
                else:
                    val = numpy.zeros(ng)
            else:
                val = val_dict[key]
            setattr(self, key, val)

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
                t, index, self.rho[index], self.jy[index], self.jz[index], self.ex[index], self.ey[index],
                self.ez[index], bx0, self.by[index], self.bz[index],
                self.f_left[index], self.f_right[index], self.g_left[index], self.g_right[index]))
