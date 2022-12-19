import numpy


class ParticleList:
    def __init__(self, x, np, vy=None, vxp=None, vb0=None,
                 x_old=None, vy_old=None,
                 vxp_old=None, vb0_old=None, left_grid=None,
                 left_grid_old=None):
        """
        Store particle data.
        :param x: position
        :param np: number of particles
        :param vy: velocity in the y direction
        :param vxp: velocity in the x' direction
        :param vb0: velocity in the B_0 direction
        :param x_old: position at previous time step
        :param vy_old: velocity in the y direction at previous time step
        :param vxp_old: velocity in the x' direction at previous time step
        :param vb0_old: velocity in the B_0 direction at previous time step
        :param left_grid: the immediate grids to the left of particles
        :param left_grid_old: the immediate grids to the left of particles at previous time step
        """
        self.x = x
        self.vy = vy
        self.vxp = vxp
        self.vb0 = vb0
        self.x_old = x_old
        self.vy_old = vy_old
        self.vxp_old = vxp_old
        self.vb0_old = vb0_old
        self.left_grid = left_grid
        self.left_grid_old = left_grid_old
        val_dict = dict(vy=vy, vxp=vxp, vb0=vb0, x_old=x_old, vy_old=vy_old, vxp_old=vxp_old, vb0_old=vb0_old,
                        left_grid=left_grid, left_grid_old=left_grid_old)
        # SET INITIAL VALUES
        for key in val_dict.keys():
            if val_dict[key] is None:
                val = numpy.zeros(np)
            else:
                val = val_dict[key]
            setattr(self, key, val)
        # self.v = numpy.sqrt(vxp ** 2 + vy ** 2 + vb0 ** 2) # TO BE DELETED
        # self.v_old = numpy.sqrt(vxp_old ** 2 + vy_old ** 2 + vb0_old ** 2) # TBD

        # https://docs.python.org/3/library/functions.html#property

    def update_nearest_grid(self, dx):
        """
        Update the nearest left grids of the particles
        :param dx: grid size
        :return: none
        """
        floored = numpy.floor(self.x / dx).astype(int)
        self.left_grid_old = self.left_grid
        self.left_grid = floored

    def nearest_grids(self, ng):
        """
        Output the nearest left and right grids of the particles
        :param ng: number of grids
        :return: nearest left grids, nearest right grids
        """
        left = self.left_grid
        right = left + 1
        right[right == ng] = 0
        return left, right

    def nearest_grids_old(self, ng):
        """
        Output the nearest old left and right grids of the particles
        :param ng: number of grids
        :return: old nearest left grids, old nearest right grids
        """
        left_old = self.left_grid_old
        right_old = left_old + 1
        right_old[right_old == ng] = 0
        return left_old, right_old

    def vx(self, sin_theta, cos_theta):
        """
        Return a list of vx (particle velocities in the x direction)
        :param sin_theta: sin(theta), theta = angle between b0 and z
        :param cos_theta: cos(theta), theta = angle between b0 and z
        :return: vx
        """
        return self.vxp * cos_theta + self.vb0 * sin_theta


class ElectronList(ParticleList):
    def __init__(self, x, ne, qe, me, qm, **kwargs):
        """
        Store PIC electron data
        :param x: particle position
        :param ne: number of PIC electrons
        :param qe: PIC electron charge
        :param me: PIC electron mass
        :param qm: electron charge-to-mass ratio
        :param kwargs: see parent class ParticleList
        """
        super().__init__(x, ne, **kwargs)
        self.q = -qe
        self.m = me
        self.qm = -qm


# DEFINE CLASS TO STORE PIC ION DATA
class IonList(ParticleList):
    def __init__(self, x, ni, qi, mi, qmi, **kwargs):
        """
        Store PIC ion data
        :param x: particle position
        :param ni: number of PIC ions
        :param qi: PIC ion charge
        :param mi: PIC ion mass
        :param qmi: ion charge-to-mass ratio
        :param kwargs: see parent class ParticleList
        """
        super().__init__(x, ni, **kwargs)
        self.q = qi
        self.m = mi
        self.qm = qmi
