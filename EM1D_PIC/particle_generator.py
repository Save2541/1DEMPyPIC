import numpy


class ParticleList:
    def __init__(self, x, np, q, m, qm, is_output=True, vy=None, vxp=None, vb0=None,
                 x_old=None, vy_old=None,
                 vxp_old=None, vb0_old=None, left_grid=None,
                 left_grid_old=None):
        """
        Store particle data.
        :param x: position
        :param np: number of particles
        :param q: PIC particle charge
        :param m: PIC particle mass
        :param qm: charge-to-mass ratio
        :param is_output: whether the specie x or v will be outputted
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
        self.q = q
        self.m = m
        self.qm = qm
        self.is_output = is_output
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
                setattr(self, key, numpy.zeros(int(np)))

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


def generate_particles(sp_list, x_list, v_list, args_list):
    """
    Generate particles
    :param sp_list: list of specie parameters
    :param x_list: list of particle positions
    :param v_list: list of particle velocities
    :param args_list: other arguments
    :return: list of particles
    """
    species = []
    for i in range(0, sp_list.n_sp):
        specie_name = sp_list.name[i]
        if i in sp_list.out_sp:
            is_output = True
        else:
            is_output = False
        species.append(ParticleList(x_list[specie_name],  # uniform position distribution
                                    *args_list[i],
                                    is_output=is_output,
                                    # gaussian velocity distribution
                                    vxp=v_list[specie_name]["vxp"],
                                    vy=v_list[specie_name]["vy"],
                                    vb0=v_list[specie_name]["vb0"]))
    return species
