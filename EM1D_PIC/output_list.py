import numpy

from . import user_input


class OutputList:
    def __init__(self, output_name_list, n_sp, n_out_sp, n_sample, nt_sample=user_input.nt_sample, ng=user_input.ng):
        """
        Generate lists of outputs
        :param output_name_list: names of requested output
        :param n_sp: number of species
        :param n_out_sp: number of species of which x and v will be outputted
        :param n_sample: number of particles to plot in phase space plot (per processor)
        :param nt_sample: number of time steps to record
        :param ng: number of grid cells
        """
        self.output_list = []
        self.output_j_list = []
        for name in output_name_list:
            if name in ["x", "v"]:
                setattr(self, name, numpy.zeros(shape=(n_out_sp, nt_sample, n_sample)))
            elif name in ["ex", "ey", "ez", "by", "bz", "rho"]:
                setattr(self, name, numpy.zeros(shape=(nt_sample, ng)))
                self.output_list.append(name)
            elif name in ["jy", "jz"]:
                setattr(self, name, numpy.zeros(shape=(nt_sample, ng)))
                self.output_j_list.append(name)
            elif name == "den":
                setattr(self, name, numpy.zeros(shape=(nt_sample, n_sp, ng)))
                self.output_list.append(name)
            else:
                assert False, "INVALID OUTPUT SELECTION!"

    def update_xv_output(self, index, species, plot_particles_id, sin_theta, cos_theta):
        """
        Update x and v data for phase space plot
        :param index: time step to update
        :param species: particle list
        :param plot_particles_id: indices of selected particles to be plotted
        :param sin_theta: sine of b0 angle
        :param cos_theta: cosine of b0 angle
        :return: none
        """
        counter = 0
        if hasattr(self, "x") and hasattr(self, "v"):
            for specie in species:
                if specie.is_output:
                    self.x[counter][index] = specie.x[plot_particles_id]
                    self.v[counter][index] = specie.vx(sin_theta, cos_theta)[plot_particles_id]
                    counter += 1
        elif hasattr(self, "x"):
            for specie in species:
                self.x[counter][index] = specie.x[plot_particles_id]
                counter += 1
        elif hasattr(self, "v"):
            for specie in species:
                self.v[counter][index] = specie.vx(sin_theta, cos_theta)[plot_particles_id]
                counter += 1

    def update_output(self, index, grids):
        """
        Update grid data for time plot and spectrum plot
        :param index: time step to update
        :param grids: grid list
        :return: none
        """
        for name in self.output_list:
            getattr(self, name)[index] = getattr(grids, name)
        for name in self.output_j_list:
            getattr(self, name)[index] = 0.5 * (getattr(grids, name + "_left") + getattr(grids, name + "_right"))

    def get_output(self, *args):
        """
        Get output arrays and store them in variables
        :param args: names of output arrays
        :return: requested output arrays
        """
        output = ()
        for attr in args:
            output += (getattr(self, attr, None),)
        return output
