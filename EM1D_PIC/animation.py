import os

import matplotlib.pyplot as plt
import numpy
import zarr
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, FFMpegWriter

from . import plot_config
from . import qol


def y_axis_limit(value_list):
    """
    Set y-axis limit for plots
    :param value_list: values to be plotted
    :return: y-axis limit
    """
    maximum = numpy.amax(value_list)
    minimum = numpy.amin(value_list)
    if minimum != maximum:
        return [minimum, maximum]
    else:
        return [-1, 1]


def set_default_parameters(specie, anim, time_step, add_plots, add_plot_keys, dot_size, plot_species, plot_colors, fps, frame_interval):
    """
    Set default parameters for the function plot_non_fourier
    :param specie: specie to focus on in the phase space animation
    :param anim: flag for animation
    :param time_step: time step to be plotted, will be set to 0 for animation
    :param add_plots: shape of the additional plots above the phase space animation
    :param add_plot_keys: values to be plotted in the additional plots
    :param dot_size: size of particles in phase space plot
    :param plot_species: species to plot in phase space
    :param plot_colors: color to use for each specie
    :param fps: frame rate per second
    :param frame_interval: number of time steps to skip per frame plus one
    :return: parameters set to user-inputted values in plot_config
    """
    if specie is None:
        specie = plot_config.specie_of_interest
    if time_step is None:
        time_step = plot_config.time
    if add_plots is None:
        add_plots = plot_config.add_plots
    if add_plot_keys is None:
        add_plot_keys = plot_config.add_plots_keys
    if dot_size is None:
        dot_size = plot_config.dot_size
    if plot_species is None:
        plot_species = plot_config.plot_species
    if plot_colors is None:
        plot_colors = plot_config.plot_colors
    if fps is None:
        fps = plot_config.fps
    if frame_interval is None:
        frame_interval = plot_config.frame_interval
    if anim:
        time_step = 0
    if specie is None:
        specie = numpy.s_[:]
    if plot_species == "all":
        plot_species = numpy.s_[:]

    return specie, time_step, add_plots, add_plot_keys, dot_size, plot_species, plot_colors, fps, frame_interval


def sanity_check(add_plot_keys, add_plots, plot_species, plot_colors, n_sp):
    """
    Check if user input is reasonable
    :param add_plot_keys: values to be plotted in the additional plots
    :param add_plots: shape of the additional plots above the phase space animation
    :param plot_species: species to be plotted in phase space
    :param plot_colors: color to use for each specie
    :param n_sp: number of species
    :return: none
    """
    assert numpy.array(add_plot_keys).shape == add_plots, "The shape of add_plot_keys is not equal to add_plots!"
    assert len(plot_colors) == len(plot_species), "The length of plot_colors is not equal to the length of plot_species!"
    assert len(plot_species) <= n_sp, "The length of plot_species is greater than the total number of species!"


def set_variables(x, ng, dx):
    """
    Set some variables
    :param x: grid positions
    :param ng: number of grids
    :param dx: grid size
    :return: number of species, number of sample particles per specie, spatial length of the simulation
    """
    return x.shape[0], x.shape[2], ng * dx


def merge_bottom_subplots(fig, axs):
    """
    Merge all subplots in the bottom row to create a long subplot for phase space animation
    :param fig: figure
    :param axs: axes
    :return: merged bottom axis
    """
    gs = axs[-1, 0].get_gridspec()
    # Remove the underlying axes
    for ax in axs[-1, :]:
        ax.remove()
    return fig.add_subplot(gs[-1, :])


def set_up_add_plots(loader, add_plots, add_plot_keys, axs, length, grid_x, time_step):
    """
    Set up the additional plots
    :param loader: loaded data file
    :param add_plots: shape of the additional plots
    :param add_plot_keys: values to be plotted in the additional plots
    :param axs: axes of subplots
    :param length: spatial length of the simulation
    :param grid_x: list of grid positions
    :param time_step: time step to be plotted, will be 0 for animation.
    :return: list of lines to be animated, list of value arrays
    """
    array_list = []
    line_list = []
    for i in range(add_plots[0]):
        for j in range(add_plots[1]):
            key = add_plot_keys[i][j]
            if key == "rho":
                array = loader['rho_list']
                title = "rho"
                units = "C/m^3"
            elif key in ["ex", "ey", "ez"]:
                array = loader[key]
                title = key.capitalize()
                units = "V/m"
            elif key in ["bx", "by", "bz"]:
                array = loader[key]
                title = key.capitalize()
                units = "T"
            elif key in ["jy", "jz"]:
                array = loader[key]
                title = key.capitalize()
                units = "A/m^2"
            elif key[:3] == "den":
                array = loader["den"][:, int(key[3]), :]
                title = key.capitalize()
                units = "1/m^3"
            elif key == "energy density":
                e_squared = loader["ex"] ** 2 + loader["ey"] ** 2 + loader["ez"] ** 2
                b_squared = (loader["b0"] * numpy.sin(loader["theta"])) ** 2 + loader["by"] ** 2 + loader["bz"] ** 2
                epsilon_0 = 8.85E-12
                mu_0 = 1.26E-6
                array = 0.5 * epsilon_0 * e_squared + 0.5 * b_squared / mu_0
                title = key.capitalize()
                units = "J/m^3"
            else:
                assert False, "Invalid plot key: {}".format(key)
            axis = axs[i, j]
            # SET FONT
            font_size = 44
            font_family = 'Palatino Linotype'
            axis.set_title(title, fontsize=font_size, fontfamily=font_family)
            axis.set_xlabel("x (m)", fontsize=font_size, fontfamily=font_family)
            axis.set_ylabel("{} ({})".format(title, units), fontsize=font_size, fontfamily=font_family)
            axis.set_xlim([0, length])
            axis.set_ylim(y_axis_limit(array))
            axis.tick_params(axis='both', which='both', labelsize=36) # SET LABEL SIZE
            array_list.append(array)
            line, = axis.plot(grid_x, array[time_step], linewidth=3)
            line_list.append(line)
    return line_list, array_list


def set_up_phase_space_plot(x, vx, ax_bottom, dot_size, n_sample, plot_species, plot_colors, length, specie, line_list, time_step=0):
    """
    Set up phase space plot in the bottom axis
    :param x: particle positions
    :param vx: particle x-velocities
    :param ax_bottom: bottom axis
    :param dot_size: particle dot size in the plot
    :param n_sample: number of sample particles per specie
    :param plot_species: species to be plotted in phase space
    :param plot_colors: color to use for each specie
    :param length: spatial length of the simulation
    :param specie: specie to focus on in the phase space animation
    :param line_list: list of lines to be animated
    :param time_step: the time step to be plotted (0 for animation)
    :return: none
    """
    init_x = x[plot_species, time_step].flatten()
    init_vx = vx[plot_species, time_step].flatten()
    bottom_line = ax_bottom.scatter(init_x, init_vx, s=dot_size)  # draw an initial scatter plot
    color_list = numpy.concatenate([([i] * n_sample) for i in plot_colors], axis=0)
    bottom_line.set_facecolors(color_list)
    line_list.append(bottom_line)
    font_size = 44
    font_family = 'Palatino Linotype'
    ax_bottom.set_title("Phase space", fontsize=font_size, fontfamily=font_family)
    ax_bottom.set_xlabel("x (m)", fontsize=font_size, fontfamily=font_family)
    ax_bottom.set_ylabel("vx (m/s)", fontsize=font_size, fontfamily=font_family)
    ax_bottom.set_xlim([0, length])
    ax_bottom.set_ylim(y_axis_limit(vx[specie]))
    ax_bottom.tick_params(axis='both', which='both', labelsize=36)


def animate_and_save_to_mp4(fig, x, vx, line_list, array_list, frame_interval, fps, nt, file_name, plot_species):
    """
    Animate plots and save the animation to mp4 file
    :param fig: canvas
    :param x: particle positions
    :param vx: particle x-velocities
    :param line_list: list of lines to be animated
    :param array_list: list of value arrays
    :param frame_interval: number of time steps to skip per frame plus one
    :param fps: frame rate
    :param nt: number of time steps
    :param file_name: name of mp4 file
    :param plot_species: species to be plotted in phase space
    :return: none
    """

    def animate(frame_number):
        """
        Animation function to be called by FuncAnimation in its loop
        :param frame_number: current frame number
        :return: list of updated lines
        """
        frame_number = frame_number * frame_interval
        print("Animation Progress: {}/{}.".format(frame_number, nt))
        for index in range(len(line_list) - 1):
            line_list[index].set_ydata(array_list[index][frame_number])
        new_x = x[plot_species, frame_number].flatten()
        new_vx = vx[plot_species, frame_number].flatten()
        line_list[-1].set_offsets(numpy.column_stack((new_x, new_vx)))
        return line_list

    # CREATE ANIMATION
    animation = FuncAnimation(fig, animate, save_count=int(nt // frame_interval))

    # SAVE ANIMATION
    f = "anim/{}.mp4".format(file_name)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    writer_video = FFMpegWriter(fps=fps)
    animation.save(f, writer=writer_video)
    plt.close()


def save_non_fourier_plot(file_name):
    """
    Save non-animation plot
    :param file_name: name of png file
    :return: none
    """
    path = 'fig/{}_NF.png'.format(file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)  # save figure


def plot_non_fourier(file_name, specie=None, anim=True, time_step=None, add_plots=None, add_plot_keys=None,
                     dot_size=None, plot_species=None, plot_colors=None, fps=None, frame_interval=None):
    """
    PLOT GRID QUANTITIES WITHOUT FOURIER TRANSFORMING
    :param file_name: name of the file which stores the arrays
    :param specie: specie to focus on in the phase space animation
    :param anim: make an animation or not
    :param time_step: time step to be plotted, only needed if not animated
    :param add_plots: shape of the additional plots above the phase space animation
    :param add_plot_keys: values to be plotted in the additional plots
    :param dot_size: size of particles in phase space plot
    :param plot_species: species to plot in phase space
    :param plot_colors: color to use for each specie
    :param fps: frame rate
    :param frame_interval: number of time steps to skip per frame plus one
    :return:
    """
    # SET DEFAULT PARAMETERS
    specie, time_step, add_plots, add_plot_keys, dot_size, plot_species, plot_colors, fps, frame_interval = set_default_parameters(specie, anim, time_step, add_plots, add_plot_keys, dot_size, plot_species, plot_colors, fps, frame_interval)

    # SET VIDEO WRITER PATH
    rcParams['animation.ffmpeg_path'] = r'ffmpeg\\bin\\ffmpeg.exe'

    # LOAD DATA FILE
    loader = zarr.load('output/{}.zip'.format(file_name))

    # GET ARRAYS AND VALUES FROM THE FILE
    (grid_x, x, vx, ng, nt, dx) = qol.read_almanac(loader, "grid_x", "x", "vx", "ng", "nt", "dx")

    # DEFINE SOME VARIABLES
    n_sp, n_sample, length = set_variables(x, ng, dx)

    # CHECK IF INPUT IS REASONABLE
    sanity_check(add_plot_keys, add_plots, plot_species, plot_colors, n_sp)

    # USE DARK BACKGROUND
    plt.style.use("dark_background")

    # SET UP CANVAS
    fig, axs = plt.subplots(add_plots[0] + 1, add_plots[1], figsize=[18, 9.6], squeeze=False)

    # MERGE BOTTOM SUBPLOTS
    ax_bottom = merge_bottom_subplots(fig, axs)

    # SET UP THE ADDITIONAL PLOTS
    line_list, array_list = set_up_add_plots(loader, add_plots, add_plot_keys, axs, length, grid_x, time_step)

    # SET PHASE SPACE PLOT
    if anim:
        set_up_phase_space_plot(x, vx, ax_bottom, dot_size, n_sample, plot_species, plot_colors, length, specie, line_list)
    else:
        set_up_phase_space_plot(x, vx, ax_bottom, dot_size, n_sample, plot_species, plot_colors, length, specie, line_list, time_step)

    # PREVENT PLOT OVERLAPS
    plt.tight_layout()

    if anim:
        # ANIMATE AND SAVE TO MP4
        animate_and_save_to_mp4(fig, x, vx, line_list, array_list, frame_interval, fps, nt, file_name, plot_species)

    else:
        # SAVE PLOT AS IS (NO ANIMATION)
        save_non_fourier_plot(file_name)
