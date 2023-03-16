import math
import os

import numpy
import scipy.fft
import zarr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from . import qol
from . import plot_config
from . import plot_generator
from . import theory
from . import fourier_transformer


def get_sample_frequencies_2d(data):
    """
    Get sample k's and omega's and a 2D array of k values (for digital filtering)
    :param data: loaded data files
    :return: sample k's and omega's
    """
    # GET VARIABLES FROM DATA
    (ng, nt, dx, dt) = qol.read_almanac(data, "ng", "nt", "dx", "dt")
    # MAKE VARIABLE INTEGER
    nt = int(nt)
    # CALCULATE SAMPLE FREQUENCIES
    sample_k = qol.get_sample_frequencies(dx, ng)
    sample_w = 2 * math.pi * scipy.fft.fftfreq(nt, dt)
    # CREATE A 2D ARRAY OF k VALUES
    k_array = numpy.tile(sample_k, (nt, 1))
    return sample_k, sample_w[0:nt // 2], k_array


def plot_spectrum(spectrum, axis, title, extent, orders_of_magnitude=None, maximum_magnitude=None):
    """
    Plot a spectrum on an axis
    :param spectrum: spectrum to be plotted
    :param axis: axis to be plotted on
    :param title: title to be shown on top of the plot
    :param extent: plot extent
    :param orders_of_magnitude: orders of magnitude to be plotted
    :param maximum_magnitude: lower the maximum magnitude to focus on weaker signals
    :return: none
    """
    # ASSIGN DEFAULT VALUES TO VARIABLES
    if orders_of_magnitude is None:
        orders_of_magnitude = plot_config.orders_of_magnitude
    if maximum_magnitude is None:
        maximum_magnitude = plot_config.maximum_magnitude
    # CALCULATE THE MAXIMUM MAGNITUDE
    try:
        maximum = math.ceil(math.log10(numpy.amax(spectrum[1:]))) - maximum_magnitude
    except ValueError:
        maximum = 0
    # PLOT SPECTRUM
    shw = axis.imshow(spectrum, cmap='plasma',
                      norm=LogNorm(vmin=10 ** (maximum - orders_of_magnitude), vmax=10 ** maximum), origin='lower',
                      extent=extent, aspect='auto', interpolation='none')
    # SET COLOR BAR
    plt.colorbar(shw, ax=axis)
    # SET TITLE
    axis.set_title(title)
    # LABEL X-AXIS
    axis.set_xlabel('k (rad/m)')
    # LABEL Y_AXIS
    axis.set_ylabel('omega (rad/s)')


def crop_plot(axis, extent, x_lim=None, y_lim=None):
    """
    Crop plot to a specified size (lower left is the origin)
    :param axis: axis to be cropped
    :param extent: original plot extent
    :param x_lim: x-axis plot range
    :param y_lim: y-axis plot range
    :return:
    """
    if x_lim is None:
        x_lim = plot_config.plot_range_x
    if y_lim is None:
        y_lim = plot_config.plot_range_y
    if x_lim == "full":
        x_lim = extent[:2]
    if y_lim == "full":
        y_lim = extent[2:]
    axis.set_xlim(left=x_lim[0], right=x_lim[1])
    axis.set_ylim(bottom=y_lim[0], top=y_lim[1])


def draw_on_axes(plot_list, axs, loader, k_array, extent):
    """
    Draw spectra and lines on each axis
    :param plot_list: list of plots
    :param axs: axes to be plotted on
    :param loader: loaded data file
    :param k_array: 2D array of k values for digital filtering
    :param extent: full plot extent
    :return:
    """
    for i in plot_list.row_range():
        for j in plot_list.column_range():
            axis = axs[i, j]
            # PLOT SPECTRUM
            spectrum_key = plot_list.keys[i][j]
            spectrum = fourier_transformer.fourier_transform_2d(spectrum_key, loader, k_array)
            plot_spectrum(spectrum, axis, plot_list.title(i, j), extent)
            # PLOT LINE
            line_key = plot_list.overlays[i][j]
            if line_key == "auto":
                k, w = theory.auto_get_line(spectrum_key, loader)
            else:
                k, w = theory.get_theory_line(line_key, loader)
            axis.plot(k, w, color='black', linewidth=1)
            crop_plot(axis, extent)


def save_figure(file_name):
    """
    Save figure as a png file
    :param file_name: name to be saved
    :return: none
    """
    path = 'fig/{}.png'.format(file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)


def plot_fourier(file_name):
    """
    Plot fourier transformed spectrum
    :param file_name: name of file to be plotted
    :return: none
    """

    # LOAD DATA FILE
    loader = zarr.load('output/{}.zip'.format(file_name))

    # CHECK IF INPUT IS REASONABLE

    # GENERATE PLOT LIST
    plot_list = plot_generator.PlotList()

    # SET UP CANVAS
    fig, axs = plt.subplots(*plot_list.shape, figsize=(21, 14))

    # GET SAMPLE FREQUENCIES
    sample_k, sample_omega, k_array = get_sample_frequencies_2d(loader)

    # GET PLOT EXTENT
    extent = qol.extent_function(sample_k, sample_omega)

    # DRAW IN CANVAS
    draw_on_axes(plot_list, axs, loader, k_array, extent)

    # PREVENT PLOT OVERLAPS
    plt.tight_layout()

    # SAVE PLOT
    save_figure(file_name+"_spec")
