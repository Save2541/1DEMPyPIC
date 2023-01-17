import io
import math
import os

import scipy.fft
import matplotlib.pyplot as plt
import numpy
import constants
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import zarr


rcParams['animation.ffmpeg_path'] = r'ffmpeg\\bin\\ffmpeg.exe'  # for saving animation in mpeg

# PHYSICAL CONSTANTS
mu = constants.mu


# PLOT GRID QUANTITIES (NON-FOURIER)

def plot_non_fourier(file_name, specie=1, anim=True, time_step=0):
    """
    PLOT GRID QUANTITIES WITHOUT FOURIER TRANSFORMING
    :param file_name: name of the file which stores the arrays.
    :param specie: specie to focus on in the phase space animation
    :param anim: make an animation or not.
    :param time_step: time step to be plotted, only needed if not animated.
    :return:
    """

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('output/{}.zip'.format(file_name))
    grid_x = loader['grid_x']
    rho_list = loader['rho_list']
    ex_list = loader['ex']
    ey_list = loader['ey']
    ez_list = loader['ez']
    by_list = loader['by']
    bz_list = loader['bz']
    jy_list = loader['jy']
    jz_list = loader['jz']
    ng = int(loader['ng'])
    nt = int(loader['nt'])
    dx = loader['dx']
    dt = loader['dt']
    x = loader['x']
    vx = loader['vx']
    n_species = x.shape[0]
    n_sample = x.shape[2]
    length = ng * dx

    # CANVAS
    fig, axs = plt.subplots(3, 4, figsize=[9, 4.8])

    # MERGE BOTTOM SUBPLOTS
    gs = axs[-1, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[-1, :]:
        ax.remove()
    axbig = fig.add_subplot(gs[-1, :])

    # INITIALIZE PLOTS
    if anim:
        time_step = 0

    def y_axis_limit(value_list):
        maximum = numpy.amax(value_list)
        minimum = numpy.amin(value_list)
        if minimum != maximum:
            return [minimum, maximum]
        else:
            return [-1, 1]

    # TOP LEFT PLOT (RHO)
    line1, = axs[0, 0].plot(grid_x, rho_list[time_step])
    axs[0, 0].set_title("rho")
    axs[0, 0].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("rho (C/m^3)")
    axs[0, 0].set_xlim([0, length])
    axs[0, 0].set_ylim(y_axis_limit(rho_list))

    # CENTER LEFT PLOT (Ex)
    line2, = axs[1, 0].plot(grid_x, ex_list[time_step])
    axs[1, 0].set_title("Ex")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 0].set_ylabel("Ex (V/m)")
    axs[1, 0].set_xlim([0, length])
    axs[1, 0].set_ylim(y_axis_limit(ex_list))

    # BOTTOM LEFT PLOT (PHASE SPACE)
    # line3 = axs[2].scatter(numpy.zeros(n_sample), numpy.zeros(n_sample))  # empty scatter plot
    init_x = x[:, 0].flatten()
    init_vx = vx[:, 0].flatten()
    line3 = axbig.scatter(init_x, init_vx, s=1)  # draw an initial scatter plot
    color_list = numpy.concatenate([([i] * n_sample) for i in ["black", "red", "yellow"]], axis=0)
    line3.set_facecolors(color_list)
    axbig.set_title("Phase space")
    axbig.set_xlabel("x (m)")
    axbig.set_ylabel("vx (m/s)")
    axbig.set_xlim([0, length])
    axbig.set_ylim(y_axis_limit(vx[specie]))

    # TOP CENTER PLOT (By)
    line4, = axs[0, 1].plot(grid_x, by_list[time_step])
    axs[0, 1].set_title("By")
    axs[0, 1].set_xlabel("x (m)")
    axs[0, 1].set_ylabel("By (T)")
    axs[0, 1].set_xlim([0, length])
    axs[0, 1].set_ylim(y_axis_limit(by_list))

    # CENTER PLOT (Ey)
    line5, = axs[1, 1].plot(grid_x, ey_list[time_step])
    axs[1, 1].set_title("Ey")
    axs[1, 1].set_xlabel("x (m)")
    axs[1, 1].set_ylabel("Ey (V/m)")
    axs[1, 1].set_xlim([0, length])
    axs[1, 1].set_ylim(y_axis_limit(ey_list))

    # TOP RIGHT PLOT (Bz)
    line6, = axs[0, 2].plot(grid_x, bz_list[time_step])
    axs[0, 2].set_title("Bz")
    axs[0, 2].set_xlabel("x (m)")
    axs[0, 2].set_ylabel("Bz (T)")
    axs[0, 2].set_xlim([0, length])
    axs[0, 2].set_ylim(y_axis_limit(bz_list))

    # CENTER RIGHT PLOT (Ez)
    line7, = axs[1, 2].plot(grid_x, ez_list[time_step])
    axs[1, 2].set_title("Ez")
    axs[1, 2].set_xlabel("x (m)")
    axs[1, 2].set_ylabel("Ez (V/m)")
    axs[1, 2].set_xlim([0, length])
    axs[1, 2].set_ylim(y_axis_limit(ez_list))

    # TOP FAR RIGHT PLOT (Jy)
    line8, = axs[0, 3].plot(grid_x, jy_list[time_step])
    axs[0, 3].set_title("Jy")
    axs[0, 3].set_xlabel("x (m)")
    axs[0, 3].set_ylabel("Jy (A m^-2)")
    axs[0, 3].set_xlim([0, length])
    axs[0, 3].set_ylim(y_axis_limit(jy_list))

    # CENTER FAR RIGHT PLOT (Jz)
    line9, = axs[1, 3].plot(grid_x, jz_list[time_step])
    axs[1, 3].set_title("Jz")
    axs[1, 3].set_xlabel("x (m)")
    axs[1, 3].set_ylabel("Jz (A m^-2)")
    axs[1, 3].set_xlim([0, length])
    axs[1, 3].set_ylim(y_axis_limit(jz_list))

    # PREVENT PLOT OVERLAPS
    plt.tight_layout()

    if anim:
        # CREATE A LIST OF LINES
        line = [line1, line2, line3, line4, line5, line6, line7, line8, line9]

        def animate(frame_number):
            frame_number = frame_number * 16
            line[0].set_ydata(rho_list[frame_number])
            line[1].set_ydata(ex_list[frame_number])
            new_x = x[:, frame_number].flatten()
            new_vx = vx[:, frame_number].flatten()
            line[2].set_offsets(numpy.column_stack((new_x, new_vx)))
            line[3].set_ydata(by_list[frame_number])
            line[4].set_ydata(ey_list[frame_number])
            line[5].set_ydata(bz_list[frame_number])
            line[6].set_ydata(ez_list[frame_number])
            line[7].set_ydata(jy_list[frame_number])
            line[8].set_ydata(jz_list[frame_number])
            return line

        # CREATE ANIMATION
        animation = FuncAnimation(fig, animate, save_count=nt // 16)

        # SAVE ANIMATION
        f = "anim/{}_anim.mp4".format(file_name)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        writervideo = FFMpegWriter(fps=120)
        animation.save(f, writer=writervideo)
        plt.close()

    else:
        # SHOW PLOT
        plt.show()


# PHASE SPACE ANIMATION PLOTTING PROGRAM
def plot_phase_space_animation(file_name, specie):
    """PLOT PHASE SPACE ANIMATION
        file_name = name of the file which stores the arrays.
        specie = particles to be plotted ("electron", "ion", "both")
    """

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('{}.zip'.format(file_name))

    ng = loader['ng']
    nt = loader['nt']
    slu = loader['slu']
    stu = loader['stu']
    x = loader['x'] * slu
    vx = loader['vx'] * slu / stu
    dx = loader['dx'] * slu
    dt = loader['dt'] * stu / 10 ** -3
    v_th = loader['v_th'][0] * slu / stu
    vi_th = loader['v_th'][1] * slu / stu
    length = ng * dx
    n_species = x.shape[0]
    n_sample = x.shape[2]

    # CREATE SCATTER PLOT FOR ANIMATION
    fig_anim = plt.figure(figsize=(14, 7))  # create figure
    ax_anim = plt.axes(xlim=(0, length),
                       ylim=(-vi_th * 20, vi_th * 20))  # draw axis on figure
    plt.title("Phase space")  # plot title
    plt.xlabel("x-Position (m)")  # x-axis label
    plt.ylabel("x-Velocity (m/s)")  # y-axis label

    if specie == "electron":
        scatter = ax_anim.scatter(numpy.zeros(n_sample), numpy.zeros(n_sample))  # empty scatter plot
        color_list = ["black"] * n_sample
    elif specie == "ion":
        scatter = ax_anim.scatter(numpy.zeros(n_sample), numpy.zeros(n_sample))  # empty scatter plot
        color_list = ["red"] * n_sample
    else:
        scatter = ax_anim.scatter(numpy.zeros(n_sample * n_species),
                                  numpy.zeros(n_sample * n_species))  # empty scatter plot
        color_list = numpy.concatenate([([i] * n_sample) for i in ["black", "red"]], axis=0)

    scatter.set_facecolors(color_list)

    def setup_plot():
        """Initial drawing of the scatter plot"""
        if specie == "electron":
            init_x = x[0, 0].flatten()
            init_vx = vx[0, 0].flatten()
        elif specie == "ion":
            init_x = x[1, 0].flatten()
            init_vx = vx[1, 0].flatten()
        else:
            init_x = x[:, 0].flatten()
            init_vx = vx[:, 0].flatten()
        scatter.set_offsets(numpy.column_stack((init_x, init_vx)))  # draw an initial scatter plot
        return scatter,

    def update(frame_number):
        """Update the scatter plot"""
        if specie == "electron":
            new_x = x[0, frame_number + 1].flatten()
            new_vx = vx[0, frame_number + 1].flatten()
        elif specie == "ion":
            new_x = x[1, frame_number + 1].flatten()
            new_vx = vx[1, frame_number + 1].flatten()
        else:
            new_x = x[:, frame_number + 1].flatten()
            new_vx = vx[:, frame_number + 1].flatten()
        scatter.set_offsets(numpy.column_stack((new_x, new_vx)))
        return scatter,

    # CREATE PHASE SPACE ANIMATION
    anim = FuncAnimation(fig_anim, update, init_func=setup_plot, save_count=nt)

    # SAVE ANIMATION
    f = "{}_phasespace.mp4"
    writervideo = FFMpegWriter(fps=3000)
    anim.save(f.format(file_name), writer=writervideo)
    plt.close()


# PLOT SPECTRUM WITH THEORETICAL LINES


def plot(file_name, specie, field):
    """PLOTTING PROGRAM
        file_name = name of the file which stores the arrays.
        specie = theoretical waves to be plotted ("electron", "ion", "both")
        field = field to be plotted (E, B, rho, rho_d)
    """

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('output/{}.zip'.format(file_name))
    mi = loader['m'][1]
    ex_list = loader['ex']
    ey_list = loader['ey']
    ez_list = loader['ez']
    by_list = loader['by']
    bz_list = loader['bz']
    rho_list = loader['rho_list']
    ng = int(loader['ng'])
    nt = int(loader['nt'])
    b0 = loader['b0']
    rho = loader['rho']
    kte = loader['kt'][0]
    kti = loader['kt'][1]
    dx = loader['dx']
    dt = loader['dt']
    wp = loader['wp'][0]
    wc = loader['wc'][0]
    wci = loader['wc'][1]
    c = loader['c']
    theta = loader['theta']
    v_th = loader['v_th'][0]
    init_d_k = loader['init_d_k']
    init_k = loader['init_v_k']
    length = ng * dx
    print(init_d_k)
    print(init_k)

    # k-DOMAIN FOR THEORETICAL WAVE PLOTS
    k_list = numpy.linspace(0.0, math.pi / dx, ng)

    # alfven_omega_list = (b0) / numpy.sqrt(mu * np / 2 / length * mi) * omega_list

    # SPECTRUM FIGURE
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(
        30, 10))  # , sharey='row')  # create a figure with one row and three columns (ax1 for
    # Ex, ax2 for Ey, ax3 for Ez)

    # PLASMA WAVES THEORETICAL LINES
    if specie == "ion":
        gamma_i = 3
        gamma_e = 1
        v_s2 = (gamma_e * kte + gamma_i * kti) / mi  # sound speed squared
        if b0 == 0:
            v_a = 0
        else:
            v_a = math.sqrt(c ** 2 / (1 + c ** 2 * rho * mu / b0 ** 2))  # b0 / math.sqrt(mu * rho)  # alfven speed
        if wci == 0:
            # Ex: ELECTROSTATIC ION WAVES (ACOUSTIC WAVE)
            ex_k_list = k_list
            ex_omega_list = numpy.sqrt(k_list ** 2 * v_s2)
            # Ey : NONE
            ey_k_list = []
            ey_omega_list = []
            # Ez : NONE
            ez_k_list = []
            ez_omega_list = []
        elif theta == 0:
            # Ex: ELECTROSTATIC ION WAVES (ELECTROSTATIC ION CYCLOTRON WAVES OR LOWER HYBRID OSCILLATION)
            ex_k_list = k_list
            # ex_omega_list = numpy.sqrt(wci**2+k_list ** 2 * v_s)
            ex_omega_list = numpy.sqrt(wci * wc + k_list ** 2 * v_s2)
            # Ey: MAGNETOSONIC WAVE
            ey_k_list = k_list
            ey_omega_list = k_list * c * math.sqrt((v_s2 + v_a ** 2) / (c ** 2 + v_a ** 2))
            # Ez: MAGNETOSONIC WAVE
            ez_k_list = k_list
            ez_omega_list = ey_omega_list
        elif theta == math.pi / 2:
            # Ex: ELECTROSTATIC ION WAVES (ACOUSTIC WAVE)
            ex_k_list = k_list
            ex_omega_list = numpy.sqrt(k_list ** 2 * v_s2)
            # Ey: ALFVEN WAVE
            ey_k_list = k_list
            ey_omega_list = k_list * v_a
            # Ez: ALFVEN WAVE
            ez_k_list = k_list
            ez_omega_list = ey_omega_list
        else:
            ex_k_list = []
            ex_omega_list = []
            ey_k_list = []
            ey_omega_list = []
            ez_k_list = []
            ez_omega_list = []

    # ELECTRON WAVES THEORETICAL LINES
    elif specie == "electron":
        if wc == 0:
            # Ex: ELECTROSTATIC ELECTRON WAVES (PLASMA OSCILLATION)
            ex_k_list = k_list
            ex_omega_list = numpy.sqrt(wp ** 2 + 3 / 2 * (k_list ** 2) * (v_th ** 2))
            # Ey: LIGHT WAVES
            ey_k_list = k_list
            ey_omega_list = numpy.sqrt(wp ** 2 + wc ** 2 + k_list ** 2 * c ** 2)
            # Ez: LIGHT WAVES
            ez_k_list = k_list
            ez_omega_list = numpy.sqrt(wp ** 2 + wc ** 2 + k_list ** 2 * c ** 2)
        elif theta == 0:
            # Ex: ELECTROSTATIC ELECTRON WAVES (UPPER HYBRID OSCILLATIONS)
            ex_k_list = k_list
            ex_omega_list = numpy.sqrt(wp ** 2 + wc ** 2 + 3 / 2 * k_list ** 2 * v_th ** 2)
            print(numpy.sqrt(wp ** 2 + wc ** 2))
            # Ey: EXTRAORDINARY WAVES
            ey_omega_list = numpy.linspace(0, math.pi / dt, nt)
            ey_k_list = numpy.sqrt(ey_omega_list ** 2 - wp ** 2 * (
                    (ey_omega_list ** 2 - wp ** 2) / (ey_omega_list ** 2 - wp ** 2 - wc ** 2))) / c
            # Ez: ORDINARY WAVES
            ez_k_list = k_list
            ez_omega_list = numpy.sqrt(wp ** 2 + wc ** 2 + k_list ** 2 * c ** 2)
        elif theta == math.pi / 2:
            # Ex: ELECTROSTATIC ELECTRON WAVES (PLASMA OSCILLATION)
            ex_k_list = k_list
            ex_omega_list = numpy.sqrt(wp ** 2 + 3 / 2 * (k_list ** 2) * (v_th ** 2))
            # Ey: L WAVE
            ey_omega_list = numpy.linspace(0, math.pi / dt, nt)
            ey_k_list = numpy.sqrt(ey_omega_list ** 2 - wp ** 2 / (1 + wc / ey_omega_list)) / c
            # Ez: R WAVE
            ez_omega_list = ey_omega_list
            ez_k_list = numpy.sqrt(ez_omega_list ** 2 - wp ** 2 / (1 - wc / ez_omega_list)) / c
        else:
            ex_k_list = []
            ex_omega_list = []
            ey_k_list = []
            ey_omega_list = []
            ez_k_list = []
            ez_omega_list = []

    # FOURIER TRANSFORMS
    print("Fourier transforming...")
    if field == 'B':
        title1 = 'Ex'
        ex_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ex_list))
        title2 = 'By'
        ey_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(by_list))
        title3 = 'Bz'
        ez_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(bz_list))
    elif field == 'rho' or field == 'rho_d':
        title1 = 'rho'
        ex_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(rho_list))
        title2 = 'Ey'
        ey_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ey_list))
        title3 = 'Ez'
        ez_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ez_list))

        # CHANGE THEORETICAL LINES
        if specie == 'electron':
            index = 0
        if specie == 'ion':
            index = 1

        # if field == 'rho':
        #     ey_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #     ey_k_list = ey_omega_list * 0 + init_k[index][1]
        #
        #     if theta == 0:
        #         ex_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #         ex_k_list = ex_omega_list * 0 + init_k[index][0]
        #         ez_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #         ez_k_list = ez_omega_list * 0 + init_k[index][2]
        #     elif theta == math.pi / 2:
        #         ex_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #         ex_k_list = ex_omega_list * 0 + init_k[index][2]
        #         ez_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #         ez_k_list = ez_omega_list * 0 + init_k[index][0]
        # else:
        #     ex_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        #     ex_k_list = ex_omega_list * 0 + init_d_k[index]

    else:
        title1 = 'Ex'
        ex_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ex_list))
        title2 = 'Ey'
        ey_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ey_list))
        title3 = 'Ez'
        ez_omega_k_list = numpy.abs(dx * dt * scipy.fft.rfft2(ez_list))

    print('Plot dimensions: x = ', len(ey_omega_k_list[0]), 'y = ', len(ey_omega_k_list) // 2, ".")  # print dimensions

    # GET SAMPLE FREQUENCIES
    sample_k = 2 * math.pi * scipy.fft.rfftfreq(ng, dx)
    sample_omega = 2 * math.pi * scipy.fft.fftfreq(nt, dt)

    # CONSTRUCT A 2D ARRAY OF K-VALUES
    k_array = numpy.tile(sample_k, (nt, 1))

    # SMOOTHING FUNCTIONS
    def smooth(fourier_plot, weight):
        """
        Apply smoothing function to fourier-transformed plot
        :param fourier_plot: fourier-transformed plot to be smoothed
        :param weight: weight of smoothing
        :return: smoothed plot
        """

        return (1 + 2 * weight * numpy.cos(k_array * dx)) / (1 + 2 * weight) * fourier_plot

    # SMOOTH PLOTS
    w_smooth = 0.5
    ex_omega_k_list = smooth(ex_omega_k_list, w_smooth)
    ey_omega_k_list = smooth(ey_omega_k_list, w_smooth)
    ez_omega_k_list = smooth(ez_omega_k_list, w_smooth)

    def extent_function(xcoords, ycoords):
        '''returns extent (to go to imshow), given xcoords, ycoords. Assumes origin='lower'.
        Use this method to properly align extent with middle of pixels.
        (Noticeable when imshowing few enough pixels that individual pixels are visible.)

        xcoords and ycoords should be arrays.
        (This method uses their first & last values, and their lengths.)

        returns extent == np.array([left, right, bottom, top]).
        '''
        Nx = len(xcoords)
        Ny = len(ycoords)
        dx = (xcoords[-1] - xcoords[0]) / Nx
        dy = (ycoords[-1] - ycoords[0]) / Ny
        return numpy.array([*(xcoords[0] + numpy.array([0 - dx / 2, dx * Nx + dx / 2])),
                            *(ycoords[0] + numpy.array([0 - dy / 2, dy * Ny + dy / 2]))])

    # GET PLOT EXTENT
    sample_omega = sample_omega[0:nt // 2]
    extent = extent_function(sample_k, sample_omega)
    # extent = [sample_k[0], sample_k[-1], sample_omega[0], -sample_omega[nt // 2]]
    y_max = nt // 2 + 1

    # PLOT SPECTRA
    maximum = math.ceil(math.log10(numpy.amax(ex_omega_k_list[1:]))) - 2
    shw1 = ax1.imshow(ex_omega_k_list[0:y_max], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (maximum - 4),
                                   vmax=10 ** maximum),
                      origin='lower',
                      extent=extent,
                      aspect='auto', interpolation='none')
    try:
        maximum = math.ceil(math.log10(numpy.amax(ey_omega_k_list[1:]))) - 2
    except ValueError:
        maximum = 0
    shw2 = ax2.imshow(ey_omega_k_list[0:y_max], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (maximum - 4),
                                   vmax=10 ** maximum),
                      origin='lower',
                      extent=extent,
                      aspect='auto', interpolation='none')
    try:
        maximum = math.ceil(math.log10(numpy.amax(ez_omega_k_list[1:]))) - 2
    except ValueError:
        maximum = 0
    shw3 = ax3.imshow(ez_omega_k_list[0:y_max], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (maximum - 4),
                                   vmax=10 ** maximum),
                      origin='lower',
                      extent=extent,
                      aspect='auto', interpolation='none')

    print(extent)  # PRINT PLOT SIZES

    # CREATE BAR PLOTS
    color_bar_text_size = 5  # set text size on color bar
    bar1 = plt.colorbar(shw1, ax=ax1)
    # bar1.ax.tick_params(labelsize=color_bar_text_size)
    bar2 = plt.colorbar(shw2, ax=ax2)
    # bar2.ax.tick_params(labelsize=color_bar_text_size)
    bar3 = plt.colorbar(shw3, ax=ax3)
    # bar3.ax.tick_params(labelsize=color_bar_text_size)

    # PLOT LABELS
    ax1.set_title(title1)
    ax1.set_xlabel('k (rad/m)')
    ax1.set_ylabel('omega (rad/s)')
    ax2.set_title(title2)
    ax2.set_xlabel('k (rad/m)')
    ax2.set_ylabel('omega (rad/s)')
    ax3.set_title(title3)
    ax3.set_xlabel('k (rad/m)')
    ax3.set_ylabel('omega (rad/s)')
    # fig.supxlabel('k (1 unit = {:.2e} m-1)'.format(1 / slu))
    # fig.supylabel('omega (1 unit = {:.2e} Hz)'.format(1 / stu))
    # fig.supxlabel('k')
    # fig.supylabel('omega')
    # bar1.set_label('Ex')
    # bar2.set_label('Ey')

    # scaled_ey_k_list = []
    # for row, position in zip(ey_omega_k_list, range(1,1+len(ey_omega_k_list))):
    #    scaled_row = row / position
    #   for element, pos in zip(scaled_row, range(0,len(scaled_row))):
    #        scaled_ey_k_list.append([pos, element])
    # plt.scatter(*numpy.array(scaled_ey_k_list).T)

    # Plot L and R waves
    # l_wv = ax.plot(l_k_list, omega_list, color='green')
    # r_wv = ax.plot(r_k_list, omega_list, color='black')
    # alfven_wv = ax.plot(omega_list, alfven_omega_list, color='white')

    # PLOT THEORETICAL WAVES
    ex_wv = ax1.plot(ex_k_list, ex_omega_list, color='black', linewidth=1)  # draw theoretical line on left plot
    ey_wv = ax2.plot(ey_k_list, ey_omega_list, color='black', linewidth=1)  # draw theoretical line on middle plot
    ez_wv = ax3.plot(ez_k_list, ez_omega_list, color='black', linewidth=1)  # draw theoretical line on right plot

    # CROP PLOT
    left, right = plt.xlim()  # get current left and right limit
    bottom, top = plt.ylim()  # get current left and right limit
    # right = 4 * math.sqrt(wp**2 + wc**2)  # set plot limit to 4x the plasma frequency
    if specie == "ion" or specie == "electron":
        height = 1E7  # 5.64E5  # plot height #math.sqrt(wp ** 2 + wc ** 2) * 5
        width = 2
    else:
        height = min(math.sqrt(wp ** 2 + wc ** 2) * 20, top)  # plot height #math.sqrt(wp ** 2 + wc ** 2) * 5
        width = height / c  # plot width

    # CROP PLOTS
    # ax1.set_xlim(right=width)  # set x limit for left plot
    # ax1.set_ylim(top=height)  # set y limit for left plot
    # ax2.set_xlim(right=width)  # set x limit for middle plot
    # ax2.set_ylim(top=height)  # set y limit for middle plot
    # ax3.set_xlim(right=width)  # set x limit for right plot
    # ax3.set_ylim(top=height)  # set y limit for right plot

    # PREVENT PLOT OVERLAPS
    plt.tight_layout()

    path = 'fig/{}_{}.png'.format(field, file_name)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path)  # save figure
    # plt.show()  # show figure


# RUN MAIN PROGRAM
# plot("20221013-122755_ni1e+08_ti1e+04_te1e+05_b00e+00_theta90", "ion", "rho")

# RUN PROGRAM WITH PROFILER
if __name__ == '__main__':
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    file_list = ["20230116-221209_nsp2_theta90_EM"]
    for file in file_list:
        plot_non_fourier(file, numpy.s_[0:2])
        # plot(file, "ion", "rho")
        # plot_phase_space_animation(file, "ion")
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    stats.print_stats()

    with open('outstats_ps.txt', 'w+') as f:
        f.write(s.getvalue())
