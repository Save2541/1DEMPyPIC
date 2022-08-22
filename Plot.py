import math
import scipy.fft
import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import zarr

rcParams['animation.ffmpeg_path'] = r'ffmpeg\\bin\\ffmpeg.exe'  # for saving animation in mpeg


# PHASE SPACE ANIMATION PLOTTING PROGRAM
def plot_phase_space_animation(file_name):
    """PLOT PHASE SPACE ANIMATION"""
    # COLOR FOR ANIMATIION FIGURE
    # color_list = []
    # for particle in particle_list[0::47]: # Every 47th particle
    #    if particle.type == "ion":
    #        color_list.append("red")
    #    else:
    #        color_list.append("black")
    # scatter.set_facecolors(color_list)

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('{}'.format(file_name))
    x = loader['x']
    vx = loader['vx']
    ng = loader['ng']
    nt = loader['nt']
    slu = loader['slu']
    stu = loader['stu']
    dx = loader['dx']
    dt = loader['dt']
    v_th = loader['v_th']
    length = ng * dx
    n_sample = len(x)

    # CREATE SCATTER PLOT FOR ANIMATION
    fig_anim = plt.figure(figsize=(14, 7))  # create figure
    ax_anim = plt.axes(xlim=(0, length),
                       ylim=(-v_th * 6, v_th * 6))  # draw axis on figure
    plt.title("Phase space")  # plot title
    plt.xlabel("x-Position (1 unit = {:.2e} m)".format(slu))  # x-axis label
    plt.ylabel("x-Velocity (1 unit = {:.2e} m/s)".format(slu / stu))  # y-axis label
    scatter = ax_anim.scatter(numpy.zeros(n_sample), numpy.zeros(n_sample))  # empty scatter plot

    def setup_plot():
        """Initial drawing of the scatter plot"""
        scatter.set_offsets(numpy.column_stack((x[0], vx[0])))  # draw an initial scatter plot
        return scatter,

    def update(frame_number):
        """Update the scatter plot"""
        scatter.set_offsets(numpy.column_stack((x[frame_number + 1], vx[frame_number + 1])))
        return scatter,

    # CREATE PHASE SPACE ANIMATION
    anim = FuncAnimation(fig_anim, update, init_func=setup_plot, interval=200, save_count=nt)

    # SAVE ANIMATION
    f = "{}_phasespace.mp4"
    writervideo = FFMpegWriter(fps=24)
    anim.save(f.format(filename), writer=writervideo)
    plt.close()


# PLOT SPECTRUM WITH THEORETICAL LINES


def plot(file_name):
    """PLOTTING PROGRAM
        file_name = name of the file which stores the arrays
    """

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('{}'.format(file_name))
    ex_list = loader['ex']
    ey_list = loader['ey']
    ez_list = loader['ez']
    ng = loader['ng']
    nt = loader['nt']
    slu = loader['slu']
    stu = loader['stu']
    dx = loader['dx']
    dt = loader['dt']
    wp = loader['wp']
    wc = loader['wc']
    theta = loader['theta']
    v_th = loader['v_th']
    c = dx / dt
    length = ng * dx

    # k-DOMAIN FOR THEORETICAL WAVE PLOTS
    k_list = numpy.arange(0.0, math.pi / dx, math.pi / length)

    # alfven_omega_list = (b0) / numpy.sqrt(mu * np / 2 / length * mi) * omega_list

    # SPECTRUM FIGURE
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                        sharey='row')  # create a figure with one row and two columns (ax1 for Ex, ax2 for
    # Ey)

    # THEORETICAL LINES
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
        # Ey: EXTRAORDINARY WAVES
        ey_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
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
        ey_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
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
    ex_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ex_list)))
    ey_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ey_list)))
    ez_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ez_list)))

    print('Plot dimensions: x = ', len(ey_omega_k_list[0]), 'y = ', len(ey_omega_k_list) // 2, ".")  # print dimensions

    # PLOT SPECTRA
    shw1 = ax1.imshow(ex_omega_k_list[0:int(len(ex_omega_k_list) / 2 + 1)], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ex_omega_k_list[0]))) - 3),
                                   vmax=10 ** math.ceil(math.log10(max(ex_omega_k_list[0])))),
                      # norm=LogNorm(),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ex_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ex_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')
    shw2 = ax2.imshow(ey_omega_k_list[0:int(len(ey_omega_k_list) / 2 + 1)], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ey_omega_k_list[0]))) - 3),
                                   vmax=10 ** math.ceil(math.log10(max(ey_omega_k_list[0])))),
                      # norm=LogNorm(),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ey_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ey_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')
    shw3 = ax3.imshow(ez_omega_k_list[0:int(len(ez_omega_k_list) / 2 + 1)], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ez_omega_k_list[0]))) - 3),
                                   vmax=10 ** math.ceil(math.log10(max(ez_omega_k_list[0])))),
                      # norm=LogNorm(),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ez_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ez_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')

    print([0, 2 * math.pi * len(ey_omega_k_list[0]) / length, 0,
           2 * math.pi * (len(ey_omega_k_list) / 2 + 1) / (dt * nt)])  # PRINT PLOT SIZES

    # CREATE BAR PLOTS
    bar1 = plt.colorbar(shw1, ax=ax1)
    bar2 = plt.colorbar(shw2, ax=ax2)
    bar3 = plt.colorbar(shw3, ax=ax3)

    # PLOT LABELS
    ax1.set_title('Ex')
    ax2.set_title('Ey')
    ax3.set_title('Ez')
    fig.supxlabel('k (1 unit = {:.2e} m-1)'.format(1 / slu))
    fig.supylabel('omega (1 unit = {:.2e} Hz)'.format(1 / stu))
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
    ex_wv = ax1.plot(ex_k_list, ex_omega_list, color='black', linewidth=0.5)  # draw theoretical line on left plot
    ey_wv = ax2.plot(ey_k_list, ey_omega_list, color='black', linewidth=0.5)  # draw theoretical line on middle plot
    ez_wv = ax3.plot(ez_k_list, ez_omega_list, color='black', linewidth=0.5)  # draw theoretical line on right plot

    # CROP PLOT
    left, right = plt.xlim()  # get current left and right limit
    bottom, top = plt.ylim()  # get current left and right limit
    # right = 4 * math.sqrt(wp**2 + wc**2)  # set plot limit to 4x the plasma frequency
    height = min(math.sqrt(wp ** 2 + wc ** 2) * 20, top)  # plot height #math.sqrt(wp ** 2 + wc ** 2) * 5
    width = height  # plot width
    ax1.set_xlim(right=width)  # set x limit for left plot
    ax1.set_ylim(top=height)  # set y limit for left plot
    ax2.set_xlim(right=width)  # set x limit for middle plot
    ax2.set_ylim(top=height)  # set y limit for middle plot
    ax3.set_xlim(right=width)  # set x limit for right plot
    ax3.set_ylim(top=height)  # set y limit for right plot
    # plt.xlim([0, 2 * math.pi * ng / 2 / length])
    # plt.ylim([0, 2 * math.pi * nt / 2 / (dt * nt)])

    plt.savefig('E_{}.png'.format(file_name), dpi=1200)  # save figure
    plt.show()  # show figure


# RUN MAIN PROGRAM
plot("ne1e+08_te1e+04_b01e-07_theta0.zip")
