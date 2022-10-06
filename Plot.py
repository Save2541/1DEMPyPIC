import math
import scipy.fft
import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import rcParams
from matplotlib.colors import LogNorm
import zarr

rcParams['animation.ffmpeg_path'] = r'ffmpeg\\bin\\ffmpeg.exe'  # for saving animation in mpeg

# PHYSICAL CONSTANTS
mu = 1.26E-6


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
    dt = loader['dt']
    v_th = loader['v_th'] * slu / stu
    length = ng * dx
    n_species = x.shape[0]
    n_sample = x.shape[2]

    # CREATE SCATTER PLOT FOR ANIMATION
    fig_anim = plt.figure(figsize=(14, 7))  # create figure
    ax_anim = plt.axes(xlim=(0, length),
                       ylim=(-v_th * 6, v_th * 6))  # draw axis on figure
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
    anim = FuncAnimation(fig_anim, update, init_func=setup_plot, interval=200, save_count=nt)

    # SAVE ANIMATION
    f = "{}_phasespace.mp4"
    writervideo = FFMpegWriter(fps=24)
    anim.save(f.format(file_name), writer=writervideo)
    plt.close()


# PLOT SPECTRUM WITH THEORETICAL LINES


def plot(file_name, specie, field):
    """PLOTTING PROGRAM
        file_name = name of the file which stores the arrays.
        specie = theoretical waves to be plotted ("electron", "ion", "both")
        field = field to be plotted (E, B, rho)
    """

    # LOAD ARRAYS FROM FILE
    loader = zarr.load('{}.zip'.format(file_name))
    mi = loader['mi']
    slu = loader['slu']
    stu = loader['stu']
    smu = loader['smu']
    scu = loader['scu']
    ex_list = loader['ex'] * slu * smu * stu / stu**3 / scu
    ey_list = loader['ey'] * slu * smu * stu / stu**3 / scu
    ez_list = loader['ez'] * slu * smu * stu / stu**3 / scu
    by_list = loader['by'] * smu / scu / stu
    bz_list = loader['bz'] * smu / scu / stu
    rho_list = loader['rho_list']
    ng = loader['ng']
    nt = loader['nt']
    b0 = loader['b0'] * smu / scu / stu
    rho = loader['rho']
    kte = loader['kte'] * smu * slu ** 2 / stu ** 2
    kti = loader['kti'] * smu * slu ** 2 / stu ** 2
    dx = loader['dx'] * slu
    dt = loader['dt'] * stu
    wp = loader['wp'] / stu
    wpi = loader['wpi'] / stu
    wc = loader['wc'] / stu
    wci = loader['wci'] / stu
    c = loader['c'] * slu / stu
    theta = loader['theta']
    v_th = loader['v_th'] * slu / stu
    init_k = loader['init_k']
    length = ng * dx
    print(init_k)

    # k-DOMAIN FOR THEORETICAL WAVE PLOTS
    k_list = numpy.arange(0.0, math.pi / dx, math.pi / length)

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
            ey_omega_list = k_list * c * math.sqrt((v_s2 ** 2 + v_a ** 2) / (c ** 2 + v_a ** 2))
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
    print("Fourier transforming...")
    if field == 'B':
        title1 = 'Ex'
        ex_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ex_list)))
        title2 = 'By'
        ey_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(by_list)))
        title3 = 'Bz'
        ez_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(bz_list)))
    elif field == 'rho':
        title1 = 'rho'
        ex_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(rho_list)))
        title2 = 'Ey'
        ey_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ey_list)))
        title3 = 'Ez'
        ez_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ez_list)))

        # CHANGE THEORETICAL LINES
        if specie == 'electron':
            index = 0
        if specie == 'ion':
            index = 1

        ey_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
        ey_k_list = ey_omega_list * 0 + init_k[index][1]

        if theta == 0:
            ex_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
            ex_k_list = ex_omega_list * 0 + init_k[index][0]
            ez_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
            ez_k_list = ez_omega_list * 0 + init_k[index][2]
        elif theta == math.pi / 2:
            ex_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
            ex_k_list = ex_omega_list * 0 + init_k[index][2]
            ez_omega_list = numpy.arange(0, math.pi / dt, math.pi / dt / nt / 10000)
            ez_k_list = ez_omega_list * 0 + init_k[index][0]

    else:
        title1 = 'Ex'
        ex_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ex_list)))
        title2 = 'Ey'
        ey_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ey_list)))
        title3 = 'Ez'
        ez_omega_k_list = numpy.abs(numpy.real(dx * dt * scipy.fft.rfft2(ez_list)))

    print('Plot dimensions: x = ', len(ey_omega_k_list[0]), 'y = ', len(ey_omega_k_list) // 2, ".")  # print dimensions

    # PLOT SPECTRA
    shw1 = ax1.imshow(ex_omega_k_list[0:int(len(ex_omega_k_list) / 2 + 1)], cmap='plasma',
                      #norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ex_omega_k_list[1]))) - 2),
                      #             vmax=10 ** math.ceil(math.log10(max(ex_omega_k_list[1])))),
                      norm=LogNorm(vmin=10 ** -6,
                                  vmax=10 ** -4),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ex_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ex_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')
    shw2 = ax2.imshow(ey_omega_k_list[0:int(len(ey_omega_k_list) / 2 + 1)], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ey_omega_k_list[1]))) - 2),
                                   vmax=10 ** math.ceil(math.log10(max(ey_omega_k_list[1])))),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ey_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ey_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')
    shw3 = ax3.imshow(ez_omega_k_list[0:int(len(ez_omega_k_list) / 2 + 1)], cmap='plasma',
                      norm=LogNorm(vmin=10 ** (math.ceil(math.log10(max(ez_omega_k_list[1]))) - 2),
                                   vmax=10 ** math.ceil(math.log10(max(ez_omega_k_list[1])))),
                      origin='lower',
                      extent=[0, 2 * math.pi * len(ez_omega_k_list[0]) / length, 0,
                              2 * math.pi * (len(ez_omega_k_list) / 2 + 1) / (dt * nt)],
                      aspect='auto', interpolation='none')

    print([0, 2 * math.pi * len(ey_omega_k_list[0]) / length, 0,
           2 * math.pi * (len(ey_omega_k_list) / 2 + 1) / (dt * nt)])  # PRINT PLOT SIZES

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
    ax1.set_xlabel('k (1/m)')
    ax1.set_ylabel('omega (Hz)')
    ax2.set_title(title2)
    ax2.set_xlabel('k (1/m)')
    ax2.set_ylabel('omega (Hz)')
    ax3.set_title(title3)
    ax3.set_xlabel('k (1/m)')
    ax3.set_ylabel('omega (Hz)')
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
    ex_wv = ax1.plot(ex_k_list, ex_omega_list, color='black', linewidth=0.5)  # draw theoretical line on left plot
    ey_wv = ax2.plot(ey_k_list, ey_omega_list, color='black', linewidth=0.5)  # draw theoretical line on middle plot
    ez_wv = ax3.plot(ez_k_list, ez_omega_list, color='black', linewidth=0.5)  # draw theoretical line on right plot

    # CROP PLOT
    left, right = plt.xlim()  # get current left and right limit
    bottom, top = plt.ylim()  # get current left and right limit
    # right = 4 * math.sqrt(wp**2 + wc**2)  # set plot limit to 4x the plasma frequency
    if specie == "ion":
        height = 1E6  # 5.64E5  # plot height #math.sqrt(wp ** 2 + wc ** 2) * 5
        width = 0.01
    else:
        height = min(math.sqrt(wp ** 2 + wc ** 2) * 20, top)  # plot height #math.sqrt(wp ** 2 + wc ** 2) * 5
        width = height / c  # plot width

    # CROP PLOTS
    ax1.set_xlim(right=width)  # set x limit for left plot
    ax1.set_ylim(top=height)  # set y limit for left plot
    ax2.set_xlim(right=width)  # set x limit for middle plot
    ax2.set_ylim(top=height)  # set y limit for middle plot
    ax3.set_xlim(right=width)  # set x limit for right plot
    ax3.set_ylim(top=height)  # set y limit for right plot
    # plt.xlim([0, 2 * math.pi * ng / 2 / length])
    # plt.ylim([0, 2 * math.pi * nt / 2 / (dt * nt)])

    # PREVENT PLOT OVERLAPS
    plt.tight_layout()

    plt.savefig('E_{}.png'.format(file_name))  # save figure
    #plt.show()  # show figure


# RUN MAIN PROGRAM
plot("20221006-181403_ni1e+08_ti1e+04_te1e+07_b01e-07_theta90", "ion", "rho")
# plot_phase_space_animation("ni1e+08_ti1e+04_b01e-07_theta90_nt2097152", "both")
