import math

import numpy

from . import plot_config
from . import qol
from . import constants


def is_parallel(theta):
    """
    Check if b0 is parallel to k
    :param theta: angle between b0 and k (x-axis)
    :return: boolean
    """
    return theta == math.radians(90)


def is_perpendicular(theta):
    """
    Check if b0 is perpendicular to k
    :param theta: angle between b0 and k (x-axis)
    :return: boolean
    """
    return theta == 0


def auto_get_line(plot_key, data, waves_of_interest=None):
    """
    Automatically get theoretical lines without user's specification
    :param plot_key: value to be plotted
    :param data: loaded data file
    :param waves_of_interest: 0 for electron waves, 1 for ion waves
    :return: theoretical line
    """
    # SET DEFAULT VALUE
    if waves_of_interest is None:
        waves_of_interest = plot_config.waves_of_interest
    # READ FROM DATA FILE
    (b0, theta) = qol.read_almanac(data, "b0", "theta")
    # CATEGORIES OF PLOTTED VALUES
    longitudinal_waves = ("rho", "ex")
    y_transverse_waves = ("ey", "by")
    z_transverse_waves = ("ez", "bz")
    # FIND AN APPROPRIATE THEORETICAL LINE
    if plot_key in longitudinal_waves or plot_key[:3] == "den":
        # ELECTROSTATIC WAVES
        if waves_of_interest == 0:
            # ELECTRON ELECTROSTATIC WAVES
            if b0 == 0 or is_parallel(theta):
                # PLASMA OSCILLATIONS
                key = 143
            elif is_perpendicular(theta):
                # UPPER HYBRID OSCILLATIONS
                key = 144
            else:
                key = None
        elif waves_of_interest == 1:
            # ION ELECTROSTATIC WAVES
            if b0 == 0 or is_parallel(theta):
                # ACOUSTIC WAVES
                key = 145
            elif is_perpendicular(theta):
                # LOWER HYBRID OSCILLATIONS
                key = 147
            else:
                key = None
        else:
            key = None
    elif plot_key in y_transverse_waves or plot_key in z_transverse_waves:
        # ELECTROMAGNETIC WAVES
        if waves_of_interest == 0:
            # ELECTRON ELECTROMAGNETIC WAVES
            if b0 == 0:
                # LIGHT WAVES
                key = 148
            elif is_perpendicular(theta):
                if plot_key in z_transverse_waves:
                    # O WAVE
                    key = 149
                else:
                    # X WAVE
                    key = 150
            elif is_parallel(theta):
                # R WAVE (WHISTLER MODE)
                key = 151
            else:
                key = None
        elif waves_of_interest == 1:
            # ION ELECTROMAGNETIC WAVES
            if is_parallel(theta):
                # ALFVEN WAVE
                key = 153
            elif is_perpendicular(theta):
                # MAGNETOSONIC WAVE
                key = 154
            else:
                key = None
        else:
            key = None
    else:
        key = None

    return get_theory_line(key, data)


def get_sound_speed(kte, kti, mi, gamma_e=1, gamma_i=3):
    """
    Calculate sound speed
    :param kte: electron thermal energy
    :param kti: ion thermal energy
    :param mi: ion mass
    :param gamma_e: electron gamma factor
    :param gamma_i: ion gamma factor
    :return: sound speed
    """
    return math.sqrt((gamma_e * kte + gamma_i * kti) / mi)


def get_alfven_speed(b0, rho, c=None, mu=None):
    """
    Calculate alfven speed
    :param b0: external magnetic field
    :param rho: plasma mass density
    :param c: speed of light
    :param mu: vacuum permeability
    :return: alfven speed
    """
    assert b0 != 0, "CAN'T CALCULATE ALFVEN SPEED WITHOUT MAGNETIC FIELD!"
    if c is None:
        c = constants.c
    if mu is None:
        mu = constants.mu
    return math.sqrt(c ** 2 / (1 + c ** 2 * rho * mu / b0 ** 2))


def get_theory_line(key, data, c=None, electron_id=None, ion_id=None):
    """
    Get theory line
    :param key: theory line to get (same key as in Chen textbook)
    :param data: loaded data file
    :param c: speed of light
    :param electron_id: index of electron
    :param ion_id: index of ion
    :return:
    """
    # SET DEFAULT VALUES
    if electron_id is None:
        electron_id = plot_config.electron_id
    if ion_id is None:
        ion_id = plot_config.ion_id
    if c is None:
        c = constants.c

    # READ DATA FROM FILE
    (b0, dx, dt, ng, nt, m, kt, wp, wc, v_th, rho) = qol.read_almanac(data, "b0", "dx", "dt", "ng", "nt", "m", "kt",
                                                                      "wp", "wc", "v_th", "rho")
    # GET THEORETICAL LINE
    if key == 143:
        # PLASMA OSCILLATIONS
        k = qol.get_sample_k(dx, ng)
        omega = numpy.sqrt(wp[electron_id] ** 2 + 3 * k ** 2 * v_th[electron_id] ** 2)
    elif key == 144:
        # UPPER HYBRID OSCILLATIONS
        k = qol.get_sample_k(dx, ng)
        omega = numpy.sqrt(wp[electron_id] ** 2 + wc[electron_id] ** 2 + 3 / 2 * k ** 2 * v_th[electron_id] ** 2)
    elif key == 145:
        # ACOUSTIC WAVES
        k = qol.get_sample_k(dx, ng)
        omega = k * get_sound_speed(kt[electron_id], kt[ion_id], m[ion_id])
    elif key == 146:
        # ELECTROSTATIC ION CYCLOTRON WAVES
        k = qol.get_sample_k(dx, ng)
        omega = numpy.sqrt(wc[ion_id] ** 2 + k ** 2 * get_sound_speed(kt[electron_id], kt[ion_id], m[ion_id]) ** 2)
    elif key == 147:
        # LOWER HYBRID OSCILLATIONS
        k = qol.get_sample_k(dx, ng)
        w_pi2 = wp[ion_id] ** 2
        wci_times_wc = wc[ion_id] * wc[electron_id]
        omega = numpy.sqrt(
            w_pi2 * wci_times_wc / (w_pi2 + wci_times_wc) + k ** 2 * get_sound_speed(kt[electron_id], kt[ion_id],
                                                                                     m[ion_id]) ** 2)
    elif key == 148:
        # LIGHT WAVES
        k = qol.get_sample_k(dx, ng)
        omega = numpy.sqrt(wp[electron_id] ** 2 + wc[electron_id] ** 2 + k ** 2 * c ** 2)
    elif key == 149:
        # ORDINARY WAVES
        k = qol.get_sample_k(dx, ng)
        omega = numpy.sqrt(wp[electron_id] ** 2 + wc[electron_id] ** 2 + k ** 2 * c ** 2)
    elif key == 150:
        # EXTRAORDINARY WAVES
        omega = qol.get_sample_w(dt, nt)
        omega_2 = omega ** 2
        w_pe2 = wp[electron_id]
        k = numpy.sqrt(omega_2 - w_pe2 * ((omega_2 - w_pe2) / (omega_2 - w_pe2 - wc[electron_id] ** 2))) / c
    elif key == 151:
        # R WAVE (WHISTLER MODE)
        omega = qol.get_sample_w(dt, nt)
        k = numpy.sqrt(omega ** 2 - wp[electron_id] ** 2 / (1 - wc[electron_id] / omega)) / c
    elif key == 152:
        # L WAVE
        omega = qol.get_sample_w(dt, nt)
        k = numpy.sqrt(omega ** 2 - wp[electron_id] ** 2 / (1 + wc[electron_id] / omega)) / c
    elif key == 153:
        # ALFVEN WAVE
        k = qol.get_sample_k(dx, ng)
        omega = k * get_alfven_speed(b0, rho, c)
    elif key == 154:
        # MAGNETOSONIC WAVE
        k = qol.get_sample_k(dx, ng)
        v_a2 = get_alfven_speed(b0, rho, c) ** 2
        omega = k * c * math.sqrt(
            (get_sound_speed(kt[electron_id], kt[ion_id], m[ion_id]) ** 2 + v_a2) / (c ** 2 + v_a2))
    else:
        assert key is None, "{} IS NOT A VALID INPUT FOR overlays!".format(key)
        k = []
        omega = []

    return k, omega
