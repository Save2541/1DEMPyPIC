import os

import numpy
import zarr

from . import qol
from . import unit_scale
from . import user_input


def unpack_data(sp_list, almanac, output, grids):
    """
    Unpack data into variables
    :param sp_list: list of specie parameters
    :param almanac: dictionary of useful values
    :param output: list of output arrays
    :param grids: grid list
    :return: variables
    """
    (n_sp, kt, m, wp, wc, v_th) = sp_list.get_info("n_sp", "kt", "real_mass", "wp", "wc", "vth")
    init_d_k = numpy.zeros(n_sp)
    i = 0
    for name in sp_list.name:
        init_d_k[i] = sp_list.init_d_wv[name]["wave number (k)"]
        i += 1
    init_v_k = numpy.zeros(shape=(n_sp, 3))
    i = 0
    for name in sp_list.name:
        v_wv = sp_list.init_v_wv[name]
        init_v_k[i][0] = v_wv["vxp"]["wave number (k)"]
        init_v_k[i][1] = v_wv["vy"]["wave number (k)"]
        init_v_k[i][2] = v_wv["vb0"]["wave number (k)"]
        i += 1
    (file_name, b0, rho, c, dx, dt, theta) = qol.read_almanac(almanac, "file name", "b0", "rho_mass", "c", "dx",
                                                              "dt_sample", "theta")
    (ex, ey, ez, by, bz, rho_list, jy, jz, den, x, vx) = output.get_output("ex", "ey", "ez", "by", "bz", "rho", "jy",
                                                                           "jz", "den", "x", "v")
    grid_x = grids.x
    return (
        n_sp, kt, m, wp, wc, v_th, init_d_k, init_v_k, file_name, b0, rho, c, dx, dt, theta, ex, ey, ez, by, bz,
        rho_list,
        jy, jz, den, x, vx, grid_x)


def convert_units(almanac, kt, wp, wc, v_th, b0, c, dx, dt, ex, ey, ez, by, bz, rho_list, jy, jz, den, x, vx, grid_x):
    """
    Convert data from simulation units into SI units
    :param almanac: dictionary of useful values
    :param kt: specie thermal energy
    :param wp: specie plasma frequency
    :param wc: specie cyclotron frequency
    :param v_th: specie thermal velocity
    :param b0: external magnetic field magnitude
    :param c: speed of light
    :param dx: grid size
    :param dt: recorded time step
    :param ex: recorded x-electric field
    :param ey: recorded y-electric field
    :param ez: recorded z-electric field
    :param by: recorded y-magnetic field
    :param bz: recorded z-magnetic field
    :param rho_list: recorded charge density
    :param jy: recorded y-current density
    :param jz: recorded z-current density
    :param den: recorded number density
    :param x: recorded particle positions
    :param vx: recorded particle x-velocity
    :param grid_x: grid positions
    :return: data in SI units
    """
    kt = unit_scale.convert_to_joules(kt, almanac)
    wp = unit_scale.convert_to_radians_per_second(wp, almanac)
    wc = unit_scale.convert_to_radians_per_second(wc, almanac)
    v_th = unit_scale.convert_to_meters_per_second(v_th, almanac)
    b0 = unit_scale.convert_to_tesla(b0, almanac)
    c = unit_scale.convert_to_meters_per_second(c, almanac)
    dx = unit_scale.convert_to_meters(dx, almanac)
    dt = unit_scale.convert_to_seconds(dt, almanac)
    ex = unit_scale.convert_to_volts_per_meter(ex, almanac)
    ey = unit_scale.convert_to_volts_per_meter(ey, almanac)
    ez = unit_scale.convert_to_volts_per_meter(ez, almanac)
    by = unit_scale.convert_to_tesla(by, almanac)
    bz = unit_scale.convert_to_tesla(bz, almanac)
    rho_list = unit_scale.convert_to_coulombs_per_cubic_meter(rho_list, almanac)
    jy = unit_scale.convert_to_amperes_per_square_meter(jy, almanac)
    jz = unit_scale.convert_to_amperes_per_square_meter(jz, almanac)
    den = unit_scale.convert_to_particles_per_cubic_meter(den, almanac)
    x = unit_scale.convert_to_meters(x, almanac)
    vx = unit_scale.convert_to_meters_per_second(vx, almanac)
    grid_x = unit_scale.convert_to_meters(grid_x, almanac)
    return kt, wp, wc, v_th, b0, c, dx, dt, ex, ey, ez, by, bz, rho_list, jy, jz, den, x, vx, grid_x


def save_to_zip(sp_list, almanac, output, grids, ng=user_input.ng, nt_sample=user_input.nt_sample,
                is_electromagnetic=user_input.is_electromagnetic):
    # UNPACK DATA
    (n_sp, kt, m, wp, wc, v_th, init_d_k, init_v_k, file_name, b0, rho, c, dx, dt, theta, ex, ey, ez, by, bz, rho_list,
     jy, jz, den, x, vx, grid_x) = unpack_data(sp_list, almanac, output, grids)
    # CONVERT TO SI UNITS
    kt, wp, wc, v_th, b0, c, dx, dt, ex, ey, ez, by, bz, rho_list, jy, jz, den, x, vx, grid_x = convert_units(almanac,
                                                                                                              kt,
                                                                                                              wp, wc,
                                                                                                              v_th,
                                                                                                              b0, c, dx,
                                                                                                              dt,
                                                                                                              ex, ey,
                                                                                                              ez, by,
                                                                                                              bz,
                                                                                                              rho_list,
                                                                                                              jy, jz,
                                                                                                              den, x,
                                                                                                              vx,
                                                                                                              grid_x)
    # CREATE OUTPUT DIRECTORY IF NOT EXIST
    path = "output/{}.zip".format(file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # SAVE TO ZIP FILE
    zarr.save(path, is_electromagnetic=is_electromagnetic, ng=ng, nt=nt_sample, n_sp=n_sp,
              kt=kt, m=m, b0=b0, rho=rho, ex=ex, ey=ey, ez=ez, by=by, bz=bz, rho_list=rho_list, jy=jy, jz=jz, den=den,
              x=x,
              vx=vx, c=c, dx=dx, dt=dt, wp=wp, wc=wc, theta=theta, v_th=v_th, grid_x=grid_x, init_d_k=init_d_k,
              init_v_k=init_v_k)
