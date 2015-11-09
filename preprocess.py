from pyhdf.SD import SD
import hdf4
import scipy.io
import numpy as np
import formatNum as fN
import re
import pandas as pd
import extract
import update
from collections import namedtuple

header_file_aerosol = '../../projects/aerosol/products/MIL2ASAE/'
header_data = '../../projects/aerosol/cache/data/'

PixelData = namedtuple('PixelData', 'reg smart')


def reg_smart(date, path, orbit, block, r):

    from constant import COMPONENT_NUM

    reg_dat = _reg_dat(date, path, orbit, block, r)
    optical_properties = particle(date, path, orbit)

    reg_is_used = reg_dat['reg'][0, 0]['reg_is_used'].T
    y, x = np.where(reg_is_used)
    ind_used = np.ravel(reg_dat['reg'][0, 0]['ind_used'], order='F') - 1
    num_reg_used = reg_dat['reg'][0, 0]['num_reg_used'][0][0]
    channel_is_used = reg_dat['reg'][0, 0]['channel_is_used']
    min_equ_ref = reg_dat['reg'][0, 0]['min_equ_ref']
    mean_equ_ref = reg_dat['reg'][0, 0]['mean_equ_ref']
    eof = reg_dat['reg'][0, 0]['eof']
    max_usable_eof = reg_dat['reg'][0, 0]['max_usable_eof'] - 1

    ss = reg_dat['smart'][0, 0]['ss']
    ms = reg_dat['smart'][0, 0]['ms']

    Q = get_q(r)

    i2d, j2d = np.nonzero(Q)
    reg_is_used = np.ravel(reg_is_used)
    mask = np.bool_(reg_is_used[i2d] & reg_is_used[j2d] & np.not_equal(i2d, j2d))
    i = ind_used[i2d[mask]]
    j = ind_used[j2d[mask]]

    tau0 = aod(date, path, orbit, block)

    tau = np.mean(tau0)*np.ones(num_reg_used)

    theta = 1.0/COMPONENT_NUM * np.ones((COMPONENT_NUM, num_reg_used), dtype=float)

    return x, y, i, j, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties


def particle(date, path, orbit):

    file_aerosol = _MIL2ASAE_fname(date, path, orbit)

    f = hdf4.HDF4_root(file_aerosol)
    str_table = f.children['Component Particle Information'].attr['Component Particle Properties - Summary Table'].value
    p1 = re.compile('Part 2 *').search(str_table).start()
    p2 = re.compile('Shape types:*').search(str_table).start()
    str_dat = str_table[p1:p2].split('\n')[8:92]
    optical_properties = np.array([map(float, re.compile('(\s\s+)').sub(',', x).split(',')[4:6]) for x in str_dat])

    return optical_properties


def aod(date, path, orbit, block):

    from constant import BAND_GREEN
    file_aerosol = _MIL2ASAE_fname(date, path, orbit)
    f = SD(file_aerosol)
    tau0 = f.select('RegMeanSpectralOptDepth').get()[block-1 , : , :, BAND_GREEN]
    tau0[tau0 == -9999] = np.mean(tau0[tau0 != -9999])

    return tau0


def _reg_dat(date, path, orbit, block, r):

    file_mat = _reg_mat_fname(date, path, orbit, block, r) # use matlab block number
    dat = scipy.io.loadmat(file_mat)

    return dat


def get_q(r):

    dat = scipy.io.loadmat('prec.mat')

    if r == 4400:
        return dat['Q_4400']
    elif r == 1100:
        return dat['Q_1100']
    else:
        print 'resolution not implemented!'


def _MIL2ASAE_fname(date, path, orbit):

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    from constant import HEADER_MIL2ASAE_FILENAME
    dir_aerosol = header_file_aerosol + date
    file_aerosol = dir_aerosol + '/' + HEADER_MIL2ASAE_FILENAME + path + '_O' + orbit + '_F12_0022.hdf'

    return str(file_aerosol)


def _reg_mat_fname(date, path, orbit, block, r):

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    return str(header_data + date + '_P' + path + '_O' + orbit + '_B' + str(block) + '_R' + str(r) + '.mat')


def merge_dict(file_xls, r):

    xls = pd.ExcelFile(file_xls)
    sheet_name = xls.sheet_names[0]
    df = xls.parse(sheet_name)

    dates = list(df['Dates'][5:6])
    paths = list(df['Paths'][5:6])
    orbits = list(df['Orbits'][5:6])
    blocks = list(df['Blocks'][5:6])

    dict_data0 = []
    dict_param0 =[]
    dict_neigh0 = []
    dict_env0 = []
    N = len(dates)
    optical_properties0 = [[]] * N
    num_reg_used0 = [0] * N
    i0 = [[]] * N
    j0 = [[]] * N

    for d in xrange(N):

        date = dates[d]
        path = paths[d]
        orbit = orbits[d]
        block = blocks[d]

        x, y, i, j, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties = \
        reg_smart(date, path, orbit, block, r)

        dict_data, dict_param = get_data_param(date, path, orbit, block,\
            x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r)
        dict_neigh = update.get_neigh(date, path, orbit, block, i, j, num_reg_used)
        dict_env = update.get_env_block(tau, theta, num_reg_used, dict_neigh)

        dict_data0 = dict_data0 + dict_data
        dict_param0 = dict_param0 + dict_param
        dict_neigh0 = dict_neigh0 + dict_neigh
        dict_env0 = dict_env0 + dict_env
        optical_properties0[d] = optical_properties
        num_reg_used0[d] = num_reg_used
        i0[d] = i
        j0[d] = j


        print date + " dictionary is done!"

    return dict_data0, dict_param0, dict_neigh0, dict_env0, optical_properties0, num_reg_used0, i0, j0, dates, paths, orbits, blocks


def get_data_param(date, path, orbit, block, x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r):

    ps = xrange(num_reg_used)
    keys = [fN.get_key(date, path, orbit, block, p) for p in ps]

    dict_data = []
    dict_param = []

    for p in ps:

        tau_p = tau[p]
        theta_p = theta[:, p]

        reg_p, smart_p = extract.pixel(x[p], y[p], channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r)
        _, _, resid_p = update.get_resid(tau_p, theta_p, reg_p, smart_p, optical_properties, r)

        dict_data.append((keys[p], PixelData(reg_p, smart_p)))
        dict_param.append((keys[p], update.PixelParam(tau_p, theta_p, resid_p)))

    return dict_data, dict_param

