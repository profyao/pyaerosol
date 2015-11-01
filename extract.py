import hdf4
import numpy as np
import formatNum as fN
import re
from pyhdf.SD import SD
import scipy.io
import math
import collections
import os

header_file_aerosol = '../../projects/aerosol/products/MIL2ASAE/'
header_data = '../../projects/aerosol/cache/data/'
header_result = "../../projects/aerosol/cache/spark/"

def reg_smart(date, path, orbit, block, r):

    from constant import COMPONENT_NUM

    #XDim_r = XDIM_R4400 * R4400/r
    #YDim_r = YDIM_R4400 * R4400/r

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

def _result_fname(date, path, orbit, block, r, Method):

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    return str(header_result + date + '_P' + path + '_O' + orbit + '_B' + str(block) + '_R' + str(r) + '_' + Method + '.mat')


def pixel(xp, yp, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r):

    from constant import R17600, COMPONENT_PARTICLE

    channel_is_used_p = np.transpose(channel_is_used[xp, yp, :, :]).astype(bool)
    min_equ_ref_p = np.transpose(min_equ_ref[xp, yp, :, :])
    mean_equ_ref_p = np.transpose(mean_equ_ref[xp, yp, :, :])

    reg_scale = R17600 / r

    if r > 1100:
        eof_p = eof[xp, yp, :, :, :]
        max_usable_eof_p = max_usable_eof[xp, yp, :]
    else:
        eof_p = eof[xp, yp, :, :]
        max_usable_eof_p = max_usable_eof[xp, yp]

    tau_cam_ss = np.asfortranarray(np.transpose(ss[:, math.ceil(xp/reg_scale), math.ceil(yp/reg_scale), COMPONENT_PARTICLE, :, :], [0, 3, 1, 2]))
    tau_cam_ms = np.asfortranarray(np.transpose(ms[:, math.ceil(xp/reg_scale), math.ceil(yp/reg_scale), COMPONENT_PARTICLE, :, :], [0, 3, 1, 2]))

    reg = collections.namedtuple('reg', 'channel_is_used min_equ_ref mean_equ_ref eof max_usable_eof')
    smart = collections.namedtuple('smart', 'ss ms')

    reg_p = reg(channel_is_used_p, min_equ_ref_p, mean_equ_ref_p, eof_p, max_usable_eof_p)
    smart_p = smart(tau_cam_ss, tau_cam_ms)

    return reg_p, smart_p


def save_tau_theta(dates, paths, orbits, blocks, r, method, paramRDD):

    N = len(dates)

    for i in xrange(N):

        date = dates[i]
        path = paths[i]
        orbit = orbits[i]
        block = blocks[i]

        tau = paramRDD.filter(lambda x: x[0][0] == date).map(lambda y: (y[0][4], y[1].tau)).collect()
        theta = paramRDD.filter(lambda x: x[0][0] == date).map(lambda y: (y[0][4], y[1].theta)).collect()

        tau = [x[1] for x in sorted(tau, key = lambda y: y[0])]
        theta = [x[1] for x in sorted(theta, key = lambda y: y[0])]

        file_result = _result_fname(date, path, orbit, block, r, method)

        if not os.path.exists(header_result):
            os.makedirs(header_result)

        scipy.io.savemat(file_result, {'tau': tau, 'theta': theta})

