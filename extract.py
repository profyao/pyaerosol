import hdf4
import numpy as np
import formatNum as fN
import re
from pyhdf.SD import SD
import scipy.io
import math
import collections



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
    tau0 = f.select('RegMeanSpectralOptDepth').get()[block-1 , : , :, BAND_GREEN-1]
    tau0[tau0 == -9999] = np.mean(tau0[tau0 != -9999])

    return tau0


def reg_dat(date, path, orbit, block, r):

    file_mat = _reg_mat_fname(date, path, orbit, block, r)
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
    dir_aerosol = '../../projects/aerosol/products/MIL2ASAE/' + date
    file_aerosol = dir_aerosol + '/' + HEADER_MIL2ASAE_FILENAME + path + '_O' + orbit + '_F12_0022.hdf'

    return file_aerosol


def _reg_mat_fname(date, path, orbit, block, r):

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    return '../../projects/aerosol/cache/data/' + date + '_P' + path + '_O' + orbit + '_B' + str(block) + '_R' + str(r) + '.mat'


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

    tau_cam_ss = np.asfortranarray(np.transpose(ss[:, math.ceil(xp/reg_scale), math.ceil(yp/reg_scale), COMPONENT_PARTICLE-1, :, :], [0, 3, 1, 2]))
    tau_cam_ms = np.asfortranarray(np.transpose(ms[:, math.ceil(xp/reg_scale), math.ceil(yp/reg_scale), COMPONENT_PARTICLE-1, :, :], [0, 3, 1, 2]))

    reg = collections.namedtuple('reg', 'channel_is_used min_equ_ref mean_equ_ref eof max_usable_eof')
    smart = collections.namedtuple('smart', 'ss ms')

    reg_p = reg(channel_is_used_p, min_equ_ref_p, mean_equ_ref_p, eof_p, max_usable_eof_p)
    smart_p = smart(tau_cam_ss, tau_cam_ms)

    return reg_p, smart_p
