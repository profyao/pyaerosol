import numpy as np
import scipy.io
import collections
import os
import math
import formatNum as fN

header_result = "../../projects/aerosol/cache/spark/"


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


def _result_fname(date, path, orbit, block, r, Method):

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    return str(header_result + date + '_P' + path + '_O' + orbit + '_B' + str(block) + '_R' + str(r) + '_' + Method + '.mat')
