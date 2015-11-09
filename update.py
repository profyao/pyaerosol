import numpy as np
import pandas as pd
from constant import CAM_DIM, BAND_DIM, MODEL_COMPONENTDIM, COMPONENT_PARTICLE, BAND_GREEN, COMPONENT_NUM, NCHANNEL, POINTS, MODEL_OPTICALDEPTHLEN
import interpol
import extract
import formatNum as fN
from collections import namedtuple
#from scipy.interpolate import griddata

PixelData = namedtuple('PixelData', 'reg smart')
PixelParam = namedtuple('PixelParam', 'tau theta resid')
PixelEnv = namedtuple('PixelEnv', 'tau_neigh theta_neigh n_neigh')


def get_resid(tau, theta, reg, smart, optical_properties, r):

    atm_path, surf = get_model(tau, theta, optical_properties, smart)

    valid = np.ravel(reg.channel_is_used)

    upbd = np.greater(np.ravel(atm_path)[valid], 0.9 * np.ravel(reg.min_equ_ref)[valid])

    if ~np.any(upbd):
        resid, surf = get_resid_eof(reg, atm_path, r)
    else:
        resid = np.inf * np.ones(NCHANNEL)
        surf = np.nan * np.ones(NCHANNEL)

    return np.ravel(atm_path, order='F'), surf, resid


def get_model(tau, theta, optical_properties, smart):

    atm_path = np.zeros((CAM_DIM, BAND_DIM))
    surf_limit = np.zeros((CAM_DIM, BAND_DIM))
    ExtCroSect = np.reshape(optical_properties[:, 0], (MODEL_COMPONENTDIM, BAND_DIM)).T
    CompSSA = np.reshape(optical_properties[:, 1], (MODEL_COMPONENTDIM, BAND_DIM)).T

    for band in xrange(BAND_DIM):

        scale_factor = np.dot(ExtCroSect[band, COMPONENT_PARTICLE] / ExtCroSect[BAND_GREEN, COMPONENT_PARTICLE], theta)
        fraction_band = ExtCroSect[band, COMPONENT_PARTICLE] / ExtCroSect[BAND_GREEN, COMPONENT_PARTICLE] * theta / scale_factor
        ssa_v = CompSSA[band, COMPONENT_PARTICLE]
        ssa_mixture = np.dot(ssa_v, fraction_band)

        tau_band = tau * scale_factor
        ss_band = smart.ss[:, :, :, band]
        ms_band = smart.ms[:, :, :, band]
        rayleigh = ms_band[0, :, :]

        if tau_band <= 0:
            ss = ss_band[0, :, :]
            ms = rayleigh
        elif tau_band >=6:
            ss = ss_band[MODEL_OPTICALDEPTHLEN-1, :, :]
            ms = ms_band[MODEL_OPTICALDEPTHLEN-1, :, :]
        else:

            ss = np.zeros((CAM_DIM,COMPONENT_NUM), order='F')
            ms = np.zeros((CAM_DIM,COMPONENT_NUM), order='F')
            interpol.interpol_3d(ss, tau_band, ss_band, np.asarray(ss_band.shape, dtype=np.int32))
            interpol.interpol_3d(ms, tau_band, ms_band, np.asarray(ms_band.shape, dtype=np.int32))
            #ss = interpol_3d(tau_band, ss_band, 'linear')
            #ms = interpol_3d(tau_band, ms_band, 'linear')

        atm_path[:, band] = (rayleigh + ss).dot(fraction_band) + (ms - rayleigh).dot( \
            ssa_mixture / ssa_v * np.exp( - tau_band * abs(ssa_mixture - ssa_v)) * fraction_band)

    return atm_path, surf_limit


"""
def interpol_3d(tau, dat, method):

    dat = np.ravel(dat, order = 'F')
    cam_i, comp_i, tau_i = np.meshgrid(np.arange(CAM_DIM), np.arange(COMPONENT_NUM), tau)

    fv = griddata(POINTS, dat, (cam_i, comp_i, tau_i), method)

    fv = np.asfortranarray(fv[:, :, 0].T)

    return fv
"""


def get_resid_eof(reg, atm_path, r):

    resid = np.nan * np.ones((CAM_DIM, BAND_DIM))
    surf = np.nan * np.ones((CAM_DIM, BAND_DIM))

    if r > 1100:

        for band in xrange(BAND_DIM):

            cam_used = reg.channel_is_used[:, band]
            num_cam_used = np.sum(cam_used)
            diff = reg.mean_equ_ref[cam_used, band] - atm_path[cam_used, band]
            eof_band = reg.eof[band, :num_cam_used, :num_cam_used]

            exp_coef = diff.dot(eof_band)
            idx = np.less_equal(np.arange(num_cam_used), reg.max_usable_eof[band])
            surf[cam_used, band] = eof_band[:, idx].dot(exp_coef[idx])
            resid[cam_used, band] = diff - surf[cam_used, band]

        return np.ravel(resid, 'F'), np.ravel(surf, 'F')

    elif r == 1100:

            cam_used = reg.channel_is_used[:, BAND_GREEN]
            num_cam_used = np.sum(cam_used)
            diff = reg.mean_equ_ref[cam_used, :] - atm_path[cam_used, :]
            eof = reg.eof[1:num_cam_used, 1:num_cam_used]

            exp_coef = diff.dot(eof)
            idx = np.less_equal(np.arange(num_cam_used), reg.max_usable_eof)
            surf[cam_used, :] = eof[:, idx].dot(exp_coef[idx, :])
            resid[cam_used, :] = diff - surf[cam_used, :]

            return np.ravel(resid), np.ravel(surf)

    else:

        print 'resolution is incorrect!'


def get_data_param(date, path, orbit, block, x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r):

    ps = xrange(num_reg_used)
    keys = [fN.get_key(date, path, orbit, block, p) for p in ps]

    dict_data = []
    dict_param = []

    for p in ps:

        tau_p = tau[p]
        theta_p = theta[:, p]

        reg_p, smart_p = extract.pixel(x[p], y[p], channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r)
        _, _, resid_p = get_resid(tau_p, theta_p, reg_p, smart_p, optical_properties, r)

        dict_data.append((keys[p], PixelData(reg_p, smart_p)))
        dict_param.append((keys[p], PixelParam(tau_p, theta_p, resid_p)))

    return dict_data, dict_param


def get_sigmasq(paramRDD, dates, paths, orbits, blocks):

    N = len(dates)
    sigmasq = np.zeros((NCHANNEL, N))

    for d in xrange(N):

        key = fN.get_date_key(dates[d], paths[d], orbits[d], blocks[d])
        residRDD = paramRDD.filter(lambda x: x[0][:4] == key).map(lambda x: x[1].resid)
        sigmasq[:, d] = get_sigmasq_block(residRDD)

    return sigmasq


def get_sigmasq_block(residRDD):

    a = residRDD.map(lambda x: rm_nan_inf(x)**2).reduce(lambda x, y: x + y)

    b = residRDD.map(lambda x: np.isfinite(x).astype(float)).reduce(lambda x, y: x + y)

    return a/(b + 2)


def rm_nan_inf(resid):

    resid[np.isnan(resid) | np.isinf(resid)] = 0

    return resid


def get_kappa_block(tau, i, j, num_reg_used):

    tau_2d = 0.5 * sum((tau[i] - tau[j])**2)

    if tau_2d == 0:
        tau_2d = 1e3

    return (num_reg_used - 3) / tau_2d


def get_kappa(paramRDD, i0, j0, num_reg_used0, dates, paths, orbits, blocks):

    N = len(num_reg_used0)
    kappa = np.zeros(N)

    for d in xrange(N):

        key = fN.get_date_key(dates[d], paths[d], orbits[d], blocks[d])
        tau = paramRDD.filter(lambda x: x[0][:4] == key).map(lambda x: x[1].tau).collect()
        kappa[d] = get_kappa_block(np.array(tau), i0[d], j0[d], num_reg_used0[d])

    return kappa


def get_neigh(date, path, orbit, block, i, j, num_reg_used):

    dict_neigh = []
    ps = xrange(num_reg_used)
    keys = [fN.get_key(date, path, orbit, block, p) for p in ps]

    for p in ps:

        idx = np.equal(j, p)

        if any(idx):
            dict_neigh.append((keys[p], i[idx]))
        else:
            dict_neigh.append((keys[p], None))

    return dict_neigh


def get_param_env(dates, pixelRDD, kappa, sigmasq, delta, num_reg_used, dict_neigh, optical_properties, r):

    paramRDD = pixelRDD.map(lambda x: get_param(x[0], dates, x[1], kappa, sigmasq, optical_properties, delta, r))

    dict_env = get_env(dates, paramRDD, num_reg_used, dict_neigh)

    return paramRDD, dict_env


def get_param(key, dates, pixel, kappa0, sigmasq0, optical_properties0, delta, r):

    d = dates.index(key[0])
    optical_properties = optical_properties0[d]
    kappa = kappa0[d]
    sigmasq = sigmasq0[:, d]

    tau = pixel[0][0].tau
    theta = pixel[0][0].theta
    resid = pixel[0][0].resid
    tau_neigh = pixel[0][1].tau_neigh
    theta_neigh = pixel[0][1].theta_neigh
    n_neigh = pixel[0][1].n_neigh
    reg = pixel[1].reg
    smart = pixel[1].smart

    tau, resid = get_tau(tau, theta, resid, tau_neigh, n_neigh, kappa, sigmasq, delta, reg, smart, optical_properties, r)
    theta, resid = get_theta(tau, theta, resid, theta_neigh, n_neigh, sigmasq, reg, smart, optical_properties, r)

    return key, PixelParam(tau, theta, resid)


def get_env(dates, paramRDD, num_reg_used, dict_neigh):

    dict_env0 = []

    for d in dates:

        tau = paramRDD.filter(lambda x: x[0][0] == d).map(lambda x: (x[0][4], x[1].tau)).collect()
        theta = paramRDD.filter(lambda x: x[0][0] == d).map(lambda x: (x[0][4], x[1].theta)).collect()
        tau = np.array([x[1] for x in sorted(tau, key = lambda y: y[0])])
        theta = np.array([x[1] for x in sorted(theta, key = lambda y: y[0])]).T

        dict_neigh_d = [dn for dn in dict_neigh if dn[0][0] == d]

        dict_env = get_env_block(tau, theta, num_reg_used[dates.index(d)], dict_neigh_d)

        dict_env0 = dict_env0 + dict_env

    return dict_env0


def get_env_block(tau, theta, num_reg_used, dict_neigh):

    dict_env = []

    for k in xrange(num_reg_used):

        neighbor = dict_neigh[k][1]

        if neighbor is not None:
            tau_neigh = tau[neighbor]
            theta_neigh = theta[:, neighbor]
            n_neigh = len(neighbor)
        else:
            tau_neigh = None
            theta_neigh = None
            n_neigh = 0

        dict_env.append((dict_neigh[k][0], PixelEnv(tau_neigh, theta_neigh, n_neigh)))

    return dict_env


def get_tau(tau, theta, resid, tau_neigh, n_neigh, kappa, sigmasq, delta, reg, smart, optical_properties, r):

    if np.isinf(resid[0]):

        tau_next = 0.8 * tau
        s0 = 0
        s1 = 0

    else:

        if n_neigh > 0:

            mu = 0.5 * (np.mean(tau_neigh) + tau)
            tau_next = mu + delta * np.random.normal()

            s0 = kappa * np.sum((tau - tau_neigh)**2)
            s1 = kappa * np.sum((tau_next - tau_neigh)**2)
        else:

            tau_next = tau + delta * np.random.normal()

            s0 = 0
            s1 = 0

    tau_next = max(0., tau_next)
    tau_next = min(3., tau_next)

    _, _, resid_next = get_resid(tau_next, theta, reg, smart, optical_properties, r)

    chisq_next = np.nansum(resid_next**2 / sigmasq.T)
    chisq = np.nansum(resid**2 / sigmasq.T)

    if (np.isinf(resid[0]) and np.isinf(resid_next[0])) or chisq + s0 > chisq_next + s1:
        return tau_next, resid_next
    else:
        return tau, resid


def get_theta(tau, theta, resid, theta_neigh, n_neigh, sigmasq, reg, smart, optical_properties, r):

    if n_neigh > 0:
        mu = np.mean(theta_neigh, axis=1)
    else:
        mu = theta

    theta_next = np.zeros(mu.shape)
    idx = mu >= 1e-6
    theta_next[idx] = np.random.gamma(mu[idx], 1, mu[idx].shape)
    theta_next = theta_next / np.sum(theta_next)

    _, _, resid_next = get_resid(tau, theta_next, reg, smart, optical_properties, r)

    chisq_next = np.nansum(resid_next**2 / sigmasq.T)
    chisq = np.nansum(resid**2 / sigmasq.T)

    if (np.isinf(resid[0]) and np.isinf(resid_next[0])) or chisq > chisq_next:
        return theta_next, resid_next
    else:
        return theta, resid


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
        extract.reg_smart(date, path, orbit, block, r)

        dict_data, dict_param = get_data_param(date, path, orbit, block,\
            x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r)
        dict_neigh = get_neigh(date, path, orbit, block, i, j, num_reg_used)
        dict_env = get_env_block(tau, theta, num_reg_used, dict_neigh)

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

