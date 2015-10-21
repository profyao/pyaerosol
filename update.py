import numpy as np
from constant import CAM_DIM, BAND_DIM, MODEL_COMPONENTDIM, COMPONENT_PARTICLE, BAND_GREEN, COMPONENT_NUM, NCHANNEL
import interpol
import extract
from collections import namedtuple

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
        else:
            ss = np.zeros((CAM_DIM,COMPONENT_NUM), order='F')
            ms = np.zeros((CAM_DIM,COMPONENT_NUM), order='F')
            interpol.interpol_3d(ss, tau_band, ss_band, np.asarray(ss_band.shape, dtype=np.int32))
            interpol.interpol_3d(ms, tau_band, ms_band, np.asarray(ms_band.shape, dtype=np.int32))

        atm_path[:, band] = (rayleigh + ss).dot(fraction_band) + (ms - rayleigh).dot( \
            ssa_mixture / ssa_v * np.exp( - tau_band * abs(ssa_mixture - ssa_v)) * fraction_band)

    return atm_path, surf_limit


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


def get_data_param(x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r):

    keys = xrange(num_reg_used)
    dict_data = []
    dict_param = []

    for p in keys:

        tau_p = tau[p]
        theta_p = theta[:, p]

        reg_p, smart_p = extract.pixel(x[p], y[p], channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r)
        _, _, resid_p = get_resid(tau_p, theta_p, reg_p, smart_p, optical_properties, r)

        dict_data.append((p, PixelData(reg_p, smart_p)))
        dict_param.append((p, PixelParam(tau_p, theta_p, resid_p)))

    return dict_data, dict_param


def get_sigmasq(paramRDD):

    a = paramRDD.map(lambda x: x[1].resid).reduce(lambda x, y: rm_nan_inf(x)**2 + rm_nan_inf(y)**2)

    b = paramRDD.map(lambda x: x[1].resid).reduce(lambda x, y: (np.isinf(x) | np.isnan(x)) + (np.isinf(y) | np.isnan(x)))

    return a/(b+2)


def rm_nan_inf(resid):

    resid[np.isnan(resid) | np.isinf(resid)] = 0

    return resid


def get_kappa(paramRDD, i, j, num_reg_used):

    tau = np.array(paramRDD.map(lambda x: x[1].tau).collect())

    tau_2d = 0.5 * sum((tau[i] - tau[j])**2)

    if tau_2d == 0:
        tau_2d = 1e3

    return (num_reg_used - 3) / tau_2d


def get_env(tau, theta, num_reg_used, dict_neigh):

    dict_env = []

    for p in xrange(num_reg_used):

        neighbor = dict_neigh[p][1]

        if neighbor is not None:
            tau_neigh = tau[neighbor]
            theta_neigh = theta[:, neighbor]
            n_neigh = len(neighbor)
        else:
            tau_neigh = None
            theta_neigh = None
            n_neigh = 0

        dict_env.append((p, PixelEnv(tau_neigh, theta_neigh, n_neigh)))

    return dict_env


def get_dict_neigh(i, j, num_reg_used):

    dict_neigh = []

    for p in xrange(num_reg_used):

        idx = np.equal(j, p)

        if any(idx):
            dict_neigh.append((p, i[idx]))
        else:
            dict_neigh.append((p, None))

    return dict_neigh

#def get_paramRDD(dataRDD, paramRDD, envRDD):



def get_tau(tau, theta, resid, tau_neigh, n_neigh, kappa, sigmasq, delta, reg, smart, optical_properties, r):

    if np.isinf(resid[0]):

        tau_next = 0.8 * tau
        s0 = 0
        s1 = 0

    else:

        if n_neigh > 0:

            mu = 0.5 * (np.mean(tau_neigh) + tau)
            tau_next = mu + delta * np.random.normal()

            print tau_next
            tau_next = max(0, tau_next)
            tau_next = min(3, tau_next)

            s0 = kappa * sum((tau - tau_neigh)**2)
            s1 = kappa * sum((tau_next - tau_neigh)**2)
        else:

            tau_next = tau + delta * np.random.normal()

            tau_next = max(0, tau_next)
            tau_next = min(3, tau_next)

            s0 = 0
            s1 = 0

    _, _, resid_next = get_resid(tau_next, theta, reg, smart, optical_properties, r)

    chisq_next = np.nansum(resid_next**2 / sigmasq)
    chisq = np.nansum(resid**2 / sigmasq)

    if (np.isinf(resid[0]) and np.isinf(resid_next[0])) or chisq + s0 > chisq_next + s1 :
        return tau_next, resid_next
    else:
        return tau, resid


def get_theta(tau, theta, resid, theta_neigh, n_neigh, sigmasq, reg, smart, optical_properties, r):

    if n_neigh > 0:
        mu = np.mean(theta_neigh, axis=1)
    else:
        mu = theta

    theta_next = map(lambda x: np.random.gamma(x, 1), mu)
    theta_next = theta_next / np.sum(theta_next)

    _, _, resid_next = get_resid(tau, theta_next, reg, smart, optical_properties, r)

    chisq_next = np.nansum(resid_next**2 / sigmasq)
    chisq = np.nansum(resid**2 / sigmasq)

    if (np.isinf(resid[0]) and np.isinf(resid_next[0])) or chisq > chisq_next:
        return theta_next, resid_next
    else:
        return theta, resid