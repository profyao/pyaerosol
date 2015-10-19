import numpy as np
from constant import CAM_DIM, BAND_DIM, MODEL_COMPONENTDIM, COMPONENT_PARTICLE, BAND_GREEN, COMPONENT_NUM, NCHANNEL
import interpol


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

        scale_factor = np.dot(ExtCroSect[band, COMPONENT_PARTICLE-1] / ExtCroSect[BAND_GREEN-1, COMPONENT_PARTICLE-1], theta)
        fraction_band = ExtCroSect[band, COMPONENT_PARTICLE-1] / ExtCroSect[BAND_GREEN-1, COMPONENT_PARTICLE-1] * theta / scale_factor
        ssa_v = CompSSA[band, COMPONENT_PARTICLE-1]
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
            idx = np.less_equal(np.arange(num_cam_used), reg.max_usable_eof[band]-1)
            surf[cam_used, band] = eof_band[:, idx].dot(exp_coef[idx])
            resid[cam_used, band] = diff - surf[cam_used, band]

        return np.ravel(resid, 'F'), np.ravel(surf, 'F')

    elif r == 1100:

            cam_used = reg.channel_is_used[:, BAND_GREEN-1]
            num_cam_used = np.sum(cam_used)
            diff = reg.mean_equ_ref[cam_used, :] - atm_path[cam_used, :]
            eof = reg.eof[1:num_cam_used, 1:num_cam_used]

            exp_coef = diff.dot(eof)
            idx = np.less_equal(np.arange(num_cam_used), reg.max_usable_eof-1)
            surf[cam_used, :] = eof[:, idx].dot(exp_coef[idx, :])
            resid[cam_used, :] = diff - surf[cam_used, :]

            return np.ravel(resid), np.ravel(surf)

    else:

        print 'resolution is incorrect!'

