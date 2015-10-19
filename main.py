import extract
import numpy as np
import update

date = '2011.06.02'
path = 16
orbit = 60934
block = 59
r = 4400

from constant import XDIM_R4400, YDIM_R4400, R4400, COMPONENT_NUM

XDim_r = XDIM_R4400 * R4400/r
YDim_r = YDIM_R4400 * R4400/r

reg_dat = extract.reg_dat(date, path, orbit, block, r)
optical_properties = extract.particle(date, path, orbit)


reg_is_used = reg_dat['reg'][0, 0]['reg_is_used']
x, y = np.where(reg_is_used)
ind_used = np.ravel(reg_dat['reg'][0, 0]['ind_used'], order='F')
num_reg_used = reg_dat['reg'][0, 0]['num_reg_used'][0][0]
channel_is_used = reg_dat['reg'][0, 0]['channel_is_used']
min_equ_ref = reg_dat['reg'][0, 0]['min_equ_ref']
mean_equ_ref = reg_dat['reg'][0, 0]['mean_equ_ref']
eof = reg_dat['reg'][0, 0]['eof']
max_usable_eof = reg_dat['reg'][0, 0]['max_usable_eof']

ss = reg_dat['smart'][0, 0]['ss']
ms = reg_dat['smart'][0, 0]['ms']

Q = extract.get_q(r)

i2d, j2d = np.nonzero(Q)
reg_is_used = np.ravel(reg_is_used, order='F')
mask = np.bool_(reg_is_used[i2d] & reg_is_used[j2d] & np.not_equal(i2d,j2d))

tau0 = extract.aod(date, path, orbit, block)

tau = np.mean(tau0)*np.ones(num_reg_used)

theta = 1.0/COMPONENT_NUM * np.ones((COMPONENT_NUM, num_reg_used),dtype=float)

p = 500
xp = x[p]
yp = y[p]
theta_p = theta[:, p]
tau_p = tau[p]

reg_p, smart_p = extract.pixel(xp, yp, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r)

atm_path, surf, resid = update.get_resid(tau_p, theta_p, reg_p, smart_p, optical_properties, r)

print atm_path, resid