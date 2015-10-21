import extract
import update
import numpy as np
from constant import XDIM_R4400, YDIM_R4400, R4400, COMPONENT_NUM
from pyspark import SparkContext

date = '2011.06.02'
path = 16
orbit = 60934
block = 59 - 1
r = 4400

XDim_r = XDIM_R4400 * R4400/r
YDim_r = YDIM_R4400 * R4400/r

reg_dat = extract.reg_dat(date, path, orbit, block, r)
optical_properties = extract.particle(date, path, orbit)

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

Q = extract.get_q(r)

i2d, j2d = np.nonzero(Q)
reg_is_used = np.ravel(reg_is_used)
mask = np.bool_(reg_is_used[i2d] & reg_is_used[j2d] & np.not_equal(i2d, j2d))
i = ind_used[i2d[mask]]
j = ind_used[j2d[mask]]

tau0 = extract.aod(date, path, orbit, block)

tau = np.mean(tau0)*np.ones(num_reg_used)

theta = 1.0/COMPONENT_NUM * np.ones((COMPONENT_NUM, num_reg_used), dtype=float)

dict_data, dict_param = update.get_data_param(x, y, num_reg_used, tau, theta, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, optical_properties, r)
dict_neigh = update.get_dict_neigh(i, j, num_reg_used)
dict_env = update.get_env(tau, theta, num_reg_used, dict_neigh)

#sc = SparkContext()
#dataRDD = sc.parallelize(dict_data)
paramRDD = sc.parallelize(dict_param)
#envRDD = sc.parallelize(dict_env)
#pixelRDD = dataRDD.join(paramRDD).join(envRDD)

p = 7
xp = x[p]
yp = y[p]
tau_p = tau[p]
theta_p = theta[:, p]

reg_p, smart_p = extract.pixel(xp, yp, channel_is_used, min_equ_ref, mean_equ_ref, eof, max_usable_eof, ss, ms, r)
tau_neigh = dict_env[p][1].tau_neigh
theta_neigh = dict_env[p][1].theta_neigh
n_neigh = dict_env[p][1].n_neigh

kappa = update.get_kappa(paramRDD, i, j, num_reg_used)
_, _, resid_p = update.get_resid(tau_p, theta_p, reg_p, smart_p, optical_properties, r)
sigmasq = update.get_sigmasq(paramRDD)

tau_next,resid_next = update.get_tau(tau_p, theta_p, resid_p, tau_neigh, n_neigh, kappa, sigmasq, 0.05, reg_p, smart_p, optical_properties, r)
theta_next,resid_next = update.get_theta(tau_p, theta_p, resid_p, theta_neigh, n_neigh, sigmasq, reg_p, smart_p, optical_properties, r)