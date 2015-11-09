import update
import time
import extract
import preprocess
from pyspark import SparkContext, SparkConf

r = 1100
file_xls = '../../projects/aerosol/src/MISR_INFO.xls'

conf = SparkConf().setMaster("local[4]").set("spark.executor.memory", "2g")
sc = SparkContext(conf = conf)

dict_data, dict_param, dict_neigh, dict_env, optical_properties, num_reg_used, i, j, dates, paths, orbits, blocks = preprocess.merge_dict(file_xls, r)

dataRDD = sc.parallelize(dict_data)
dataRDD.cache()
paramRDD = sc.parallelize(dict_param)
envRDD = sc.parallelize(dict_env)
delta = 0.05
kappa = update.get_kappa(paramRDD, i, j, num_reg_used, dates, paths, orbits, blocks)
sigmasq = update.get_sigmasq(paramRDD, dates, paths, orbits, blocks)

t0 = time.time()

for iter in xrange(1):
    pixelRDD = paramRDD.join(envRDD).join(dataRDD)
    paramRDD, dict_env = update.get_param_env(dates, pixelRDD, kappa, sigmasq, delta, num_reg_used, dict_neigh, optical_properties, r)
    envRDD = sc.parallelize(dict_env)
    kappa = update.get_kappa(paramRDD, i, j, num_reg_used, dates, paths, orbits, blocks)
    sigmasq = update.get_sigmasq(paramRDD, dates, paths, orbits, blocks)
    
    print 'Iteration: ' + str(iter) + 'done!'

print time.time() - t0

extract.save_tau_theta(dates, paths, orbits, blocks, r, 'CD-random', paramRDD)
