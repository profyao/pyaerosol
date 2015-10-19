import formatNum as fN
import os
import urllib

__author__ = 'syao'


def _fetch(file_url, local_file_name):
    print 'start download from {}'.format(file_url)

    if not os.path.exists(local_file_name):
        urllib.urlretrieve(file_url, local_file_name)
        print '{} is completed!'.format(local_file_name)
    else:
        print '{} already exists!'.format(local_file_name)


def mi1b2t(date, path, orbit, cam):

    from constant import HEADER_MI1B2T_FILENAME
    from constant import HEADER_MI1B2T_URL

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    dir_radiance = '../../projects/aerosol/products/MI1B2T/' + date
    if not os.path.exists(dir_radiance):
        os.makedirs(dir_radiance)

    file_name = HEADER_MI1B2T_FILENAME + path + '_O' + orbit + '_' + cam + '_F03_0024.hdf'
    local_file_name = dir_radiance + '/' + file_name
    file_url = HEADER_MI1B2T_URL + date + '/' + file_name

    _fetch(file_url, local_file_name)


def mil2asae(date, path, orbit):

    from constant import HEADER_MIL2ASAE_FILENAME, HEADER_MIL2ASAE_URL

    orbit = fN.orbit2str(orbit)
    path = fN.path2str(path)

    dir_aerosol = '../../projects/aerosol/products/MIL2ASAE/' + date

    if not os.path.exists(dir_aerosol):
        os.makedirs(dir_aerosol)

    file_name = HEADER_MIL2ASAE_FILENAME + path + '_O' + orbit + '_F12_0022.hdf'
    local_file_name = dir_aerosol + '/' + file_name
    file_url = HEADER_MIL2ASAE_URL + date + '/' + file_name

    _fetch(file_url, local_file_name)


def miancagp(path):

    from constant import HEADER_MIANCAGP_FILENAME, HEADER_MIANCAGP_URL1, HEADER_MIANCAGP_URL2

    path = fN.path2str(path)

    dir_geo = 'product/MIANCAGP'

    if not os.path.exists(dir_geo):
        os.makedirs(dir_geo)

    file_name = HEADER_MIANCAGP_FILENAME + path + '_F01_24.hdf'
    local_file_name = dir_geo + '/' + file_name
    file_url = [HEADER_MIANCAGP_URL1+file_name,HEADER_MIANCAGP_URL2+file_name]

    for i in [0, 1]:
        try:
            _fetch(file_url[i], local_file_name)
            break

        except IOError:
            print "file doesn't exist at {}".format(file_url[i])


