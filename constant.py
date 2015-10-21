import numpy as np

__author__ = 'syao'

# file HEADER
HEADER_MI1B2T_URL = 'ftp://l5eil01.larc.nasa.gov/MISR/MI1B2T.003/'
HEADER_MI1B2T_FILENAME = 'MISR_AM1_GRP_TERRAIN_GM_P'
HEADER_MIL2ASAE_URL = 'ftp://l5eil01.larc.nasa.gov/MISR/MIL2ASAE.002/'
HEADER_MIL2ASAE_FILENAME = 'MISR_AM1_AS_AEROSOL_P'
HEADER_MIL2ASAF = 'ftp://l5eil01.larc.nasa.gov/MISR/MIL2ASAF.001/'
MIANSMT_SS_FILENAME = 'MISR_AM1_SMART_TOA_RHO_ATM_SS_F02_0009.hdf'
MIANSMT_MS_FILENAME = 'MISR_AM1_SMART_TOA_RHO_ATM_MS_F02_0009.hdf'
MIANSMT_TDIFF_FILENAME = 'MISR_AM1_SMART_TDIFF_F02_0009.hdf'
MIANSMT_EDIFF_FILENAME = 'MISR_AM1_SMART_BOA_EDIFF_F02_0009.hdf'
HEADER_MIANCAGP_URL1 = 'ftp://l5eil01.larc.nasa.gov/MISR/MIANCAGP.001/1999.11.07/'
HEADER_MIANCAGP_URL2 = 'ftp://l5eil01.larc.nasa.gov/MISR/MIANCAGP.001/1999.11.08/'
HEADER_MIANCAGP_FILENAME = 'MISR_AM1_AGP_P'

# MISR camera parameters
CAM_DF = 0
CAM_CF = 1
CAM_BF = 2
CAM_AF = 3
CAM_AN = 4
CAM_AA = 5
CAM_BA = 6
CAM_CA = 7
CAM_DA = 8

CAM_NAME = {'DF', 'CF', 'BF', 'AF', 'AN', 'AA', 'BA', 'CA', 'DA'}
CAM_DIM = len(CAM_NAME)

# MISR spatial resolutions
R275 = 275
R1100 = 1100
R2200 = 2200
R4400 = 4400
R8800 = 8800
R17600 = 17600

XDIM_R1100 = 128
XDIM_R2200 = 64
XDIM_R4400 = 32
XDIM_R8800 = 16
XDIM_R17600 = 8

YDIM_R1100 = 512
YDIM_R2200 = 256
YDIM_R4400 = 128
YDIM_R8800 = 64
YDIM_R17600 = 32

# r = r4400 # default resolution for retrieval
# XDIM_r = XDIM_r4400 # default X dimension
# YDIM_r = YDIM_r4400 # default Y dimension

# Number of sub-regions in a region
# RegSize = r / r1100
# Scale factor to the 17.6-km standard region
# RegScale = r17600 / r

# XDIM is the number of rows in a block, depending on the resolution
# YDIM is the number of columns in a block,  depending on the resolution

# MISR bands parameters
BAND_BLUE = 0
BAND_GREEN = 1
BAND_RED = 2
BAND_NIR = 3
BAND_DIM = 4

NCHANNEL = BAND_DIM*CAM_DIM
BAND_NAME = {'BlueBand', 'GreenBand', 'RedBand', 'NIRBand'}
BAND_USED = np.ones(BAND_DIM)
CHANNEL_USED = map(bool, (np.kron(BAND_USED, np.ones(CAM_DIM))))

BAND_RADIANCE = ['Blue Radiance/RDQI', 'Green Radiance/RDQI', 'Red Radiance/RDQI', 'NIR Radiance/RDQI']
CONFIG_RDQI1 = 1
CONFIG_C_LAMBDA = np.array([5.67e-6, 1.04e-4, 4.89e-5, 3.94e-6])
CONFIG_SPECTRAL_CORR_MATRIX = np.array([[1.0106, -0.0057, -0.0038, -0.0011],
                                        [-0.0080, 1.0200, -0.0086, -0.0034],
                                        [-0.0060, -0.0048, 1.0145, -0.0036],
                                        [-0.0048, -0.0033, -0.0136, 1.0217]])

# sample_size = RegSize*RegSize
# CONFIG_MIN_HET_SUBR_THRESH = sample_size/4

MIN_CAM_USED = 2
CONFIG_FIRST_EIGENVALUE_FOR_EOFS = 1
CONFIG_EIGENVECTOR_VARIANCE_THRESH = 0.95

# MISR aerosol model parameters
MODEL_COMPONENTDIM = 21
MODEL_MIXTUREDIM = 74
MODEL_PRESSURE = np.array([607.95, 1013.25])

MODEL_MU0GRID = np.arange(0.2, 1.0, 0.01)
MODEL_MUGRID = np.array([0.31, 0.32, 0.33, 0.34, 0.35, 0.47, 0.48, 0.49, 0.5, 0.51, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1])
MODEL_SCATTERANGLEGRID = np.array([-1, 0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5,
                                   85, 87.5, 90, 92.5, 95, 97.5, 100, 102.5, 105, 107.5, 110, 112.5, 115, 117.5, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
                                   145, 146, 147, 148, 149, 150, 152.5, 155, 157.5, 160, 162.5, 165, 167.5, 170, 172.5, 175, 176, 177, 178, 179, 180, 181])
MODEL_OPTICALDEPTHGRID = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6])
MODEL_OPTICALDEPTHLEN = len(MODEL_OPTICALDEPTHGRID)

MODEL_AOTGRIDGAP = 0.025
MODEL_OPTICALDEPTHFINERGRID = np.arange(0, 3, MODEL_AOTGRIDGAP)
MODEL_OPTICALDEPTHFINERGRIDLEN = len(MODEL_OPTICALDEPTHFINERGRID)

COMPONENT_PARTICLE = np.array([1, 2, 3, 6, 8, 14, 19, 21]) - 1
COMPONENT_NUM = len(COMPONENT_PARTICLE)

CONFIG_ALBEDO_THRESH_LAND = 0.015
