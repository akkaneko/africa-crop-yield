import util_processing as util
import features
import numpy as np

#util.save_all_features(features.get_histogram, 'MODIS', verbose = True, append_temp = True)
util.save_all_features(features.get_feature_mean, 'NDVI',  verbose = True, append_temp = False)
util.save_all_features(features.get_feature_mean, 'TEMP',  verbose = True, append_temp = False)

