import sys
import os
sys.path.append(os.path.abspath("baseline/"))
from constants import HIST_BINS_LIST, NUM_IMGS_PER_YEAR, NUM_TEMP_BANDS, NUM_REF_BANDS, CROP_SENTINEL, GBUCKET, RED_BAND, NIR_BAND, BLUE_BAND, L, C1, C2, G 
import numpy as np
import time
import matplotlib.pyplot as plt




def get_histogram_mode(image, num_bands):
    histogram = calc_histograms(image, HIST_BINS_LIST[:num_bands], num_bands) # (32,1,7)
    squeeze_histograms = histogram.squeeze() # (32,7)
    idx = np.argmax(squeeze_histograms, axis=0) # (7,)
    return idx

def get_histogram(image, num_bands):
    histogram = calc_histograms(image, HIST_BINS_LIST[:num_bands], num_bands) # (32,1,7)
    shape = histogram.shape
    histogram = np.reshape(histogram, [-1, shape[0], shape[2]])
    return histogram

def get_temp_histogram(image, num_bands):
    bin_seq=np.linspace(1,4999,33)
    image_temp = self.calc_histograms(image_temp,bin_seq,2)
    image_temp[np.isnan(image_temp)] = 0
    print image_temp
    return image_temp

def get_feature_mean(image, _):
    image = image.astype('float')
    image[image==0] = np.nan
    mean1 = np.nanmean(image[:,:,0])
    mean2 = np.nanmean(image[:,:,1])
    return [mean1, mean2]

def get_total_mean(image, _):
    image = image.astype('float')
    image[image==0] = np.nan
    mean = np.nanmean(image)
    return mean

def get_veg_means(image, _):
    image = image.astype('float')
    image[image == 0] = np.nan
    nir = image[:,:,NIR_BAND]
    red = image[:,:,RED_BAND]
    blue = image[:,:,BLUE_BAND]
    ndvi = (nir - red)/(nir + red)
    ndvi_denom = (nir + red) 
    ndvi_mean = np.nanmean(ndvi)
    evi_num = G*(nir - red)
    evi_denom = (nir + C1*red - C2*blue + L)
    evi_denom[evi_denom == 0] = np.nan
    evi = evi_num/evi_denom
    evi_mean = np.nanmean(evi)
    return [ndvi_mean, evi_mean]



"""
Returns the calculated histograms for a given image

Args:
    image:     3D image in tensor form [H, W, time/num_bands + band]
    num_bands: number of bands for the given image
    num_bins:  number of bins to separate the histograms
    
Returns: 
    3D tensor of pixel histograms [normalized bin values, time, band]
    
"""

def calc_histograms(image, bin_seq_list, num_bands, num_bins=32):

    if image.shape[2] % num_bands != 0:
        raise Exception('Number of bands does not match image depth.')
    num_times = image.shape[2]/num_bands
    hist = np.zeros([num_bins, int(num_times), num_bands]) # (32,1.0,7)
    for i in range(image.shape[2]):
        band = i % num_bands
        density, _ = np.histogram(image[:, :, i], bin_seq_list[band], density=False)
        total = density.sum() # normalize over only values in bins
        hist[:, int(i / num_bands), band] = density/float(total) if total > 0 else 0
    return hist

        
