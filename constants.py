import numpy as np
import os
HIST_BINS_LIST = [np.linspace(1, 2200, 33),
                  np.linspace(900, 4999, 33), 
                  np.linspace(1, 1250, 33),
                  np.linspace(150, 1875, 33),
                  np.linspace(750, 4999, 33),
                  np.linspace(300, 4999, 33),
                  np.linspace(1, 4999, 33),
                  np.linspace(13000,16500,33),
                  np.linspace(13000,15500,33)]

RED_BAND = 0 # 0-indexed, so this is band 1
NIR_BAND = 1 # Near infrared.
BLUE_BAND = 2
#EVI constants:
L = 1
C1 = 6
C2 = 7.5
G = 2.5
SW_IR_BAND = 6 # Shortwave infrared
NUM_IMGS_PER_YEAR = 45
NUM_REF_BANDS = 7
NUM_TEMP_BANDS = 2
CROP_SENTINEL = 12
LOCAL_DATA_DIR = './static_data_files/'
GBUCKET = os.path.expanduser('~/bucket2/')
VISUALIZATIONS = 'visualizations'
DATASETS = 'datasets'
BASELINE_DIR = 'baseline_results'

ALL_TRANSFER = [("MW", "Main"),("KE", "Annual"), ("NG","Wet" ), ("TZ", "Annual"), ("ZM", "Annual"), ("ET", "Meher")]
ALL_THRESHES = [38, 18]

MW_TRANSFER = [("MW", "Main"), ("ZM", "Annual")]
ZM_TRANSFER = [("ZM", "Annual"), ("MW", "Main")]
MW_ZM_THRESHES = [38, 18]

KE_TRANSFER = [("KE", "Annual"), ("TZ", "Annual")]
TZ_TRANSFER = [("TZ", "Annual"), ("KE", "Annual")]
KE_TZ_THRESHES = [76, 38]
