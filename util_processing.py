import numpy as np
import pandas as pd
import time
import gdal
import ee
import json
from datetime import date 
from scipy.stats import mode
from sklearn.metrics import r2_score,mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import operator

ee.Initialize()
gdal.SetCacheMax(2**32)




######## All code beyond this point are used to save features ###

## A dict mapping img collection codes to the necessary attributes
imgcoll_dict = {
    'MODIS': ('MODIS/006/MOD09A1', '', 7),
    'NDVI' : ('MODIS/006/MOD13Q1', 'VEG_', 2),
    'TEMP' : ('MODIS/006/MOD11A2', 'TEMP_', 2),
    'AQUA' : ('MODIS/006/MYD11A2', 'AQUA_TEMP_', 2),
    'NEW_NDVI': ('MODIS/006/MOD09A1', '', 7)
}

## dict mapping countries to target admin and season
country_dict = {
    'ET': ('admin2', 'Meher', 8),
    'ZM': ('admin2', 'Annual', 3), 
    'TZ': ('admin1', 'Annual', 4),
    'KE': ('admin2', 'Annual', 5),
    'NG': ('admin1', 'Wet', 8),
    'MW': ('admin2', 'Main', 3),
    'SU': ('admin2', 'Annual', 7)
            }

## dict mapping countries and admin to the span 
## of years where data is available

years_dict = {
    "NG_admin1" : 17,
    "ET_admin1" : 16,
    "ET_admin2" : 16,
    "KE_admin2" : 16,
    "MW_admin2" : 17,
    "ZM_admin2" : 17,
    "TZ_admin1" : 15,
    "SU_admin2" : 5
}

fusion_dict = {
    'ET_admin2': ("users/tkenned97/ET_Admin2_simp_2k",'2001-1-1','2016','admin2'),
    'TZ_admin1': ("users/tkenned97/TZ_Admin1_simp_2k", '2000-2-18','2015','admin1'),
    'ZM_admin2': ("users/tkenned97/ZM_Admin2_simp_2k",'2000-2-18','2017','admin2'),
    'NG_admin1': ("users/tkenned97/NG_Admin1_simp_2k",'2000-2-18','2017','admin1'),
    'MW_admin2': ("users/tkenned97/MW_Admin2_simp_2k",'2000-2-18','2017','admin2'),
    'KE_admin2': ("users/tkenned97/KE_Admin2_simp_2k",'2000-2-18','2016','admin2'),
    'SU_admin2': ('users/akkstuff/sudan_admin2', '2013-1-1', '2018', 'ADM2_EN')
}


"""
Once all .tif data is downloaded, this function will download
all the features among all countries (specified in data_dict)
between the data's given timespan. It will download the 
saved features into the given bucket path, in the file "saved features"
Args:
    feature_function:   func, chosen feature function, taken from features.py
    coll_code:          str,  collection code, 'MODIS'/'TEMP'/'NDVI'
    append_temp:        bool, if downloading MODIS features, append the TEMP data.
    verbose:            bool, activate printstubs
    path_to_bucket:     str,  path to the gcloud bucket
Returns:
    None. Saves feature into file. The saved features will be in 
    a dict of the form dict[county][year] to get the desired features. 
    
"""
def save_all_features(feature_function, coll_code, append_temp = False, verbose = False, path_to_bucket='/mnt/mounted_bucket'):

    df = pd.read_csv('fulldata.csv')
    countries =['ET', 'ZM', 'TZ', 'KE', 'NG', 'MW']


    for country in countries:
        print '##############################'
        print 'saving data for ' + str(country)
        admin, season, _ = country_dict[country]
        save_features(country, admin, season, feature_function, coll_code = coll_code, append_temp = append_temp, verbose = verbose, path_to_bucket = path_to_bucket)
        

        
"""
Given a country, admin, and season, apply and save all target
features. It will download the saved features into the given bucket path, 
in the file "saved features"
Args:
    country:            str, country code
    admin:              str, str, admin level
    function:           func, chosen feature function, taken from features.py
    coll_code:          str,  collection code, 'MODIS'/'TEMP'/'NDVI'
    append_temp:        bool, if downloading MODIS features, append the TEMP data.
    verbose:            bool, activate printstubs
    path_to_bucket:     str,  path to the gcloud bucket
Returns:
    List of array-like images. 
"""       
def save_features(country, admin, season, function, coll_code = 'MODIS', append_temp = False, verbose = False, path_to_bucket = '/mnt/mounted_bucket'):
    country_admin = "%s_%s" % (country, admin)
    img_code = imgcoll_dict[coll_code][1]
    

    path_to_bucket = '/mnt/mounted_bucket'
    if country == 'SU':
        df = pd.read_csv('sudan_data.csv')
    else:
        df = pd.read_csv('fulldata.csv')

    df = df[df['country'] == country]
    counties = df[admin].unique()
    
    final_dict = {}
        
    for i, c in enumerate(counties):

        try:
            curpath = '%s/tif_data_final/NEW_MASKED_%s%s/%s.tif'%(path_to_bucket, img_code, country_admin, c)
            print "Creating features for %s" % (c)


            feature_dict = save_helper(country, admin, c, season, function, curpath, coll_code, '/mnt/mounted_bucket', append_temp, verbose)

            final_dict[c] = feature_dict

        except Exception as e:
            print "Failed to get data for %s" % c
            print e
    

    save_path = '%s/saved_features/%s/%s%s_%s' % (path_to_bucket, country, img_code, country, season)
    with open(save_path, "w") as write_file:
        np.save(save_path, final_dict)
        
        
"""
Helper function. Returns a dictionary mapping years to
features given the target data. 
Args: 
    country:            str, country code
    admin_level:        str, admin level
    county_name:        str, target county name
    season:             str, target season
    function:           func, chosen feature function, taken from features.py
    MODIS_path:         str, path to the county tif data
    path_to_bucket:     str, path to the gcloud bucket
    append_temp:        bool, if downloading MODIS features, append the TEMP data.
    verbose:            bool, activate printstubs
    
Returns: 
    dictionary mapping from year to features.
"""

def save_helper(country, admin_level, county_name, season, function, MODIS_path, coll_code = 'MODIS', path_to_bucket = '/mnt/mounted_bucket',  append_temp = False, verbose = False):
    
    use_mask = True
    
    if verbose:
        print 'Getting .tif data at ' + MODIS_path 
    
    _,  coll_id, num_bands= imgcoll_dict[coll_code]
    country_admin = "%s_%s" % (country, admin_level)
    start = time.time()
    
    
    try: 
   # print MODIS_path
        MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
    except AttributeError:
        if verbose: print 'File too large. Failed.'
        MODIS_temp_path = '%s/tif_data_final/NEW_MASKED_%s%s/new_masked_%s.npy' % (path_to_bucket, coll_id, country_admin, county_name)
        MODIS_img = np.load(MODIS_temp_path)

    years_in_df = get_years_df(country, admin_level, season, county_name)
    num_years = years_dict[country_admin]
    if coll_code is 'AQUA':
        num_years -= 1
    
    num_images_year = 46
    if coll_code == 'NDVI':
        num_images_year = 23
        
    
    MODIS_img_list = divide_image(MODIS_img, 0, num_bands*num_images_year, num_years)
    MODIS_img = 0
    additional_bands = 0

    if append_temp == True:
        MODIS_temp_path = '%s/tif_data_final/NEW_MASKED_%s%s/%s.tif' % (path_to_bucket, 'TEMP_', country_admin, county_name)


        MODIS_temp_img = np.transpose(np.array(gdal.Open(MODIS_temp_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        MODIS_temp_img_list=divide_image(MODIS_temp_img, 0, 2*46, num_years)
        additional_bands = 2
        MODIS_img_list = merge_image(MODIS_img_list,MODIS_temp_img_list)
        MODIS_temp_img_list = None
        MODIS_temp_img = None
        
    if use_mask == True:
        MODIS_mask_path = '%s/tif_data_final/%s/%s/%s.tif' % (path_to_bucket, 'crop_masks', country_admin, county_name)
        MODIS_mask_img = np.transpose(np.array(gdal.Open(MODIS_mask_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        MODIS_mask_img[MODIS_mask_img != 12] = 0
        MODIS_mask_img[MODIS_mask_img == 12] = 1
        MODIS_mask_img_list = divide_image(MODIS_mask_img, 0, 1, num_years)
        MODIS_img_list = mask_image(MODIS_img_list, MODIS_mask_img_list)
        
    MODIS_img_list = list_from_year_blocks(MODIS_img_list, num_bands + additional_bands)
    
    imgcoll = get_imgcoll(country_admin, path_to_bucket, coll_code)
    dates = get_years(imgcoll)
    assert len(dates) == len(MODIS_img_list), 'Image collection dates and TIF data dates are not the same length'

    histogram_dict = {}
    #Loop over data and check dates, add to dataset if matching season and year
    
    new_list = []
    
    for i in years_in_df:
        new_list.append(i)
        if i+1 not in new_list:
            new_list.append(i+1)
        if i-1 not in new_list:
            new_list.append(i-1)
            
    years_in_df = new_list
    
    for i in range(len(MODIS_img_list)):
        cur_year = dates[i].year
        cur_month = dates[i].month
        curimage = MODIS_img_list[i]
        

        if cur_year in years_in_df:
            if (cur_year, cur_month) not in histogram_dict:
                histogram_dict[(cur_year, cur_month)] = [function(curimage, num_bands + additional_bands)]
            else:
                histogram_dict[(cur_year, cur_month)].append(function(curimage, num_bands + additional_bands))            
                
    if verbose:
        print "took " + str( time.time() - start )
                            
    return histogram_dict




def get_metrics(pred, test_labels, country, RMSE = True, verbose = True):
    from sklearn.metrics import r2_score

    if RMSE:
        mse = np.sqrt(np.mean(np.square(test_labels - pred)))
    else:
        mse = np.mean(np.square(test_labels - pred))
    coefficient_of_dermination = r2_score(test_labels, pred)
    r2,_ = stats.pearsonr(test_labels, pred)

    if verbose:
        print("RMSE: %.2f" % mse) 
        print("R2: %.2f" % coefficient_of_dermination)
        print("r: %.2f" % r2) 
        fig, ax = plt.subplots()
        ax.scatter(test_labels, pred)
        ax.plot([0.5, 4.0], [0.5, 4.0], 'k--', lw=3)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        #plt.title(country)
        plt.ylim(0, 4)
        plt.show()
    
    return mse, coefficient_of_dermination, r2


def mask_image(MODIS_list,MODIS_mask_img_list):
    MODIS_list_masked = []
    for i in range(0, len(MODIS_list)):
        mask = np.tile(MODIS_mask_img_list[i],(1,1,MODIS_list[i].shape[2]))
        masked_img = MODIS_list[i]*mask
        MODIS_list_masked.append(masked_img)
    return MODIS_list_masked



def get_modis_feature(country, before_peak = 0, after_peak = 0, splits = None, modis_path = None):

    admin, season, peak_month = country_dict[country]
    path_to_bucket = '/mnt/mounted_bucket'
    if modis_path == None: 
        modis_path =  '%s/saved_features/%s/%s_%s.npy' % (path_to_bucket, country, country, season)
    feature_dict = np.load(modis_path).item()
    
            
    if country == 'TZ':
        min_value = 23
    else:
        min_value = 22
        
    df = pd.read_csv('fulldata.csv')
    df = df[df['country'] == country]
    years = sorted(df['year'].unique())
    
    max_years = years[-6:]
    
    if splits == None:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s_%s_split.npy' % (country, country, season)).item()
    else:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s' % (country, splits)).item()
    location_dict = np.load('/mnt/mounted_bucket/saved_features/location_data.npy').item()
    
    ft_data = []
    train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, train_years, val_years, test_years = [],[],[],[],[],[],[],[],[]
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    train_YI, val_YI, test_YI = [], [], []
    
    year_data, locx_data, locy_data, county_data, YI_data = [], [], [], [], []
    train_meta, val_meta, test_meta = [], [], []
    
    metadata = []
    for c in feature_dict:    
        years_df = get_years_df(country, admin, season, c)
        x,y = location_dict[country][c]
        
        for year in years_df:
            
            ft = []
            target_month = []

            for month in range(peak_month - before_peak, peak_month + after_peak + 1):
                #print month
                if month < 1:
                    target_month.append((year-1, 12 + month))
                elif month > 12:
                    target_month.append((year + 1, month - 12))
                else:
                    target_month.append((year, month))
                    
            for pair in target_month:
                try:
                    cur_feature = np.swapaxes(np.array(feature_dict[c][pair]), 0, 1)
                except:
                    continue
                ft.append(cur_feature)
                
                
            if len(ft) == 0:
                continue
                
            ft = np.concatenate(ft, axis = 1)



            if year in max_years:
                YI = max_years.index(year)/10.0
            else:
                YI = 0 
                
            year_data.append(year)
            locx_data.append(x)
            locy_data.append(y)
            county_data.append(c)
            ft_data.append(ft)
            YI_data.append(YI)
            
    time_lengths = [x.shape[1] for x in ft_data]
    #min_value = min(time_lengths)
    min_value = 22
    ft_data = clip_arrays(ft_data, min_value)
            
    ft_data, other_data  = filter_array(ft_data, [year_data, locx_data, locy_data, county_data, YI_data], min_value) 
    year_data, locx_data, locy_data, county_data, YI_data = other_data

    year_norm = normalize(np.array(year_data))
    locx_data = normalize(np.array(locx_data))
    locy_data = normalize(np.array(locy_data))
    
    
    for i in range(len(year_data)):
        year = year_data[i]
        county = county_data[i]
        for s in ['train', 'val', 'test']:
            if year in splits[s][county].keys():
                if s == 'train':
                    train_ft.append(ft_data[i])
                    train_lb.append(splits[s][county][year])
                    train_YI.append(YI_data[i])
                    train_years.append(year_norm[i])
                    train_x.append(locx_data[i])
                    train_y.append(locy_data[i])
                    train_meta.append((county, year))
                    
                    break
                if s == 'val':
                    val_ft.append(ft_data[i])
                    val_lb.append(splits[s][county][year])
                    val_YI.append(YI_data[i])
                    val_years.append(year_norm[i])
                    val_x.append(locx_data[i])
                    val_y.append(locy_data[i]) 
                    val_meta.append((county, year))
                    break
                if s == 'test':
                    test_ft.append(ft_data[i])
                    test_lb.append(splits[s][county][year])
                    test_YI.append(YI_data[i])
                    test_years.append(year_norm[i])
                    test_x.append(locx_data[i])
                    test_y.append(locy_data[i])
                    test_meta.append((county, year))

                    break
                    
    train_ft = np.concatenate(train_ft)
    val_ft = np.concatenate(val_ft)
    test_ft = np.concatenate(test_ft)  
    
    train_years = np.reshape(train_years, (-1, 1))
    val_years = np.reshape(val_years, (-1, 1))
    test_years = np.reshape(test_years, (-1, 1))
    
    train_x = np.reshape(train_x, (-1, 1))
    val_x = np.reshape(val_x, (-1, 1))
    test_x = np.reshape(test_x, (-1, 1))

    train_y = np.reshape(train_y, (-1, 1))
    val_y = np.reshape(val_y, (-1, 1))
    test_y = np.reshape(test_y, (-1, 1))
    
    

    #print train_x.shape
    
    train_other = np.concatenate([train_years, train_x, train_y], axis = 1)
    val_other = np.concatenate([val_years, val_x, val_y], axis = 1)
    test_other = np.concatenate([test_years, test_x, test_y], axis = 1)
    
    return (train_ft, np.array(train_lb), train_other, train_meta), (val_ft, np.array(val_lb), val_other, val_meta), (test_ft, np.array(test_lb), test_other, test_meta)

def get_new_baseline_feature(country, before_peak = 0, after_peak = 0, splits = None, normalized = True):

    admin, season, peak_month = country_dict[country]
    path_to_bucket = '/mnt/mounted_bucket'
    modis_path =  '%s/saved_features/%s/FINAL_%s_%s.npy' % (path_to_bucket, country, country, season)
    feature_dict = np.load(modis_path).item()
    
    if before_peak == 0:
        before_peak = season_span
    if after_peak == 0:
        after_peak = season_span
           
        
    df = pd.read_csv('fulldata.csv')
    df = df[df['country'] == country]
    years = sorted(df['year'].unique())
        
    if splits == None:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s_%s_split.npy' % (country, country, season)).item()
    else:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s' % (country, splits)).item()
    location_dict = np.load('/mnt/mounted_bucket/saved_features/location_data.npy').item()
    
    ft_data = []
    train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, train_years, val_years, test_years = [],[],[],[],[],[],[],[],[]
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []    
    year_data, locx_data, locy_data, county_data = [], [], [], []
    train_meta, val_meta, test_meta = [], [], []
    metadata = []
    for c in feature_dict:    
        years_df = get_years_df(country, admin, season, c)

        x,y = location_dict[country][c]
        
        for year in years_df:
            
            target_month = []

            for month in range(peak_month - before_peak, peak_month+1+after_peak):

                #print month
                if month < 1:
                    target_month.append((year-1, 12 + month))
                elif month > 12:
                    target_month.append((year + 1, month - 12))
                else:
                    target_month.append((year, month))
                                    
            year_data.append(year)
            locx_data.append(x)
            locy_data.append(y)
            county_data.append(c)
            
    other_data = [year_data, locx_data, locy_data, county_data]
    year_norm, locx_data, locy_data, county_data = other_data

    if normalized == True:
        year_norm = normalize(np.array(year_data))
        locx_data = normalize(np.array(locx_data))
        locy_data = normalize(np.array(locy_data))
    
    for i in range(len(year_data)):
        year = year_data[i]
        county = county_data[i]
        for s in ['train', 'val', 'test']:
            if year in splits[s][county].keys():
                if s == 'train':
                    train_lb.append(splits[s][county][year])
                    train_years.append(year_norm[i])
                    train_x.append(locx_data[i])
                    train_y.append(locy_data[i])
                    train_meta.append((county, year))
                    break
                if s == 'val':
                    val_lb.append(splits[s][county][year])
                    val_years.append(year_norm[i])
                    val_x.append(locx_data[i])
                    val_y.append(locy_data[i]) 
                    val_meta.append((county, year))
                    break
                if s == 'test':
                    test_lb.append(splits[s][county][year])
                    test_years.append(year_norm[i])
                    test_x.append(locx_data[i])
                    test_y.append(locy_data[i])
                    test_meta.append((county, year))
                    break
                    
    train_years = np.reshape(train_years, (-1, 1))
    val_years = np.reshape(val_years, (-1, 1))
    test_years = np.reshape(test_years, (-1, 1))
    
    train_x = np.reshape(train_x, (-1, 1))
    val_x = np.reshape(val_x, (-1, 1))
    test_x = np.reshape(test_x, (-1, 1))

    train_y = np.reshape(train_y, (-1, 1))
    val_y = np.reshape(val_y, (-1, 1))
    test_y = np.reshape(test_y, (-1, 1))
    
    
    train_other = np.concatenate([train_years, train_x, train_y], axis = 1)
    val_other = np.concatenate([val_years, val_x, val_y], axis = 1)
    test_other = np.concatenate([test_years, test_x, test_y], axis = 1)
    

    return ( train_other, np.array(train_lb), train_meta), (val_other, np.array(val_lb), val_meta), (test_other, np.array(test_lb), test_meta)



def get_error_feature(country, season_span = 0, before_peak = 0, after_peak = 0, splits = None):
    admin, season, peak_month = country_dict[country]
    path_to_bucket = '/mnt/mounted_bucket'
    modis_path =  '%s/saved_features/%s/FINAL_%s_%s.npy' % (path_to_bucket, country, country, season)
    feature_dict = np.load(modis_path).item()
    
    #assert not(season_span==0 and before_peak == 0 and after_peak == 0)
    
    if before_peak == 0:
        before_peak = season_span
    if after_peak == 0:
        after_peak = season_span
    
    if country == 'TZ':
        min_value = 23
    else:
        min_value = 22
        
    df = pd.read_csv('fulldata.csv')
    df = df[df['country'] == country]
    years = sorted(df['year'].unique())
    
    max_years = years[-6:]
    
    if splits == None:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s_%s_split.npy' % (country, country, season)).item()
    else:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s' % (country, splits)).item()
    location_dict = np.load('/mnt/mounted_bucket/saved_features/location_data.npy').item()
    
    ft_data = []
    train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, train_years, val_years, test_years = [],[],[],[],[],[],[],[],[]
    train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
    
    year_data, locx_data, locy_data, county_data,  [], [], [], []
    train_meta, val_meta, test_meta = [], [], []
    
    for c in feature_dict:    
        years_df = get_years_df(country, admin, season, c)
        
        x,y = location_dict[country][c]
        
        for year in years_df:
  
            ft = []
            target_month = []

            for month in range(peak_month - before_peak, peak_month+1+after_peak):
                if month < 1:
                    target_month.append((year-1, 12 + month))
                elif month > 12:
                    target_month.append((year + 1, month - 12))
                else:
                    target_month.append((year, month))
                    
            for pair in target_month:
                try:
                    cur_feature = np.swapaxes(np.array(feature_dict[c][pair]), 0, 1)
                except:
                    continue

                ft.append(cur_feature)
            if len(ft) == 0:
                continue
                
            ft = np.concatenate(ft, axis = 1)

                
            year_data.append(year)
            locx_data.append(x)
            locy_data.append(y)
            county_data.append(c)
            ft_data.append(ft)
            
    time_lengths = [x.shape[1] for x in ft_data]
    min_value = stats.mode(time_lengths)[0][0]
    ft_data = clip_arrays(ft_data, min_value)
            
    ft_data, other_data  = filter_array(ft_data, [year_data, locx_data, locy_data, county_data], min_value) 
    year_data, locx_data, locy_data, county_data = other_data

    year_norm = normalize(np.array(year_data))
    locx_data = normalize(np.array(locx_data))
    locy_data = normalize(np.array(locy_data))
    
    mean_histograms = {}
    
    for i in range(len(year_data)):
        year = year_data[i]
        county = county_data[i]
        if county not in mean_histograms:
            mean_histograms[county] = []
        for s in ['train', 'val', 'test']:
            if year in splits[s][county].keys():
                if s == 'train':
                
                    mean_histograms[county].append(ft_data[i])
                    train_ft.append(ft_data[i])
                    train_lb.append(get_error(country, county, year))
                    train_years.append(year_norm[i])
                    train_x.append(locx_data[i])
                    train_y.append(locy_data[i])
                    train_meta.append((county, year))
                    break
                if s == 'val':
                    val_ft.append(ft_data[i])
                    val_lb.append(get_error(country, county, year))
                    val_years.append(year_norm[i])
                    val_x.append(locx_data[i])
                    val_y.append(locy_data[i]) 
                    val_meta.append((county, year))
                    break
                if s == 'test':
                    test_ft.append(ft_data[i])
                    test_lb.append(get_error(country, county, year))
                    test_years.append(year_norm[i])
                    test_x.append(locx_data[i])
                    test_y.append(locy_data[i])
                    test_meta.append((county, year))
                    break
                    
    for county in mean_histograms:
        allhists = np.squeeze(np.concatenate([mean_histograms[county]]))
        mean_histograms[county] = np.mean(allhists, axis = 0)
        
    train_ft = np.concatenate(train_ft)
    val_ft = np.concatenate(val_ft)
    test_ft = np.concatenate(test_ft)  
    
        
    return (train_ft, np.array(train_lb), train_meta), (val_ft, np.array(val_lb), val_meta), (test_ft, np.array(test_lb), test_meta), mean_histograms



        
def get_error(country, county, year):
    df = pd.read_csv('%s_error_data.csv' % country)
    df = df[df['county'] == county]
    df = df[df['year'] == year]
    return float(df['error'])



def clip_arrays(array, cutoff):
    for i in range(len(array)):
        if array[i].shape[1] < cutoff:
            continue
        array[i] = array[i][:, -cutoff:]
        
    return array

def filter_array(ft_array, other_arrays, mode):
    
    for i in range(len(ft_array)-1, -1, -1):
        if ft_array[i].shape[1] != mode: 
            del ft_array[i]
            for array in other_arrays:
                del array[i]

    return ft_array, other_arrays
            


def normalize_modis(data):
    concat_data = np.concatenate(data)
    data_modis = normalize(concat_data[:,:,:,0:7])
    data_temp = normalize(concat_data[:,:,:,7:9])
    data_ft = np.concatenate([data_modis, data_temp], axis = 3)
    return data_ft



def normalize(dataset):
    mu = np.nanmean(dataset,axis=0)
    sigma = np.nanstd(dataset,axis=0)
    normalized = (dataset - mu)/sigma
    bool_ind = np.isnan(normalized)
    normalized[bool_ind] = 0
    return normalized


    
def get_anomaly_regression_features(country, season_span, simplified = False, normalization = True, splits = None, get_metadata = False, subtract_num = 0):

    admin, season, peak_month = country_dict[country]
    path_to_bucket = '/mnt/mounted_bucket'
    ndvi_path = '%s/saved_features/%s/%s%s_%s_new_manual_masked.npy' % (path_to_bucket, country, 'VEG_', country, season)
    temp_path = '%s/saved_features/%s/%s%s_%s.npy' % (path_to_bucket, country, 'TEMP_', country, season)
    ndvi_dict = np.load(ndvi_path).item()
    temp_dict = np.load(temp_path).item()
    if splits == None:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s_%s_split.npy' % (country, country, season)).item()
    else:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s' % (country, splits)).item()
    
    split_dict = {
        
        'train':[],
        'val'  :[],
        'test' :[]
    }
   
    df = pd.read_csv('fulldata.csv')
    df = df[df['country'] == country]
    years = sorted(df['year'].unique())
    
    max_years = years[-6:]
    
    location_dict = np.load('/mnt/mounted_bucket/saved_features/location_data.npy').item()
    county_list = []
    out = []
    #train_ft, train_lb, val_ft, val_lb, test_ft, test_lb = [],[],[],[],[],[]
    mean_ndvi_dict = {}
    mean_temp_dict = {}
    for c in ndvi_dict:
        if c not in mean_ndvi_dict:
            mean_ndvi_dict[c] = []
        if c not in mean_temp_dict:
            mean_temp_dict[c] = []
            
        x,y = location_dict[country][c]
        
        years_df = get_years_df(country, admin, season, c)
        for year in years_df:           
            ftndvi, fttemp, ftyear, ftYI = [], [], [], []
            target_month = []
#             print("range: {}").format(range(peak_month-season_span, peak_month+season_span-subtract_num))
            for month in range(peak_month-season_span, peak_month+season_span-subtract_num):
                #print month
                if month < 1:
                    target_month.append((year-1, 12 + month))
                elif month > 12:
                    target_month.append((year + 1, month - 12))
                else:
                    target_month.append((year, month))

            for pair in target_month:
                try:
                    month_feature = ndvi_dict[c][pair]
                    temp_feature = temp_dict[c][pair]
                except:
                    #print pair
                    continue
                for i in range(len(month_feature)):
                    if simplified:
                        ftndvi.append(month_feature[i][0])
                        fttemp.append(temp_feature[i][0])
                    else:
                        for val in month_feature[i]:
                            ftndvi.append(val)
                        for val in temp_feature[i]:
                            fttemp.append(val)
                            
                            
                        
            ftyear.append(year)
            if year in max_years:
                YI = max_years.index(year)/10.0
            else:
                YI = 0
            
            ftYI.append(YI)
            
            try:
                for s in ['train', 'val', 'test']:
                    if year in splits[s][c].keys():
                        if s == 'train':                          
                            county_list.append(c)
                        if simplified:
                            if s == 'train':
                                mean_ndvi_dict[c].append([max(ftndvi)])
                                mean_temp_dict[c].append([np.nanmean(fttemp)])
                            cur = ([max(ftndvi)], [np.nanmean(fttemp)], get_error(country, c, year),(c,year))
                        else:
                            if s == 'train':
                                mean_ndvi_dict[c].append(ftndvi)
                                mean_temp_dict[c].append(fttemp)
                            cur = (ftndvi, fttemp, get_error(country, c, year), (c,year))                       
                        split_dict[s].append(cur)
                        if s == 'test':
                            out.append((c, year))
            except:
                continue
                
    out_dict = {}
    for county in mean_ndvi_dict:       
        mean_ndvi_dict[county] = [np.nanmean(col) for col in zip(*mean_ndvi_dict[county])]
        mean_temp_dict[county] = [np.nanmean(col) for col in zip(*mean_temp_dict[county])]
    
    
    
    ndvi_thresh = 100000
    temp_thresh = 100000
    for i in split_dict['train']:
        ndvi_len = len(mean_ndvi_dict[i[3][0]])
        if ndvi_len < ndvi_thresh:
            ndvi_thresh = ndvi_len
        temp_len = len(mean_temp_dict[i[3][0]])
        if temp_len < temp_thresh:
            temp_thresh = temp_len

    
    for s in ['train', 'val', 'test']:
        
        ndvi_ft = [np.subtract(i[0][-ndvi_thresh:], mean_ndvi_dict[i[3][0]][-ndvi_thresh:])  for i in split_dict[s]]
        temp_ft = [np.subtract(i[1][-temp_thresh:], mean_temp_dict[i[3][0]][-temp_thresh:]) for i in split_dict[s]]
        lb   = [i[2] for i in split_dict[s]]
        
#         if s =='train':
#             ndvi_thresh = min([len(i) for i in ndvi_ft])
#             temp_thresh = min([len(i) for i in temp_ft])
        
                    
#         ndvi_ft = np.stack([i[-ndvi_thresh:] for i in ndvi_ft])
#         temp_ft = np.stack([i[-temp_thresh:] for i in temp_ft])
              
        if normalization:
            ndvi_norm = normalize(ndvi_ft)
            temp_norm = normalize(temp_ft)
            feature_list = [ndvi_norm,temp_norm]
            out_dict[s] = (np.concatenate(feature_list, axis = 1), lb)           
        else:
            out_dict[s] = (np.concatenate([ndvi_ft], axis = 1), lb)

    
        
    
    
    train_ft, train_lb = out_dict['train']
    val_ft, val_lb = out_dict['val']
    test_ft, test_lb = out_dict['test']
    if get_metadata:
        metadata = [i[3] for i in split_dict['test']]    
        return train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, metadata
    else:
        return train_ft, train_lb, val_ft, val_lb, test_ft, test_lb
    

def get_baseline_features(country, season_span, simplified = False, normalization = True, splits = None, get_metadata = False, subtract_num = 0):

    admin, season, peak_month = country_dict[country]
    path_to_bucket = '/mnt/mounted_bucket'
    ndvi_path = '%s/saved_features/%s/%s%s_%s_new_manual.npy' % (path_to_bucket, country, 'VEG_', country, season)
    temp_path = '%s/saved_features/%s/%s%s_%s.npy' % (path_to_bucket, country, 'TEMP_', country, season)
    ndvi_dict = np.load(ndvi_path).item()
    temp_dict = np.load(temp_path).item()
    if splits == None:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s_%s_split.npy' % (country, country, season)).item()
    else:
        splits = np.load('/mnt/mounted_bucket/saved_features/%s/%s' % (country, splits)).item()
    
    split_dict = {
        
        'train':[],
        'val'  :[],
        'test' :[]
    }
   
    df = pd.read_csv('fulldata.csv')
    df = df[df['country'] == country]
    years = sorted(df['year'].unique())
    
    max_years = years[-6:]
    
    location_dict = np.load('/mnt/mounted_bucket/saved_features/location_data.npy').item()
    county_list = []
    out = []

    #train_ft, train_lb, val_ft, val_lb, test_ft, test_lb = [],[],[],[],[],[]
    for c in ndvi_dict:
        
        x,y = location_dict[country][c]
        
        years_df = get_years_df(country, admin, season, c)
        for year in years_df:         
            ftndvi, fttemp, ftyear, ftYI = [], [], [], []
            target_month = []
#             print("range: {}").format(range(peak_month-season_span, peak_month+season_span-subtract_num))
            for month in range(peak_month-season_span, peak_month+season_span-subtract_num):
                #print month
                if month < 1:
                    target_month.append((year-1, 12 + month))
                elif month > 12:
                    target_month.append((year + 1, month - 12))
                else:
                    target_month.append((year, month))

            for pair in target_month:
                try:
                    month_feature = ndvi_dict[c][pair]
                    temp_feature = temp_dict[c][pair]
                except:
                    #print pair
                    continue
                for i in range(len(month_feature)):
                    if simplified:
                        ftndvi.append(month_feature[i][0])
                        fttemp.append(temp_feature[i][0])
                    else:
                        for val in month_feature[i]:
                            ftndvi.append(val)
                        for val in temp_feature[i]:
                            fttemp.append(val)
                            
                            
                        
            ftyear.append(year)
            if year in max_years:
                YI = max_years.index(year)/10.0
            else:
                YI = 0
            
            ftYI.append(YI)
            
            try:
                for s in ['train', 'val', 'test']:
                    if year in splits[s][c].keys():
                        if s == 'train':
                            county_list.append(c)
                        if simplified:
                            cur = ([max(ftndvi)], [np.nanmean(fttemp)], ftyear, ftYI, [x], [y], splits[s][c][year], (c,year))
                        else:
                            cur = (ftndvi, fttemp, ftyear, ftYI, [x], [y], splits[s][c][year], (c,year))                       
                        split_dict[s].append(cur)
                        
                        if s == 'test':                        
                            out.append((c, year))
                            
            except:
                continue
                
    out_dict = {}
    for s in ['train', 'val', 'test']:
        ndvi_ft = [i[0] for i in split_dict[s]]
        temp_ft = [i[1] for i in split_dict[s]]
        year_ft = [i[2] for i in split_dict[s]]
        YI_ft = [i[3] for i in split_dict[s]]
        ftx =     [i[4] for i in split_dict[s]]
        fty =     [i[5] for i in split_dict[s]]
        lb   = [i[6] for i in split_dict[s]]
        
        if s =='train':
            ndvi_thresh = min([len(i) for i in ndvi_ft])
            temp_thresh = min([len(i) for i in temp_ft])
        
                    
        ndvi_ft = np.stack([i[-ndvi_thresh:] for i in ndvi_ft])
        temp_ft = np.stack([i[-temp_thresh:] for i in temp_ft])
        year_ft = np.stack(year_ft)
        ftx = np.stack(ftx)
        fty = np.stack(fty)
        #print ndvi_ft.shape, temp_ft.shape, year_ft.shape, ftx.shape, fty.shape
      #  year_ft = np.stack([i[:-year_thresh] for i in year_ft])
        

        if normalization:
            ndvi_norm = normalize(ndvi_ft)
            temp_norm = normalize(temp_ft)
            year_norm = normalize(year_ft)
            x_norm = normalize(ftx)
            y_norm = normalize(fty)
#             feature_list = [ndvi_norm,temp_norm,year_norm, YI_ft, x_norm, y_norm]
            feature_list = [ndvi_norm,temp_norm]

#             feature_list = [ x_norm, y_norm]
            out_dict[s] = (np.concatenate(feature_list, axis = 1), lb)           
        else:
            out_dict[s] = (np.concatenate([ndvi_ft, temp_ft, year_ft], axis = 1), lb)
#             out_dict[s] = (ndvi_ft, year_ft, lb)

    
        
    

    train_ft, train_lb = out_dict['train']
    val_ft, val_lb = out_dict['val']
    test_ft, test_lb = out_dict['test']


    if get_metadata:
        metadata = [i[7] for i in split_dict['test']]    
        return train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, metadata
    else:
        return train_ft, train_lb, val_ft, val_lb, test_ft, test_lb
    

# def admin1_testing(country,admin2_predictions,metadata):
    
    
def combine_tifs(tif_path_list, save_name):
    print("combining:")
    print(tif_path_list)
    tif_list = []
    tif_y_splits = []
    tif_x_splits = []
    for tif_path in sorted(tif_path_list):
        dash_index = tif_path.index('-')
        dot_index = tif_path.index('.')
        y_int_index = tif_path.index('0')
        y_split_int = int(tif_path[y_int_index:dash_index])
        x_split_int = int(tif_path[dash_index + 1:dot_index])
        tif_y_splits.append(y_split_int)
        tif_x_splits.append(x_split_int)
        tif = np.transpose(np.array(gdal.Open(tif_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
        print("Opened " + tif_path)
        tif_list.append(tif)

    
    num_y_splits = len(np.unique(np.array(tif_y_splits)))
    num_x_splits = len(np.unique(np.array(tif_x_splits)))
    tif_matrix = np.zeros((num_y_splits, num_x_splits), dtype=object)
    
    for i, tif in enumerate(tif_list):
        tif_matrix[i/num_x_splits, i%num_x_splits] = tif

    tif_list = None
    print("Created Matrix.")
    
    full_rows = []
    for row in range(0,tif_matrix.shape[0]):
        full_row = tif_matrix[row,0]
        for col in range(1,tif_matrix.shape[1]):
            full_row = np.concatenate((full_row, tif_matrix[row][col]), axis = 1)
        full_rows.append(full_row)

    full_img = full_rows[0]
    for i in range(1,len(full_rows)):
        full_img = np.concatenate((full_img,full_rows[i]), axis = 0)
    
    print("final tif dimensions: (%d,%d,%d)" % full_img.shape)
    print("Saving...")
    np.save(save_name, full_img)
    print("Saved file: " + save_name)

        
"""
Given the specifications, looks through the dataframe
and returns the years in the dataframe that contains
yield data in a sorted list. 
Args:
    country:            str, country code
    admin_level:        str, admin level
    season:             str, target season
    county_name:        str, target county name
Returns:
years in the dataframe corresponding to the target country,
and season in a sorted list.
"""               

def get_years_df(country, admin_level, season, county_name):
    path_to_bucket = '/mnt/mounted_bucket'
    df = pd.read_csv('fulldata.csv')
#     df = df[df['country'] == country]
#     df = df[df['season'] == season]
#     df = df[df[admin_level] == county_name]
#     df = df[df['season'] == season]
    years_in_df = df['year'].unique()
    years_in_df = [int(y) for y in years_in_df]
    years_in_df = sorted(years_in_df)
    return years_in_df

"""
Takes two lists of image-arrays broken up into years.
Concatenates the corresponding images in the lists together. 
Used to concatenate the temperature data and MODIS data.
Args:
    MODIS_img_list:              list, list of image arrays, broken up into yearly intervals
    MODIS_temperature_img_list:  list, str, admin level
Returns:
list of array images where each element contains the concatenation
of the images in both lists. 
"""  

def merge_image(MODIS_img_list,MODIS_temperature_img_list):
    MODIS_list=[]
    for i in range(0,len(MODIS_img_list)):
        img_shape=MODIS_img_list[i].shape
        img_temperature_shape=MODIS_temperature_img_list[i].shape
        img_shape_new=(img_shape[0],img_shape[1],img_shape[2]+img_temperature_shape[2])
        merge=np.empty(img_shape_new)
        for j in range(0,img_shape[2]/7):
            img=MODIS_img_list[i][:,:,(j*7):(j*7+7)]
            temperature=MODIS_temperature_img_list[i][:,:,(j*2):(j*2+2)]
            merge[:,:,(j*9):(j*9+9)]=np.concatenate((img,temperature),axis=2)
        MODIS_list.append(merge)
    return MODIS_list


"""
Returns a boolean corresponding to if the given dates are within 
the season or not. 
Args:
    spans_year:   bool,  if the given season spans over new years
    dates:        date,  date to be checked
    season_range: tuple, season range
    cur_year:     int,   current year
Returns:
list of array images where each element contains the concatenation
of the images in both lists. 
"""
def in_season(spans_year, dates, season_range, cur_year):

    if not spans_year :
        return dates.year == cur_year and season_range[0] <= dates.month <= season_range[1]
    else:
        return ((dates.year == cur_year - 1 and season_range[0] < dates.month)\
                or (dates.year == cur_year and season_range[1] > dates.month))
        

        

"""
Takes in an concatenated image and splits them into
individual images. 
Args:
    img:   the concatenated img in array form
    first: first pixel to divide from
    step:  Number of bands in each image
    num:   number of images in the concatenated image
    
Returns:
    List of array-like images. 
"""

def divide_image(img,first,step,num):
    image_list=[]
    for i in range(0,num-1):
        image_list.append(img[:, :, first:first+step])
        first+=step
    image_list.append(img[:, :, first:])
    return image_list


"""
Takes in a list of concatenated images broken up by year,
and turns it into a list of individual images.
Args:
    year_list: list of concatenated images broken up by year
    num_bands: number of bands in each image
    
Returns:
    List of array-like images. 
"""

def list_from_year_blocks(year_list, num_bands):
    full_modis_list = []
    for year in year_list:
        year_list = divide_image(year, 0, num_bands, year.shape[2]/num_bands)
        full_modis_list = full_modis_list + year_list
    return full_modis_list

"""
Get the desired ImageCollection object
Args: 
    country_admin:   String in the form "COUNTRY_ADMIN", eg: ET_admin1
    path_to_bucket:  path to the bucket holding feature_table.json
    coll_code:       code for the correct band mode
    
Returns:
ImageCollection object.
    
"""

def get_imgcoll(country_admin, path_to_bucket = None, coll_code = 'MODIS'):
    
    
    img_code = imgcoll_dict[coll_code][0]
    
    if path_to_bucket == None:
        path_to_bucket = '/mnt/mounted_bucket'
    
    fid, start_date, end_date, admin_type = fusion_dict[country_admin]
    print start_date, end_date
    imgcoll = ee.ImageCollection(img_code).filterDate(start_date, end_date + '-12-31').sort('system:time_start').filterBounds(ee.Geometry.Rectangle(-22, 38, 60, -38))
    return imgcoll


"""
Takes in an image collection and outputs all the dates when
the photos were taken. 
Args: 
    imgcoll: Image Collection, filtered by years
Returns: 
    list of datetime objects
"""

def get_years(imgcoll):

    def ymdList(image, newlist):
        date = ee.Number.parse(image.date().format("YYYYMMdd"))
        newlist = ee.List(newlist)
        return ee.List(newlist.add(date).sort())

    yearlist = imgcoll.iterate(ymdList, ee.List([])).getInfo()
    yearlist = [date(int(str(y)[:4]), int(str(y)[4:6]), int(str(y)[6:8])) for y in yearlist]
    return yearlist



"""
Returns the yield label given the target data.
Args: 
    country:            str, country code
    admin_level:        str, admin level
    county:             str, target county name
    season:             str, target season
    year:               int, target year
Returns: 
    list of datetime objects
"""

def get_labels(country, admin_level, county, season, year):

    path_to_bucket = '/mnt/mounted_bucket'
    path ='fulldata.csv'
    
    df = pd.read_csv(path)
    
    df = df[df['country'] == country] 
    df = df[df[admin_level] == county]
    df = df[df['season'] == season]
    df = df[df['year'] == year]['yield']
#     print year, county
#     print df

    return list(df)[0]

def admin1_testing(country,predictions,metadata): 
    label_dat = pd.read_csv('fulldata_w_area.csv')
    label_dat = label_dat[label_dat['country'] == country]
    season = country_dict[country][1]
    label_dat = label_dat[label_dat['season'] == season]
    label_dat = label_dat.drop(columns = ["Unnamed: 0"])
    label_dat["prediction"] = -1


    for i in range(len(predictions)):
        year = metadata[i][1]
        admin2_county = metadata[i][0]
        ind = label_dat.index[(label_dat['year'] == year) & (label_dat["admin2"] == admin2_county)]
        label_dat.at[ind, 'prediction'] = predictions[i]

    label_dat = label_dat[label_dat['prediction'] != -1]

    new_predictions = []
    new_labels = []
    # print(label_dat.sort_values(by=['year']))
#     print(label_dat.head())
    grouped_predictions_df = label_dat.groupby(["year","admin1"]).apply(lambda dfx: (dfx["prediction"] * dfx["area"]).sum() / dfx["area"].sum())
    print(grouped_predictions_df)
    grouped_yields_df = label_dat.groupby(["year","admin1"]).apply(lambda dfx: (dfx["yield"] * dfx["area"]).sum() / dfx["area"].sum())
    print(grouped_yields_df)
    
    agged_predictions = grouped_predictions_df.reset_index()[0]
    agged_labels = grouped_yields_df.reset_index()[0]

    coefficient_of_dermination = r2_score(agged_labels, agged_predictions)
    mse = mean_squared_error(agged_labels, agged_predictions) 
    r2,_ = stats.pearsonr(agged_labels, agged_predictions)

    fig, ax = plt.subplots()
    ax.scatter(agged_labels, agged_predictions)
    ax.plot([0.5, 3.5], [0.5, 3.5], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')

    print("MSE: %.4f" % mse) 
    print("R2: %.4f" % coefficient_of_dermination)
    print("r: %.4f" %r2)

    plt.show()
    print("{} ADMIN 1 RESULTS").format(country)

def get_label_data(country): 
    label_dat = pd.read_csv('fulldata_w_area.csv')
    label_dat = label_dat[label_dat['country'] == country]
    season = country_dict[country][1]
    label_dat = label_dat[label_dat['season'] == season]
    label_dat = label_dat.drop(columns = ["Unnamed: 0"])
    return label_dat
