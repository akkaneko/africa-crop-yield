import pandas as pd
import numpy as np
import util_processing as util
import matplotlib.pyplot as plt
from random import shuffle
import random


country_dict = {
    'ET': ('admin2', 'Meher'),
    'ZM': ('admin2', 'Annual'), 
    'TZ': ('admin1', 'Annual'),
    'KE': ('admin2', 'Annual'),
    'NG': ('admin1', 'Wet'),
    'MW': ('admin2', 'Main'),
#     'SU' : ('admin2', 'Annual')
            }

def shuffle_in_unison(a, b):
    assert a.shape[0] == b.shape[0]
    shuffled_a = np.empty(a.shape)
    shuffled_b = np.empty(b.shape)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def percentage_split(seq, percentages):
    cdf = np.cumsum(percentages)
    assert cdf[-1] == 1.0
    stops = map(int, cdf * len(seq))
    return [seq[a:b] for a, b in zip([0]+stops, stops)]



for country in country_dict:
    
    for target_year in range(-1, -5, -1):

        admin, season= country_dict[country]

        data_dict = np.load('/mnt/mounted_bucket/saved_features/%s/VEG_%s_%s.npy' % (country, country, season))
        data_dict = data_dict.item()

        for county in data_dict:
            data_dict[county] = set([i[0] for i in data_dict[county].keys()])

        train_dict = {}
        val_dict = {}
        test_dict = {}

        df = pd.read_csv('fulldata.csv')
        df = df[df['country'] == country]
        df = df[df['season'] == season]
        all_years = list(df['year'].unique())
        all_years = sorted(all_years)

    #     years = all_years[-5:]
    #     train_years = years[:3]
    #     test_year = all_years[-1]
    #     val_year = all_years[-2]

    #     print train_years, val_year, test_year

        count = 0
        data = []
        for county in data_dict:

            train_dict[county] = {}
            val_dict[county] = {}
            test_dict[county] = {}

            for year in data_dict[county]:
                try:
                    cur_label = util.get_labels(country, admin, county, season, year)
                    if cur_label == np.nan:
                        continue
                except:
                    continue
                data.append((county, year, cur_label))
        train = []
        test = []
        val = []
        rest = []  

        ###RANDOMIZED SPLITS###
    #     shuffle(data)
        #splits = percentage_split(data, [0.2, 0.2, 0.2, 0.2, 0.2])

        ###LAST TWO SPLITS###

        test_year = all_years[target_year]
        val_year = all_years[-5]
        train_years = all_years[:-5] + all_years[-5:target_year]


        #test_year = random.sample(all_years, 1)
        #print test_year
    #     all_years.remove(test_year)
    #     val_year = max(all_years)

    #     test, val, train = [], [], [] 


        for i in data:
            if i[1] == test_year:
                test.append(i)
            elif i[1] == val_year:
                val.append(i)
            else:
                if i[1] in train_years:
                    train.append(i)

    #     shuffle(rest)
    #     train = rest[:int(0.8*len(rest))]
    #     val   = rest[int(0.8*len(rest)):]


        for i in train:
            train_dict[i[0]][i[1]] = i[2]

        for i in val:
            val_dict[i[0]][i[1]] = i[2]

        for i in test:
            test_dict[i[0]][i[1]] = i[2]

        split_dict = {
            'train' : train_dict,
            'val'   : val_dict,
            'test'  : test_dict
        }
        path = '/mnt/mounted_bucket/saved_features/%s/chrono_%d' % (country, test_year)
        #path  = '/mnt/mounted_bucket/saved_features/%s/%s_%s_split' % (country, country, season)
        #path   =  '/mnt/mounted_bucket/saved_features/%s/%s_%s_chrono_5year_train'% (country, country, season)
        #path = '/mnt/mounted_bucket/saved_features/%s/%s_%s_chrono_split_randval.npy' % (country, country, season)
        #path = '/mnt/mounted_bucket/saved_features/%s/%s_%s_chrono_split_fiveyears.npy' % (country, country, season)
        print path
        np.save(path, split_dict)
    