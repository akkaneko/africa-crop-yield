import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
from random import shuffle
import matplotlib.pyplot as plt
import util_models as util
import util_processing as utilp
from LSTM_model import LSTMNetwork
from numpy import random
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from CNN_class import *
from baseline_class import *
import GaussianProcessing as gp

season_dict = {
    'ET' : 'Meher',
    'ZM' : 'Annual', 
    'MW' : 'Main',
    'TZ' : 'Annual',
    'NG' : 'Wet',
    'KE' : 'Meher',
    'SU' : 'Annual'
}

def train_LSTM(country, splits, hyperparameter_path, save_path, 
               tensorboard_id = None, before_peak = 3, after_peak = 2, modis_path = None):
    
    if tensorboard_id == None:
        tensorbaord_id = country
        
    train_data, val_data, test_data = utilp.get_modis_feature(country, before_peak=before_peak, 
                                                              after_peak=after_peak, splits=splits, modis_path = modis_path)
    param_dict = np.load(hyperparameter_path).item()
    
    LSTM = LSTMNetwork(param_dict, train_data, val_data, save_path = save_path) 
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            LSTM.run(s, country, tensorboard_id)
            
            
def test_LSTM(country, splits, hyperparameter_path, restore_path, GP = True,
              before_peak = 3, after_peak = 2, modis_path = None):
        
    train_data, val_data, test_data = utilp.get_modis_feature(country, before_peak=before_peak, 
                                                              after_peak=after_peak, splits=splits, modis_path = modis_path)
    param_dict = np.load(hyperparameter_path).item()
    
    if not GP:
        param_dict['batch_size'] = len(test_data[1])
        NN = LSTMNetwork(param_dict, train_data, test_data, restore_path = restore_path) 
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            test_pred = NN.predict(s)
        test_labels = test_data[1]
        return utilp.get_metrics(test_pred, test_labels, country, verbose = True)
        
    else:
        param_dict['batch_size'] = len(val_data[1])
        NN = LSTMNetwork(param_dict, train_data, val_data, restore_path = restore_path) 
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            NN.predict(s)
            _, _, val_ft = NN.get_GP_features(s, 'test')

        param_dict['batch_size'] = len(train_data[1])
        NN = LSTMNetwork(param_dict, train_data, val_data, restore_path = restore_path) 
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            NN.predict(s, 'train')
            _, _, train_ft = NN.get_GP_features(s, 'train')

        param_dict['batch_size'] = len(test_data[1])
        NN = LSTMNetwork(param_dict, train_data, test_data, restore_path = restore_path) 
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            NN.predict(s)
            test_weights, test_bias, test_ft = NN.get_GP_features(s, 'test')
            
        test_bias = np.reshape(test_bias, (1,-1))
        train_ft = np.concatenate([train_ft, val_ft])

        param_dict = {
        'weight_out':test_weights,
        'b_out':test_bias,
        'feature_out_train': train_ft,
        'feature_out_test' : test_ft,
        'country': country
        }
        
        test_labels = test_data[1]
        out_pred = gp.GaussianProcess(param_dict, mode = 'test', before_peak = 3, after_peak=2, splits = splits) 
        pred = np.squeeze(out_pred)
        return utilp.get_metrics(country=country, pred=pred, test_labels=test_labels, RMSE = True)

def lstm_hyperparamter_search(country, splits, num_epochs = 250, num_trials = 50, tensorboard_id = None, 
                              before_peak = 3, after_peak = 2, modis_path = None):
   
    season = data_dict[country][1]
    train_data, val_data, test_data = utilp.get_modis_feature(country, before_peak=3, after_peak=2,
                                                              splits=splits, modis_path = modis_path)
    
    if tensorboard_id == None:
        tensorbaord_id = country

    for i in range(num_trials):
        
        lr = random.uniform(0.0001, 0.005)
        hs = random.randint(100, 400)
        do = random.uniform(0.01, 1)
        dl = random.randint(50, 250)
        bs = 32
        
        print "\n ###############################"
        print "Country:%s, Run %d" % (country, i)
        print "Testing: lr = %f, bs = %d" % (lr, bs)
        print "hs = %d, do = %f, dl = %d"  % (hs, do, dl)
        print "############################### \n"
        
        param_dict = {'batch_size' : bs,
                      'lr'         : lr,
                      'num_epochs' : num_epochs,
                      'hidden_size': hs,
                      'activation' : tf.nn.relu,
                      'initializer': tf.contrib.layers.xavier_initializer,
                      'dropout'    : do,
                      'dense_size' : dl,
                      'tensorboard': tensorboard_id
                     }
        NN = LSTMNetwork(param_dict, train_data, val_data)
        name = 'hyperparameters: %f_%d_%d_%f_%d' % (lr, bs, hs, do, dl)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as s:
            NN.run(s, country, name)
            
def save_hyperparameter_file(country, hyperparameter_path, learning_rate, lstm_size, dropout, num_epochs, 
                             dense_size, batch_size = None, tensorboard_id = None, 
                             initializer = None, activation = None):
    
    if initializer == None:
        initializer = tf.contrib.layers.xavier_initializer
    if activation == None:
        activation = tf.nn.relu
    if tensorboard_id == None:
        tensorboard_id = country
    if batch_size == None:
        batch_size = 32
        
    param_dict = {'batch_size' : batch_size,
                  'lr'         : learning_rate,
                  'num_epochs' : num_epochs,
                  'hidden_size': lstm_size,
                  'activation' : tf.nn.relu,
                  'initializer': tf.contrib.layers.xavier_initializer,
                  'dropout'    : dropout,
                  'dense_size' : dense_size,
                  'tensorboard': tensorboard_id
                  }
    
    np.save(hyperparameter_path, param_dict)
        
            

def run_baseline(country, mode, month_span = 3, learning_rate = 0.001, epochs = 4000, regularizer = 0.5,
                    train_verbose = False, test_verbose = True, simplified = False, save_path = None, 
                    restore_path=None, splits = None, get_metadata = False, subtract_num = 0):
    
    season = season_dict[country]
    if get_metadata:
        train_ft, train_lb, val_ft, val_lb, test_ft, test_lb, metadata = utilp.get_baseline_features(country, month_span, simplified = simplified, get_metadata = get_metadata, splits = splits, subtract_num = subtract_num)
    else:
        train_ft, train_lb, val_ft, val_lb, test_ft, test_lb = utilp.get_baseline_features(country, month_span, simplified = simplified, get_metadata = get_metadata, splits = splits, subtract_num = subtract_num)

    if mode == 'val' or mode == 'train':
        test_features = val_ft 
        test_labels = val_lb
    elif mode == 'test':
        test_features = test_ft
        test_labels = test_lb
        

    model = baseline(country, X_train = train_ft, y_train=train_lb, X_test = test_features, y_test = test_labels, learning_rate = learning_rate, train_epochs = epochs, reg_scale = regularizer, save_path = save_path, restore_path = restore_path)

    model.train(verbose = train_verbose)

    predict_result = model.predict(verbose = test_verbose)
    model.reset()
    
    if get_metadata:
        return predict_result, metadata
    else:
        return predict_result


def run_transfer_baseline(train_country, test_country, save_path, month_span = 3, learning_rate = 0.001, epochs = 4000, regularizer = 0.5, train_verbose = False, test_verbose = True):
    
    train_ft, train_lb, val_ft, val_lb, test_ft, test_lb = utilp.get_baseline_features(train_country, month_span, simplified = True)
    
    train_features = np.concatenate([train_ft, val_ft, test_ft])
    train_labels = np.concatenate([train_lb, val_lb, test_lb])
    
    train_ft, train_lb, val_ft, val_lb, test_ft, test_lb = utilp.get_baseline_features(test_country, month_span, simplified = True)
    
    test_features = np.concatenate([train_ft, val_ft, test_ft])
    test_labels   = np.concatenate([train_lb, val_lb, test_lb])
    
    print train_features.shape, train_labels.shape, test_features.shape, test_labels.shape
    
    cutoff = min([test_features.shape[1], train_features.shape[1]])
    test_features = test_features[:, -cutoff:]
    train_features = train_features[:, -cutoff:]
          
    
    #print train_features.shape
    model = baseline(train_country, X_train = train_features, y_train=train_labels, X_test = test_features, y_test = test_labels, learning_rate = learning_rate, train_epochs = epochs, reg_scale = regularizer, save_path = save_path)
    
    model.train(verbose = train_verbose)
    pred = model.predict(verbose = test_verbose)
    model.reset()
    return pred

    #print test_features.shape
    
#     model = baseline(train_country, X_train = train_features, y_train=train_labels, X_test = test_features, y_test = test_labels, learning_rate = learning_rate, train_epochs = epochs, reg_scale = regularizer,  restore_path = save_path)
    
#     return model.predict(verbose = test_verbose)

    

    
    


