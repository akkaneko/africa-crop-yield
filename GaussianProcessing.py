import util_processing as utilp
import util_models as utilm
import numpy as np
import pandas as pd
from numpy import random
import tensorflow as tf
from LSTM_model import LSTMNetwork
import matplotlib.pyplot as plt
from scipy import stats
import sys
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform


def GaussianProcess(param_dict, after_peak = None, before_peak = None, mode = 'val', splits = None, modis_path = None,
                   sigma = 1, l_s = 0.5, l_t = 1.5, noise =0.1, const = 0.01, data = None):
    reload(utilp)
    
    
    
    country = param_dict['country']
    X_train = param_dict['feature_out_train']
    X_test = param_dict['feature_out_test']
    W = param_dict['weight_out']

    if data != None:
        train_data, val_data, test_data = data
    else:
        train_data, val_data, test_data = utilp.get_modis_feature(country, after_peak = after_peak, before_peak=before_peak, splits=splits, modis_path = modis_path)
    _, Y_train, other_train, _ = train_data
    _, Y_val, other_val, _ = val_data
    
    Y_train = np.concatenate([Y_train, Y_val])
    other_train = np.concatenate([other_train, other_val])
    
    year_train = np.reshape(other_train[:, 0], [-1, 1])
    x_train = np.reshape(other_train[:, 1], [-1, 1])
    y_train = np.reshape(other_train[:, 2], [-1, 1])    
    
    if mode == 'test':
        _, _, other_test, _ = test_data
    else:
        _, _, other_test, _ = val_data
        
    year_test = np.reshape(other_test[:, 0], [-1, 1])
    x_test = np.reshape(other_test[:, 1], [-1, 1])
    y_test = np.reshape(other_test[:, 2], [-1, 1])
        
        
    n1 = X_train.shape[0]
    n2 = X_test.shape[0]
        
    #X = np.concatenate((X_train,X_test),axis=0)
    XLOC = np.concatenate((x_train,x_test),axis=0)
    YLOC = np.concatenate((y_train,y_test),axis=0)
    YEAR = np.concatenate((year_train,year_test),axis=0)
    
    
    pairwise_dists_xloc = squareform(pdist(XLOC, 'euclidean'))**2/l_s**2
    pairwise_dists_yloc = squareform(pdist(YLOC, 'euclidean'))**2/l_s**2
    pairwise_dists_year = squareform(pdist(YEAR, 'euclidean'))**2/l_t**2

    n=np.zeros([n1+n2,n1+n2])
    n[0:n1,0:n1] += noise*np.identity(n1)

    kernel_mat_3 = sigma*(np.exp(-pairwise_dists_xloc)*np.exp(-pairwise_dists_yloc)*np.exp(-pairwise_dists_year))+n
  
    b = W
    B = np.identity(X_train.shape[1])

    B /= const # B is diag, inverse is simplified
    
    
    K_inv = np.linalg.inv(kernel_mat_3[0:n1,0:n1])

    beta = np.linalg.inv(B+X_train.T.dot(K_inv).dot(X_train)).dot(
            X_train.T.dot(K_inv).dot(Y_train.reshape([n1,1]))+B.dot(b))
        
    Y_pred_3 = X_test.dot(beta) + kernel_mat_3[n1:(n1+n2),0:n1].dot(K_inv\
            ).dot(Y_train.reshape([n1,1])-X_train.dot(beta))
    
    return Y_pred_3
