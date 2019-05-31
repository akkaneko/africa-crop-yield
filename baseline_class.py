import features
import util_models
import gdal
import sys
import numpy as np
import os
import importlib
sys.path.append(os.path.abspath("baseline/"))
from constants import HIST_BINS_LIST, NUM_IMGS_PER_YEAR, NUM_TEMP_BANDS, NUM_REF_BANDS, CROP_SENTINEL, GBUCKET, RED_BAND, NIR_BAND
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy import stats

class baseline:
    def __init__(self, country, season = None, learning_rate = 0.001, train_epochs = 4000, restore_path = None,
                 save_path = None, X_train=None, X_test=None, y_train=None, y_test=None, reg_scale = 0.5):
           
        self.country = country
        
        self.train_x, self.train_y = self.append_bias_reshape(X_train,y_train)
        self.test_x, self.test_y = self.append_bias_reshape(X_test, y_test)
        
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.cost_history = np.empty(shape=[1],dtype=float)
        self.r2_history = np.empty(shape=[1], dtype=float)
        
        n_dim = np.concatenate([self.train_x, self.test_x]).shape[1]
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
        self.X = tf.placeholder(tf.float32,[None,n_dim])
        self.Y = tf.placeholder(tf.float32,[None,1])
        with tf.variable_scope("w", reuse = tf.AUTO_REUSE):
            self.W = tf.get_variable('W', shape=(n_dim,1), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32,regularizer=regularizer)
        
        self.init = tf.global_variables_initializer()
        self.y_ = tf.matmul(self.X, self.W)

        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.Y)) + reg_term

        total_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, tf.reduce_mean(self.Y))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(self.Y, self.y_)))
        self.R_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))

        self.training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.saver = tf.train.Saver()
        self.save_path = save_path
        self.restore_path = restore_path

        
    def train(self, verbose = False):
        self.sess = tf.Session()
        self.sess.run(self.init)

        for epoch in range(self.train_epochs):
            self.sess.run(self.training_step,feed_dict={self.X:self.train_x,self.Y:self.train_y})
            self.cost_history = np.append(self.cost_history,self.sess.run(self.cost,feed_dict={self.X: self.train_x,self.Y: self.train_y}))
            self.r2_history = np.append(self.r2_history, self.sess.run(self.R_squared, feed_dict={self.X: self.train_x,self.Y: self.train_y}))
                        
            
        if self.save_path != None:
                self.saver.save(self.sess,  self.save_path)
                     
        if verbose:
            print 'R2: ',  self.r2_history[-1]
            print 'MSE: ', self.cost_history[-1]   

            plt.plot(range(len(self.cost_history)),self.cost_history)
            plt.axis([0,self.train_epochs,0,np.max(self.cost_history)])
            plt.xlabel("Epochs")
            plt.ylabel("Train MSE")
            plt.show()
            #plt.savefig("./result_figures/baseline/Train_MSE_"+ self.country + "_" + self.season)
            plt.gcf().clear()
            plt.plot(range(len(self.r2_history)),self.r2_history)
            plt.axis([0,self.train_epochs,-1,1])
            plt.xlabel("Epochs")
            plt.ylabel("R2 Train")
#             print 'Train R2: ',  self.r2_history[-1]
#             print 'Train MSE: ', self.cost_history[-1]
            plt.show()
        
            

        
        #plt.savefig("./result_figures/baseline/Train_R2_"+ self.country + "_" + self.season)
            
    def predict(self, verbose = True):
        
        if self.restore_path != None:
            self.saver.restore(self.sess, self.restore_path)
        
        pred_y = self.sess.run(self.y_, feed_dict={self.X: self.test_x})
        if np.isnan(pred_y).any():
            return float('inf')
        
        coefficient_of_dermination = r2_score(self.test_y, pred_y)
        mse = tf.reduce_mean(tf.square(pred_y - self.test_y))
        r2,_ = stats.pearsonr(self.test_y, pred_y)

        
        if verbose:
            
            fig, ax = plt.subplots()
            ax.scatter(self.test_y, pred_y)
            ax.plot([0.5, 3.5], [0.5, 3.5], 'k--', lw=3)
            ax.set_xlabel('Measured')
            ax.set_ylabel('Predicted')
            
            print("MSE: %.4f" % self.sess.run(mse)) 
            print("R2: %.4f" % coefficient_of_dermination)
            print("r: %.4f" %r2)

            plt.show()
        


        return pred_y
#         return [('R2', coefficient_of_dermination), ("MSE" , self.sess.run(mse)), ('r2', r2)] 

        #plt.savefig("./result_figures/baseline/Results_"+ self.country + "_" + self.season)
        
            
    def append_bias_reshape(self, features,labels):
        n_training_samples = features.shape[0]
        n_dim = features.shape[1]
        f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
        l = np.reshape(labels,[n_training_samples,1])
        return f, l
    
    def reset(self):
        tf.reset_default_graph()


    
    
def main(argv):
    country = argv[0]
    season = argv[1]
    model = baseline(country, season)
    model.train()
    model.predict()
        
if __name__ == "__main__":
    main(sys.argv[1:])
