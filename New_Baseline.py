import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os

    
class Baseline(object):
    
    def __init__(self, param_dict, train_data, test_data, save_path = None, restore_path = None):
        
        self.param_dict = param_dict
        
        self.save_path = save_path
        self.restore_path = restore_path

        self.X_train, self.y_train  = train_data[0], train_data[1]
        self.X_test,  self.y_test   = test_data[0], test_data[1] 
        
        self.mse_test_total, self.mse_train_total, self.r2_train_total,  self.r2_test_total = 0, 0, 0, 0
        
        self.max_r2 = (0, 0)
        self.min_mse = (float('inf'),0)
        self.train = True
        
        self.prev_r2 = -float('inf')
        self.prev_mse = float('inf')
        self.tensorboard_path = param_dict['tensorboard']
        

            
        self.build_graph(param_dict)
        
    def flatten(self, array):
        return np.reshape(array, (-1, array.shape[1], self.dim))
        
        
    def shuffle_in_unison(self,  b, c):
        assert b.shape[0] == c.shape[0]
        shuffled_b = np.empty(b.shape)
        shuffled_c = np.empty(c.shape)
        permutation = np.random.permutation(len(b))
        for old_index, new_index in enumerate(permutation):
            shuffled_b[new_index] = b[old_index]
            shuffled_c[new_index] = c[old_index]
        return shuffled_b, shuffled_c


    def batch_generator(self, data, labels, num_epochs):
        epoch = 0
        
        over_data = []
        over_labels = []
        decrease = 0
        
        while epoch < num_epochs:
                
            data, labels = self.shuffle_in_unison(data, labels)
            over = len(over_data)
           
            
            if over != 0:
                               
                yield(np.concatenate([over_data, data[0:self.batch_size-over]]), 
                      np.concatenate([over_labels, labels[0:self.batch_size-over]]),
                      epoch)
                prev = over
                      
            else:
                prev = 0
                      
            for i in range(prev + self.batch_size, len(data), self.batch_size):
                cur_data = data[prev:i]
                cur_labels = labels[prev:i]
                                
                yield (data[prev:i], labels[prev:i], epoch)
                prev = i
                      
            over_data = data[prev:len(data)]
            over_labels = labels[prev:len(labels)]
            epoch += 1
            if epoch % 10 == 0:
             
                print self.report(epoch)

                if self.prev_r2 > self.r2_test and self.prev_mse < self.test_mse:
                    decrease += 1
                    print decrease
                else:
                    decrease = 0 
                    
#                 if decrease > 5: 
#                     epoch = num_epochs + 1
                    
                self.prev_r2 = self.r2_test
                self.prev_mse = self.test_mse
                   
                
                



    def get_batch(self, data, labels):
        data, labels = self.shuffle_in_unison(data, labels)
        return(data[0:self.batch_size], labels[0:self.batch_size])

    def build_graph(self, param_dict):
        with tf.device('/gpu:0'):
            tf.reset_default_graph()

            self.batch_size = param_dict['batch_size']
            lr = param_dict['lr']
            keep_prob = param_dict['dropout']
            dense_size = param_dict['dense_size']


            #graph = tf.Graph()

            with tf.name_scope('placeholders'):
                self.labels = tf.placeholder(tf.float32, shape = (None))
                self.features = tf.placeholder(tf.float32, shape = (None, 3))                
    
            with tf.variable_scope('fc1'):
                self.dense = tf.layers.dense(self.features, dense_size, activation = tf.nn.relu, kernel_initializer =  tf.contrib.layers.xavier_initializer(), name = 'dense')
                
            with tf.variable_scope('logit'):
                self.logit =  tf.layers.dense(self.dense, 1, activation = None)
                self.pred = tf.squeeze(self.logit)
                
            with tf.name_scope('loss'):
                self.loss = tf.nn.l2_loss(self.pred - self.labels)
                tf.summary.scalar('loss', self.loss)
                
            with tf.name_scope('mse'):
                self.cur_mse = tf.reduce_mean(tf.square(self.pred - self.labels))
                tf.summary.scalar('mse', self.cur_mse)
                
            with tf.name_scope('r2'):
                total_error = tf.reduce_sum(tf.square(self.labels - tf.reduce_mean(self.labels)))
                unexplained_error = tf.reduce_sum(tf.square(self.labels - self.pred))
                self.cur_r2 = 1 - tf.div(unexplained_error, total_error)
                tf.summary.scalar('r2', self.cur_r2)
                
                #self.cur_r2 = r2_score(self.labels, self.pred)

            with tf.name_scope('optim'):
                self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

            self.saver = tf.train.Saver()
            
            self.merged = tf.summary.merge_all()
            

            
    def predict(self, session, mode = 'test'):
        self.train = False

        if self.restore_path != None:
            self.saver.restore(session, '/home/akkaneko/africa-yield/code/saved_models/%s' % self.restore_path)
            
        if mode == 'test':
            ft_batch = self.X_test
            lb_batch = self.y_test
        else:
            ft_batch = self.X_train
            lb_batch = self.y_train
            
        feed_dict = {self.features : ft_batch, self.labels : lb_batch}
        self.pred = session.run([self.pred], 
                                         feed_dict = feed_dict)

        return np.squeeze(self.pred)
    
    def get_GP_features(self, session, mode):
        
        if mode == 'test':    
            ft_batch = self.X_test
            lb_batch = self.y_test
        if mode == 'train':
            ft_batch = self.X_train
            lb_batch = self.y_train
            
        feed_dict = {self.features : ft_batch, self.labels : lb_batch}

        weights = tf.get_default_graph().get_tensor_by_name('fc1/dense/kernel:0')
        bias = tf.get_default_graph().get_tensor_by_name('fc1/dense/bias:0')
            
        features = session.run(self.dense, feed_dict = feed_dict)

        #os.path.split(self.logit.name)[0] + 
        return weights.eval(), bias.eval(), features 



    def run(self, session, country, name):
        session.run(tf.global_variables_initializer())
        
        if self.restore_path != None:
            self.saver.restore(session, '/home/akkaneko/africa-yield/code/saved_models/%s' % self.restore_path)

        cur_epoch = 0

        num_epochs = self.param_dict['num_epochs']
        acc_test = 0
        acc_train = 0

        step = 0
        total_train_loss = 0
        total_test_loss = 0        
        decrease = 0
        total_train_r2 = 0
        total_train_mse = 0
        total_test_r2 = 0
        total_test_mse = 0
        self.r2_test = 0
        
        best_r2 = 0
        best_mse = float('inf')

        
        train_writer = tf.summary.FileWriter('./tensorboard/%s%s/%s/train' % (country, self.tensorboard_path, name),
                                                  session.graph)
        test_writer = tf.summary.FileWriter('./tensorboard/%s%s/%s/test' % (country, self.tensorboard_path, name))
        tf.global_variables_initializer().run()


        for batch in self.batch_generator(self.X_train, self.y_train, num_epochs):

            ft_batch, lb_batch, cur_epoch = batch
            self.cur_train_labels = lb_batch
            self.train = True
            feed_dict = {self.features : ft_batch, self.labels : lb_batch}
            _, curloss, cur_train_mse, cur_train_r2 = session.run([self.optimizer, 
                                                                              self.loss, 
                                                                              self.cur_mse, self.cur_r2], 
                                                                              feed_dict = feed_dict)

            
            test_ft, test_lb = self.get_batch(self.X_test, self.y_test)
            test_dict = {self.features : test_ft, self.labels: test_lb}

            self.cur_test_labels = test_lb
            self.train = False

            cur_test_loss, cur_test_mse, cur_test_r2 = session.run([self.loss, self.cur_mse,
                                                                                  self.cur_r2],
                                                                     feed_dict = test_dict)
            
            
            total_train_loss += curloss
            total_test_loss += cur_test_loss
            total_train_mse += cur_train_mse
            total_train_r2 += cur_train_r2
            total_test_mse += cur_test_mse
            total_test_r2 += cur_test_r2
            
            step += 1

            self.average_loss = total_train_loss/step 
            self.test_loss = total_test_loss/step
            self.r2_test = total_test_r2/step
            self.r2_train = total_train_r2/step
            self.test_mse = total_test_mse/step
            self.train_mse = total_train_mse/step
            
            if self.save_path != None:
                #path = '/mnt/mounted_bucket/saved_models/%s' % self.save_path
                path = '/home/akkaneko/africa-yield/code/saved_models/%s' % self.save_path
                self.saver.save(session, path)
                
            if step % 10 == 0:
                train_summary = session.run(self.merged, feed_dict = feed_dict)
                test_summary = session.run(self.merged, feed_dict = test_dict)
                test_writer.add_summary(test_summary, step)
                train_writer.add_summary(train_summary, step)

    def report(self, epoch):
        return [epoch, ('loss',self.average_loss), ('test_loss',self.test_loss), ("Test R2: %.4f" % self.r2_test), ("Train R2: %.4f" % self.r2_train), ("test MSE: %f" % (self.test_mse), ("train MSE: %f" % (self.train_mse)))] #('train_acc',self.accuracy_train)]
    
    def get_metrics(self):
        metrics = {}
        
        metrics['training loss'] = self.average_loss
        metrics['test loss'] = self.test_loss
        metrics['train r2'] = self.r2_train
        metrics['test r2'] = self.max_r2
        metrics['train mse'] = self.train_mse
        metrics['test mse'] = self.min_mse
        
        return metrics
        


