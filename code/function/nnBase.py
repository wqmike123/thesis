# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 07:37:39 2017

@author: wqmike123
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing import sequence
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
#%%
#class basicRNN(object):
    
class nnBase(object):   
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,hidden_dims = [],\
                 batch_size = 30,epochs_number = 20, lstm_units= 512, dropout=0.2,learning_rate = 0.1,decay_rate = 1e-4,trainable=True):
        pass
## early stop       
#        cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
#                callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        #get all available data samples from data iterators


    def fit(self,x_train,y_train,x_valid,y_valid,class_weight = None):
        if class_weight:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs_number,
                      validation_data=(x_valid, y_valid),class_weight = class_weight, shuffle=True)
        else:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs_number,
                      validation_data=(x_valid, y_valid), shuffle=True)           
    def predict(self,test_x):
        return self.model.predict(test_x)
    
    def load_weight(self,fadd):
        self.model.load_weights(fadd)
        
    def save_model(self,fpath):
        self.model.save(fpath)
    @staticmethod
    def padding(x,maxlen):
        return sequence.pad_sequences(x, maxlen=maxlen)  
    
    def plot(self,to_file = None):
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
        if to_file:
            from keras.utils import plot_model
            plot_model(self.model, to_file=to_file)
        else:
            from IPython.display import SVG
            from keras.utils.vis_utils import model_to_dot
            SVG(model_to_dot(self.model).create(prog='dot', format='svg'))


           
#class aelstm(mylstm):
#    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,hidden_dims = [],\
#                 batch_size = 30,epochs_number = 20, lstm_units= 512, dropout=0.2,learning_rate = 0.1,decay_rate = 1e-4,trainable=True):
#        self.batch_size = batch_size
#        self.epochs_number = epochs_number
#        model = Sequential()
#
#
#    
#    def __init__