# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 07:37:39 2017

@author: wqmike123
"""
from nnBase import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.preprocessing import sequence
import keras.backend as tf
#%%
#class basicRNN(object):
    
class FCNN(nnBase):   
    def __init__(self,input_dim,hidden_dims=[],\
                 batch_size = 20,epochs_number = 20, dropout=None, 
                 learning_rate = 0.1,decay_rate = 1e-4):
        self.batch_size = batch_size
        self.epochs_number = epochs_number
        model = Sequential()
        model.add(Dense(hidden_dims[0],input_dim = input_dim,activation='relu'))
        for ihidden in hidden_dims[1:]:
            model.add(Dense(ihidden,activation='relu'))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(3, activation='softmax'))
        opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        self.model = model
 
## early stop       
#        cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', save_best_only=True),
#                callbacks.EarlyStopping(monitor='val_loss', patience=3)]
        #get all available data samples from data iterators


        
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


class FCReg(nnBase):
    def __init__(self,input_dim,hidden_dims=[],\
                 batch_size = 20,epochs_number = 20, dropout=None, 
                 learning_rate = 0.1,decay_rate = 1e-4):
        self.batch_size = batch_size
        self.epochs_number = epochs_number
        model = Sequential()
        model.add(Dense(hidden_dims[0],input_dim = input_dim,activation='relu'))
        for ihidden in hidden_dims[1:]:
            model.add(Dense(ihidden,activation='relu'))
        if dropout:
            model.add(Dropout(dropout))
        model.add(Dense(1))
        opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=[self.cosine])
        self.model = model
        
    def cosine(self,y_true, y_pred):
        return tf.tf.reduce_sum(tf.tf.multiply(y_true,y_pred))/tf.sqrt( tf.tf.reduce_sum(tf.square(y_true)))/tf.sqrt(tf.tf.reduce_sum(tf.square(y_pred)))