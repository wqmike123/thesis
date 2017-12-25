# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:18:06 2017

@author: wqmike123
"""
#%% build a simple CNN with gloVec as initial
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import optimizers
from keras.callbacks import EarlyStopping
#%%
class cnn:

    def __init__(self,maxlen,max_voc,embedweight = None,embedding_dims = 300, batch_size = 30,\
                 filters = 1024, conv_kernel = 3,hidden_dim = 2048,epochs = 20,\
                 output_dim = 2,dropout = 0.1,trainable=False):

        self.epochs = epochs
        self.batch_size = batch_size
        model = Sequential()
        
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        if not isinstance(embedweight,type(None)):
            model.add(Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen,weights = [embedweight],trainable = trainable))
        else:
            model.add(Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen))            
        model.add(Dropout(dropout))
        
        # we add a Convolution1D, which will learn filters
        # word group filters of size filter_length:
        model.add(Conv1D(filters,
                         conv_kernel,
                         padding='valid',
                         activation='relu',
                         strides=1))
        # we use max pooling:
        model.add(GlobalMaxPooling1D())
        
        # We add a vanilla hidden layer:
        model.add(Dense(hidden_dim))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))
        
        model.add(Dense(512))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))
        
        model.add(Dense(128))
        model.add(Dropout(dropout))
        model.add(Activation('relu'))
        
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(output_dim))
        model.add(Activation('softmax'))
        opt = optimizers.SGD(lr=0.1,decay = 1e-4,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        self.model = model
        
    @staticmethod
    def padding(x,maxlen):
        return sequence.pad_sequences(x, maxlen=maxlen)      
    
    def fit(self,x_train,y_train,x_valid,y_valid,class_weight = None,earlyStopping = True):
        callback_ = None
        if earlyStopping:
            callback_ = EarlyStopping(monitor='val_loss', patience=10)
        if class_weight:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(x_valid, y_valid),class_weight = class_weight, shuffle=True,callbacks=[callback_])
        else:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs,
                      validation_data=(x_valid, y_valid), shuffle=True,callbacks=[callback_])     
#    def fit(self,x_train,y_train,x_valid,y_valid,class_weight = None):
#        if class_weight:
#            self.model.fit(x_train, y_train,
#                      batch_size=self.batch_size,
#                      epochs=self.epochs,
#                      validation_data=(x_valid, y_valid),class_weight = class_weight)
#        else:
#            self.model.fit(x_train, y_train,
#                      batch_size=self.batch_size,
#                      epochs=self.epochs,
#                      validation_data=(x_valid, y_valid))        
    def load_weight(self,fadd):
        self.model.load_weights(fadd)
        
    def save_model(self,fpath):
        self.model.save(fpath)
        
    def predict(self,test_x):
        return self.model.predict(test_x)
    