# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 07:37:39 2017

@author: wqmike123
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Embedding
from keras import optimizers
from keras.preprocessing import sequence

#%%
#class basicRNN(object):
    
class mylstm(object):   
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,hidden_dims = [],\
                 batch_size = 30,epochs_number = 20, lstm_units= 512, dropout=0.2,learning_rate = 0.1,decay_rate = 1e-4,trainable=True):
        self.batch_size = batch_size
        self.epochs_number = epochs_number
        model = Sequential()
        if not isinstance(embedweight,type(None)):
            model.add(Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen,weights = [embedweight],trainable = trainable))
        else:
            model.add(Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen))

        model.add(LSTM(lstm_units,  input_shape=(maxlen, embedding_dims), activation = 'tanh' ,return_sequences=False)) # attention layer?
        model.add(Dropout(dropout))
        for ihidden in hidden_dims:
            model.add(Dense(ihidden,activation='relu'))
        model.add(Dense(2, activation='softmax'))
        opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        self.model = model
 
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