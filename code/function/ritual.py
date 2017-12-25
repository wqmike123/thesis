# -*- coding: utf-8 -*-
"""
Replicate the model done by Lancaster group at SemEval2017 Task 5

Created on Tue Oct 24 08:48:03 2017

@author: wqmike123
"""

from nnBase import *
#from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout,Embedding,Bidirectional,Activation,Concatenate,Conv1D,Multiply,Input,Multiply
from keras import optimizers
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import keras.backend as tf
#%%
class uhSystem(nnBase):
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,hidden_dims = 21,filters = 512, cnn_ngram = 4,\
                 batch_size = 32,epochs_number = 25, lstm_units= 200, dropout=0.5,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 1e-4,trainable=True):
        self.batch_size = batch_size
        self.epochs_number = epochs_number
        
        inputs = Input((maxlen,))
        if not isinstance(embedweight,type(None)):
            embedding = (Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen,weights = [embedweight],trainable = trainable))(inputs)      
        else:
            embedding = (Embedding(max_voc,
                                embedding_dims,
                                input_length=maxlen))(inputs)        
        #model.add(Dropout(dropout))
        cnn_ngram_list = []
        for i in range(1,cnn_ngram+1):
            convi = Conv1D(filters,
                         i,
                         padding='valid',
                         activation='relu',
                         strides=1)(embedding)
            attention = uhSystem._attention(convi,filters)
            cnn_ngram_list.append(attention)
        concat = Concatenate()(cnn_ngram_list)
        output_cnn = Dense(128)(concat)

        # bi-gru
        bi_h = Bidirectional(GRU(lstm_units,return_sequences=True))(embedding) #activation='softsign',
                                     #dropout=dropout,
        output_bi = uhSystem._attention(bi_h,lstm_units*2)                           
                                #input_shape=(maxlen, embedding_dims))) # attention layer?
        
        #opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        #model.compile(loss="mean_square_error", optimizer=opt, metrics=[self.])
        opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=clipvalue)
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics=['cosine_proximity'])
        self.model = model
        
    def fit(self,x_train,y_train,x_valid,y_valid,class_weight = None,earlyStopping = True):
        callback_ = None
        if earlyStopping:
            callback_ = EarlyStopping(monitor='val_loss', patience=10)
        if class_weight:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs_number,
                      validation_data=(x_valid, y_valid),class_weight = class_weight, shuffle=True,callbacks=[callback_])
        else:
            self.model.fit(x_train, y_train,
                      batch_size=self.batch_size,
                      epochs=self.epochs_number,
                      validation_data=(x_valid, y_valid), shuffle=True,callbacks=[callback_])    
    @staticmethod
    def _attention(x,nfilter):
        """ the pooling is over second dim """
        attention_u = Dense(1,activation='tanh')(x)
        attention_alpha = Dense(1,activation='softmax')(attention_u)
        attention = tf.repeat_elements(attention_alpha,nfilter,axis=2)
        return tf.sum(Multiply()([x,attention]),axis=1)
