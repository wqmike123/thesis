# -*- coding: utf-8 -*-
"""
Replicate the model done by Lancaster group at SemEval2017 Task 5

Created on Tue Oct 24 08:48:03 2017

@author: wqmike123
"""

from nnBase import *
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Embedding,Bidirectional,Activation
from keras import optimizers
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

#%%
class blstm(nnBase):
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,hidden_dims = 21,\
                 batch_size = 32,epochs_number = 25, lstm_units= 32, dropout=0.5,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 1e-4,trainable=True):
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
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(lstm_units, activation='softsign',
                                     #dropout=dropout,
                                     return_sequences=True)))
                                #input_shape=(maxlen, embedding_dims))) # attention layer?
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(hidden_dims, activation='softsign',
                                     )))                                    
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.add(Activation('linear'))
        #opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        #model.compile(loss="mean_square_error", optimizer=opt, metrics=[self.])
        opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=clipvalue)
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics=['cosine_proximity'])
        self.model = model
        
    def fit(self,x_train,y_train,x_valid = None,y_valid = None,validation_split = 0.1,class_weight = None,earlyStopping = True):
        callback_ = None
        if earlyStopping:
            callback_ = [EarlyStopping(monitor='val_loss', patience=10)]
        if x_valid is not None:
            if class_weight:
                self.model.fit(x_train, y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs_number,
                          validation_data=(x_valid, y_valid),class_weight = class_weight, shuffle=True,callbacks=callback_)
            else:
                self.model.fit(x_train, y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs_number,
                          validation_data=(x_valid, y_valid), shuffle=True,callbacks=callback_)
        else:
            if class_weight:
                self.model.fit(x_train, y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs_number,
                          validation_split=validation_split,class_weight = class_weight, shuffle=True,callbacks=callback_)
            else:
                self.model.fit(x_train, y_train,
                          batch_size=self.batch_size,
                          epochs=self.epochs_number,
                          validation_split=validation_split, shuffle=True,callbacks=callback_)            
            
    def cross_validate(self, train_text, train_sentiments, n_folds=10,
                       shuffle=True, score_function=None):

        all_results = []
        train_text_array = numpy.asarray(train_text)
        train_sentiments_array = numpy.asarray(train_sentiments)

        kfold = KFold(n_splits=n_folds, shuffle=shuffle)
        for train, test in kfold.split(train_text_array, train_sentiments_array):
            self.fit(train_text_array[train], train_sentiments_array[train])
            predicted_sentiments = self.predict(train_text_array[test])
            result = score_function(predicted_sentiments, train_sentiments_array[test])
            all_results.append(result)
        return all_results