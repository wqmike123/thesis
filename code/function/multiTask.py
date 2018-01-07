#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 00:57:00 2017

@author: wq
"""
import tensorflow as tf



#%%

class multiTask:

    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,event_embedding = 300,filter_sizes = [3,4,5],\
                 num_filters = 64,predict_state_dim = 300, lstm_units= 256, pred_dense_dim = 64,target_dim = 2,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 0.9,momentum = 0.0,epsilon = 1e-6,trainable=True,core_function = 'cnn'):  
        self.inputs = tf.placeholder(tf.int64, [None, maxlen],name='input')
        self.outputAux = tf.placeholder(tf.float32,[None,target_dim],name = 'output_auxiliary')
        self.y = tf.placeholder(tf.float32, [None,target_dim],name = 'target')
        self.dropout = tf.placeholder(tf.float32,shape={},name = 'dropout')
        #
        self.plot_in =  tf.placeholder(tf.float32,shape={},name = 'plot')
        # news model
        with tf.variable_scope('Embedding'):
            if embedweight is not None:
                word_embeddings = tf.get_variable("word_embeddings",
                                      initializer = embedweight,trainable = trainable)
            else:
                word_embeddings = tf.get_variable("word_embeddings",
                                     [max_voc,embedding_dims],trainable = trainable)

        def BiRNN(x,returnSeq = True):
            #x = tf.split(x, maxlen,axis=1)
            #for i in range(len(x)):
            #    x[i] = tf.squeeze(x[i],axis=1)
            x = tf.unstack(x,axis=1)
        
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units, forget_bias = 1.0)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_units, forget_bias = 1.0)
        
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell, x,
                                                                    dtype = tf.float64)
            if returnSeq:
                return tf.transpose(tf.nn.softsign(outputs),(1,0,2))
            else:
                return tf.nn.softsign(outputs[-1])
            
        def CNN(x):
            pooled_outputs = []
            x_exp = tf.expand_dims(x,-1)
            for i, filter_size in enumerate(filter_sizes):              
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_dims, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        x_exp,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, maxlen - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
 
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs,axis=3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            return h_pool_flat


        def newAnalysis(x,word_embeddings):
            """
            parameter:
            ----------
                x: tensor (number of news,maxlen)
            returns:
            ----------
                tensor (number of news,event_embedding_dim)
            """
            embed = tf.nn.embedding_lookup(word_embeddings,x)
            embed_drop = tf.nn.dropout(embed,self.dropout)
            if core_function == 'cnn':
                with tf.variable_scope('cnn'):
                    hidden = CNN(embed)
                    hidden_drop = tf.nn.dropout(hidden,self.dropout)
            elif core_function == 'birnn':
                with tf.variable_scope('birnn1'):
                    bi1 = BiRNN(embed_drop,returnSeq=True)
                    bi1_drop = tf.nn.dropout(bi1,self.dropout)
                with tf.variable_scope('birnn2'):
                    hidden = BiRNN(bi1_drop,returnSeq=False)
                    hidden_drop = tf.nn.dropout(hidden,self.dropout)
            with tf.variable_scope('fc'):
                event = tf.layers.dense(hidden_drop,pred_dense_dim,tf.nn.relu)
                event_drop = tf.nn.dropout(event,self.dropout)
            with tf.variable_scope('fc2'):
                predRes = tf.layers.dense(event_drop,target_dim)
            return predRes

        with tf.variable_scope('news'):
            self.predict = newAnalysis(self.inputs,word_embeddings)
            predict_final = tf.multiply(self.predict,self.outputAux)
        target_final = tf.multiply(self.y,self.outputAux)
        self.loss_op = tf.losses.mean_squared_error(target_final, predict_final)
        optimizer =  tf.train.RMSPropOptimizer(learning_rate,decay = decay_rate,momentum = momentum,epsilon = epsilon)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, clipvalue)
        self.optimize = optimizer.apply_gradients(zip(gradients, variables))
        
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss",self.loss_op)
            tf.summary.histogram("hist_loss",self.loss_op)
            self.summary_op = tf.summary.merge_all()
        with tf.name_scope('summaries'):
            plt_obj = tf.summary.scalar("plot",tf.squeeze(self.plot_in))
            self.summary_op_aux = tf.summary.merge([plt_obj])