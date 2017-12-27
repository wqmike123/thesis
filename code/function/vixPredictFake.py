# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 00:14:51 2017

@author: wqmike123
"""
import tensorflow as tf
#%%

class vixPredict_blstm:
    
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,event_embedding = 500,\
                 predict_state_dim = 300, lstm_units= 256, pred_dense_dim = 64,target_dim = 1,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 0.9,momentum = 0.0,epsilon = 1e-6,trainable=True):        

        self.inputs = tf.placeholder(tf.int64, [None, maxlen])
        self.state_cell = tf.placeholder(tf.float64,[predict_state_dim])
        self.state_hidden = tf.placeholder(tf.float64,[predict_state_dim])
        self.y = tf.placeholder(tf.float64, [target_dim])
        self.dropout = tf.placeholder(tf.float64,shape={})
        # news model
        with tf.variable_scope('newsAnalysis'):
            if embedweight is not None:
                word_embeddings = tf.get_variable("word_embeddings",
                                      initializer = embedweight,trainable = trainable)
            else:
                word_embeddings = tf.get_variable("word_embeddings",
                                     [max_voc,embedding_dims],trainable = trainable)

        with tf.variable_scope('predict'):
            attention_W = tf.get_variable('attention_W',[event_embedding,predict_state_dim],
                                          initializer = tf.truncated_normal_initializer,dtype=tf.float64)

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
            with tf.variable_scope('birnn1'):
                bi1 = BiRNN(embed_drop,returnSeq=False)
            bi1_drop = tf.nn.dropout(bi1,self.dropout)
            #with tf.variable_scope('birnn2'):
            #    bi2 = BiRNN(bi1_drop,returnSeq=False)
            #bi2_drop = tf.nn.dropout(bi2,self.dropout)
            with tf.variable_scope('fc'):
                event = tf.layers.dense(bi1_drop,event_embedding,tf.nn.tanh)
            return event
            
        with tf.variable_scope('news'):
            news_embedding = newAnalysis(self.inputs,word_embeddings)
            
        with tf.variable_scope('predict'):
            alpha = tf.matmul(tf.matmul(news_embedding,attention_W),tf.expand_dims(self.state_hidden,axis=1))
            scale_alpha = tf.nn.softmax(alpha,dim=0)
            news_input = tf.reduce_sum(tf.multiply(news_embedding,scale_alpha),axis=0)
            lstm = tf.contrib.rnn.BasicLSTMCell(predict_state_dim,forget_bias=1.0)
            output_pred,state = lstm(tf.expand_dims(news_input,axis=0),
                                                  (tf.expand_dims(self.state_cell,axis=0),(tf.expand_dims(self.state_hidden,axis=0))))
            pred_dense = tf.layers.dense(output_pred,pred_dense_dim,tf.nn.tanh)
            self.price = tf.squeeze(tf.layers.dense(pred_dense,target_dim),axis=1)
        self.loss_op = tf.losses.mean_squared_error(self.y, self.price)
        optimizer =  tf.train.RMSPropOptimizer(learning_rate,decay = decay_rate,momentum = momentum,epsilon = epsilon)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, clipvalue)
        self.optimize = optimizer.apply_gradients(zip(gradients, variables))
        self.state = state
        

        #opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        #model.compile(loss="mean_square_error", optimizer=opt, metrics=[self.])
        #opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=clipvalue)
#%%

 
class vixPredict_cnn:
    
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,event_embedding = 300,filter_sizes = [3,4,5,6],\
                 num_filters = 64,predict_state_dim = 300, lstm_units= 256, pred_dense_dim = 64,target_dim = 1,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 0.9,momentum = 0.0,epsilon = 1e-6,trainable=True):        

        self.inputs = tf.placeholder(tf.int64, [None, maxlen],name='input')
        self.state_cell = tf.placeholder(tf.float32,[predict_state_dim],name = 'state_cell')
        self.state_hidden = tf.placeholder(tf.float32,[predict_state_dim],name = 'state_hidden')
        self.y = tf.placeholder(tf.float32, [target_dim],name = 'target')
        self.dropout = tf.placeholder(tf.float32,shape={},name = 'dropout')
        self.dropout_event = tf.placeholder(tf.float32,shape={},name = 'dropout_event')
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
            #embed_drop = tf.nn.dropout(embed,self.dropout)
            with tf.variable_scope('cnn'):
                cnn = CNN(embed)
            #cnn_drop = tf.nn.dropout(cnn,self.dropout)
            #with tf.variable_scope('birnn2'):
            #    bi2 = BiRNN(bi1_drop,returnSeq=False)
            #bi2_drop = tf.nn.dropout(bi2,self.dropout)
            with tf.variable_scope('fc'):
                event = tf.layers.dense(cnn,event_embedding,tf.nn.relu)
            event_drop = tf.nn.dropout(event,self.dropout_event)
            return event_drop
            
        with tf.variable_scope('news'):
            self.news_embedding = newAnalysis(self.inputs,word_embeddings)
            
        with tf.variable_scope('predict'):
            attention_W = tf.get_variable('attention_W',[event_embedding,predict_state_dim],
                              initializer = tf.truncated_normal_initializer,dtype=tf.float32)
            alpha = tf.matmul(tf.matmul(self.news_embedding,attention_W),tf.expand_dims(self.state_hidden,axis=1))
            scale_alpha = tf.nn.softmax(alpha,dim=0)
            news_input = tf.reduce_sum(tf.multiply(self.news_embedding,scale_alpha),axis=0)
            #lstm = tf.contrib.rnn.BasicLSTMCell(predict_state_dim,forget_bias=1.0)
            #output_pred,state = lstm(tf.expand_dims(news_input,axis=0),
            #                                      (tf.expand_dims(self.state_cell,axis=0),(tf.expand_dims(self.state_hidden,axis=0))))
            #output_pred_drop = tf.nn.dropout(output_pred,self.dropout)
            
            pred_dense1 = tf.layers.dense(news_input,lstm_units,tf.nn.relu)
            pred_dense2 = tf.layers.dense(pred_dense1,pred_dense_dim,tf.nn.relu)
            output_pred_drop = tf.nn.dropout(pred_dense2,self.dropout)
            self.price = tf.squeeze(tf.layers.dense(output_pred_drop,target_dim),axis=1)
        self.loss_op = tf.losses.mean_squared_error(self.y, self.price)#absolute_difference
        optimizer =  tf.train.GradientDescentOptimizer(learning_rate = learning_rate)#RMSPropOptimizer(learning_rate,decay = decay_rate,momentum = momentum,epsilon = epsilon)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, clipvalue)
        self.optimize = optimizer.apply_gradients(zip(gradients, variables))
        self.state = state
        
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss",self.loss_op)
            tf.summary.histogram("hist_loss",self.loss_op)
            self.summary_op = tf.summary.merge_all()
        with tf.name_scope('plot'):
            plt_obj = tf.summary.scalar("plot",tf.squeeze(self.plot_in))
            self.summary_op_aux = tf.summary.merge([plt_obj])

        #opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        #model.compile(loss="mean_square_error", optimizer=opt, metrics=[self.])
        #opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=clipvalue)


#%%
        
 
class vixClassify_cnn:
    
    def __init__(self,maxlen,max_voc,embedweight=None,embedding_dims = 300,event_embedding = 300,filter_sizes = [3,4,5],\
                 num_filters = 64,predict_state_dim = 300, lstm_units= 256, pred_dense_dim = 64,target_dim = 3,learning_rate = 0.001,\
                 clipvalue = 5,decay_rate = 0.9,momentum = 0.0,epsilon = 1e-6,trainable=True):        

        self.inputs = tf.placeholder(tf.int64, [None, maxlen],name='input')
        self.state_cell = tf.placeholder(tf.float32,[predict_state_dim],name = 'state_cell')
        self.state_hidden = tf.placeholder(tf.float32,[predict_state_dim],name = 'state_hidden')
        self.y = tf.placeholder(tf.float32, [target_dim],name = 'target')
        self.dropout = tf.placeholder(tf.float32,shape={},name = 'dropout')
        self.dropout_event = tf.placeholder(tf.float32,shape={},name = 'dropout_event')
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
            #embed_drop = tf.nn.dropout(embed,self.dropout)
            with tf.variable_scope('cnn'):
                cnn = CNN(embed)
            #cnn_drop = tf.nn.dropout(cnn,self.dropout)
            #with tf.variable_scope('birnn2'):
            #    bi2 = BiRNN(bi1_drop,returnSeq=False)
            #bi2_drop = tf.nn.dropout(bi2,self.dropout)
            with tf.variable_scope('fc'):
                event = tf.layers.dense(cnn,event_embedding)
                event_drop = tf.nn.dropout(event,self.dropout_event)
            return event_drop
            
        with tf.variable_scope('news'):
            self.news_embedding = newAnalysis(self.inputs,word_embeddings)
            
        with tf.variable_scope('predict'):
            attention_W = tf.get_variable('attention_W',[event_embedding,predict_state_dim],
                              initializer = tf.truncated_normal_initializer,dtype=tf.float32)
            alpha = tf.matmul(tf.matmul(self.news_embedding,attention_W),tf.expand_dims(self.state_hidden,axis=1))
            scale_alpha = tf.nn.softmax(alpha,dim=0)
            news_input = tf.reduce_sum(tf.multiply(self.news_embedding,scale_alpha),axis=0)
            #lstm = tf.contrib.rnn.BasicLSTMCell(predict_state_dim,forget_bias=1.0)
            #output_pred,state = lstm(tf.expand_dims(news_input,axis=0),
            #                                      (tf.expand_dims(self.state_cell,axis=0),(tf.expand_dims(self.state_hidden,axis=0))))
            #output_pred_drop = tf.nn.dropout(output_pred,self.dropout)
            
            pred_dense1 = tf.layers.dense(news_input,lstm_units,tf.nn.relu)
            pred_dense2 = tf.layers.dense(pred_dense1,pred_dense_dim,tf.nn.relu)
            output_pred_drop = tf.nn.dropout(pred_dense2,self.dropout)
            self.price = tf.layers.dense(output_pred_drop,target_dim)
        self.loss_op = tf.squeeze(tf.nn.softmax_cross_entropy_with_logits(labels = self.y,logits= self.price))
        optimizer =  tf.train.RMSPropOptimizer(learning_rate,decay = decay_rate,momentum = momentum,epsilon = epsilon)#tf.train.GradientDescentOptimizer(learning_rate = learning_rate)#
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, clipvalue)
        self.optimize = optimizer.apply_gradients(zip(gradients, variables))
        self.state = state
        
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss",self.loss_op)
            tf.summary.histogram("hist_loss",self.loss_op)
            self.summary_op = tf.summary.merge_all()
        with tf.name_scope('summaries'):
            plt_obj = tf.summary.scalar("plot",tf.squeeze(self.plot_in))
            self.summary_op_aux = tf.summary.merge([plt_obj])

        #opt = optimizers.SGD(lr=learning_rate,decay = decay_rate,momentum=0.9) #optimizers.adam(lr=0.01, decay=1e-6)
        #model.compile(loss="mean_square_error", optimizer=opt, metrics=[self.])
        #opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0, clipvalue=clipvalue)


 