# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:04:15 2017

@author: wqmike123
"""

import tensorflow as tf


#%%

maxlen = 30
max_voc = 1000
embedding_dims = 300
event_embedding = 500
filter_sizes = [3,4,5]
num_filters = 2
predict_state_dim = 300
lstm_units= 256
pred_dense_dim = 64
target_dim = 1
learning_rate = 0.001
clipvalue = 5
decay_rate = 0.9
momentum = 0.0
epsilon = 1e-6
trainable=True

#%%
g_1 = tf.Graph()
with g_1.as_default() as gf:
    inp = tf.placeholder(tf.int64, [None, maxlen])
    dropout = tf.placeholder(tf.float64,shape={})
    y = tf.placeholder(tf.float32, [target_dim])
    state_cell = tf.placeholder(tf.float32,[predict_state_dim])
    state_hidden = tf.placeholder(tf.float32,[predict_state_dim])
    word_embeddings = tf.get_variable("word_embeddings",
                                         [max_voc,embedding_dims],trainable = trainable)
    with tf.variable_scope('predict'):
        attention_W = tf.get_variable('attention_W',[event_embedding,predict_state_dim],
                                      initializer = tf.truncated_normal_initializer,dtype=tf.float32)
    x = tf.nn.embedding_lookup(word_embeddings,inp)
    x = tf.expand_dims(x,-1)
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):              
            # Convolution Layer
            filter_shape = [filter_size, embedding_dims, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                x,
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
    
    
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs,axis=3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    with tf.variable_scope('fc'):
        event = tf.layers.dense(h_pool_flat,event_embedding,tf.nn.tanh)
        event_drop = tf.nn.dropout(event,0.5)
    
    
        
    with tf.variable_scope('predict'):
        alpha = tf.matmul(tf.matmul(event_drop,attention_W),tf.expand_dims(state_hidden,axis=1))
        scale_alpha = tf.nn.softmax(alpha,dim=0)
        news_input = tf.reduce_sum(tf.multiply(event_drop,scale_alpha),axis=0)
        lstm = tf.contrib.rnn.BasicLSTMCell(predict_state_dim,forget_bias=1.0)
        output_pred,state = lstm(tf.expand_dims(news_input,axis=0),
                                              (tf.expand_dims(state_cell,axis=0),(tf.expand_dims(state_hidden,axis=0))))
        pred_dense = tf.layers.dense(output_pred,pred_dense_dim,tf.nn.tanh)
        price = tf.squeeze(tf.layers.dense(pred_dense,target_dim),axis=1)
    loss_op = tf.losses.mean_squared_error(y, price)
    optimizer =  tf.train.RMSPropOptimizer(learning_rate,decay = decay_rate,momentum = momentum,epsilon = epsilon)
    gradients, variables = zip(*optimizer.compute_gradients(loss_op))
    gradients, _ = tf.clip_by_global_norm(gradients, clipvalue)
    optimize = optimizer.apply_gradients(zip(gradients, variables))
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    logDir = './temp_res/log/'
    train_writer = tf.summary.FileWriter(logDir + '/tb/cnn/train',graph = gf)
    train_writer.close()
