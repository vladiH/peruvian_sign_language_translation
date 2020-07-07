#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np


# In[ ]:


class identity_block(tf.keras.Model):
    def __init__(self, f, filters, stage, block, **kwargs):
        super(identity_block, self).__init__(**kwargs)
        # defining name basis
        conv_name_base = 'conv'+str(stage)+'_block'+str(block)
        # Retrieve Filters
        F1, F2, F3 = filters
        
        initializer = tf.initializers.glorot_uniform(seed=0)
        self.conv1 = tf.keras.layers.Conv2D(F1, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '_1_conv', kernel_initializer = initializer)
        self.batch1 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_1_bn')
        
        self.conv2 = tf.keras.layers.Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '_2_conv', kernel_initializer = initializer)
        self.batch2 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_2_bn')
        
        self.conv3 = tf.keras.layers.Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '_3_conv', kernel_initializer = initializer)
        self.batch3 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_3_bn')
        
    def call(self, x, training):
        X_shortcut = x
        x = self.conv1(x)
        x = self.batch1(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.conv3(x)
        x = self.batch3(x, training=training)
        x = tf.keras.layers.add([x, X_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x


# In[ ]:


class conv_block(tf.keras.Model):
    def __init__(self, f, filters, stage, block, s = 2, **kwargs):
        super().__init__(**kwargs)
       # defining name basis
        conv_name_base = 'conv'+str(stage)+'_block'+str(block)

        # Retrieve Filters
        F1, F2, F3 = filters
        
        initializer = tf.initializers.glorot_uniform(seed=0)
        self.conv1 = tf.keras.layers.Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '_1_conv', kernel_initializer = initializer)
        self.batch1 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_1_bn')
        
        self.conv2 = tf.keras.layers.Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '_2_conv', kernel_initializer = initializer)
        self.batch2 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_2_bn')
        
        self.conv3 = tf.keras.layers.Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '_3_conv', kernel_initializer = initializer)
        self.batch3 = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_3_bn')
        
        self.conv_shortcut = tf.keras.layers.Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '_0_conv', kernel_initializer = initializer)
        self.batch_shortcut = tf.keras.layers.BatchNormalization(axis = 3, name = conv_name_base + '_0_bn')
        
    def call(self, x, training):
        x_shortcut = x
        x = self.conv1(x)
        x = self.batch1(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = tf.keras.layers.ReLU()(x)
        
        x = self.conv3(x)
        x = self.batch3(x, training=training)
        
        x_shortcut =  self.conv_shortcut(x_shortcut)
        x_shortcut = self.batch_shortcut(x_shortcut, training=training)
        
        x = tf.keras.layers.add([x, x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x


# In[ ]:


#https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735
class BLSTM(tf.keras.layers.Layer):
    def __init__(self, units=500, dropout = .5, recurrent_dropout = .5,
                return_sequences = True, return_state = False, merge = "concat"):
        super().__init__()
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.merge = merge
        self.f_lstm = tf.keras.layers.LSTM(self.units,  
                                                      kernel_initializer='glorot_uniform', 
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros',
                                                      dropout=self.dropout,
                                                      recurrent_dropout=self.recurrent_dropout,
                                                      return_sequences=self.return_sequences,
                                                      return_state=self.return_state,
                                                      go_backwards=False)
        self.b_lstm = tf.keras.layers.LSTM(self.units,  
                                                      kernel_initializer='glorot_uniform', 
                                                      recurrent_initializer='orthogonal',
                                                      bias_initializer='zeros',
                                                      dropout=self.dropout,
                                                      recurrent_dropout=self.recurrent_dropout,
                                                      return_sequences=self.return_sequences,
                                                      return_state=self.return_state,
                                                      go_backwards=True)
        self.blstm = tf.keras.layers.Bidirectional(layer=self.f_lstm, 
                                                       backward_layer=self.b_lstm, 
                                                       merge_mode= self.merge)
    def build(self, input_shape):
        self.batch_size, self.time_step, self.input_dim = input_shape
        super(BLSTM,self).build(input_shape)
        
    def call(self, x, states):
        if self.return_state:
            x, forward_h, forward_c, backward_h, backward_c = self.blstm(x, initial_state=states)
            states_f = [forward_h, backward_h]
            states_b = [forward_c, backward_c]
            states = [states_f, states_b]
        else:
            x = self.blstm(x, initial_state=states)
        return x, states


# In[ ]:


#https://www.mitpressjournals.org/doi/10.1162/neco.1997.9.8.1735
class ResidualBLSTM(tf.keras.layers.Layer):
    def __init__(self, units=500, dropout = .5, recurrent_dropout = .5,
                return_sequences = True, return_state = False, merge = "concat", **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.merge = merge
        self.blstm = BLSTM(self.units,
                                self.dropout, 
                                self.recurrent_dropout, 
                                self.return_sequences, 
                                self.return_state, 
                                self.merge)
        
    def call(self, x, states, apply_dropout=True):
        if self.return_state:
            lstm, states = self.blstm(x, states)
        else:
            lstm,_ = self.blstm(x, states)
        if apply_dropout:
            lstm = tf.keras.layers.Dropout(self.dropout)(lstm)
        x = tf.keras.layers.add([lstm, x])
        return x, states


# In[ ]:


#https://arxiv.org/abs/1409.0473
#https://arxiv.org/abs/1508.04025 (optional)
#https://arxiv.org/abs/1512.08756
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units=64):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    #h=hidden cell from lstm, x, encoder_output
    def call(self, h, x):
        h = tf.keras.layers.RepeatVector(x.shape[1])(h)#(batch,time,units)
        x_h = tf.keras.layers.Concatenate(axis=-1)([h,x])#(batch,time_step,h_units+x_seq_units)
        # score shape == (batch_size, max_length, 1)
        score = self.V(tf.nn.relu(self.W1(x_h)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * x 
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = tf.math.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
#x = tf.cast(np.random.rand(1,5,1000), tf.float32)
#h = tf.cast(np.random.rand(1,900), tf.float32)
#att = BahdanauAttention(units=64)
#ctx, w = att(h,x)
#ctx
#att.summary()


# In[ ]:


#http://proceedings.mlr.press/v28/goodfellow13.html
#https://doi.org/10.1016/j.specom.2015.12.003
class Maxout(tf.keras.layers.Layer):
    def __init__(self, ln_unit):
        super(Maxout, self).__init__()
        self.ln_unit=ln_unit
        
    def call(self, x, axis =None):
        shape = x.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % self.ln_unit:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, self.ln_unit))
        shape[axis] = shape[axis]//self.ln_unit
        shape += [self.ln_unit]
        outputs = tf.keras.backend.max(tf.reshape(x, shape), -1, keepdims=False)
        return outputs
#inp = tf.cast(np.random.rand(1,1,900), tf.float32)
#maxout = Maxout(ln_unit=5)
#out = maxout(inp)
#out
#maxout.summary()


