#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import numpy as np
from layers.layers import*


# In[ ]:


#https://arxiv.org/abs/1512.03385
class ResNet50(tf.keras.Model):
    #paramas:(int, string, boolean)
        #units: units only will be considerate when include_top is True, it will be used in last layer dense for clasification mode
        #pooling: it can be ["avg", "max", "None"]
            #avg: The final output from encoder (None, 7, 7, 2048) passes through GlobalAVGPooling if include_top ==FALSE
            #max: The final output from encoder (None, 7, 7, 2048) passes through GlobalMAXPooling if include_top ==FALSE
            #None: The final output from encoder is (None, 7, 7, 2048) if include_top ==FALSE
        #include_top: If True,the model will be used as clasification otherwise it will be used as feature extractor
        
    def __init__(self, units, pooling, include_top=False):
        super().__init__()
        self.units = units
        self.pooling = pooling
        self.include_top = include_top
        self.initializer = tf.initializers.glorot_uniform(seed=0)
        #encoder
        self.layer1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1_conv', kernel_initializer = self.initializer)
        self.layer2 = tf.keras.layers.BatchNormalization(axis = 3, name = 'conv1_bn')
        
        self.layer3 = conv_block(3, [64, 64, 256], stage=2, block='1', s=1)
        self.layer4 = identity_block(3, [64, 64, 256], stage=2, block='2')
        self.layer5 = identity_block(3, [64, 64, 256], stage=2, block='3')

        self.layer6 = conv_block(3, [128, 128, 512], stage=3, block='1')
        self.layer7 = identity_block(3, [128, 128, 512], stage=3, block='2')
        self.layer8 = identity_block(3, [128, 128, 512], stage=3, block='3')
        self.layer9 = identity_block(3, [128, 128, 512], stage=3, block='4')

        self.layer10 = conv_block(3, [256, 256, 1024], stage=4, block='1')
        self.layer11 = identity_block(3, [256, 256, 1024], stage=4, block='2')
        self.layer12 = identity_block(3, [256, 256, 1024], stage=4, block='3')
        self.layer13 = identity_block(3, [256, 256, 1024], stage=4, block='4')
        self.layer14 = identity_block(3, [256, 256, 1024], stage=4, block='5')
        self.layer15 = identity_block(3, [256, 256, 1024], stage=4, block='6')

        self.layer16 = conv_block(3, [512, 512, 2048], stage=5, block='1')
        self.layer17 = identity_block(3, [512, 512, 2048], stage=5, block='2')
        self.layer18 = identity_block(3, [512, 512, 2048], stage=5, block='3')
        
        if self.include_top:
            self.layer19 = tf.keras.layers.Dense(self.units, activation='softmax', name='probs')
    #apply_dropout: if true, dropout is applied before moving on to the last fully connected layer               
    def call(self, x, training=True, apply_dropout=False):
        #encoder resnet50
        x = tf.keras.layers.ZeroPadding2D(padding=(3,3), name='conv1_pad')(x)
        x = self.layer1(x) #(None, 230, 230, 64) 
        x = self.layer2(x, training=training)#(None, 112, 112, 64) 
        x = tf.keras.layers.ReLU()(x)#(None, 112, 112, 64) 
        
        x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name="pool1_pad")(x)#(None, 114, 114, 64) 
        x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)#(None, 56, 56, 64)
        
        x = self.layer3(x, training) 
        x = self.layer4(x, training)
        x = self.layer5(x, training)#(None, 56, 56, 256)
        
        x = self.layer6(x, training)
        x = self.layer7(x, training)
        x = self.layer8(x, training)
        x = self.layer9(x, training)#(None, 28, 28, 512)

        x = self.layer10(x, training)
        x = self.layer11(x, training)
        x = self.layer12(x, training)
        x = self.layer13(x, training)
        x = self.layer14(x, training)
        x = self.layer15(x, training)#(None, 14, 14, 1024)

        x = self.layer16(x, training)
        x = self.layer17(x, training)
        x = self.layer18(x, training)#(None, 7, 7, 2048) Final output from encoder
        
        if self.include_top:
            x = tf.keras.layers.AveragePooling2D(pool_size=(7,7), name='avg_pool')(x)
            x = tf.keras.layers.Flatten()(x)
            if apply_dropout:
                x = tf.keras.layers.Dropout(0.2)(x)
            x = self.layer19(x)
        else:
            if self.pooling == "avg":
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
            elif self.pooling == 'max':
                x = tf.keras.layers.GlobalMaxPool2D()(x)
            elif self.pooling== None:
                pass
        return x
    
    # get the second last index from S with regard to CH  (e.g ("model/id/conv/kernel:0", "/") it return 9)
    @staticmethod
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch][-2]
    
    #loadWeight: load previously trained weights
        #params:(string, MODEL, int, int, int, string)
            #wightPath: train weight path
            #model: Self model
            #height, width, channels: input img sizes
    
    def loadWeight(self, weightPath, height=224, width=224, channels=3, model_name="rgb"):
        checkpoint_prefix = weightPath
        if checkpoint_prefix is not None:
            assert (model_name in ('rgb', 'depth', 'skeleton'))
            if model_name == "rgb":
                checkpoint = tf.train.Checkpoint(rgbResnet50=self)
            if model_name == "depth":
                checkpoint = tf.train.Checkpoint(depthResnet50=self)
            if model_name == "skeleton":
                checkpoint = tf.train.Checkpoint(skeletonResnet50=self)
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()#assert_existing_objects_matched()#assert_consumed()
            if checkpoint_manager.latest_checkpoint:
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                print("Initialized from scratch")
                
        else:
            in_width, in_height, in_channels = height, width, channels
            num = tf.cast(tf.constant(np.random.randint(0,255, (1,in_width,in_height,in_channels))),dtype=tf.float32)
            self(num,False, False)
            if self.include_top:
                None if self.units == 1000 else "Warning!!, model has {} output units and it must be 1000".format(self.units)  
            pretrained_resnet = tf.keras.applications.ResNet50(
                        weights="imagenet",
                        include_top=self.include_top,
                        input_shape=(in_width, in_height, 3),
                    )  
            imagenet = {}
            for i, var in enumerate(pretrained_resnet.variables):
                imagenet[var.name] = i
                
            contador = 0
            for var in self.variables:
                name = var.name[self.find(var.name,"/")+1:]
                if (name in imagenet) and var.get_shape()==pretrained_resnet.variables[imagenet[name]].get_shape():
                    var.assign(tf.identity(pretrained_resnet.variables[imagenet[name]]))
                    contador+= 1
            print("Original resnet50 has {} trainable variables and {} trainable variables were transferred".format(len(pretrained_resnet.variables),contador)) 
                
            del pretrained_resnet
                       

#model = ResNet50(21, "avg", True)
#model.loadWeight(None,  height=224, width=224, channels=1)
#a = tf.constant(tf.cast(np.random.randint(0,255,(1,224,224,1)),dtype=tf.float32))
#model(a, training=True, apply_dropout=False)


# In[ ]:


#https://arxiv.org/pdf/1705.04524.pdf
#https://arxiv.org/pdf/1303.5778.pdf
#https://arxiv.org/pdf/1411.4389.pdf
#http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
##http://jmlr.org/papers/v15/srivastava14a.html
class EncoderLSTM(tf.keras.Model):
    def __init__(self, units=500, dropout = .5, recurrent_dropout = .5,
                return_sequences = True, return_state = False, merge = "concat"):
        super().__init__()
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
        self.residual_blstm = ResidualBLSTM(self.units, 
                          self.dropout, 
                          self.recurrent_dropout, 
                          self.return_sequences, 
                          self.return_state, 
                          self.merge)
        self.batch_input = tf.keras.layers.BatchNormalization(axis = -1)
        
    def call(self, x, states, training=True, apply_dropout=True):
        x = self.batch_input(x, training=training)
        if self.return_state:
            x, states1 = self.blstm(x, states)
            x, states2 = self.residual_blstm(x, states, apply_dropout)
            states = [states1, states2]
        else:
            x, _ = self.blstm(x, states)
            x, _ = self.residual_blstm(x, states, apply_dropout)
        return x, states
    
    def initial_state(self, batch_size, units):
        states = [tf.zeros((batch_size, units)),tf.zeros((batch_size, units)), #h
                  tf.zeros((batch_size, units)),tf.zeros((batch_size, units))] #c
        return states
    
    def loadWeight(self, weightPath):
        checkpoint_prefix = weightPath        
        if checkpoint_prefix is not None:
            try:
                checkpoint = tf.train.Checkpoint(encoderLSTM=self)
                checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
                checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()#assert_existing_objects_matched()#assert_consumed()
                if checkpoint_manager.latest_checkpoint:
                    print("Ecoder lstm restored from {}".format(checkpoint_manager.latest_checkpoint))
                #del inp, states
            except:
                raise ValueError("An exception occurred")
        else:
            print("Encoder lstm is initialised from scratch")
        
    
#inp = tf.cast(np.random.rand(3,183,4096), tf.float32)
#encoder = EncoderLSTM(units=500)
#states = encoder.initial_state(inp.shape[0],500)
#out = encoder(inp, states, training=True, apply_dropout=True)
#encoder.summary()


# In[ ]:


#https://arxiv.org/pdf/1502.03044.pdf
class DecoderLSTM(tf.keras.layers.Layer):
    def __init__(self, dec_units=900, vocab_size=23, att_units=64, maxout_units=5, max_dropout = 0.3):
        super(DecoderLSTM, self).__init__()
        self.dec_units = dec_units
        self.att_units = att_units
        self.maxout_units = maxout_units
        self.max_dropout = max_dropout 
        self.cell_lstm = tf.keras.layers.LSTMCell(self.dec_units)

        # used for attention
        self.attention = BahdanauAttention(self.att_units)
        
        #maxout network
        self.maxout = Maxout(self.maxout_units)
        
        self.fc = tf.keras.layers.Dense(vocab_size, activation='softmax')
        
        self.state_size = [self.dec_units,self.dec_units]
    def build(self, input_shape):
        self.batch_size,  self.input_dim = input_shape
        super(DecoderLSTM,self).build(input_shape)
        
    def call(self, x, states, constants):
        #x=Y=(batch_size,vocabulary_size)
        #states= [h,c]
        # constants shape == tuple((batch_size, max_length, enc_units),)
        #context_vector = (batch_size, enc_units), attention_weights =(batch_size, time, 1)
        context_vector, attention_weights = self.attention(states[0], constants[0])
        # x shape after concatenation == (batch_size, context_vector + hidden_size)
        x = tf.keras.layers.Concatenate(axis=-1)([context_vector, x])
        # passing the concatenated vector to the LSTM
        x, states = self.cell_lstm(x, states = states)
        
              
        maxout = self.maxout(x, axis=-1)
        
        maxout = tf.nn.dropout(maxout, rate=self.max_dropout)
        # output shape == (batch_size, vocab)
        x = self.fc(maxout)
        
        return x, states, attention_weights
    
    def initial_state(self, batch_size, units):
        states = [tf.zeros((batch_size, units)),tf.zeros((batch_size, units))]
        return states
    
    def loadWeight(self, weightPath):
        checkpoint_prefix = weightPath
        if checkpoint_prefix is not None:
            try:
                checkpoint = tf.train.Checkpoint(decoderLSTM=self)
                checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
                checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()#assert_existing_objects_matched()#assert_consumed()
                if checkpoint_manager.latest_checkpoint:
                    print("Decoder lstm restored from {}".format(checkpoint_manager.latest_checkpoint))
                #del inp, states
            except:
                raise ValueError("An exception occurred")
        else:
            print("Decoder lstm is initialised from scratch")
            
#x = tf.cast(np.random.rand(2,23), tf.float32)
#states = [tf.cast(np.random.rand(2,900), tf.float32), tf.cast(np.random.rand(2,900), tf.float32)]
#enc_out = tf.cast(np.random.rand(2,5,1000), tf.float32)
#dec = DecoderLSTM()
#x, state = dec(x, states, [enc_out])
#x
#dec.summary()


# In[ ]:


def restoreEncoderDecoderWeight(weightPath, encoder, decoder):
    checkpoint_prefix = weightPath
    if checkpoint_prefix is not None:
        try:
            checkpoint = tf.train.Checkpoint(encoderLSTM= encoder, decoderLSTM=decoder)
            checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
            checkpoint.restore(checkpoint_manager.latest_checkpoint).assert_existing_objects_matched()#assert_existing_objects_matched()#assert_consumed()
            if checkpoint_manager.latest_checkpoint:
                print("Encoder_decoder lstm restored from {}".format(checkpoint_manager.latest_checkpoint))
                #del inp, states
        except:
            raise ValueError("An exception occurred")
    else:
        print("Encoder_decoder lstm is initialised from scratch")

