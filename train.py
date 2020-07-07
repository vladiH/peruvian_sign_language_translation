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
import tensorflow as tf
import numpy as np
import time
import datetime
import h5py
from data import utils
from data.mydecorator import scope
from data import generators
from data.preprocessing import changeLabels
from data.rawToHdf5 import DepthLSP, SkeletonLSP, LSP10
from model.builders import*
from tqdm.notebook import tnrange
from tqdm.notebook import tqdm as tqdm_notebook


# In[ ]:


class Train:
    def __init__(self, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs):
        self.dataset_path = dataset_path
        self.h5_file_name = h5_file_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir
    
    def initMetrics(self):
        self.epoch_train_acc = tf.keras.metrics.Mean()
        self.epoch_train_loss = tf.keras.metrics.Mean()
        self.epoch_validation_acc = tf.keras.metrics.Mean()
        self.epoch_validation_loss = tf.keras.metrics.Mean()
            
    def saveMetrics(self, epoch):
        tf.summary.scalar('train_acc',self.epoch_train_acc.result(), step=epoch)
        tf.summary.scalar('train_loss',self.epoch_train_loss.result() , step=epoch)
        tf.summary.scalar('val_acc', self.epoch_validation_acc.result(), step=epoch)
        tf.summary.scalar('val_loss', self.epoch_validation_loss.result(), step=epoch)
    
    def resetMetrics(self):
        self.epoch_train_acc.reset_states()
        self.epoch_train_loss.reset_states()
        self.epoch_validation_acc.reset_states()
        self.epoch_validation_loss.reset_states()
        
    @staticmethod
    def adamOptimizer(lr, beta_1, beta_2, decay):
        return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, decay= decay)
    
    @staticmethod
    def rmspOtimizer(lr, rho, momentum, decay):
        return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, momentum=momentum, decay= decay)
    
    @staticmethod
    def sgdOptimizer(lr, momentun):
        return tf.keras.optimizers.SGD(lr, momentun)
    
    def makeSummaryDir(self, name):
        output_dir = os.path.join(self.output_dir, "summary")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.join(output_dir, name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        return self.output_dir
    
    @staticmethod
    def summaryWriter(summary_dir):
        log_dir = os.path.join(summary_dir, "log")
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        return tf.summary.create_file_writer(
              log_dir + "/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    @staticmethod
    def lossFunction(target, predicted):
        return tf.reduce_mean(tf.losses.categorical_crossentropy(target, predicted))
    
    @staticmethod
    def accuracyFunction(target, predicted):
        target = tf.argmax(target,-1)
        predicted = tf.argmax(predicted,-1)
        return tf.reduce_mean(tf.cast(tf.math.equal(target,predicted), dtype=tf.float32))


# In[ ]:


class ResnetTrain(Train):
    def __init__(self, output_units, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs):
        super().__init__(dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs)
        self.confusionMatrix = utils.confusionMatrix()
        self.labels = None #vocabulary
        self.weight_path = None
        self.output_units = output_units
        self.train_data
        self.validation_data
        self.optimizer
        self.model
        self.ckpt
    
    @scope
    def train_data(self):
        "WARNING"
        "if you want to use this class you must develop RGB generator and replace genDepth with your class"
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genDepth('train', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def validation_data(self):
        "WARNING"
        "if you want to use this class you must develop RGB generator and replace genDepth with your class"
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genDepth('validation', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def optimizer(self):
        return self.sgdOptimizer(self.learning_rate, 0.9)
    @scope
    def model(self):
        return build_rgbResnet50(self.weight_path, self.output_units)
    @scope
    def ckpt(self):
        summary_dir = self.makeSummaryDir("rgb")
        checkpoint_prefix = os.path.join(summary_dir, "ckpt")
        if not os.path.isdir(checkpoint_prefix):
            os.mkdir(checkpoint_prefix)
        checkpoint = tf.train.Checkpoint(rgbResnet50=self.model)
        return tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    
    @tf.function   
    def trainStep(self,inputs, targets):
        with tf.GradientTape() as tape:
            model_output = self.model(inputs, training = True, apply_dropout=True)
            loss = self.lossFunction(targets, model_output)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        acc = self.accuracyFunction(targets, model_output)
        return loss, acc
    
    def valStep(self, inputs, targets):
        model_output = self.model(inputs, training = True, apply_dropout=True)
        loss = self.lossFunction(targets, model_output)
        acc = self.accuracyFunction(targets, model_output)
        self.confusionMatrix.targets = targets
        self.confusionMatrix.predicteds = model_output
        return loss, acc
    
    def train(self, early_stop):
        self.initMetrics()
        summary_dir = self.output_dir
        summary_writer = self.summaryWriter(summary_dir)
        best_val_loss, best_val_epoch = None, None
        for epoch in tnrange(self.epochs, desc="Epoch: ", position=0, leave=True):
            start = time.time()
            
            for inputs, targets in tqdm_notebook(self.train_data, desc="Train: ", position=0, leave=False):
                loss, acc = self.trainStep(inputs, targets)
                self.epoch_train_loss(loss)
                self.epoch_train_acc(acc)
                
            for inputs, targets in tqdm_notebook(self.validation_data, desc="Val: ", position=0, leave=False):
                loss, acc = self.valStep(inputs, targets)
                self.epoch_validation_loss(loss)
                self.epoch_validation_acc(acc)
                
            tf.print ("Time taken for epoch {0} is {1:.3f} sec, train_acc is {2:.4f}, train_loss is {3:.4f}, val_acc is {4:.4f}, val_loss is {5:.4f}\n".format((epoch +1) ,
                                                                   time.time()-start, self.epoch_train_acc.result(),self.epoch_train_loss.result(),self.epoch_validation_acc.result(),self.epoch_validation_loss.result()))
            
            if early_stop!=0:
                validation_loss = self.epoch_validation_loss.result()
                if best_val_loss is None or best_val_loss > validation_loss:
                    best_val_loss, best_val_epoch = validation_loss, epoch
                    self.ckpt.save()
                    self.confusionMatrix.restore()
                    with summary_writer.as_default():
                        self.saveMetrics(epoch)
                if best_val_epoch <= epoch - early_stop:
                    
                    self.confusionMatrix.saveFigure(self.output_dir,self.labels)
                    break 
            self.resetMetrics()
        summary_writer.close()
            


# In[ ]:


class depthResnetTrain(ResnetTrain):
    def __init__(self, output_units, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs):
        super().__init__(output_units, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs)
        self.labels = list(DepthLSP.getLabelToInt("", vocabulary=True).keys())#vocabulary
        self.train_data
        self.validation_data
        self.optimizer
        self.model
        self.ckpt
    @scope
    def train_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genDepth('train', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def validation_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genDepth('validation', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def optimizer(self):
        return self.sgdOptimizer(self.learning_rate, 0.9)
    @scope
    def model(self):
        return build_depthResnet50(self.weight_path, self.output_units)
    @scope
    def ckpt(self):
        summary_dir = self.makeSummaryDir("depth")
        checkpoint_prefix = os.path.join(summary_dir, "ckpt")
        if not os.path.isdir(checkpoint_prefix):
            os.mkdir(checkpoint_prefix)
        checkpoint = tf.train.Checkpoint(depthResnet50=self.model)
        return tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)


# In[ ]:


class skeletonResnetTrain(ResnetTrain):
    def __init__(self, output_units, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs):
        super().__init__(output_units, dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs)
        self.labels = list(SkeletonLSP.getLabelToInt("", vocabulary=True).keys())#vocabulary
        self.train_data
        self.validation_data
        self.optimizer
        self.model
        self.ckpt
    @scope
    def train_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genSkeleton('train', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def validation_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        return generators.genSkeleton('validation', h5_file_path).fromGenerator(batch_size=self.batch_size)
    @scope
    def optimizer(self):
        return self.sgdOptimizer(self.learning_rate, 0.9)
    @scope
    def model(self):
        return build_skeletonResnet50(self.weight_path, self.output_units)
    @scope
    def ckpt(self):
        summary_dir = self.makeSummaryDir("skeleton")
        checkpoint_prefix = os.path.join(summary_dir, "ckpt")
        if not os.path.isdir(checkpoint_prefix):
            os.mkdir(checkpoint_prefix)
        checkpoint = tf.train.Checkpoint(skeletonResnet50=self.model)
        return tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)


# In[ ]:


class LspTrain(Train):
    def __init__(self, dataset_path, h5_file_name, output_dir, batch_size,  learning_rate, epochs,
                tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units,attention_units,
                 maxout_linear_units, max_dropout, optimizer_name,
                low_vram):
        super().__init__(dataset_path, h5_file_name, output_dir, batch_size, learning_rate, epochs)
        
        self.tx = tx
        self.ty = ty
        self.enc_units = enc_units
        self.enc_dropout = enc_dropout
        self.recurrent_dropout = recurrent_dropout
        
        self.dec_units = dec_units
        self.attention_units = attention_units
        self.maxout_linear_units = maxout_linear_units
        self.max_dropout = max_dropout
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.low_vram = low_vram
    
        self.vocabulary_size = len(LSP10.getLabelToInt("",vocabulary=True))
    
        self.optimizer
        self.encoder
        self.decoder
        self.ckpt
    
    @scope
    def optimizer(self):
        if self.optimizer_name == "adam":
            return self.adamOptimizer(self.learning_rate, beta_1=0.9, beta_2=0.999, decay=0.1)
        if self.optimizer_name == "rmsp":
            return self.rmspOtimizer(self.learning_rate, rho=0.94, momentum=0., decay=0.0)
    
    @scope
    def encoder(self):
        return build_encoderLSTM(None, units=self.enc_units, dropout = self.enc_dropout,
                                 recurrent_dropout = self.recurrent_dropout, merge = "concat")
    @scope
    def decoder(self):
        return build_decoderLSTM(None, units=self.dec_units, vocabulary_size=self.vocabulary_size, 
                                 attention_units=self.attention_units, maxout_linear_units=self.maxout_linear_units,
                                 max_dropout= self.max_dropout)
    
    @scope
    def ckpt(self):
        summary_dir = self.makeSummaryDir("lsp")
        checkpoint_prefix = os.path.join(summary_dir, "ckpt")
        if not os.path.isdir(checkpoint_prefix):
            os.mkdir(checkpoint_prefix)
        checkpoint = tf.train.Checkpoint(encoderLSTM = self.encoder,
                                        decoderLSTM = self.decoder)
        return tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    
    @staticmethod
    def lossFunction(target, predicted):
        mask = tf.math.logical_not(tf.reduce_all(tf.math.equal(target, tf.one_hot([20], 23)),axis=-1))
        mask = tf.cast(mask, dtype=tf.float32)
        loss = tf.keras.losses.categorical_crossentropy(target, predicted)
        loss *= mask
        return tf.reduce_mean(loss)
    
    def selectInput(self, inputs):
        output = []
        for inp in inputs:
            if inp is not None:
                output.append(inp)
        return output
    
    def inputLowVram(self, inputs):
        return inputs
    
    def inputNoLowVram(self, inputs):   
        return inputs
    
    def inputs(self, inputs):
        if self.low_vram:
            inputs = self.inputLowVram(inputs)
        else:
            inputs = self.inputNoLowVram(inputs)
        return inputs
    
    @tf.function         
    def trainStep(self, inputs, targets, enc_states, dec_states):
        loss = 0.
        acc = 0.
        with tf.GradientTape() as tape:
            tape.watch(self.encoder.trainable_variables + self.decoder.trainable_variables)
            enc_output, _ = self.encoder(inputs, enc_states)
            dec_input = targets[:, 0]
            for t in tf.range(1, targets.shape[1]):
                predictions, dec_states, _ = self.decoder(dec_input, dec_states, [enc_output])
                loss += self.lossFunction(targets[:, t], predictions)
                dec_input = targets[:, t]
                acc += self.accuracyFunction(targets[:, t], predictions)
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss/targets.shape[1], acc/targets.shape[1]
    
    def valStep(self, inputs, targets, enc_states, dec_states):
        loss = 0.
        acc = 0.
        enc_output,_ = self.encoder(inputs, enc_states)
        dec_input = targets[:, 0]
        for t in tf.range(1, self.ty):
            # passing enc_output to the decoder x, hidden, enc_output
            predictions, dec_states, _ = self.decoder(dec_input, dec_states, [enc_output])
            loss += self.lossFunction(targets[:, t], predictions)
            acc += self.accuracyFunction(targets[:, t], predictions)
            predictions = tf.argmax(predictions,axis=-1)
            predictions = changeLabels(predictions,self.vocabulary_size, op="categorical")
            dec_input =predictions
        return loss/targets.shape[1], acc/targets.shape[1]
                    
    def train(self, early_stop):
        self.initMetrics()
        summary_dir = self.output_dir
        summary_writer = self.summaryWriter(summary_dir)
        best_val_loss, best_val_epoch = None, None
        for epoch in tnrange(self.epochs, desc="Epoch: ", position=0, leave=True):
            start = time.time()
            for r,d,s, targets in tqdm_notebook(self.train_data, desc="Train: ", position=0, leave=False):
                inputs = [r,d,s]
                inputs = self.selectInput(inputs)
                inputs = self.inputs(inputs)
                enc_states = self.encoder.initial_state(targets.shape[0],self.enc_units)
                dec_states = self.decoder.initial_state(targets.shape[0],self.dec_units)
                loss, acc = self.trainStep(inputs, targets, enc_states, dec_states)
                self.epoch_train_loss(loss)
                self.epoch_train_acc(acc)
                
            for r,d,s, targets in tqdm_notebook(self.validation_data, desc="Val: ", position=0, leave=False):
                inputs = [r,d,s]
                inputs = self.selectInput(inputs)
                inputs = self.inputs(inputs)
                enc_states = self.encoder.initial_state(targets.shape[0],self.enc_units)
                dec_states = self.decoder.initial_state(targets.shape[0],self.dec_units)
                loss, acc = self.valStep(inputs, targets, enc_states, dec_states)
                self.epoch_validation_loss(loss)
                self.epoch_validation_acc(acc)
                
            tf.print ("Time taken for epoch {0} is {1:.3f} sec, train_acc is {2:.4f}, train_loss is {3:.4f}, val_acc is {4:.4f}, val_loss is {5:.4f}\n".format((epoch +1) ,
                                                                   time.time()-start, self.epoch_train_acc.result(),self.epoch_train_loss.result(),self.epoch_validation_acc.result(),self.epoch_validation_loss.result()))
            
            if early_stop!=0:
                validation_loss = self.epoch_validation_loss.result()
                if best_val_loss is None or best_val_loss > validation_loss:
                    best_val_loss, best_val_epoch = validation_loss, epoch
                    self.ckpt.save()
                    with summary_writer.as_default():
                        self.saveMetrics(epoch)
                if best_val_epoch <= epoch - early_stop:
                    break 
            self.resetMetrics()
        summary_writer.close()


# In[ ]:


class rgbLspTrain(LspTrain):
    def __init__(self, dataset_path, h5_file_name, output_dir, batch_size,  learning_rate, epochs,
                tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units, attention_units, 
                maxout_linear_units, max_dropout, optimizer_name,
                low_vram, rgb_weight_path=None):
        super().__init__(dataset_path, h5_file_name, output_dir, batch_size,  learning_rate, epochs,
                         tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units,
                         attention_units, maxout_linear_units, max_dropout, optimizer_name,
                low_vram)
        self.rgb_weight_path = rgb_weight_path
        self.train_data
        self.validation_data
        self.fe_rgb
        
    @scope
    def train_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        if self.low_vram:
            data = generators.genLSP10.initGenLSPlowVram("train", self.tx, self.ty, h5_file_path,
                                                         rgb=True, depth=False, skeleton=False).fromGenerator(batch_size=self.batch_size)
        else:
            data = generators.genLSP10.initGenLSP("train", self.tx, self.ty, h5_file_path,
                                                  rgb=True, depth=False, skeleton=False).fromGenerator(batch_size=self.batch_size)
        return data
    
    @scope
    def validation_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        if self.low_vram:
            data = generators.genLSP10.initGenLSPlowVram("validation", self.tx, self.ty, h5_file_path,
                                                         rgb=True, depth=False, skeleton=False).fromGenerator(batch_size=self.batch_size) 
        else:
            data = generators.genLSP10.initGenLSP("validation", self.tx, self.ty, h5_file_path,
                                                  rgb=True, depth=False, skeleton=False).fromGenerator(batch_size=self.batch_size)
        return data
    
    @scope
    def fe_rgb(self):
        if self.low_vram:
            return None
        model = build_rgbFeatureExtractor(weight_path=self.rgb_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    
    def inputLowVram(self, inputs):
        return inputs[0]
    
    def inputNoLowVram(self, inputs):   
        return self.fe_rgb(inputs[0])


# In[ ]:


class fullLspTrain(LspTrain):
    def __init__(self, dataset_path, h5_file_name, output_dir, batch_size,  learning_rate, epochs,
                tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units,
                attention_units, maxout_linear_units, max_dropout, optimizer_name,
                low_vram, depth_weight_path, skeleton_weight_path, rgb_weight_path=None):
        super().__init__(dataset_path, h5_file_name, output_dir, batch_size,  learning_rate, epochs,
                tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units,
                attention_units, maxout_linear_units, max_dropout, optimizer_name,
                low_vram)
        self.depth_weight_path = depth_weight_path
        self.skeleton_weight_path = skeleton_weight_path
        self.rgb_weight_path = rgb_weight_path
        
        self.train_data
        self.validation_data
        self.fe_rgb
        self.fe_depth
        self.fe_skeleton
    @scope
    def train_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        if self.low_vram:
            data = generators.genLSP10.initGenLSPlowVram("train", self.tx, self.ty, h5_file_path,
                                                         rgb=True, depth=True, skeleton=True).fromGenerator(batch_size=self.batch_size)
        else:
            data = generators.genLSP10.initGenLSP("train", self.tx, self.ty, h5_file_path,
                                                  rgb=True, depth=True, skeleton=True).fromGenerator(batch_size=self.batch_size)
        return data
    
    @scope
    def validation_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        if self.low_vram:
            data = generators.genLSP10.initGenLSPlowVram("validation", self.tx, self.ty, h5_file_path,
                                                         rgb=True, depth=True, skeleton=True).fromGenerator(batch_size=self.batch_size) 
        else:
            data = generators.genLSP10.initGenLSP("validation", self.tx, self.ty, h5_file_path,
                                                  rgb=True, depth=True, skeleton=True).fromGenerator(batch_size=self.batch_size)
        return data
    
    @scope
    def fe_rgb(self):
        if self.low_vram:
            return None
        model = build_rgbFeatureExtractor(weight_path=self.rgb_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    @scope
    def fe_depth(self):
        if self.low_vram:
            return None
        model =  build_depthFeatureExtractor(weight_path=self.depth_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    @scope
    def fe_skeleton(self):
        if self.low_vram:
            return None
        model = build_skeletonFeatureExtractor(weight_path=self.skeleton_weight_path, pooling_last_layer="avg")
        return model
    
    def inputLowVram(self, inputs):
        skeleton_rgb = tf.concat([np.expand_dims(inputs[2],axis=1), inputs[0]], axis=1)
        skeleton_depth = tf.concat([np.expand_dims(inputs[2],axis=1), inputs[1]], axis=1)
        return tf.concat([skeleton_rgb,skeleton_depth], axis=-1)
    
    def inputNoLowVram(self, inputs): 
        fe_rgb = self.fe_rgb(inputs[0])
        fe_depth = self.fe_depth(inputs[1])
        fe_skeleton = self.fe_skeleton(inputs[2])
        skeleton_rgb = tf.concat([fe_skeleton, fe_rgb], axis=1)
        skeleton_depth = tf.concat([fe_skeleton, fe_depth], axis=1)
        return tf.concat([skeleton_rgb,skeleton_depth], axis=-1)


# In[ ]:


class TransformToLowVram():
    def __init__(self, dataset_path, h5_file_name, output_dir,
                 rgb_weight_path, depth_weight_path, skeleton_weight_path):
        self.dataset_path = dataset_path
        self.h5_file_name = h5_file_name
        self.output_dir = output_dir
        self.depth_weight_path = depth_weight_path
        self.skeleton_weight_path = skeleton_weight_path
        self.rgb_weight_path = rgb_weight_path
        
        self.train_data
        self.validation_data
        self.fe_rgb
        self.fe_depth
        self.fe_skeleton
        
    @scope
    def train_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        data = generators.Lsp10ToLowRamH5("train", h5_file_path,
                                                  rgb=True, depth=True, skeleton=True).generateData()
        return data
    
    @scope
    def validation_data(self):
        h5_file_path = os.path.join(self.dataset_path,self.h5_file_name)
        data = generators.Lsp10ToLowRamH5("validation", h5_file_path,
                                                  rgb=True, depth=True, skeleton=True).generateData()
        return data
    
    @scope
    def fe_rgb(self):
        model = build_rgbFeatureExtractor(weight_path=self.rgb_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    @scope
    def fe_depth(self):
        model =  build_depthFeatureExtractor(weight_path=self.depth_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    @scope
    def fe_skeleton(self):
        model = build_skeletonFeatureExtractor(weight_path=self.skeleton_weight_path, pooling_last_layer="avg")
        return model
    
    def createInputs(self, file, path_dataset_h5,sign_name,stream):
        dataset=file[path_dataset_h5].create_dataset(sign_name,stream.shape, np.float32, compression="gzip")
        dataset[...]=stream
        
    def compressData(self):
        output_dir = os.path.join(self.output_dir,"LSP10C.h5")
        file=h5py.File(output_dir,'w')
        
        group=file.create_group('train')  
        group.create_group('rgb')    
        group.create_group('depth')
        group.create_group('skeleton')   

        group=file.create_group('validation')  
        group.create_group('rgb')    
        group.create_group('depth')
        group.create_group('skeleton')
        for i, ((r,d,s), tag) in enumerate(self.validation_data):
            print("compressing data for vald...{}".format(i), end='\r', flush=True)
            stream = self.fe_rgb(r)
            self.createInputs(file,'/validation/rgb/',tag,stream)
            stream = self.fe_depth(d)
            self.createInputs(file,'/validation/depth/',tag,stream)
            stream = self.fe_skeleton(s)
            self.createInputs(file,'/validation/skeleton/',tag,stream)
            
        for i, ((r,d,s), tag) in enumerate(self.train_data):
            print("compressing data for train...{}".format(i), end='\r', flush=True)
            stream = self.fe_rgb(r)
            self.createInputs(file,'/train/rgb/',tag,stream)
            stream = self.fe_depth(d)
            self.createInputs(file,'/train/depth/',tag,stream)
            stream = self.fe_skeleton(s)
            self.createInputs(file,'/train/skeleton/',tag,stream)
        file.close()

