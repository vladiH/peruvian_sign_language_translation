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
import os, sys
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
import matplotlib.pyplot as plt
from rawToHdf5 import LSP10, SkeletonLSP, DepthLSP
from mydecorator import scope
from math import floor, ceil

from utils import augmentPad
from preprocessing import resize, jointToRgb, changeLabels, depthPreprocessing, rgbPreprocessing


# In[ ]:


class genSkeleton:
    def __init__(self, dataset_name, h5_file_path=None, height=224, width=224, minimun=0, maximun=1, method="nearest"):
        self.h5_file_path = h5_file_path
        self.height = height
        self.width = width
        self.minimun =minimun
        self.maximun = maximun
        self.method = method
        self.dataset_name = dataset_name
        self.file
        self.inputs
    @scope
    def file(self):
        assert self.h5_file_path != None, "unspecified dataset path"
        return h5py.File(self.h5_file_path, 'r+')
    @scope
    def inputs(self):
        return self.file["{0}".format(self.dataset_name)]
    
    def closeFile(self):
        self.file.close()
    
    def walkOverData(self, name_data):
        images= None
        label = []
        for name in name_data:
            img = jointToRgb(self.inputs[name][()],self.minimun,self.maximun)
            if images is not None:
                images = np.vstack((images,resize(img, self.height, self.width ,method=self.method)))
            else:
                images = resize(img, self.height, self.width ,method=self.method)
            label.append(SkeletonLSP.getLabelToInt(name.split('_')[0], False))
        return images, label
    
    def generateData(self, batch_size):
        dataset_size = len(self.inputs)
        examples_name = np.array(self.inputs)
        np.random.shuffle(examples_name)
        vocabulary_size = len(SkeletonLSP.getLabelToInt("", vocabulary=True))
        batchs = np.arange(0, ceil(dataset_size/batch_size))
        np.random.shuffle(batchs)
        for batch in batchs:
            init = int(batch*batch_size)
            end = int(np.minimum((init+batch_size), dataset_size))
            imgs, labels = self.walkOverData(examples_name[init:end])
            labels = changeLabels(labels,vocabulary_size ,"categorical")
            yield imgs, labels
        
    def fromGenerator(self, batch_size):
        return tf.data.Dataset.from_generator(self.generateData, 
                                  (tf.float32, tf.float32),
                                  output_shapes=(tf.TensorShape([None,self.height,self.width,3]),
                                  tf.TensorShape([None,len(SkeletonLSP.getLabelToInt("", vocabulary=True))])),
                                  args= [batch_size])
    


# In[ ]:


class genDepth:
    def __init__(self, dataset_name, h5_file_path=None, height=224, width=224, crop=1., 
                 applyDist=True, minimun=-1, maximun=1, method="nearest", viridis=True, measure="cm"):
        self.h5_file_path = h5_file_path
        self.height = height
        self.width = width
        self.crop = crop
        self.applyDist = applyDist
        self.minimun =minimun
        self.maximun = maximun
        self.method = method
        self.viridis = viridis
        self.measure = measure
        self.dataset_name = dataset_name
        self.file
        self.inputs
        self.outputs
    @scope
    def file(self):
        assert self.h5_file_path != None, "unspecified dataset path"
        return h5py.File(self.h5_file_path, 'r+')
    
    @scope
    def inputs(self):
        return self.file["{0}/depth/depth_raw".format(self.dataset_name)]
    @scope
    def outputs(self):
        return self.file["{0}/label/label_raw".format(self.dataset_name)]
        
    def closeFile(self):
        self.file.close()

    #Function:generate data from .h5 with especific batch
    #batch_size:Int => 
    def generateData(self, batch_size):
        lenDataInput = self.inputs.shape[0]
        lenDataOutput = self.outputs.shape[0]
        assert lenDataInput ==lenDataOutput
        size_vocabulary = len(DepthLSP.getLabelToInt("", vocabulary=True))
        batchs = np.arange(0, ceil(lenDataInput/batch_size))
        np.random.shuffle(batchs)
        for batch in batchs:
            init = int(batch*batch_size)
            end = int(np.minimum((init+batch_size), lenDataInput))
            imgs = self.inputs[init:end,...][()]
            imgs = depthPreprocessing(imgs, height=self.height, width=self.width, crop=self.crop, applyDist=self.applyDist,
                                      min=self.minimun, max=self.maximun, method=self.method, viridis=self.viridis, measure=self.measure)
            labels = changeLabels(self.outputs[init:end],size_vocabulary ,"categorical")
            yield imgs, labels
    
    def fromGenerator(self, batch_size):
        return tf.data.Dataset.from_generator(self.generateData, 
                                  (tf.float32, tf.float32),
                                  output_shapes=(tf.TensorShape([None,224,224,3]),
                                  tf.TensorShape([None,len(DepthLSP.getLabelToInt("", vocabulary=True))])),
                                  args= [batch_size])



# In[ ]:


class genLSP10:
    def __init__(self, dataset_name, tx, ty, h5_file_path, height, width, minimun, maximun, method,
                 crop, apply_dist, viridis, measure, rgb, depth, skeleton, low_vram):
        self.dataset_name = dataset_name
        self.tx = tx
        self.ty = ty
        self.h5_file_path = h5_file_path
        self.height = height
        self.width = width
        self.minimun =minimun
        self.maximun = maximun
        self.method = method
        self.crop = crop
        self.apply_dist = apply_dist
        self.viridis = viridis
        self.measure = measure
        self.rgb = rgb
        self.depth = depth
        self.skeleton = skeleton
        self.low_vram = low_vram
        
        self.file
        self.inpRGB
        self.inpDepth
        self.inpSkeleton
            
    @classmethod
    def initGenLSP(cls, dataset_name, tx, ty, h5_file_path=None, height=224, width=224, minimun=0, maximun=1, method="nearest",
                 crop=1., apply_dist=True, viridis=True, measure="cm", rgb=True, depth=True, skeleton=True):
        return cls(dataset_name=dataset_name,
                  tx= tx, ty=ty,
                  h5_file_path= h5_file_path,
                  height=height, width=width, minimun=minimun,
                  maximun=maximun, method=method,crop=crop,apply_dist=apply_dist,
                  viridis=viridis, measure=measure, rgb=rgb, depth=depth, skeleton=skeleton, low_vram=False) 
    
    @classmethod
    def initGenLSPlowVram(cls, dataset_name, tx, ty, h5_file_path=None, rgb=True, depth=True, skeleton=True):
        return cls(dataset_name=dataset_name, ty=ty,
                  h5_file_path= h5_file_path, tx= tx,
                  height=0, width=0, minimun=0,
                  maximun=0, method="",crop=0,apply_dist=False,
                  viridis=False, measure="", rgb=rgb, depth=depth, skeleton=skeleton, low_vram=True) 
    
    @scope
    def file(self):
        assert self.h5_file_path != None, "unspecified dataset path"
        return h5py.File(self.h5_file_path, 'r+')
    
    @scope
    def inpRGB(self):
        return self.file["{0}/rgb/".format(self.dataset_name)]
    
    @scope
    def inpDepth(self):
        return self.file["{0}/depth/".format(self.dataset_name)]
    
    @scope
    def inpSkeleton(self):
        return self.file["{0}/skeleton/".format(self.dataset_name)]
        
    def closeFile(self):
        self.file.close()
    
    def batchRgb(self, name_data):
        rgb = None
        for name in name_data:
            stream = self.inpRGB[name][()]
            if not self.low_vram:
                stream = rgbPreprocessing(stream, 
                            height = self.height, width = self.width, crop = self.crop, mode="caffe", 
                            method = self.method)
            stream = augmentPad(stream, self.tx)
                
            if rgb is not None:
                rgb = np.concatenate((rgb, stream),axis=0)
            else:
                rgb = stream
                
        return rgb
    
    def batchDepth(self, name_data):
        depth = None
        for name in name_data:
            stream = self.inpDepth[name][()]
            if not self.low_vram:
                stream = depthPreprocessing(stream,
                            height=self.height, width=self.width, crop=self.crop, 
                            applyDist=self.apply_dist, min=self.minimun, max=self.maximun,
                            method=self.method, viridis=self.viridis, measure=self.measure)   
            stream = augmentPad(stream, self.tx)
            if depth is not None:
                depth = np.concatenate((depth,stream),axis=0)
            else:
                depth = stream
        return depth
    
    def batchSkeleton(self, name_data):
        skeleton = None
        for name in name_data:
            img = self.inpSkeleton[name][()]
            if not self.low_vram:
                img = jointToRgb(img, self.minimun, self.maximun)
                img = resize(img, self.height, self.width ,method=self.method)
            if skeleton is not None:
                skeleton = np.concatenate((skeleton, img),axis=0)
            else:
                skeleton = img
        return skeleton
        
    def batchLabel(self, name_data, vocabulary_size):
        batch_labels = list()
        vocab=list(map(lambda x:LSP10.getIntToSentences(LSP10.getSentencesToInt(x[0:x.rfind("_")])),name_data))
        for i in vocab:
            parse_vocab=i.split('_')
            vocab_int = list(map(lambda x: LSP10.getLabelToInt(x), parse_vocab))
            bos = [LSP10.getLabelToInt('<bos>')]
            vocab_int = bos + vocab_int
            vocab_int += [LSP10.getLabelToInt('<eos>')]
            if len(vocab_int) < self.ty:
                vocab_int += [LSP10.getLabelToInt('<pad>')] * (self.ty - len(vocab_int))
            batch_labels.append(vocab_int)
        return changeLabels(batch_labels,vocabulary_size ,"categorical")
       
    def getData(self, name_data, vocabulary_size):
        rgb, depth, skeleton, label = [None]*4
        if self.rgb:
            rgb = self.batchRgb(name_data)
        if self.depth:
            depth = self.batchDepth(name_data)
        if self.skeleton:
            skeleton = self.batchSkeleton(name_data)
        
        label = self.batchLabel(name_data, vocabulary_size)
        return rgb, depth, skeleton, label
            
    def generateData(self,batch_size):
        lenDataInput = len(self.inpRGB)
        name_data = list(self.inpRGB)
        np.random.shuffle(name_data)
        vocabulary_size = len(LSP10.getLabelToInt("", vocabulary=True))
        batchs = np.arange(0, np.ceil(lenDataInput/batch_size))
        np.random.shuffle(batchs)
        for batch in batchs:
            init = int(batch*batch_size)
            end = int(np.minimum((init+batch_size), lenDataInput))
            result = self.getData(name_data[init:end], vocabulary_size)
            yield result#(rgb,depth,skeleton,label)
            
            
    def fromGenerator(self, batch_size):
        rgb, depth, skeleton = [tf.TensorShape(None,)]*3
        if self.low_vram:
            if self.rgb:
                rgb = tf.TensorShape([None, self.tx, 2048])
            if self.depth:
                depth = tf.TensorShape([None, self.tx, 2048])
            if self.skeleton:
                skeleton = tf.TensorShape([None, 1, 2048])
                
            return tf.data.Dataset.from_generator(self.generateData, 
                                  (tf.float32, tf.float32, tf.float32, tf.float32),
                                  output_shapes=(rgb,depth,skeleton,
                                  tf.TensorShape([None, self.ty, len(LSP10.getLabelToInt("", vocabulary=True))])),
                                  args= [batch_size])
        else:
            if self.rgb:
                rgb = tf.TensorShape([None, self.tx, self.height,self.width,3])
            if self.depth:
                depth = tf.TensorShape([None, self.tx, self.height,self.width,3])
            if self.skeleton:
                skeleton = tf.TensorShape([None, self.height,self.width,3])
            
            return tf.data.Dataset.from_generator(self.generateData, 
                                  (tf.float32, tf.float32, tf.float32, tf.float32),
                                  output_shapes=(rgb,depth,skeleton,
                                  tf.TensorShape([None,self.ty, len(LSP10.getLabelToInt("", vocabulary=True))])),
                                  args= [batch_size])



# In[ ]:


class Lsp10ToLowRamH5:
    def __init__(self, dataset_name, h5_file_path=None, height=224, width=224, minimun=0, maximun=1, method="nearest",
                 crop=1., apply_dist=True, viridis=True, measure="cm", rgb=True, depth=True, skeleton=True):
        self.dataset_name = dataset_name
        self.h5_file_path = h5_file_path
        self.height = height
        self.width = width
        self.minimun =minimun
        self.maximun = maximun
        self.method = method
        self.crop = crop
        self.apply_dist = apply_dist
        self.viridis = viridis
        self.measure = measure
        self.rgb = rgb
        self.depth = depth
        self.skeleton = skeleton
        
        self.file
        self.inpRGB
        self.inpDepth
        self.inpSkeleton
    
    @scope
    def file(self):
        assert self.h5_file_path != None, "unspecified dataset path"
        return h5py.File(self.h5_file_path, 'r+')
    
    @scope
    def inpRGB(self):
        return self.file["{0}/rgb/".format(self.dataset_name)]
    
    @scope
    def inpDepth(self):
        return self.file["{0}/depth/".format(self.dataset_name)]
    
    @scope
    def inpSkeleton(self):
        return self.file["{0}/skeleton/".format(self.dataset_name)]
        
    def closeFile(self):
        self.file.close()
    
    def batchRgb(self, name_data):
        stream = self.inpRGB[name_data][()]
        stream = rgbPreprocessing(stream, 
                            height = self.height, width = self.width, crop = self.crop, mode="caffe", 
                            method = self.method)       
        return np.expand_dims(stream,axis=0)
    
    def batchDepth(self, name_data):
        stream = self.inpDepth[name_data][()]
        stream = depthPreprocessing(stream,
                            height=self.height, width=self.width, crop=self.crop, 
                            applyDist=self.apply_dist, min=self.minimun, max=self.maximun,
                            method=self.method, viridis=self.viridis, measure=self.measure)   
        return np.expand_dims(stream,axis=0)
    
    def batchSkeleton(self, name_data):
        img = self.inpSkeleton[name_data][()]
        img = jointToRgb(img, self.minimun, self.maximun)
        img = resize(img, self.height, self.width ,method=self.method)
        return img
       
    def getData(self, name_data):
        rgb, depth, skeleton = [None]*3
        if self.rgb:
            rgb = self.batchRgb(name_data)
        if self.depth:
            depth = self.batchDepth(name_data)
        if self.skeleton:
            skeleton = self.batchSkeleton(name_data)
        return rgb, depth, skeleton
            
    def generateData(self):
        name_data = list(self.inpRGB)
        np.random.shuffle(name_data)
        for name in name_data:
            result = self.getData(name)
            yield result, name


# In[ ]:




