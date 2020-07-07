#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import multiprocessing as mp
from data.rawToHdf5 import LSP10
from data.utils import*
from data.preprocessing import*
from data.mydecorator import scope
from model.builders import*


# In[ ]:


class Test():
    def makeParallel(self, type_data, labels, blocks=True):
        if type_data == "rgb":
            loader = loadRGBFile
        if type_data == "depth":
            loader = loadDepthFile
        if type_data == "skeleton":
            loader = loadSkeletonFile
        cores = mp.cpu_count()
        if blocks:
            with mp.Pool(processes=cores) as pool:
                data = pool.map(loader, labels)
            pool.close()
            return data
        else:
            return mp.Pool(cores).imap(loader, labels)    
        
    def loadInputData(self, input_dir, type_data, end_with):
        file_name = getFileNames(input_dir, ends_with=(end_with))
        dir_file_name = [os.path.join(input_dir, name)for name in file_name]
        return self.makeParallel(type_data, dir_file_name, blocks=True)


# In[ ]:


class evaluateRgbLSP(Test):
    def __init__(self,lstm_weight_path, rgb_weight_path, tx, ty, enc_units, enc_dropout, recurrent_dropout, dec_units, attention_units,
             maxout_linear_units, max_dropout):
        self.tx = tx
        self.ty = ty
        self.enc_units = enc_units
        self.enc_dropout = enc_dropout
        self.recurrent_dropout = recurrent_dropout
        
        self.dec_units = dec_units
        self.attention_units = attention_units
        self.maxout_linear_units = maxout_linear_units
        self.max_dropout = max_dropout
        
        self.rgb_weight_path = rgb_weight_path
        self.lstm_weight_path = lstm_weight_path
        self.labelToInt= LSP10.getLabelToInt("",vocabulary=True) 
        self.intToLabel = LSP10.getIntToLabel(self.labelToInt, None, reverse=True)
        self.vocabulary_size = len(self.labelToInt)
        self.fe_rgb
        self.encoder_decoder
    @scope
    def fe_rgb(self):
        model = build_rgbFeatureExtractor(weight_path=self.rgb_weight_path, pooling_last_layer="avg")
        return tf.keras.layers.TimeDistributed(model)
    @scope
    def encoder_decoder(self):
        return build_encoder_decoderLSTM(self.lstm_weight_path, enc_units=self.enc_units, dec_units=self.dec_units,
                                         dropout = self.enc_dropout, recurrent_dropout = self.recurrent_dropout,
                                         vocabulary_size=self.vocabulary_size, attention_units=self.attention_units,
                                         maxout_linear_units=self.maxout_linear_units, max_dropout=self.max_dropout,
                                         merge="concat")
   
    def initStates(self):
        #1 means only for one batch
        enc_states = self.encoder_decoder[0].initial_state(1,self.enc_units)
        dec_states = self.encoder_decoder[1].initial_state(1,self.dec_units)
        return enc_states, dec_states
    
    def preprocessing(self, inputs):
        return rgbPreprocessing(inputs)
    
    def featureExtractor(self, inputs):
        output = self.fe_rgb(inputs)
        return output
    
    def inference(self, inputs):
        inputs = self.preprocessing(inputs)
        inputs = np.expand_dims(inputs, axis = 0)
        inputs = self.featureExtractor(inputs)
        enc_states, dec_states = self.initStates()
        enc_output, _ = self.encoder_decoder[0](inputs, enc_states)
        result = ''
        dec_input = changeLabels([self.labelToInt["<bos>"]], self.vocabulary_size ,"categorical")
        for t in range(1, self.ty):
            predictions, dec_states, attention = self.encoder_decoder[1](dec_input, dec_states, [enc_output])
            predicted_id = tf.argmax(predictions, axis=-1).numpy()
            label = self.intToLabel[predicted_id[0]]
            result += label + ' '
            if label == '<eos>':
                return result
            dec_input = changeLabels(predicted_id,self.vocabulary_size, op="categorical")
        return result
    
    def predictions(self, input_dir, type_data, end_with):
        inputs = np.array(self.loadInputData(input_dir, type_data, end_with), dtype=np.float32)
        print("The model inference is: ", self.inference(inputs))
        

