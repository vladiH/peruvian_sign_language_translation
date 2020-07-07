#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ResNet50, EncoderLSTM, DecoderLSTM, restoreEncoderDecoderWeight


# In[ ]:


def changeState(model, trainable=False):
    for layers in model.layers:
        layers.trainable = trainable
    return model


# In[ ]:


def build_rgbResnet50(weight_path, output_units):
    assert output_units >0, "output_units must be more than 0!!"
    model = ResNet50(output_units, pooling=None, include_top=True)
    model._name = 'rgbResnet50'
    model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="rgb")
    return model


# In[ ]:


def build_depthResnet50(weight_path, output_units):
    assert output_units >0, "output_units must be more than 0!!"
    model = ResNet50(output_units, pooling=None, include_top=True)
    model._name = 'depthResnet50'
    model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="depth")
    return model


# In[ ]:


def build_skeletonResnet50(weight_path, output_units):
    assert output_units >0, "output_units must be more than 0!!"
    model = ResNet50(output_units, pooling=None, include_top=True)
    model._name = 'skeletonResnet50'
    model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="skeleton")
    return model


# In[ ]:


def build_rgbFeatureExtractor(weight_path, pooling_last_layer):
    if pooling_last_layer in ("avg", "max", None):
        model = ResNet50(0, pooling=pooling_last_layer, include_top=False)
        model._name = 'rgbResnet50'
        model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="rgb")
        model = changeState(model,False)
        return model
    else:
        raise ValueError("pooling_last_layer must be avg, max or None")


# In[ ]:


def build_depthFeatureExtractor(weight_path, pooling_last_layer):
    if pooling_last_layer in ("avg", "max", None):
        model = ResNet50(0, pooling=pooling_last_layer, include_top=False)
        model._name = 'depthResnet50'
        model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="depth")
        model = changeState(model,False)
        return model
    else:
        raise ValueError("pooling_last_layer must be avg, max or None")


# In[ ]:


def build_skeletonFeatureExtractor(weight_path, pooling_last_layer):
    if pooling_last_layer in ("avg", "max", None):
        model = ResNet50(0, pooling=pooling_last_layer, include_top=False)
        model._name = 'skeletonResnet50'
        model.loadWeight(weightPath=weight_path, height=224, width=224, channels=3, model_name="skeleton")
        model = changeState(model,False)
        return model
    else:
        raise ValueError("pooling_last_layer must be avg, max or None")

# In[ ]:


def build_encoderLSTM(weight_path, units=500, dropout = .5, recurrent_dropout = .5, merge = "concat"):
    model = EncoderLSTM(units=units, dropout = dropout, recurrent_dropout = recurrent_dropout,
                return_sequences = True, return_state = False, merge = merge)
    model._name = 'encoderLSTM'
    model.loadWeight(weightPath=weight_path)
    return model



# In[ ]:


def build_decoderLSTM(weight_path, units=900, vocabulary_size=23, 
                      attention_units=64, maxout_linear_units=5, max_dropout=.3):
    model = DecoderLSTM(dec_units= units, vocab_size=vocabulary_size, 
                            att_units=attention_units, maxout_units=maxout_linear_units, max_dropout = max_dropout)
    model._name = 'decoderLSTM'
    model.loadWeight(weightPath=weight_path)
    return model
        
# In[ ]:


def build_encoder_decoderLSTM(weight_path, enc_units, dec_units, dropout, recurrent_dropout,
                          vocabulary_size, attention_units, maxout_linear_units, max_dropout, merge="concat"):
    encoder_model = EncoderLSTM(units=enc_units, dropout = dropout, recurrent_dropout = recurrent_dropout,
                return_sequences = True, return_state = False, merge = merge)
    
    decoder_model = DecoderLSTM(dec_units=dec_units, vocab_size=vocabulary_size, 
                            att_units=attention_units, maxout_units=maxout_linear_units, max_dropout = max_dropout)
    encoder_model._name = 'encoderLSTM'
    decoder_model._name = 'decoderLSTM'
    restoreEncoderDecoderWeight(weight_path, encoder_model,decoder_model)
    return (encoder_model, decoder_model)


# In[ ]: