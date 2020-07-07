
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
import numpy  as np
import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input


# In[ ]:


def resize(img,h,w, method="nearest"):
    return tf.image.resize(img, size=(h,w), method=method)


# In[ ]:


def rgbPreprocessing(img, height=224, width=224, crop=1., mode="caffe", method="nearest"):
    assert img.ndim == 4, "Rank should be 4 dims"
    if crop!=1.:
        img = tf.image.central_crop(img, crop)
    shape = img.shape
    if shape[1]!=height or shape[2]!=width:
        img = resize(img, height, width, method)
    img = preprocess_input(img, data_format ='channels_last', mode=mode)
    return img


# In[ ]:


def distance(mat, measure="cm"):
    #assert tf.rank(mat).numpy() == 4, "Rank should be 4 dims"
    r1 = tf.zeros(mat.shape)
    mat = tf.where(mat<0, r1, mat)
    mat = mat/8.
    for i in tf.range(0,13):
        r1 = r1 + tf.cast(tf.pow(2,i), dtype=tf.float32)*tf.math.mod(mat,2)
        mat = mat/2
    if measure == "cm":
        return r1/10
        #centimeters
    if measure == "m":
        return r1/1000
        #meters
    else:
        return r1
        #milimeters


# In[ ]:


def escaleData(mat, a=None, b = None):
    #assert tf.rank(mat).numpy() == 4, "Rank should be 4 dims"
    if a !=None and b !=None:
        minItem = tf.reduce_min(mat,axis=[1,2,3], keepdims=True)
        maxItem = tf.reduce_max(mat, axis=[1,2,3], keepdims=True)
        #[a,b]->(b-a)((m-min)/(max-min))+a
        tmpMat = (b-a)*((mat - minItem) / (maxItem - minItem))+a # scale to [-1,1]
        return tmpMat
    else:
        return mat


# In[ ]:


def viridisMap(mat):
    #assert tf.rank(mat).numpy() == 4, "Rank should be 4 dims"
    cmap=plt.cm.viridis
    im=cmap(mat)
    return tf.squeeze(im,axis=3)[:,:,:,:3]


# In[ ]:


def depthPreprocessing(mat, height=224, width=224, crop=1., applyDist=False, min=0, max=1, method="nearest", viridis=False, measure="cm"):
    #assert tf.rank(mat).numpy() == 4, "Rank should be 4 dims"
    if applyDist:
        mat = distance(mat, measure)
    mat = escaleData(mat, min, max)
    if viridis:
        mat = viridisMap(mat)
    if crop!=1.:
        mat = tf.image.central_crop(img, crop)
    shape = mat.shape
    if shape[1]!=height or shape[2]!=width:
        mat = resize(mat,height,width, method)
    return mat


# In[ ]:


def changeLabels(labels,numClass, op="categorical"):
    if op=="categorical":
        return tf.one_hot(labels, numClass)


# In[ ]:


def jointToRgb(frame, min, max):
    r =  []
    g =  []
    b =  []
    for i in range(0,frame.shape[1],3):
        r.append(frame[:,i])
        g.append(frame[:,i+1])
        b.append(frame[:,i+2])
    x = np.array(list(filter(lambda a: a.all() != 0, r)))
    y = np.array(list(filter(lambda a: a.all() != 0, g)))
    z = np.array(list(filter(lambda a: a.all() != 0, b)))
    
    #normalize r,g ,b in to 0 to 1
    s = np.expand_dims(np.dstack((x,y,z)),axis=0)
    s =  escaleData(s, min, max)
    #r = escaleData(x, min, max)
    #g = escaleData(y, min, max)
    #b = escaleData(z, min, max)
    
    #repeat row m times m = frames/point
    i_c = np.ceil(224/s.shape[2])
    i_r = np.ceil(224/s.shape[1])
    #set type unsigned integer 
    s = np.repeat(np.repeat(s,i_r, axis=1),i_c, axis=2)
    #g = np.repeat(np.repeat(g,i_r, axis=0),i_c, axis=1)
    #b = np.repeat(np.repeat(b,i_r, axis=0),i_c, axis=1)
    
    return s


# In[ ]:


#preprocessing depth
#def skeletonPreprocessing(frames, height=224, width=224, min=0, max=1):
    #img = jointToRgb(frames, min, max)
    #img = resize(img,height, width, method="nearest")
    #return img

