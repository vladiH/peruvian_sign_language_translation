#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import itertools
import xml.etree.cElementTree as ET
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


def getDirNames(dir_input):
    dir_names = [dir_name for dir_name in os.listdir(dir_input) 
                if os.path.isdir(os.path.join(dir_input,dir_name))]
    dir_names.sort(key=lambda item: (len(item), item))
    return dir_names
    
def getFileNames(dir_input, ends_with=('.txt','.Jpg','.xml')):
    file_names = [file_name for file_name in os.listdir(dir_input) 
                 if (os.path.isfile(os.path.join(dir_input, file_name)) and file_name.endswith(ends_with))]
    file_names.sort(key=lambda item: (len(item), item))
    return file_names

def openTxt(file_dir, file_name):
    file_tags = os.path.join(file_dir, file_name)
    with open(file_tags, "r") as f:
        lines = f.read().splitlines()
    return lines

def augmentPad(stream, tx): # pad img = "<pad>"
    if stream.ndim == 4:
        if stream.shape[0] < tx:
            zeros = np.zeros(stream.shape[1:],dtype=np.float32)
            zeros = np.expand_dims(zeros,axis = 0)
            zeros = np.repeat(zeros,(tx-stream.shape[0]),axis=0)
            stream = np.concatenate((stream,zeros), axis =0)
        else:
            stream = stream[0:tx,...]
        return np.expand_dims(stream,axis=0)
    elif stream.ndim == 3:
        if stream.shape[1] < tx:
            zeros = np.zeros((1,stream.shape[2]),dtype=np.float32)
            zeros = np.expand_dims(zeros,axis = 0)
            zeros = np.repeat(zeros,(tx-stream.shape[1]),axis=1)
            stream = np.concatenate((stream, zeros), axis =1)
        else:
            stream = stream[:,:tx,...]
        return stream
    else:
        raise ValueError("stream have {} dimensions insteand of 3 or 4".format(stream.ndim))
        

def augmentPadLowVram(stream, tx): # pad img = "<pad>"
    assert stream.ndim == 3, "stream have {} dimensions insteand of 3".format(stream.ndim)
    

def loadDepthFile(path_input_file):
    tree = ET.parse(path_input_file)
    filename, _ = os.path.splitext(Path(path_input_file).name)
    elem = tree.find('%s/data' % filename) #busca la etiqueta "data" dentro del xml
    height= int(tree.find('%s/height' % filename).text) #busca la etiqueta "alto" dentro del xml
    width= int(tree.find('%s/width' % filename).text) #busca la etiqueta "ancho" dentro del xml
    strData = elem.text
    floatData = list(map(lambda x: np.int16(x), strData.split()))
    return np.array(floatData).reshape((height, width,1))

def loadRGBFile(path_input_file):
    img = cv2.imread(path_input_file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def loadSkeletonFile(path_input_file):
    elem=open(path_input_file, "r") 
    strData = elem.read()
    fixed=strData.split()
    floatData = list(map(lambda x: np.float32(x.replace(',','.')), fixed))
    return np.array(floatData)[0:60]#gracias_u.53 and gracias_u.54 have 120 coordinates, so that we select explicity this range dor avoiding some error


# In[ ]:


class confusionMatrix():
    def __init__(self):
        self._targ = []
        self._pred = []
        
    def makeImgDir(self, output_dir):
        img_dir = os.path.join(output_dir, "img")
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        return img_dir

    @property
    def targets(self):
        return self._targ
    
    @property
    def predicteds(self):
        return self._pred
    
    @targets.setter
    def targets(self, one_hot):
        one_hot = np.array(one_hot)
        self._targ.extend(np.argmax(one_hot,axis=-1))
    
    @predicteds.setter
    def predicteds(self, one_hot):
        one_hot = np.array(one_hot)
        self._pred.extend(np.argmax(one_hot,axis=-1))
    
    def restore(self):
        self._targ = []
        self._pred = []
        
    def plot_confusion_matrix(self, cm, tags, output_dir, normalize=False, title="Confusion Matrix",cmap=plt.cm.Blues):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_mark = np.arange(len(tags))
        plt.xticks(tick_mark, tags, rotation=80)
        plt.yticks(tick_mark,tags)

        if normalize:
            cm = np.around((cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]),3)
        thresh =cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                    color="white" if cm[i, j]> thresh else "black")

        plt.tight_layout()
        plt.ylabel("true label")
        plt.xlabel('predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(os.path.join(output_dir,'confusion.png'), bbox_inches='tight')
        plt.close()
        
    def metrics(self, cm, tags):
        label = ['','Precision', 'Recall', 'F1 score', '#Data']
        footer = ['Avg/total','', '', '', '']
        precision = []
        exhaustividad = []
        f1 = []
        total = []
        for x in range(cm.shape[0]):
            tp = cm[x,x]
            fn = np.sum(cm[:x,x]) + np.sum(cm[x,x+1:])
            fp = np.sum(cm[x,:x]) + np.sum(cm[x+1:,x])
            prec = round(tp/(tp+fp+1e-07),3)
            rec = round(tp/(tp+fn+1e-07),3)
            precision.append(prec)
            exhaustividad.append(rec)
            f1.append(round(2*((prec*rec)/(prec+rec+1e-07)),2))
            total.append(np.sum(cm[x,:]))

        precision = np.reshape(precision,(-1,1))
        exhaustividad = np.reshape(exhaustividad,(-1,1))
        f1 =  np.reshape(f1,(-1,1))
        total =  np.reshape(total,(-1,1))

        a = np.concatenate((precision, exhaustividad, f1, total), axis=-1)
        for i in range(1,len(footer)):
            if i<len(footer)-1:
                footer[i] = round(np.sum(a[:,i-1])/(a.shape[0]),3)
            else:
                footer[i] = round(np.sum(a[:,i-1]),0)
        a = np.concatenate((np.reshape(tags,(-1,1)),a), axis=1)
        a = np.concatenate((np.reshape(label,(1,-1)),a), axis=0)
        a = np.concatenate((a,np.reshape(footer,(1,-1))), axis=0)
        return a
    
    def render_mpl_table(self, data_frame, output_dir, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#2A6283', row_colors=['#91D3FC', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
        
        if ax is None:
            size = (np.array(data_frame.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data_frame.values, bbox=bbox, colLabels=data_frame.columns, rowLabels=data_frame.index,**kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in  mpl_table.get_celld().items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
        fig.savefig(os.path.join(output_dir,'metrics.png'), bbox_inches='tight')
        plt.close()
    
    def saveFigure(self, output_dir, tags, normalize=False, title="Confusion Matrix"):
        output_dir = self.makeImgDir(output_dir)
        cm = confusion_matrix(self.targets, self.predicteds)
        self.plot_confusion_matrix(cm, tags, output_dir, normalize = normalize, title=title)
        metrics=self.metrics(cm, tags)
        data = np.array(metrics)   
        data_frame = pd.DataFrame(data=data[1:,1:], index=data[1:,0], columns=data[0,1:])
        self.render_mpl_table(data_frame, output_dir, header_columns=0, col_width=2.0)





