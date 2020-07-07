#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys,inspect
import numpy  
import h5py
import random
import multiprocessing as mp
from math import floor, ceil
if '__file__' in globals():
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import*
from mydecorator import scope


# In[ ]:


class File():
    #params:(string, string, string, float)
        #dir_inp: directory name which contain our raw data for train and validation
        #dir_out: directory name where our preprocessed dataset is saved (hdf5)
        #output_name_h5data: output name for peprocessed dataset (.h5)
        #train_percentage: percentage used for train and the remaining is for test
    def __init__(self, dir_inp, dir_out, output_name_h5data, train_percentage):
        self.dir_inp = dir_inp
        self.dir_out = dir_out
        self.output_name_h5data = output_name_h5data
        self.train_percentage = train_percentage
        self.createFile
        
    @scope
    def createFile(self):
        path = os.path.join(self.dir_out,self.output_name_h5data)
        if os.path.exists(path):
            op = input("press y/n if you want to delete the {} file".format(path))
            if op == "y":
                os.remove(path)
            if op == "n":
                raise Exception("please, change your out_nameH5_data")
        file = h5py.File(path,'w')
        return file
    
    def closeFile(self):
        self.createFile.close()
    
    def splitDataset(self, number_examples, number_sign):
        aux = floor((number_examples*(1-self.train_percentage))/number_sign)
        validation_length = aux*number_sign
        train_length = number_examples - validation_length
        return train_length, validation_length
        
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


# In[ ]:


class LSP10(File):
    def __init__(self, dir_inp, dir_out, output_name_h5data="LSP10.h5", train_percentage=0.833):
        super().__init__(dir_inp, dir_out, output_name_h5data, train_percentage)
        
    @staticmethod    
    def getFileNameLSP(sign_dir, end_with, beginFrm, endFrm) :
        assert sign_dir
        assert end_with
        assert beginFrm
        assert endFrm
        assert (int(beginFrm) < int(endFrm)), "beginFrm should be less than endFrm"
        file_names = getFileNames(sign_dir, ends_with=(end_with))
        if end_with not in (".txt",".Jpg",".xml"):
            raise ValueError("end_with must be .txt, .xml or .Jpg")
        if end_with == ".txt":
            beginFrmFile = "%s.txt" %beginFrm
            endFrmFile = "%s.txt" %endFrm
        if end_with == ".xml":
            beginFrmFile = "depthImg%s.xml" %beginFrm
            endFrmFile = "depthImg%s.xml" %endFrm
        if end_with == ".Jpg":
            beginFrmFile = "%s.Jpg" %beginFrm
            endFrmFile = "%s.Jpg" %endFrm
            
        beginIdx = file_names.index(beginFrmFile)
        endIdx = file_names.index(endFrmFile)
        return file_names[beginIdx:endIdx+1]

    def count(self, input_list):
        data  = {}
        for i in range(1, len(input_list),11):
            user = input_list[i-1] #take head
            for j in range(i, 10+i):
                line = input_list[j]
                splitIdx = line.index(":")
                sign = line[:splitIdx]#get label
                init_end = line[splitIdx+1:].split() # get init and end
                sign_init_end = (sign ,init_end[0],init_end[1])#AYUDAME, begin, end
                if user in data:
                    data[user] += [sign_init_end]
                else:
                    data[user] = [sign_init_end]
        tuple_items = list(data.items())
        random.shuffle(tuple_items)
        data = dict(tuple_items)
        return data
    
    def contentSelection(self, data_type, dir_inp, init, end):
        if data_type == "rgb":
            file_names = self.getFileNameLSP(dir_inp, ".Jpg", beginFrm=init, endFrm=end)
            labels = [os.path.join(dir_inp,name)for name in file_names]
            print("Load data from {}, examples:{}".format(dir_inp,len(labels)), end="\r")
            stream = self.makeParallel(type_data='rgb', labels = labels, blocks=True)
        if data_type == "depth":
            file_names = self.getFileNameLSP(dir_inp, ".xml", beginFrm=init, endFrm=end)
            labels = [os.path.join(dir_inp,name)for name in file_names]
            print("Load data from {}, examples:{}".format(dir_inp,len(labels)), end="\r")
            stream = self.makeParallel(type_data='depth', labels = labels, blocks=True)
        if data_type == "skeleton":
            file_names = self.getFileNameLSP(dir_inp, ".txt", beginFrm=init, endFrm=end)
            labels = [os.path.join(dir_inp,name)for name in file_names]
            print("Load data from {}, examples:{}".format(dir_inp,len(labels)), end="\r")
            stream = self.makeParallel(type_data='skeleton', labels = labels, blocks=True)
        return stream
            
    def createGroupInFile(self, group_name):
        group= self.createFile.create_group(group_name)
        group.create_group('rgb')    
        group.create_group('depth')
        group.create_group('skeleton')
        
    def addIntoGroupFromFile(self,group_name,subgroup_name, stream_name,stream):
        if subgroup_name in ("rgb", "depth", "skeleton"):
            group = group_name+"/"+subgroup_name
            array = np.array(stream)
            dataset = self.createFile[group].create_dataset(stream_name, array.shape, np.float32, compression="gzip")
            dataset[...]=array
        else:
            raise ValueError("{} should be rgb, depth or skeleton".format(subgroup_name))
        
    @staticmethod
    def getLabelToInt(value, vocabulary=False):
        vocab ={'ayudame':0,'como':1,'cual':2,'disculpame':3,'donde':4,'entiendo':5,'es':6,'estas':7,'porfavor':8,'gracias':9,'haces':10,
            'hasta':11,'hola':12,'manana':13,'no':14,'nombre':15,'que':16,'tu':17,'vives':18, '?':19,'<pad>':20, '<bos>':21, '<eos>':22}
        if vocabulary:
            return vocab
        return vocab[value]
    
    @staticmethod        
    def getIntToLabel(vocabulary, value=None, reverse=False):
        vocab = {v:k for k,v in vocabulary.items() }
        if reverse:
            return vocab
        else:
            return vocab[value]
    
    @staticmethod
    def getSentencesToInt(string, length=False):
        vocab ={'ayudame':0,'por_favor':1,'disculpame':2,'cual_es_tu_nombre':3,
            'donde_vives_tu':4,'no_entiendo':5,'que_haces_tu':6,
            'hola_como_estas_tu':7,'gracias':8,'hasta_manana':9}
        if length:
            return len(vocab)
        return vocab[string]
    
    #getIntToLabel: return the gesture name which match with the parameter
        #params:(int)
            #integer: gesture number 
    @staticmethod
    def getIntToSentences(integer, length=False):
        vocab ={0:'ayudame', 1:'porfavor', 2:'disculpame',3:'cual_es_tu_nombre_?',
            4:'donde_vives_tu_?',5:'no_entiendo',6:'que_haces_tu_?',
            7:'hola_como_estas_tu_?',8:'gracias',9:'hasta_manana'}
        if length:
            return vocab
        return vocab[integer]
    
    
    def convertData(self,tag_file_dir, tag_file_name="label.txt"):
        if (tag_file_dir is not None) and (tag_file_name is not None):
            open_tag_file = openTxt(file_dir=tag_file_dir, file_name=tag_file_name)
            
            total_tag = self.count(open_tag_file) #return dict
            total_examples = sum(list(map(lambda x:len(x),list(total_tag.values()))))#total data
            sentences_size = self.getSentencesToInt("", length=True)
            total_train, total_validation = self.splitDataset(total_examples, sentences_size)
    
            print("#data for train:{0}, #data for test:{1}".format(total_train,total_validation))
            self.createGroupInFile("train")
            self.createGroupInFile("validation")
            validation_exm_by_sign = total_validation/sentences_size
            for i, (key, val) in enumerate(total_tag.items()):
                np.random.shuffle(val)
                for name_init_end in val:   
                    for data_type in ['rgb','depth','skeleton']:
                        if data_type == "rgb":
                            sign_dir_name = name_init_end[0]+"_"+"r.{}".format(key.split("_")[1])
                        if data_type == "depth":
                            sign_dir_name = name_init_end[0]+"_"+"d.{}".format(key.split("_")[1])
                        if data_type == "skeleton":
                            sign_dir_name = name_init_end[0]+"_"+"u.{}".format(key.split("_")[1])    
                        dir_inp = os.path.join(os.path.join(self.dir_inp, data_type),sign_dir_name)
                        stream = self.contentSelection(data_type, dir_inp, name_init_end[1], name_init_end[2]) 
                        sign_dir_name = name_init_end[0]+"_"+"{}".format(key.split("_")[1]) 
                        group_name = "train"
                        if i < validation_exm_by_sign:
                            group_name = "validation"
                        self.addIntoGroupFromFile(group_name,data_type, sign_dir_name,stream)
            self.closeFile()
            print("", end="\r")
            print("process completed!!!")
        
        else:
            raise ValueError("tag_file_dir or tag_file_name is None")


# In[ ]:


class SkeletonLSP(File):
    def __init__(self, dir_inp, dir_out, output_name_h5data="skeletonLSP.h5", train_percentage=0.90):
        super().__init__(dir_inp, dir_out, output_name_h5data, train_percentage)
    
    def count(self, input_list):
        data  = {}
        for line in input_list: 
            label_range = line.index("_")
            label = line[:label_range]
            if label in data:
                data[label] += [line]
            else:
                data[label] = [line]
        return data
    
    def createGroupInFile(self, group_name):
        self.createFile.create_group(group_name)    
        
    def addIntoGroupFromFile(self,group_name,stream_name,stream):
        array = np.array(stream)
        dataset = self.createFile[group_name].create_dataset(stream_name, array.shape, np.float32,compression="gzip")
        dataset[...]=array
        
    @staticmethod
    def getLabelToInt(value, vocabulary=False):
        vocab ={'iarriba':0, 'darriba':1, 'aarriba':2, 'iadelante':3, 'dadelante':4, 'aadelante':5, 'ii':6, 
           'di':7, 'ai':8, 'id':9, 'dd':10, 'ad':11, 'icabeza':12, 'dcabeza':13,
                   'acabeza':14, 'iboca':15, 'dboca':16, 'aboca':17, 'icentro':18, 'dcentro':19, 'acentro':20}
        if vocabulary:
            return vocab
        return vocab[value]
    
    @staticmethod        
    def getIntToLabel(vocabulary, value):
        return {v:k for k,v in vocabulary.items() }[value]
    
    def convertData(self):
        dir_names = getDirNames(self.dir_inp)
        total_tag = self.count(dir_names) #return dict
        vocabulary_size = len(self.getLabelToInt("",True))
        total_train, total_validation = self.splitDataset(len(dir_names), vocabulary_size)
        
        print("#data for train:{0}, #data for test:{1}".format(total_train,total_validation))
        self.createGroupInFile("train")
        self.createGroupInFile("validation")
        
        validation_exm_by_sign = total_validation/vocabulary_size
        
        for _, val in total_tag.items():
            np.random.shuffle(val)
            for i, dir_name in enumerate(val):            
                dir_inp = os.path.join(self.dir_inp, dir_name)
                txt_file = getFileNames(dir_inp, ends_with=(".txt"))
                labels = [os.path.join(dir_inp,name)for name in txt_file]
                print("Load data from {}, examples:{}".format(dir_name,len(labels)), end="\r")
                stream = self.makeParallel(type_data='skeleton', labels = labels, blocks=True)
                group_name = "train"
                if i < validation_exm_by_sign:
                    group_name = "validation"
                self.addIntoGroupFromFile(group_name,stream_name=dir_name,stream=stream)
        self.closeFile()
        print("", end="\r")
        print("process completed!!!")


# In[ ]:


class DepthLSP(File):
    def __init__(self, dir_inp, dir_out, output_name_h5data="depthLSP.h5", train_percentage=0.96):
        super().__init__(dir_inp, dir_out, output_name_h5data, train_percentage)
     
    #count: return dict
        #params: (array)
            #input_list: labels.txt loaded on memory(Array)
    def count(self, input_list):
        data  = {}
        for line in input_list: 
            label_range = line.index(":")
            label = line[:label_range]
            init,end = line[label_range+1:].split() # Genera el arreglo de inicio y fin
            if label in data:
                data[label] += ["depthImg%s.xml" %xmlNumber for xmlNumber in np.arange(int(init), int(end)+1)]
            else:
                data[label] = ["depthImg%s.xml" %xmlNumber for xmlNumber in np.arange(int(init), int(end)+1)]
        return data
    
    def createGroupInFile(self, group_name, shape):
        group= self.createFile.create_group(group_name)    
        group.create_group('depth')
        group.create_group('label')
        
        self.createFile[group_name+'/depth'].create_dataset('depth_raw',shape, np.float32, compression="gzip")
        self.createFile[group_name+'/label'].create_dataset('label_raw',(shape[0],), np.uint8, compression="gzip")
        
    @staticmethod
    def getLabelToInt(value, vocabulary=False):
        vocab ={'pos1':0, 'pos2':1, 'pos3':2, 'pos4':3, 'pos5':4, 'pos6':5, 'pos7':6, 
           'pos8':7, 'pos9':8, 'pos10':9, 'pos11':10, 'pos12':11, 'pos13':12, 'pos14':13}
        if vocabulary:
            return vocab
        return vocab[value]
    
    @staticmethod        
    def getIntToLabel(vocabulary, value):
        return {v:k for k,v in vocabulary.items() }[value]
    
    def convertData(self, tag_file_dir, tag_file_name="label.txt"):
        if (tag_file_dir is not None) and (tag_file_name is not None):
            open_tag_file = openTxt(file_dir=tag_file_dir, file_name=tag_file_name)
            
            total_tag = self.count(open_tag_file) #return dict
            total_examples = sum(list(map(lambda x:len(x),list(total_tag.values()))))#total data
            vocabulary_size = len(total_tag)
            total_train, total_validation = self.splitDataset(total_examples, vocabulary_size)
            
            poss_train = random.sample(range(total_train), total_train)#index where an image will be placed
            poss_validation = random.sample(range(total_validation), total_validation)#index where an image will be placed
            print("#data for train:{0}, #data for test:{1}".format(total_train,total_validation))
            
            self.createGroupInFile("train", (total_train,480,640,1))
            self.createGroupInFile("validation", (total_validation,480,640,1))
            
            i = 0
            j = 0
            validation_exm_by_sign = total_validation/vocabulary_size
            for key, val in total_tag.items():
                val = [os.path.join(self.dir_inp,name)for name in val]
                key = self.getLabelToInt(key, False)
                for k, data in enumerate(self.makeParallel(type_data="depth", labels=val, blocks=False)):
                    if (k +1) > validation_exm_by_sign:
                        print("Load data for train..img :{}, examples:{}".format(key,i), end="\r")
                        self.createFile['train/label/label_raw'][poss_train[i],...]=int(key)
                        self.createFile['train/depth/depth_raw'][poss_train[i],...]=data
                        i +=1
                    else:
                        print("Load data for validation.. img :{}, examples:{}".format(key, j), end="\r")
                        self.createFile['validation/label/label_raw'][poss_validation[j],...]=int(key)
                        self.createFile['validation/depth/depth_raw'][poss_validation[j],...]=data
                        j +=1           
            self.closeFile()
            print("", end="\r")
            print("process completed!!!")
        else:
            raise ValueError("tag_file_dir or tag_file_name is None")


# In[ ]:




