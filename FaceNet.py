from test_fddb import detect
import align

import numpy as np
import os
import cv2 
from PIL import Image
import mxnet as mx

from sklearn import preprocessing

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.score = None # Score of bbox
        self.id = None # index of faces
        self.image = None
        self.container_image = None
        self.embedding = None

class Recognition:
    def __init__(self, feature_path):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier_IMDB(feature_path)

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i,face in enumerate(faces):       
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)
            
        return faces


class Detection():

    def find_faces(self,image):
        faces = []
        aligned_list,bboxes,scores = align.Align(image)
        if aligned_list and bboxes:
            index = 1
            for bb,cropped_face,score in zip(bboxes,aligned_list,scores):
                face = Face()
                face.container_image = image
                face.bounding_box = np.zeros(4, dtype=np.int32)
                for i in range(0,4):
                    face.bounding_box[i] = bb[i]
                face.score = score
                face.id = index
                index = index + 1
                # new_image = np.transpose(cropped_face, (1, 2, 0))[:, :, ::-1]
                # out = Image.fromarray(new_image)
                # out = out.resize((112, 112))
                # out = np.asarray(out)
                # face.image = out
                face.image = cropped_face
                faces.append(face)

        return faces

class Encoder():
    def __init__(self):
        
        ctx = mx.cpu()
        image = [112,112]
        layer = 'fc1'
        
        #model_str = './insightface_weights/model,0'
        model_str = './weights/face_embedding_model/R100_MS1MV2_ArcFace/model-r100-ii/model,0'
        self.model = self.get_model(ctx,image,layer,model_str) 

    def get_feature(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding
    
    def get_model(self,ctx,image,layer,model_str):
        vec = model_str.split(',')
        prefix = vec[0]
        epoch = int(vec[1])

        sym,arg_params,aux_params = mx.model.load_checkpoint(prefix,epoch)
        all_layer = sym.get_internals()
        sym = all_layer[layer+'_output']
        model = mx.mod.Module(symbol=sym,context=ctx,label_names=None)
        model.bind(data_shapes=[('data',(1,3,image[0],image[1]))])
        model.set_params(arg_params,aux_params)
        return model

    def generate_embedding(self,face):
        embedding = self.get_feature(face.image)
        return embedding





class Identifier():

    def __init__(self):
        self.total_features = np.empty((128,),np.float32)
        self.label = None

        name_list = os.listdir("./features/")
        path_list = [os.path.join('./features',name) for name in name_list]
        for i in path_list:
            temp = np.load(i)
            self.total_features = np.vstack((self.total_features,temp))

        self.total_features = self.total_features[1:]
        self.label = name_list
        self.threshold = 0.5

    def identify(self, face):
        if not self.label:
            name = 'None'
            return name

        cosin_metric = self.cosin_metric(self.total_features,face.embedding)
        index = np.argmax(cosin_metric)
        if not cosin_metric[index]>self.threshold:
            name = 'None'
        else:
            name = self.label[index][:-4]
    
        return name
    
    def cosin_metric(self,x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1,axis=1) * np.linalg.norm(x2))


class Identifier_IMDB(Identifier):

    def __init__(self, feature_path):
        self.total_features = np.empty((128,),np.float32)
        #self.label = None
        fea_to_label = {}
        path_list = []
        if not feature_path == '':
            name_list = os.listdir(feature_path)
            for name in name_list:
                fea_list = os.listdir(os.path.join(feature_path, name))
                for fea in fea_list:
                    fea_name = os.path.join(os.path.join(feature_path, name, fea))
                    fea_to_label[fea_name] = name
                    path_list.append(fea_name)
            print('The face library length is {} ids'.format(len(name_list)))
            #path_list = [os.path.join('./features',name) for name in name_list]
            for i in path_list:
                temp = np.load(i)
                self.total_features = np.vstack((self.total_features,temp))
            print('The face library length is {} features'.format(len(path_list)))
            self.total_features = self.total_features[1:]
        else:
            print('There is no gallery features')
            self.total_features = None
        self.fea_to_label = fea_to_label
        self.path_list = path_list
        self.threshold = 0.5

    def identify(self, face):
        if len(self.path_list) < 1:
            name = 'None'
            return name

        cosin_metric = self.cosin_metric(self.total_features,face.embedding)
        index = np.argmax(cosin_metric)
        print('The max distance is {}'.format(cosin_metric[index]))
        if not cosin_metric[index]>self.threshold:
            name = 'None'
        else:
            name = self.fea_to_label[self.path_list[index]]
    
        return name
