# from flask import Flask, render_template, Response
# from keras.models import load_model
# from matplotlib import pyplot
# import os
# import numpy as np
import pickle
import cv2
from PIL import Image
from numpy import asarray, expand_dims
# from keras.models import load_model
from keras_facenet import FaceNet
# from os import listdir
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition():
    def __init__(self, embed_model = "../model/haarcascade_frontalface_default.xml",
                 labels_file_name = "../dataset/label/data.pkl" ):
        
       
       self.label_file = labels_file_name
       self.harcascade = cv2.CascadeClassifier(embed_model)
       self.face_net = FaceNet()
       
    def load_labels(self):
        return pickle.load(open(self.label_file, "rb"))
       
       
       
    def get_signature_from_frame(self, frame):
        wajah = self.harcascade.detectMultiScale(frame, 1.1, 4)
        if len(wajah) > 0:
            x1, y1, width, height = wajah[0]
        else:
            x1, y1, width, height = 1, 1, 10, 10

        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        gbr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)
        gbr_array = asarray(gbr)

        face = gbr_array[y1:y2, x1:x2]
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)

        signature = self.face_net.embeddings(face)
        return signature, x1, x2, y1, y2
    
    
    def recognize_faces(self, signature, database):
        max_similarity = -1
        identity = ' '
        uk = 'Unknown'
        label_file = self.load_labels()
        for key, value in database.items():
            similarity = cosine_similarity(value.reshape(1, -1), signature.reshape(1, -1))[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                identity = key

        confidence = max_similarity * 100  # Calculate confidence in percentage
        if confidence < 50:
            return f'{uk} ({confidence:.2f}%)'
        else:
            return f'{identity} ({confidence:.2f}%)'