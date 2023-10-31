from flask import Flask, render_template, Response
from keras.models import load_model
from matplotlib import pyplot
import os
import numpy as np
import pickle
import cv2
from PIL import Image
from numpy import asarray, expand_dims
from keras.models import load_model
from keras_facenet import FaceNet
from os import listdir
from sklearn.metrics.pairwise import cosine_similarity

HaarCascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')
MyFaceNet   = FaceNet()

myfile = open("../dataset/label/data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def recognize_faces_from_video():
    while True:
        _, frame = camera.read()

        wajah = HaarCascade.detectMultiScale(frame, 1.1, 4)

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

        signature = MyFaceNet.embeddings(face)

        max_similarity = -1
        identity = ' '
        uk = 'Unknown'
        for key, value in database.items():
            similarity = cosine_similarity(value.reshape(1, -1), signature.reshape(1, -1))[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                identity = key

        confidence = max_similarity * 100  # Calculate confidence in percentage

        if confidence < 50:
            cv2.putText(frame, f'{uk} ({confidence:.2f}%)', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f'{identity} ({confidence:.2f}%)', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(recognize_faces_from_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
app.run(host="0.0.0.0") #jgn ikutkan host=0.0.0.0 jika hanya ingin server berjalan di local
