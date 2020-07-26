import warnings

warnings.filterwarnings("ignore")

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

import pandas as pd
from glob import glob
from tqdm import tqdm
from time import time

import sys, getopt
import os

from insightface.embedder import InsightfaceEmbedder
from flask_restful import reqparse, abort, Api, Resource
from flask.views import MethodView
from flask import Flask

app = Flask(__name__)
api = Api(app)

model_path = "models/model-r34-amf/model"
embedder = InsightfaceEmbedder(model_path=model_path, epoch_num='0000', image_size=(112, 112),
                               no_face_raise=False,
                               MTCNN_min_face_size=50, MTCNN_steps_threshold=[0.7, 0.7, 0.9])
print("Face Recognition Model: Initialized successfully")
faceDetection_imgHeight = 256

import face_detection


# faceDetector = face_detection.build_detector(
#     "RetinaNetResNet50", confidence_threshold=0.5, nms_iou_threshold=0.3)
# print('Face Detector initialized successfully')


def recognize(img_path, dfPeople):
    img = cv2.imread(img_path)

    face_encodings, face_locations, face_points = embedder.embed_image(img,
                                                                       faceDetection_imgHeight=faceDetection_imgHeight)
    if face_encodings != None:
        face_names = []
        dists_list = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            tolerance = 1
            distances = np.linalg.norm(dfPeople['embedding'].values.tolist() - face_encoding, axis=1)
            matches = distances <= tolerance
            min_dist = min(distances)
            dists_list.append(min_dist)
            name = 'Unknown'
            if matches.sum() > 0:
                name = dfPeople['name'][matches].value_counts().index[0]
            face_names.append(name)

        print('face_names', face_names)
        return face_names


def get_embeddings_from_storage(path):
    path = 'data/People/'
    people_list = os.listdir(path)
    count = 0
    print(f'Обнаружено {len(people_list)} индивидов')
    print(f'Создание базы эмбеддингов')

    dfPeople = pd.DataFrame()
    dfPeople['name'] = 'object'
    dfPeople['path'] = 'object'
    dfPeople['embedding'] = 'object'
    dfPeople['bbox'] = 'object'

    for man in tqdm(people_list):
        photos = os.listdir(f'data/People/{man}')
        for photo in photos:
            img_path = f'data/People/{man}/{photo}'
            img = cv2.imread(img_path)
            emb_img, bboxes_img, points = embedder.embed_image(img, faceDetection_imgHeight=faceDetection_imgHeight)
            dfPeople.at[count, 'name'] = man
            dfPeople.at[count, 'path'] = img_path
            dfPeople.at[count, 'embedding'] = emb_img[0]
            dfPeople.at[count, 'bbox'] = bboxes_img[0]
            count += 1

    print('dfPeople.shape', dfPeople.shape)
    return dfPeople


class FaceRecognition(Resource):
    def __init__(self):
        self.output = {}
        self.output['success'] = True

        path = 'data/People'
        self.dfPeople = get_embeddings_from_storage(path)

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('path', default='data/People/')
        parser.add_argument('command', default='scan_storage')
        args = parser.parse_args()
        path = args['path']
        command = args['command']

        self.dfPeople = get_embeddings_from_storage(path)
        self.output['dfPeople.shape'] = self.dfPeople.shape

        if command == 'scan_storage':
            self.dfPeople = get_embeddings_from_storage(path)
            return self.output

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('command', default='scan_storage')
        parser.add_argument('img_path')
        args = parser.parse_args()
        img_path = args['img_path']
        command = args['command']

        if command == 'recognize':
            face_names = recognize(img_path, self.dfPeople)
            self.output['face_names'] = face_names
            return self.output

    # def inser_person(self):


    # def add_photo_to_exisng_person(self):

    # def recognize_person():

    # def add_person():


api.add_resource(FaceRecognition, '/', methods=['GET', 'POST'])
if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1")
