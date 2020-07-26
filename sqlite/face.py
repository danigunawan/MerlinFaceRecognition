from db import DBObject
from psycopg2.extensions import AsIs
from settings import DB_NAME, USER, PASSWORD, TABLE
import numpy as np
import logging
import cv2
from scipy.spatial.distance import euclidean
from insightface.embedder import InsightfaceEmbedder
import pandas as pd
from glob import glob
from tqdm import tqdm
from time import time
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource


model_path = "models/model-r34-amf/model"
embedder = InsightfaceEmbedder(model_path=model_path, epoch_num='0000', image_size=(112, 112),
                               no_face_raise=False,
                               MTCNN_min_face_size=50, MTCNN_steps_threshold=[0.7, 0.7, 0.9])


class FeceRecognizer(Resource):

    def get_embeddings_from_folder_df(self, path='data/People/', faceDetection_imgHeight=512):
        dfPeople = pd.DataFrame()
        dfPeople['name'] = 'object'
        dfPeople['path'] = 'object'
        dfPeople['embedding'] = 'object'
        dfPeople['bbox'] = 'object'

        people_list = os.listdir(path)

        count = 0
        print(f'Обнаружено {len(people_list)} индивидов')
        print(f'Создание базы эмбеддингов')

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

        return dfPeople


    def save_to_db(self, table, mapper, db):
        """
            table: table name
            mapper: dictionary mapping from person name to vector representation
            db: db object
        """
        for i, key in enumerate(mapper.keys()):
            _name = key
            _url = 'gt_db/' + _name
            _vector = AsIs('cube(ARRAY[' + str(mapper[key].tolist()).strip('[|]') + '])')
            db.make_query("INSERT INTO %s (name, url, vector) VALUES (%s, %s, %s)",
                        (AsIs(table), _name, _url, _vector))

            print str(i) + ' records inserted!'


    def insert_features(self, path, table):
        batches, predictions = get_batch_predictions(path)

        name2vector = {}
        for i, prediction in enumerate(predictions):
            name2vector[batches.filenames[i].split('/')[-1]] = prediction

        db = DBObject(db=DB_NAME, user=USER, password=PASSWORD)
        save_to_db(table, name2vector, db)


