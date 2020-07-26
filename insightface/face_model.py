from __future__ import absolute_import

import os
import numpy as np
import mxnet as mx
from mtcnn.mtcnn import MTCNN
import cv2
from sklearn.preprocessing import normalize
import insightface.face_preprocess as face_preprocess
from insightface.helper import resize_by_height
# from insightface.mtcnn_detector import MtcnnDetector


def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model


class FaceModel:
  def __init__(self, args):
    self.args = args
    _vec = args.image_size.split(',')
    image_size = (int(_vec[0]), int(_vec[1]))
    self.min_face_size = args.MTCNN_min_face_size #100
    self.steps_threshold = args.MTCNN_steps_threshold #[0.3, 0.6, 0.6]
    detector = MTCNN(steps_threshold=self.steps_threshold, min_face_size=self.min_face_size)
    self.image_size = image_size
    self.detector = detector

    ctx = mx.cpu()
    self.model = get_model(ctx, image_size, args.model, 'fc1')

  def process_point(self, point, scale_factor):
    left_eye    = point['left_eye']
    right_eye   = point['right_eye']
    nose        = point['nose']
    mouth_left  = point['mouth_left']
    mouth_right = point['mouth_right']
    point = []
    point.append(left_eye)
    point.append(right_eye)
    point.append(nose)
    point.append(mouth_left)
    point.append(mouth_right)
    point = np.array(point).astype(np.int32) // scale_factor
    return point


  def get_input(self, face_img, faceDetection_imgHeight):
    resizedImg, scale_factor = resize_by_height(face_img, faceDetection_imgHeight)
    anfas_faces = self.detector.detect_faces(resizedImg)

    if len(anfas_faces) == 0:
      return None, None, None

    aligned_faces = []
    bboxes = []
    points = []
    for enum, bbox in enumerate(anfas_faces):
      bbox = anfas_faces[enum]['box']
      bbox = [int(x/scale_factor) for x in bbox]
      bbox[2] = bbox[0]+bbox[2]
      bbox[3] = bbox[1]+bbox[3]
      bboxes.append(bbox)

      point = anfas_faces[enum]['keypoints']
      point = self.process_point(point, scale_factor)
      points.append(point)

      nimg = face_preprocess.preprocess(face_img, bbox, point, image_size=self.args.image_size)
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      aligned_faces.append(aligned)

    return aligned_faces, bboxes, points

  def get_feature(self, aligned_faces):
    embeddings = []
    for face in aligned_faces:
      input_blob = np.expand_dims(face, axis=0)
      data = mx.nd.array(input_blob)
      db = mx.io.DataBatch(data=(data,))
      self.model.forward(db, is_train=False)
      embedding = self.model.get_outputs()[0].asnumpy()
      embedding = normalize(embedding).flatten()
      embeddings.append(embedding)
    return embeddings

# # Original version
# class FaceModel:
#   def __init__(self, args):
#     self.args = args
#     ctx = mx.cpu()
#     _vec = args.image_size.split(',')
#     assert len(_vec)==2
#     image_size = (int(_vec[0]), int(_vec[1]))
#     self.model = None
#     self.ga_model = None
#     if len(args.model)>0:
#       self.model = get_model(ctx, image_size, args.model, 'fc1')
#     if len(args.ga_model)>0:
#       self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')
#
#     self.threshold = args.threshold
#     self.det_minsize = 50
#     self.det_threshold = [0.6,0.7,0.8]
#     self.image_size = image_size
#     mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
#     detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
#     self.detector = detector
#
#   def get_input(self, face_img, faceDetection_imgHeight):
#     ret = self.detector.detect_face(face_img, det_type = self.args.det)
#     if ret is None:
#       return None, None
#     bboxes_raw, points = ret
#     if bboxes_raw.shape[0] == 0:
#       return None, None
#
#     aligned_faces = []
#     bboxes = []
#     for enum, bbox in enumerate(bboxes_raw):
#       bbox = [int(i) for i in bbox[:4]]
#       bboxes.append(bbox)
#       point = points[enum,:].reshape((2,5)).T
#
#       nimg = face_preprocess.preprocess(face_img, bbox, point, image_size=self.args.image_size)
#       nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
#       aligned = np.transpose(nimg, (2,0,1))
#       aligned_faces.append(aligned)
#     return aligned_faces, bboxes, points
#
#   def get_feature(self, aligned_faces):
#     embeddings = []
#     for face in aligned_faces:
#       input_blob = np.expand_dims(face, axis=0)
#       data = mx.nd.array(input_blob)
#       db = mx.io.DataBatch(data=(data,))
#       self.model.forward(db, is_train=False)
#       embedding = self.model.get_outputs()[0].asnumpy()
#       embedding = normalize(embedding).flatten()
#       embeddings.append(embedding)
#     return embeddings
