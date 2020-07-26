import insightface.face_model as face_model
import argparse


class InsightfaceEmbedder:
    def __init__(self, model_path, epoch_num='0000', image_size=(112, 112),
                 no_face_raise=True, MTCNN_min_face_size=100, MTCNN_steps_threshold=[0.5, 0.6, 0.7]):
        self.model_path = ','.join([model_path, epoch_num])
        self.no_face_raise = no_face_raise
        args = argparse.Namespace()
        args.model = self.model_path
        args.det = 0
        args.flip = 0
        args.threshold = 1.24
        args.ga_model = ''
        args.image_size = ",".join([str(i) for i in image_size])
        args.MTCNN_min_face_size = MTCNN_min_face_size
        args.MTCNN_steps_threshold = MTCNN_steps_threshold
        self.model = face_model.FaceModel(args)

    def embed_image(self, image, faceDetection_imgHeight):
        aligned_faces, bboxes, points = self.model.get_input(image, faceDetection_imgHeight)
        if aligned_faces is None:
            if self.no_face_raise:
                raise Exception("No face detected!")
            else:
                return None, None, None

        embeddings = self.model.get_feature(aligned_faces)
        return embeddings, bboxes, points


