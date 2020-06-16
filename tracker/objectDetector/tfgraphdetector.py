# Object Detector
import cv2
import tensorflow as tf
import numpy as np
import warnings
import six

from object_detection.utils import label_map_util
from track import Track as tempTrack
from utils import xyxy2xywh
from encoder import create_box_encoder
warnings.filterwarnings('ignore')  # Ignore Warnings


class ObjectDetector(object):
    def __init__(self, model_path, label_path, encoder_path, min_conf):
        # Initialise Session
        self.detection_graph = tf.compat.v1.Graph()
        self.tensor_dict = {}
        self.sess = self.initializeSession(model_path)
        # Initialise Encoder
        self.encoder = create_box_encoder(encoder_path, batch_size=1)
        # Load the Labels
        self.category_index = label_map_util.create_category_index_from_labelmap(
            label_path, use_display_name=True)
        self.min_conf = min_conf

    def initializeSession(self, model_path):
        # Check for GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(physical_devices))
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # Load the model as graph
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session()
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}

            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            return sess

    def close(self):
        self.sess.close()

    def _runInference(self, image):
        with self.detection_graph.as_default():
            if 'detection_masks' in self.tensor_dict:
                raise NotImplementedError

            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = self.sess.run(self.tensor_dict,
                                        feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            # return output_dict
            boxes = output_dict['detection_boxes']
            scores = output_dict['detection_scores']
            classes = output_dict['detection_classes']
            nums = output_dict['num_detections']
            return boxes, scores, classes, nums

    def inferImage(self, image):
        frame = image.copy()
        hw = frame.shape[0:2]
        detections = []
        boxes, scores, classes, nums = self._runInference(frame)
        # boxes = xyxy2xywh(boxes)
        bboxes = []
        sscores = []
        for i in range(nums):
            if classes[i] in six.viewkeys(self.category_index):
                class_name = self.category_index[classes[i]]['name']
                if class_name == "person":  # Detect Only Person
                    # Consider Boxes with minimum Detections
                    if scores[i] >= self.min_conf:
                        ymin, xmin = boxes[i][0:2] * hw
                        ymax, xmax = boxes[i][2:4] * hw
                        width = xmax-xmin
                        height = ymax-ymin
                        bboxes.append(
                            np.array([int(xmin), int(ymin), int(width), int(height)]))
                        # bboxes.append(boxes[i])
                        sscores.append(scores[i])

        # Apply NMS Here

        image_patches, features = self.encoder(frame, bboxes)

        detections = [tempTrack(box, score, f, 30) for (
            box, score, f) in zip(bboxes, sscores, features)]

        return detections
