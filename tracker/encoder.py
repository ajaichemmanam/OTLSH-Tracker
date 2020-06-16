# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf
import binascii
import scipy
from scipy import cluster
# Split Data to run in Batches


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


#### Extract image patch from bounding box and optionally reshaped to patch_shape ###
def extract_image_patch(image, bbox, patch_shape):
    imgHeight, imgWidth, imgChannel = image.shape
    bbox = np.array(bbox)
    xmin = bbox[0]
    ymin = bbox[1]
    width = bbox[2]
    height = bbox[3]
    xmax = xmin+width
    ymax = ymin+height

    if(xmin < 0):
        xmin = 0
    if(ymin < 0):
        ymin = 0

    if (xmax > imgWidth):
        xmax = imgWidth
    if (ymax > imgHeight):
        ymax = imgHeight

    try:
        patch = image[ymin:ymax, xmin:xmax]
        # cv2.imshow('patch', image)
        if patch_shape is not None:
            patch = cv2.resize(
                patch, (patch_shape[1], patch_shape[0]), interpolation=cv2.INTER_AREA)
    except:
        patch = np.asarray([])
    return patch


class ImageEncoder(object):
    # Initialize the CNN Model
    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.compat.v1.Session()
        # Read Model and Create GraphDef
        with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        # Import GraphDef
        tf.import_graph_def(graph_def, name="net")
        # Declare Input and Output Layers
        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)
        # Verify the shape of layers of the model
        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

# Creates encoded version of the input Images


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape  # image_shape = [128,64,3]

    def encoder(image, boxes):
        image_patches = []
        for idx, box in enumerate(boxes):
            # image_shape[:2] = [128,64]
            # print(image.size, box, image_shape[:2])
            patch = extract_image_patch(image, box, image_shape[:2])
            if (patch.size == 0):
                print("WARNING: Failed to extract image patch: %s." % str(box))
                # Generate a random image to be used instead of the patch
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_patches, image_encoder(image_patches, batch_size)
    return encoder


# Test Image Encoder
if __name__ == "__main__":
    img = cv2.imread('./patch0.jpg')
    encoder = create_box_encoder(
        'PATHTOENCODER.pb', batch_size=1)
    xmin = 0
    xmax = img.shape[1]  # Width of image
    ymin = 0
    ymax = img.shape[0]  # Height of Image
    bboxes = [np.array([xmin, ymin, xmax, ymax])]
    features = encoder(img, bboxes)
    print(features)
