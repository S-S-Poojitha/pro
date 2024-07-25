import tensorflow as tf
import tensorflow_hub as hub

# Example URL to download a pre-trained model
model_url = 'https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
model_dir = tf.keras.utils.get_file('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', model_url, untar=True)

# Load the model
model = tf.saved_model.load(model_dir + "/saved_model")
