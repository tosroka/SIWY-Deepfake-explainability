import tensorflow as tf

m = tf.keras.models.load_model('/net/people/plgrid/plgtsroka/SIWY/deepfake-detector/specnn_amplitude')

m.summary() # last is conv2d_5