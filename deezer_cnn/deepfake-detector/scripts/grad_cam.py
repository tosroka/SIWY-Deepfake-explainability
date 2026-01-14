"grad cam for the especcn model checkpoint"

from loader.global_variables import *
import tensorflow as tf
from loader.audio import AudioLoader, EvalAugmenter
from loader.config import ConfLoader
from model.simple_cnn import SimpleSpectrogramCNN
import os
from pathlib import Path
import numpy as np

from . import grad_cam_funcs

os.environ["KERAS_BACKEND"] = "tensorflow"

config = "specnn_amplitude"
GPU = -1
CODEC = ''
CODEC_EXTENSION = ''
ENCODER = ''

grad_cam_count = 10

print("loading config from", CONF_PATH)
configuration = ConfLoader(CONF_PATH) # loads base.json
configuration.load_model(config) # loads model specific config on top of base 
global_conf = configuration.conf
global_conf['batch_size'] = 1 # only one image for grad cam
global_conf['audio_slice'] = 10 # 10 seconds for full time
global_conf['shuffle'] = grad_cam_count
global_conf['repeat'] = 1

loader = AudioLoader(POS_DB_PATH, NEG_DB_PATH, global_conf, split_path = SPLIT_PATH,
        codec = CODEC_EXTENSION)
augmenter = EvalAugmenter(global_conf)

if GPU>=0:
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU], True)

it_test = loader.create_tf_iterator('test', augmenter = augmenter)

@tf.function
def one_hot_encoder(y, depth):
    y1, y2 = y
    y1 = tf.cast(y1, tf.int32)
    idx = (1 - y1) * (y2 + 1)
    return y1, tf.one_hot(idx, depth)


it_test = it_test.map(lambda x, y: (x, one_hot_encoder(y, 10)) ) # oader.n_encoders+1 #missing data is padded!

# --- SIWY addition - model expects 64 bins ---

#loader.params['patch_size_t'] = 640
#loader.params['patch_size_f'] = 1000
#loader.params['patch_size_f_min'] = 1000 # don't randomize...

outpath = Path("/net/people/plgrid/plgtsroka/SIWY/SIWY-Deepfake-explainability/deezer_cnn/deepfake-detector/grad_cam_outputs")

#it_test = it_test.map(loader.patch_batch_spec)

# Ensure patch_batch_spec returns a value and not None
assert it_test is not None, "it_test is None after mapping patch_batch_spec"
for i in range(grad_cam_count):
    _it = iter(it_test)
    input = next(_it)
    assert input is not None, "Next element from iterator is None"
    input_batch, y_batch = input
    sample_class = np.argmax(y_batch[1])
    print(y_batch[0],y_batch[1], sample_class)

    input_image = input_batch[0] # no batch because no need anyway
    print("input shape",input_image.shape)

    model = SimpleSpectrogramCNN(input_batch.shape[1:], global_conf, detect_encoder = True)

    model.m = tf.keras.models.load_model('/net/people/plgrid/plgtsroka/SIWY/SIWY-Deepfake-explainability/deezer_cnn/deepfake-detector/specnn_amplitude/')

    for layer in ["conv2d_5","conv2d_4","conv2d_3"]:
        if sample_class>0:
            grad_cam_funcs.run_grad_cam_on_image(input_image, model, layer, output_dir=outpath / f"{i}_enc{sample_class}" / layer)
        else:
            grad_cam_funcs.run_grad_cam_on_image(input_image, model, layer, output_dir=outpath / f"{i}_real" / layer)

# model.m.summary()