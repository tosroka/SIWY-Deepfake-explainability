
"grad cam for the especcn model checkpoint"

from loader.global_variables import *
import tensorflow as tf
from loader.audio import AudioLoader, EvalAugmenter
from loader.config import ConfLoader
from model.simple_cnn import SimpleSpectrogramCNN
import os
from pathlib import Path

from . import grad_cam_funcs

os.environ["KERAS_BACKEND"] = "tensorflow"

config = "specnn_amplitude"
GPU = -1
CODEC = ''
CODEC_EXTENSION = ''
ENCODER = ''

print("loading config from", CONF_PATH)
configuration = ConfLoader(CONF_PATH) # loads base.json
configuration.load_model(config) # loads model specific config on top of base 
global_conf = configuration.conf
global_conf['batch_size'] = 1 # only one image for grad cam

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

# --- SIWY addition - model expects 64 bins---

#loader.params['patch_size_t'] = 64
#loader.params['patch_size_f'] = 640
#loader.params['patch_size_f_min'] = 640 # don't randomize...


outpath = Path("grad_cam_outputs_short_time")

#it_test = it_test.map(loader.patch_batch_spec)

# Ensure patch_batch_spec returns a value and not None
assert it_test is not None, "it_test is None after mapping patch_batch_spec"
for i in range(3):
    _it = iter(it_test)
    input = next(_it)
    assert input is not None, "Next element from iterator is None"
    input_batch, y_batch = input
    input_image = input_batch[0] # no batch because no need anyway
    print("input shape",input_image.shape)

    model = SimpleSpectrogramCNN(input_batch.shape[1:], global_conf, detect_encoder = True)

    model.m = tf.keras.models.load_model('/net/people/plgrid/plgtsroka/SIWY/deepfake-detector/specnn_amplitude')

    for layer in ["conv2d_5","conv2d_4","conv2d_3"]:
        grad_cam_funcs.run_grad_cam_on_image(input_image, model, layer, output_dir=outpath / str(i) / layer)

# model.m.summary()