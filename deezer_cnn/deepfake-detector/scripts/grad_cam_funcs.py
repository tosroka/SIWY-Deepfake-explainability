"""From keras tutorial"""
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

EXCERPT_LENGTH = 10 # 10 seconds

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        fc_out, encoder_out = preds
        if pred_index is None:
            pred_index = tf.argmax(encoder_out[0])
        # Ensure preds is a tensor for correct indexing
        class_channel = encoder_out[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_float: np.ndarray, heatmap_float: np.ndarray, cam_path: Path, alpha=0.4):

    # Rescale heatmap and image to a range 0-255
    heatmap = (255 * heatmap_float).astype(np.uint8)
    img = (255 * img_float).astype(np.uint8)

    # Use jet colormap to colorize heatmap
    jet = matplotlib.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    height = superimposed_img.size[1]
    width = superimposed_img.size[0]

    plt.figure()
    plt.imshow(superimposed_img)
    plt.axis('on')

    # 1. FIX X-TICKS (Time: 0 to 10s)
    x_ticks = np.linspace(0, width, 5)
    x_labels = [f"{int(val)}s" for val in np.linspace(0, 10, 5)] # Map to 0-10
    plt.xticks(x_ticks, labels=x_labels)
    plt.xlabel("Time (s)")

    # 2. FIX Y-TICKS (Frequency: 16kHz at top, 4kHz at bottom)
    # Note: imshow puts 0 at the top. 
    # To have 16kHz at the top and 4kHz at bottom:
    y_ticks = np.linspace(0, height, 5)
    y_labels = [f"{int(val)}kHz" for val in np.linspace(16, 4, 5)] # Reverse the range
    plt.yticks(y_ticks, labels=y_labels)
    plt.ylabel("Frequency (kHz)")

    plt.savefig(cam_path)
    plt.close()

    # Save the superimposed image
    #superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))

import numpy as np

def run_grad_cam_on_image(input_image: np.ndarray, model, last_conv_name: str, output_dir: Path):
    # input image is (64, 743, 1), 64 comes from the "audio_slice": 0,78 in global config
    # since its (64,743,1), we need to iterate over 64x64 patches:

    outdir = Path(output_dir)
    print("outputting to ",outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    heatmaps = []
    patch_size = 64
    stride = 64
    h, w, _ = input_image.shape

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            print("i,j", i,j)
            patch = input_image[i:i+64, j:j+64, :]
            print("patch shape",patch.shape)
            patch = patch[None,...]
            print("batched patch",patch.shape)
            # Do something with the patch

            # bring back the batch for this moment
            heatmap = make_gradcam_heatmap(patch, model.m, last_conv_name)
            # save to file for inspection
            heatmap_img = heatmap[...,None]
            heatmap_img = np.repeat(heatmap_img, 3, axis=-1)
            #plt.imshow(heatmap_img, cmap='jet')
            #plt.savefig(outdir / f"heatmap_{i}_{j}.png")

            # collect the heatmaps
            print("heatmap shape:", heatmap.shape)
            heatmaps.append(heatmap)


    # repeat last dimension so it's not (64,743,1) but (64,743,3)
    input_image = np.repeat(input_image, 3, axis=-1)
    # map to 0,1 range
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
    # rotate input_image 90 degrees
    input_image = np.rot90(input_image)
    plt.imsave(outdir / "spectrogram.png", input_image, cmap='gray')

    # merge the individual heatmaps into one image, one patch gets turned into a smaller precision heatmap,
    # but their count is the same, so final big heatmap will have size h_count*heatmap_size
    heatmap_size = heatmaps[0].shape[0]
    # we divide by patch_size, because this was the original patching way, which generated smaller heatmaps
    h_count = h // patch_size
    w_count = w // patch_size

    merged_heatmap = np.zeros((h_count*heatmap_size, w_count*heatmap_size))
    for i, heatmap in enumerate(heatmaps):
        # find the position to place the heatmap
        row = i // w_count
        col = i % w_count
        merged_heatmap[row*heatmap_size:(row+1)*heatmap_size, col*heatmap_size:(col+1)*heatmap_size] = heatmap

    merged_heatmap = (merged_heatmap - np.min(merged_heatmap)) / (np.max(merged_heatmap) - np.min(merged_heatmap) + 1e-8)
    merged_heatmap = np.rot90(merged_heatmap)
    # save merged heatmap to image
    plt.imsave(outdir / "merged_heatmap.png", merged_heatmap, cmap='jet')
    save_and_display_gradcam(input_image, merged_heatmap, cam_path=outdir/"gradcam.png")
