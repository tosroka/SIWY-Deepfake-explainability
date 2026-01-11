# CNN Grad-CAM

Performing Grad-CAM algorithm on top of deezer's CNN (original repo: [link](https://github.com/deezer/deepfake-detector)

## Setup

1. Install requirements.txt

```sh

pip3 install -r requirements.txt
```

2. Download `FMA_meduim` dataset and unpack it. Point the paths to correct places in `loader/global_bariables.py`
3. Run at least some of the codecs, `encodec.sh` and `identity.sh` will produce various version of the datased passed through "EnCodec", and a plain resampled dataset.

4. Run `gradcam.sh` for full song spectrogram output, or `gradcam_short_time.sh`for the default 64 stft time bins

5. Outputs will appear in `grad_cam_outputs`. The structure is `<song_id>/<layer_name>. We analyze last three layers for comparison, the more coarse the activations, the higher concepts they utilize.

6. Driver code is in `scripts/grad_cam.py`, you can increase analysed songs count by editing the loop
