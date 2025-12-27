from argparse import Namespace
from preprocessing.mtat_read import Processor
from training.eval import Predict
import numpy as np
import os
import torch


def composition_fn(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return x


# The following is required to work with:
# https://github.com/minzwon/sota-music-tagging-models
base_dir = os.path.dirname(os.path.abspath(__file__))
path_to_tags = os.path.join(base_dir, "split", "msd", "50tagList.txt")
tag_file = open(path_to_tags, "r")
tags_msd = [t.replace("\n", "") for t in tag_file.readlines()]
path_models = os.path.join("./models")


won2020_default_config = Namespace()
won2020_default_config.num_workers = 10
won2020_default_config.dataset = "msd"
won2020_default_config.model_type = "musicnn"
won2020_default_config.batch_size = (
    16  # this is actually the nr. of chunks analysed per song
)
won2020_default_config.data_path = "placeholder"
won2020_default_config.load_data = True
won2020_default_config.input_length = 3 * 16000

audio_length_per_model = {
    "fcn": 29 * 16000,
    "crnn": 29 * 16000,
    "musicnn": 3 * 16000,
    "attention": 15 * 16000,
    "hcnn": 5 * 16000,
    "sample": 59049,
}


def prepare_audio(
    audio_path, input_length, nr_chunks=None, return_snippet_starts=False
):
    # based on code in Minz Won's repo
    if nr_chunks is None:
        nr_chunks = input_length
    processor = Processor()
    raw = processor.get_npy(audio_path)
    length = len(raw)
    snippet_starts = []

    hop = (length - input_length) // nr_chunks
    x = torch.zeros(nr_chunks, input_length)
    for i in range(nr_chunks):
        snippet_starts.append(i * hop)
        x[i] = torch.tensor(raw[i * hop : i * hop + input_length]).unsqueeze(0)
    if return_snippet_starts:
        return x, snippet_starts
    return x


def create_predict_fn(model, config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    def predict_fn(x_array):
        x = torch.zeros(len(x_array), config.input_length)

        for i in range(len(x_array)):
            signal = x_array[i]

            if len(signal) > config.input_length:
                signal = signal[: config.input_length]

            x[i, : len(signal)] = (
                torch.from_numpy(signal)
                if isinstance(signal, np.ndarray)
                else torch.Tensor(signal)
            )

        x = x.to(device)

        with torch.no_grad():
            y = model(x)

        return y.detach().cpu().numpy()

    return predict_fn


def get_model(config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = Predict.get_model(config)
    state_dict = torch.load(config.model_load_path, map_location=device)

    if model is not None:
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    return model
