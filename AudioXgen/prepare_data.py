import os

import librosa
import numpy as np
import torch
from tqdm import tqdm

import utils

OUTPUT_DIR = "data_embeddings"
NUM_CROPS = 3  # Multi-Crop strategy

# Label 0 = Real, Label 1 = Fake
DATA_SOURCES = [
    ("../data/lastfm/audio", 0),
    ("../data/suno/audio", 1),
    ("../data/udio/audio", 1),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

model, processor = utils.load_encodec_model()


def process_audio_multicrop(file_path):
    """
    Extracts 'NUM_CROPS' segments from one audio file using Stratified Sampling.
    """
    embeddings_list = []
    try:
        wav_np, sr = librosa.load(file_path, sr=utils.TARGET_SR, mono=True)
        wav = torch.from_numpy(wav_np).unsqueeze(0)

        total_samples = wav.shape[1]
        target_samples = int(utils.DURATION_SEC * utils.TARGET_SR)

        # A. Audio too short -> Pad
        if total_samples < target_samples:
            pad_size = target_samples - total_samples
            wav_padded = torch.nn.functional.pad(wav, (0, pad_size))
            embeddings_list.append(
                utils.get_encodec_embedding(wav_padded, model, processor)
            )

        # B. Audio long enough -> Multi-Crop
        else:
            valid_range = total_samples - target_samples
            if valid_range > 0:
                sectors = np.array_split(range(valid_range), NUM_CROPS)
                for sector_indices in sectors:
                    if len(sector_indices) == 0:
                        continue

                    start = np.random.choice(sector_indices)
                    wav_chunk = wav[:, start : start + target_samples]
                    embeddings_list.append(
                        utils.get_encodec_embedding(wav_chunk, model, processor)
                    )
            else:
                embeddings_list.append(
                    utils.get_encodec_embedding(
                        wav[:, :target_samples], model, processor
                    )
                )

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return embeddings_list


if __name__ == "__main__":
    X_list = []
    y_list = []
    valid_extensions = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

    for folder_path, label in DATA_SOURCES:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        print(f"--- Processing label {label} from: {folder_path} ---")
        files = [
            f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)
        ]

        for f in tqdm(files):
            file_path = os.path.join(folder_path, f)
            embs = process_audio_multicrop(file_path)

            for e in embs:
                X_list.append(e)
                y_list.append(label)

    if len(X_list) > 0:
        print("Concatenating tensors...")
        X_tensor = torch.cat(X_list, dim=0)
        y_tensor = torch.tensor(y_list)

        print(f"Saving dataset to {OUTPUT_DIR}...")
        torch.save(X_tensor, os.path.join(OUTPUT_DIR, "train_data_X.pt"))
        torch.save(y_tensor, os.path.join(OUTPUT_DIR, "train_data_y.pt"))

        print(
            f"SUCCESS! Class balance: {y_tensor.sum().item()} Fakes / {len(y_tensor) - y_tensor.sum().item()} Reals"
        )
    else:
        print("ERROR: No files processed.")
