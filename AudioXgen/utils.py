import torch
from transformers import AutoProcessor, EncodecModel

TARGET_SR = 24000
DURATION_SEC = 5.0
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

DATA_X_PATH = "data_embeddings/train_data_X.pt"
DATA_Y_PATH = "data_embeddings/train_data_y.pt"

print(f"Global Device set to: {DEVICE}")


def load_encodec_model():
    """Loads the EnCodec model and processor"""
    print(f"Loading EnCodec model on {DEVICE}...")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE)
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model.eval()
    return model, processor


def get_encodec_embedding(wav_tensor, model, processor):
    """
    Encodes raw audio into continuous latent space
    """
    inputs = processor(
        raw_audio=wav_tensor.squeeze().cpu().numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
        codes = encoder_outputs.audio_codes[0].transpose(0, 1)
        embeddings = model.quantizer.decode(codes)

    return embeddings.cpu()


def get_data_splits(test_size=0.2, seed=42):
    """
    Loads dataset and splits it deterministically into Train and Test.
    Uses a fixed seed so the split is ALWAYS the same across scripts.
    """
    print("Loading full dataset from disk...")
    try:
        X = torch.load(DATA_X_PATH)
        y = torch.load(DATA_Y_PATH)
    except FileNotFoundError:
        print(f"Error: Files not found in data_embeddings/. Run prepare_data.py first!")
        exit()

    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    train_samples = total_samples - test_samples

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)

    # Slice indices
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]

    # Create tensors
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Data Split (Seed={seed}):")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")

    return (X_train, y_train), (X_test, y_test)
