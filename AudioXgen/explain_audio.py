import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from audio_transformer import AudioTransformer
from captum.attr import IntegratedGradients

import utils

OUTPUT_FILE = "explanation_result.wav"
MODEL_PATH = "models/transformer_classifier.pth"
KEEP_RATIO = 0.15


def find_best_test_sample(classifier):
    """
    Loads the TEST set, scans FAKE samples IN BATCHES
    and returns the embedding that the model is MOST confident is Fake
    """
    print("Loading Test Data...")
    (_, _), (X_test, y_test) = utils.get_data_splits(test_size=0.2)

    X_test = X_test.to(utils.DEVICE)
    y_test = y_test.to(utils.DEVICE)

    # Look only at samples labeled as FAKE
    indices_fake = (y_test == 1).nonzero(as_tuple=True)[0]
    X_fake = X_test[indices_fake]

    if len(X_fake) == 0:
        print("Error: No Fake samples found in Test Set.")
        exit()

    BATCH_SIZE = 32
    print(f"Scanning {len(X_fake)} FAKE samples (Batch size: {BATCH_SIZE})...")

    best_score = -1.0
    best_embedding = None

    classifier.eval()

    with torch.no_grad():
        # Loop through data in chunks
        for i in range(0, len(X_fake), BATCH_SIZE):
            batch = X_fake[i : i + BATCH_SIZE]

            # Predict
            outputs = classifier(batch)
            probs = F.softmax(outputs, dim=1)
            fake_probs = probs[:, 1]

            # Find max in THIS batch
            batch_max_score, batch_max_idx = torch.max(fake_probs, dim=0)
            current_score = batch_max_score.item()

            # Compare with global best
            if current_score > best_score:
                best_score = current_score
                # Retrieve the specific embedding from the batch
                best_embedding = batch[batch_max_idx]

    print(f"--> Found best candidate! Model Confidence: {best_score*100:.2f}%")

    return best_embedding.unsqueeze(0)


if __name__ == "__main__":
    encodec_model, _ = utils.load_encodec_model()

    print("Loading classifier...")
    classifier = AudioTransformer(feature_size=128, seq_length=375, num_classes=2).to(
        utils.DEVICE
    )

    if os.path.exists(MODEL_PATH):
        classifier.load_state_dict(torch.load(MODEL_PATH, map_location=utils.DEVICE))
        classifier.eval()
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        exit()

    target_embedding = find_best_test_sample(classifier)

    target_embedding = target_embedding.detach()
    target_embedding.requires_grad = True

    print("Computing attributions (Integrated Gradients)...")
    ig = IntegratedGradients(classifier)

    # Target=1 means we explain class "Fake"
    attributions = ig.attribute(target_embedding, target=1)
    attributions = attributions.detach().cpu().numpy()

    # Masking Logic (Keep only top X% artifacts)
    print("Synthesizing explanation...")
    attr_flat = np.abs(attributions).flatten()
    threshold = np.percentile(attr_flat, 100 * (1 - KEEP_RATIO))

    mask = torch.from_numpy((np.abs(attributions) >= threshold).astype(np.float32)).to(
        utils.DEVICE
    )

    # Baseline is silence (zeros)
    baseline = torch.zeros_like(target_embedding)
    explanation_latent = target_embedding * mask + baseline * (1 - mask)

    print("Decoding audio...")
    with torch.no_grad():
        # Decode the explanation (Artifacts only)
        audio_expl = encodec_model.decoder(explanation_latent)

        # Decode the original (The unmodified test sample)
        audio_orig = encodec_model.decoder(target_embedding)

    sf.write(OUTPUT_FILE, audio_expl.squeeze().cpu().numpy(), utils.TARGET_SR)
    print(f"SAVED Explanation: {OUTPUT_FILE}")

    original_filename = "original_" + OUTPUT_FILE
    sf.write(original_filename, audio_orig.squeeze().cpu().numpy(), utils.TARGET_SR)
    print(f"SAVED Original:    {original_filename}")

    print("\nDONE! You can now listen to:")
    print(f"1. {original_filename} (A sample from Test Set detected as FAKE)")
    print(f"2. {OUTPUT_FILE} (The artifacts that gave it away)")
