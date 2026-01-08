import os

import numpy as np
import torch
from audio_transformer import AudioTransformer
from captum.attr import IntegratedGradients

import utils

MODEL_PATH = "models/transformer_classifier.pth"
SAMPLE_LIMIT = 100


def run_faithfulness_test():
    print(f"--- Running Eval on {utils.DEVICE} ---")

    # Get TEST data
    (_, _), (X_test, y_test) = utils.get_data_splits(test_size=0.2)

    X_test = X_test.to(utils.DEVICE)
    y_test = y_test.to(utils.DEVICE)

    # Use only FAKE samples from the TEST set
    indices_fake = (y_test == 1).nonzero(as_tuple=True)[0]
    X_fake = X_test[indices_fake]

    if len(X_fake) == 0:
        print("Error: No Fake samples in the test set! Try increasing dataset size.")
        exit()

    if SAMPLE_LIMIT:
        X_fake = X_fake[:SAMPLE_LIMIT]
        print(f"Evaluating on {len(X_fake)} separate TEST samples.")

    # Load Model
    model = AudioTransformer(feature_size=128, seq_length=375, num_classes=2).to(
        utils.DEVICE
    )
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=utils.DEVICE))
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
        exit()
    model.eval()

    # Setup XAI
    ig = IntegratedGradients(model)
    ratios_to_remove = [0.0, 0.1, 0.2, 0.3, 0.5]

    print("\n" + "=" * 60)
    print("STARTING FIDELITY EVALUATION (Positive Occlusion Test)")
    print("Using ONLY Test Data (Unseen by model)")
    print("=" * 60)
    print(f"{'Ratio Removed':<15} | {'Avg Fake Prob':<15} | {'Acc (Is Detected?)':<20}")
    print("-" * 60)

    # Main Loop (Positive Only Deletion)
    for ratio in ratios_to_remove:
        current_probs = []
        current_accs = []

        for i in range(len(X_fake)):
            inputs = X_fake[i].unsqueeze(0).detach()
            inputs.requires_grad = True

            attributions = ig.attribute(inputs, target=1)
            attr_np = attributions.detach().cpu().numpy().flatten()

            # Highest POSITIVE first
            sorted_indices = np.argsort(attr_np)[::-1]

            n_remove = int(ratio * len(attr_np))
            mask_flat = np.ones_like(attr_np)
            if n_remove > 0:
                indices_to_remove = sorted_indices[:n_remove]
                mask_flat[indices_to_remove] = 0

            mask = (
                torch.from_numpy(mask_flat.reshape(1, 128, 375))
                .float()
                .to(utils.DEVICE)
            )
            modified_input = inputs * mask

            with torch.no_grad():
                output = model(modified_input)
                probs = torch.nn.functional.softmax(output, dim=1)
                fake_prob = probs[0][1].item()
                is_detected = 1 if fake_prob > 0.5 else 0

                current_probs.append(fake_prob)
                current_accs.append(is_detected)

        avg_prob = np.mean(current_probs)
        avg_acc = np.mean(current_accs)
        print(
            f"{ratio*100:>3.0f}%            | {avg_prob:.4f}          | {avg_acc:.4f}"
        )


if __name__ == "__main__":
    run_faithfulness_test()
