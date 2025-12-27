from argparse import ArgumentParser
from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization_spleeter import SpleeterFactorization
from audioLIME.lime_audio import LimeAudioExplainer
import librosa.display
import matplotlib.pyplot as plt
import os
import soundfile as sf
import torch

from utils import (
    audio_length_per_model,
    composition_fn,
    create_predict_fn,
    get_model,
    path_models,
    prepare_audio,
    tags_msd as tags,
    won2020_default_config as config,
)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--samples_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="fcn")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_temporal_segments", type=int)
    parser.add_argument("--n_display_components", type=int, default=3)
    parser.add_argument("--n_chunks", type=int, default=16)
    parser.add_argument("--use_global_tag", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    path_experiments = args.out_dir
    n_display_components = args.n_display_components
    batch_size = args.batch_size
    num_samples = args.num_samples
    n_segments = args.n_temporal_segments
    samples_path = args.samples_dir
    available_samples = os.listdir(args.samples_dir)
    sample_rate = 16000

    config.model_type = args.model_type
    config.model_load_path = os.path.join(
        path_models, config.dataset, config.model_type, "best_model.pth"
    )
    config.input_length = audio_length_per_model[args.model_type]
    config.batch_size = args.n_chunks

    model = get_model(config)
    if model is None:
        raise Exception("Could not fetch the chosen model")

    for sample in available_samples:
        audio_path = os.path.join(samples_path, sample)
        data_provider = RawAudioProvider(audio_path)
        x, snippet_starts = prepare_audio(
            audio_path,
            config.input_length,
            nr_chunks=config.batch_size,
            return_snippet_starts=True,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        x = x.to(device)
        outputs = model(x)
        top_tag_per_snippet = torch.argmax(outputs.detach().cpu(), dim=1)

        print(
            "top_tag_per_snippet",
            len(top_tag_per_snippet),
            top_tag_per_snippet,
        )

        sorted_args = torch.argsort(
            outputs.detach().cpu().mean(axis=0), descending=True
        )

        print([tags[t] for t in sorted_args[0:3]])

        top_idx = sorted_args[0].item()
        sorted_snippets = torch.argsort(
            outputs[:, top_idx].detach().cpu()
        ).numpy()
        print("top idx:", top_idx)
        print("top segments:", sorted_snippets)

        predict_fn = create_predict_fn(model, config)
        print(data_provider.get_mix())
        spleeter_factorization = SpleeterFactorization(
            data_provider.get_mix(),
            temporal_segmentation_params=n_segments,
            composition_fn=composition_fn,
            model_name="spleeter:5stems",
        )

        explainer = LimeAudioExplainer(
            verbose=True, absolute_feature_sort=False
        )

        for sn in range(len(snippet_starts)):
            if args.use_global_tag:
                labels = [top_idx]
            else:
                snippet_tag = top_tag_per_snippet[sn].item()
                labels = [snippet_tag]

            print("processing {}_{}".format(sample, sn))

            explanation_name = (
                "{}/{}_cls{}_sntag{}_nc{}_sn{}_seg{}_smp{}_nd{}".format(
                    config.model_type,
                    sample,
                    top_idx,
                    labels[0],
                    config.batch_size,
                    sn,
                    n_segments,
                    num_samples,
                    n_display_components,
                )
            )
            explanation_path = os.path.join(
                path_experiments, explanation_name + ".pt"
            )

            # Create any missing directories
            explanation_dir = os.path.dirname(explanation_path)
            os.makedirs(explanation_dir, exist_ok=True)

            spleeter_factorization.set_analysis_window(
                snippet_starts[sn], config.input_length
            )
            print(
                "mix length",
                len(data_provider.get_mix()),
            )

            print("Computing explanation ...")
            explanation = explainer.explain_instance(
                factorization=spleeter_factorization,
                predict_fn=predict_fn,
                labels=labels,
                num_samples=num_samples,
                batch_size=batch_size,
            )

            print(explanation.local_exp)

            top_components, component_indeces = (
                explanation.get_sorted_components(
                    labels[0],
                    positive_components=True,
                    negative_components=False,
                    num_components=n_display_components,
                    return_indeces=True,
                )
            )
            if len(top_components) == 0:
                print("No positive components found")
                continue
            if len(top_components) > 1:
                summed_components = sum(top_components)
                fig = plt.figure()
                for comp in range(len(top_components)):
                    plt.subplot(len(top_components), 1, comp + 1)
                    librosa.display.waveshow(
                        top_components[comp], sr=sample_rate
                    )
                plt.savefig(explanation_path.replace(".pt", "fig1.png"))
            else:
                summed_components = top_components[0]
                fig = plt.figure()
                librosa.display.waveshow(summed_components, sr=sample_rate)
                plt.savefig(explanation_path.replace(".pt", "fig1.png"))

            sf.write(
                explanation_path.replace(".pt", ".wav"),
                summed_components,
                sample_rate,
            )
