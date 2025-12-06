from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import TypedDict, Protocol, cast
import argparse
import json
import requests


class SunoDataObject(TypedDict, total=False):
    audio_url: str
    id: str


class UdioDataObject(TypedDict, total=False):
    id: str
    song_path: str


class Arguments(Protocol):
    input: str
    limit: int
    output: str
    test_ratio: float
    train_ratio: float
    val_ratio: float
    workers: int


def download_audio_file(
    data_object: SunoDataObject | UdioDataObject, output_dir: Path
) -> str | None:
    audio_id = data_object.get("id")
    audio_url = None

    if "audio_url" in data_object:
        audio_url = data_object["audio_url"]
    elif "song_path" in data_object:
        audio_url = data_object["song_path"]

    if not audio_url or not audio_id:
        return None

    file_name = f"{audio_id}.mp3"
    file_path = output_dir / file_name

    if file_path.exists():
        return audio_id

    try:
        response = requests.get(audio_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(file_path, "wb") as out_f:
            for chunk in response.iter_content(chunk_size=8192):
                out_f.write(chunk)

        return audio_id

    except Exception as e:
        print(f"Error for {file_name}: {e}\n")
        return None


def main(args: Arguments):
    input_path = Path(args.input)
    output_dir = Path(args.output) / "audio"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {input_path.resolve()}")
    print(f"Target directory: {output_dir.resolve()}")
    print(f"Max workers: {args.workers}")

    if args.limit >= 0:
        print(
            f"Download limit: {args.limit} record{'s' if args.limit > 1 else ''}"
        )

    all_data_objects = []
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data_object = json.loads(line)
                    all_data_objects.append(data_object)

                except Exception as e:
                    print(f"Failed to parse JSON: {e}")
                    continue

        print(f"Found {len(all_data_objects)} records in the file")
        if args.limit >= 0:
            all_data_objects = all_data_objects[: args.limit]

        downloaded_ids = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(download_audio_file, data_object, output_dir)
                for data_object in all_data_objects
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(all_data_objects),
                desc="Downloading Audio Files",
                unit="files",
            ):
                audio_id = future.result()

                if audio_id:
                    downloaded_ids.append(audio_id)

        if downloaded_ids:
            unique_ids = list(set(downloaded_ids))

            train_ids = []
            test_ids = []
            val_ids = []

            if len(unique_ids) > 3:
                test_val_pool_ratio = args.test_ratio + args.val_ratio
                train_ids, temp_pool_ids = train_test_split(
                    unique_ids,
                    random_state=42,
                    test_size=test_val_pool_ratio,
                )

                val_split_ratio = args.val_ratio / test_val_pool_ratio
                test_ids, val_ids = train_test_split(
                    temp_pool_ids, test_size=val_split_ratio, random_state=42
                )

            base_size = int(len(unique_ids) * 0.015)
            sample_size = (
                base_size
                if base_size > 150
                else max(1, int(0.25 * len(unique_ids)))
            )

            splits = {
                "sample.txt": unique_ids[:sample_size],
                "test.txt": test_ids,
                "train.txt": train_ids,
                "val.txt": val_ids,
            }

            print("\nSaving IDs to files:")
            for filename, ids in splits.items():
                file_path = Path(args.output) / filename
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(ids) + "\n")
                print(f"  -> {filename}: {len(ids)} IDs saved.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concurrently download audio files based on a JSONL metadata file."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the input JSONL metadata file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Name of the output directory where audio files will be saved. (Default: Current folder)",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=32,
        help="Maximum number of concurrent download threads. (Default: 32)",
    )

    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of records to attempt to download. Set to -1 to process all records. (Default: -1)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Ratio of data for the training set (Default: 0.70)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for the testing set (Default: 0.15)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio of data for the validation set (Default: 0.15)",
    )

    args = parser.parse_args()

    if not args.input:
        print("Provide input JSONL file!")
        exit()

    main(cast(Arguments, args))
