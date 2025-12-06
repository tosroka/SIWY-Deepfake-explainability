from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Protocol, cast
import argparse
import json
import os
import random
import re
import requests
import sqlite3
import subprocess


class Arguments(Protocol):
    input: str
    limit: int
    output: str
    test_ratio: float
    train_ratio: float
    val_ratio: float
    workers: int


def download_song(
    row: list,
    columns: list,
    audio_dir: Path,
    metadata_dir: Path,
) -> str | None:
    track_id = row[0]
    lastfm_url = row[3]
    output_filepath = audio_dir / f"{track_id}.mp3"

    if (
        output_filepath.exists()
        and (metadata_dir / f"{track_id}.json").exists()
    ):
        return track_id

    try:
        response = requests.get(lastfm_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return None

    youtube_match = re.search(
        r'href="https://www.youtube.com/watch\?v=(.*?)"', response.text
    )
    if not youtube_match:
        return None

    youtube_id = youtube_match.group(1)
    youtube_url = f"https://www.youtube.com/watch?v={youtube_id}"

    if not output_filepath.exists():
        try:
            command = [
                "yt-dlp",
                "--extract-audio",
                "--audio-format",
                "mp3",
                "--output",
                f"{audio_dir}/{track_id}.%(ext)s",
                youtube_url,
            ]
            subprocess.run(
                command, check=True, timeout=90, capture_output=True
            )
        except Exception as e:
            print(f"Unexpected error during download for {track_id}: {e}")
            return None

    try:
        metadata = dict(zip(columns, row))
        metadata["youtube_url"] = youtube_url

        json_file_path = metadata_dir / f"{track_id}.json"
        with open(json_file_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return track_id

    except Exception as e:
        print(f"Error saving metadata for {track_id}: {e}")
        return None


def main(args: Arguments):
    db_file_path = args.input

    if not os.path.exists(db_file_path):
        print(f"Database file '{db_file_path}' not found. Downloading...")
        os.system(
            "wget https://github.com/renesemela/lastfm-dataset-2020/raw/master/datasets/lastfm_dataset_2020/lastfm_dataset_2020.db"
        )
        if not os.path.exists(db_file_path):
            print("Failed to download database. Exiting.")
            return

    output_root = Path(args.output)
    audio_dir = output_root / "audio"
    metadata_dir = output_root / "metadata"

    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(db_file_path)
        sql_query = "SELECT * FROM metadata"
        rows = conn.execute(sql_query).fetchall()
        columns = [
            description[0]
            for description in conn.execute(sql_query).description
        ]
        conn.close()
    except Exception as e:
        print(f"Error querying the database: {e}")
        return

    random.seed(0)
    random.shuffle(rows)

    print(f"Input file: {db_file_path}")
    print(f"Target directory: {audio_dir.resolve()}")
    print(f"Max workers: {args.workers}")

    all_data_objects = rows
    if args.limit >= 0:
        all_data_objects = all_data_objects[: args.limit]
        print(
            f"Download limit: {args.limit} record{'s' if args.limit > 1 else ''}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_song,
                row,
                columns,
                audio_dir,
                metadata_dir,
            )
            for row in all_data_objects
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(all_data_objects),
            desc="Downloading Audio & Saving Metadata",
            unit="files",
        ):
            future.result()

    unique_ids = [f.stem for f in audio_dir.glob("*.mp3")]

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
        base_size if base_size > 150 else max(1, int(0.25 * len(unique_ids)))
    )

    splits = {
        "sample.txt": unique_ids[:sample_size],
        "test.txt": test_ids,
        "train.txt": train_ids,
        "val.txt": val_ids,
    }

    print("\nSaving IDs to files:")
    for filename, ids in splits.items():
        file_path = output_root / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        print(f"  -> {filename}: {len(ids)} IDs saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and process Last.fm audio data concurrently from the SQL DB."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to the SQLite database file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./",
        help="Root output directory where audio and metadata folders will be created. (Default: Current folder)",
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
        default=10000,
        help="Maximum number of records to process from the database. Set to -1 for all records. (Default: 10000)",
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
        print("Provide input DB file!")
        exit()

    main(cast(Arguments, args))
