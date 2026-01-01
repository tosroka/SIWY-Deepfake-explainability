from pathlib import Path
import argparse
import json
import numpy as np
import re
import shutil
import sqlite3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize audio files into genres based on metadata."
    )

    parser.add_argument(
        "--source",
        default="./data",
        help="Source folder to scan for audio files",
    )
    parser.add_argument(
        "--dest",
        default="./songs",
        help="Destination folder for organized audio",
    )
    parser.add_argument(
        "--output-npy",
        default="./songs/song_labels.npy",
        help="Output filename for the labels file",
    )
    parser.add_argument(
        "--suno",
        default="./suno.jsonl",
        help="Path to Suno metadata JSONL file",
    )
    parser.add_argument(
        "--udio",
        default="./metadata-udio.jsonl",
        help="Path to Udio metadata JSONL file",
    )
    parser.add_argument(
        "--lastfm",
        default="./lastfm_dataset_2020.db",
        help="Path to LastFM metadata database",
    )

    return parser.parse_args()


def simplify_genre(raw_genre: str | None) -> str:
    if not raw_genre:
        return "other"

    # We clean the input of any weird symbols
    genre = re.sub(r"['\[\]\"]", "", str(raw_genre).lower())
    genre = re.sub(r"[_-]", " ", genre)

    if any(
        keyword in genre
        for keyword in [
            "metal",
            "djent",
            "grind",
            "death",
            "thrash",
            "doom",
            "heavy",
            "sludge",
            "breakdown",
            "scream",
            "growl",
            "distortion",
            "hardcore",
        ]
    ):
        return "metal"

    if any(
        keyword in genre
        for keyword in [
            "hip hop",
            "hip-hop",
            "rap",
            "trap",
            "drill",
            "grime",
            "phonk",
            "flow",
            "beatbox",
            "gangsta",
            "boom bap",
        ]
    ):
        return "hip-hop"

    if any(
        keyword in genre
        for keyword in [
            "electronic",
            "techno",
            "house",
            "edm",
            "trance",
            "dubstep",
            "drum and bass",
            "dnb",
            "jungle",
            "garage",
            "disco",
            "eurodance",
            "synth",
            "vaporwave",
            "lo-fi",
            "chill",
            "club",
            "rave",
            "hardstyle",
            "bass",
            "beat",
            "glitch",
            "electro",
            "idm",
        ]
    ):
        return "electronic"

    if any(
        keyword in genre
        for keyword in [
            "rock",
            "punk",
            "grunge",
            "indie",
            "alternative",
            "emo",
            "shoegaze",
            "guitar",
            "riff",
            "psychedelic",
            "surf",
            "britpop",
            "new wave",
        ]
    ):
        return "rock"

    if any(
        keyword in genre
        for keyword in [
            "classical",
            "orchestra",
            "symphony",
            "concerto",
            "sonata",
            "baroque",
            "opera",
            "piano",
            "violin",
            "cello",
            "harp",
            "cinematic",
            "score",
            "ost",
        ]
    ):
        return "classical"

    if any(
        keyword in genre
        for keyword in [
            "reggae",
            "dub",
            "ska",
            "dancehall",
            "rocksteady",
            "ragga",
        ]
    ):
        return "reggae"

    if any(
        keyword in genre
        for keyword in [
            "jazz",
            "swing",
            "big band",
            "bebop",
            "fusion",
            "smooth jazz",
            "saxophone",
            "trumpet",
        ]
    ):
        return "jazz"

    if any(
        keyword in genre for keyword in ["blues", "delta", "chicago", "boogie"]
    ):
        return "blues"

    if any(
        keyword in genre for keyword in ["soul", "funk", "motown", "groove"]
    ):
        return "soul"

    if any(
        keyword in genre
        for keyword in [
            "country",
            "western",
            "honky tonk",
            "americana",
            "bluegrass",
            "cowboy",
        ]
    ):
        return "country"

    if any(
        keyword in genre
        for keyword in [
            "folk",
            "acoustic",
            "singer-songwriter",
            "roots",
            "traditional",
            "celtic",
            "irish",
        ]
    ):
        return "folk"

    if any(
        keyword in genre
        for keyword in [
            "pop",
            "k-pop",
            "j-pop",
            "boy band",
            "girl group",
            "chart",
            "radio",
            "mainstream",
            "ballad",
            "schlager",
            "catchy",
            "melodic",
        ]
    ):
        return "pop"

    return "other"


def load_jsonl_metadata(file_paths: list[str]) -> dict[str, str]:
    lookup = {}

    for path in file_paths:
        p = Path(path)
        if not p.exists():
            continue

        try:
            with open(p, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        id = data.get("id")
                        if not id:
                            continue

                        genre = None
                        if "tags" in data:
                            genre = data["tags"]
                        elif "metadata" in data and "tags" in data["metadata"]:
                            genre = data["metadata"]["tags"]

                        if isinstance(genre, list):
                            genre = ", ".join(genre)

                        if genre:
                            lookup[id] = genre
                        else:
                            lookup[id] = "other"

                    except Exception:
                        continue

        except Exception:
            pass

    return lookup


def get_lastfm_genre_for_file(
    cursor: sqlite3.Cursor, file_id: str, tag_columns: list[str]
) -> str | None:
    try:
        query = "SELECT * FROM tags WHERE id_dataset = ?"
        cursor.execute(query, (file_id,))
        row = cursor.fetchone()

        if row:
            active_tags = []

            for col_name in tag_columns:
                val = row[col_name]

                if val and val > 0:
                    active_tags.append(col_name)

            return ", ".join(active_tags)

    except Exception as e:
        print(f"SQL Error on {file_id}: {e}")

    return None


def organize_music(args: argparse.Namespace):
    dest_path = Path(args.dest)
    lastfm_db = args.lastfm
    numpy_output_file = args.output_npy
    source_path = Path(args.source)
    suno_jsonl = args.suno
    udio_jsonl = args.udio

    print("Loading metadata...")
    json_map = load_jsonl_metadata([suno_jsonl, udio_jsonl])

    db_path = Path(lastfm_db)
    conn = None
    cursor = None
    tag_columns = []

    if db_path.exists():
        print("Connecting to LastFM database...")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Here we fetch the tag columns for later use
        cursor.execute("SELECT * FROM tags LIMIT 1")
        sample_row = cursor.fetchone()
        if sample_row:
            tag_columns = [
                key for key in sample_row.keys() if key != "id_dataset"
            ]
    else:
        print(f"LastFM database file not found at: {db_path}")

    if not source_path.exists():
        print(f"Source folder missing: {source_path}")
        return

    audio_ai_classification = []
    genre_count = {}
    count = 0
    print("Scanning for audio files...")

    for file_path in source_path.rglob("*.mp3"):
        if file_path.name.startswith("."):
            continue

        if dest_path.resolve() in file_path.resolve().parents:
            continue

        file_id = file_path.stem
        raw_genre = None
        is_ai = -1

        if file_id in json_map:
            raw_genre = json_map[file_id]
            is_ai = 1

        if not raw_genre and conn and cursor:
            genre = get_lastfm_genre_for_file(cursor, file_id, tag_columns)

            if genre is not None:
                raw_genre = genre
                is_ai = 0

        if is_ai < 0:
            print(f"Unidentified file: {file_path.name}")
            continue

        audio_ai_classification.append([file_path.name, is_ai])
        simple_genre = simplify_genre(raw_genre)

        if simple_genre not in genre_count:
            genre_count[simple_genre] = {"AI": 0, "nonAI": 0}

        genre_count[simple_genre]["AI" if is_ai == 1 else "nonAI"] += 1

        target_folder = dest_path / simple_genre
        target_folder.mkdir(parents=True, exist_ok=True)
        new_location = target_folder / file_path.name

        if not new_location.exists():
            try:
                shutil.copy2(str(file_path), str(new_location))
                count += 1

            except Exception as e:
                print(f"Error: {e}")
        else:
            pass

    if audio_ai_classification:
        print(count)
        print("Saving audio classification data...")
        arr = np.array(audio_ai_classification, dtype=object)
        np.save(numpy_output_file, arr)

    if conn:
        conn.close()

    print("\nAI/nonAI breakdown by genre:")
    for genre in genre_count:
        print(
            f"{genre}: AI = {genre_count[genre]['AI']}, "
            f"nonAI = {genre_count[genre]['nonAI']}"
        )


if __name__ == "__main__":
    args = parse_arguments()
    organize_music(args)
