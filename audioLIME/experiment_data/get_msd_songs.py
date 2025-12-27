import os
import sqlite3
import yt_dlp
import argparse


def get_metadata(db_path, track_id):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = "SELECT artist_name, title FROM songs WHERE track_id = ?"
        cursor.execute(query, (track_id,))

        result = cursor.fetchone()
        conn.close()

        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def download_audio(output_dir, artist, title, track_id):
    search_query = f"{artist} - {title} official audio"
    output_path = os.path.join(output_dir, f"{track_id}")

    ydl_opts: yt_dlp._Params = {
        "format": "bestaudio/best",
        "outtmpl": output_path + ".%(ext)s",
        "default_search": "ytsearch1",
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading: {artist} - {title}...")
            ydl.download([search_query])
        except Exception as e:
            print(f"Failed to download {track_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download songs using Million Song Dataset track IDs."
    )

    parser.add_argument(
        "--db_path",
        type=str,
        default="track_metadata.db",
        help="Path to the MSD SQLite database (track_metadata.db)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./songs",
        help="Directory where MP3 files will be saved",
    )
    parser.add_argument(
        "--txt_file",
        type=str,
        default="file_list.txt",
        help="Path to the TXT file containing track IDs",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not os.path.exists(args.txt_file):
        print(f"Error: {args.txt_file} not found.")
        exit(1)

    # Read song track IDs
    with open(args.txt_file, "r") as f:
        track_ids = [line.strip() for line in f if line.strip()]

    # Get info about the song and attempt to download it
    for tid in track_ids:
        metadata = get_metadata(args.db_path, tid)

        if metadata:
            artist_name, song_title = metadata
            download_audio(args.out_dir, artist_name, song_title, tid)
        else:
            print(f"ID {tid} not found in database.")
