import json
import os
import re
import time
import argparse
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dotenv import load_dotenv


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'-+', '-', text)
    return text[:80]


def search_album_cover(sp: spotipy.Spotify, artist: str, song: str) -> str | None:
    query = f'track:"{song}" artist:"{artist}"'
    results = sp.search(q=query, type='track', limit=5)
    tracks = results.get('tracks', {}).get('items', [])

    if not tracks:
        query = f'{song} {artist}'
        results = sp.search(q=query, type='track', limit=5)
        tracks = results.get('tracks', {}).get('items', [])

    if not tracks:
        return None

    album_images = tracks[0].get('album', {}).get('images', [])
    if not album_images:
        return None

    return album_images[0]['url']


def download_and_resize(url: str, save_path: str, size: int = 512) -> bool:
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/covers')
    parser.add_argument('--manifest', type=str, default='data/covers_manifest.json')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()

    load_dotenv()
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise ValueError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ),
        requests_timeout=15
    )

    with open(args.dataset, 'r', encoding='utf-8') as f:
        songs = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    manifest = []
    processed = set()
    if os.path.exists(args.manifest):
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
            for entry in manifest:
                processed.add((entry['artist'].lower(), entry['song'].lower()))

    found = len(manifest)

    for song in tqdm(songs, desc="Fetching covers"):
        artist = song['artist']
        title = song['song']

        if (artist.lower(), title.lower()) in processed:
            continue

        try:
            cover_url = search_album_cover(sp, artist, title)
        except Exception:
            time.sleep(2)
            continue

        if not cover_url:
            continue

        filename = f"{slugify(artist)}_{slugify(title)}.jpg"
        filepath = os.path.join(args.output_dir, filename)

        if download_and_resize(cover_url, filepath, size=args.size):
            found += 1
            entry = {
                'artist': artist,
                'song': title,
                'year': song.get('year'),
                'annotations': song.get('annotations', {}),
                'image_path': filepath,
                'image_filename': filename
            }
            manifest.append(entry)
            processed.add((artist.lower(), title.lower()))

            if found % 20 == 0:
                with open(args.manifest, 'w') as f:
                    json.dump(manifest, f, indent=2)

        time.sleep(0.15)

    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()
