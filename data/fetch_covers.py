"""
Step 1: Fetch album covers from Spotify for all songs in the dataset.

Uses the Spotify Web API (via spotipy) to search for each song and download
the album art. Images are saved as 512x512 JPEGs.

Usage:
    python data/fetch_covers.py --dataset /path/to/cleaned_combined.json

Requires .env with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.
"""

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
    """Convert text to filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '_', text)
    text = re.sub(r'-+', '-', text)
    return text[:80]


def search_album_cover(sp: spotipy.Spotify, artist: str, song: str) -> str | None:
    """
    Search Spotify for a track and return the highest-res album cover URL.
    Returns None if not found.
    """
    # Try exact search first
    query = f'track:"{song}" artist:"{artist}"'
    results = sp.search(q=query, type='track', limit=5)
    tracks = results.get('tracks', {}).get('items', [])

    # Fallback: looser search
    if not tracks:
        query = f'{song} {artist}'
        results = sp.search(q=query, type='track', limit=5)
        tracks = results.get('tracks', {}).get('items', [])

    if not tracks:
        return None

    # Get the album images from the first result (sorted by size descending)
    album_images = tracks[0].get('album', {}).get('images', [])
    if not album_images:
        return None

    # First image is the largest (usually 640x640)
    return album_images[0]['url']


def download_and_resize(url: str, save_path: str, size: int = 512) -> bool:
    """Download image from URL, resize to size x size, save as JPEG."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(save_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Fetch album covers from Spotify')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to cleaned_combined.json')
    parser.add_argument('--output_dir', type=str, default='data/covers',
                        help='Directory to save cover images')
    parser.add_argument('--manifest', type=str, default='data/covers_manifest.json',
                        help='Path to save the manifest (image paths + metadata)')
    parser.add_argument('--size', type=int, default=512,
                        help='Output image size (default: 512)')
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise ValueError("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")

    # Init Spotify client
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ),
        requests_timeout=15
    )

    # Load dataset
    with open(args.dataset, 'r', encoding='utf-8') as f:
        songs = json.load(f)
    print(f"Loaded {len(songs)} songs from {args.dataset}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support: load existing manifest
    manifest = []
    processed = set()
    if os.path.exists(args.manifest):
        with open(args.manifest, 'r') as f:
            manifest = json.load(f)
            for entry in manifest:
                processed.add((entry['artist'].lower(), entry['song'].lower()))
        print(f"Resuming: {len(manifest)} covers already fetched")

    # Stats
    found = len(manifest)
    missed = 0
    errors = 0

    for i, song in enumerate(tqdm(songs, desc="Fetching covers")):
        artist = song['artist']
        title = song['song']

        # Skip already processed
        if (artist.lower(), title.lower()) in processed:
            continue

        # Search Spotify
        try:
            cover_url = search_album_cover(sp, artist, title)
        except Exception as e:
            print(f"\n  Spotify API error for '{title}' by {artist}: {e}")
            errors += 1
            time.sleep(2)
            continue

        if not cover_url:
            missed += 1
            continue

        # Download and save
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

            # Checkpoint every 20
            if found % 20 == 0:
                with open(args.manifest, 'w') as f:
                    json.dump(manifest, f, indent=2)
                tqdm.write(f"  Checkpoint: {found} covers saved")

        # Rate limiting (Spotify allows ~30 req/s but be polite)
        time.sleep(0.15)

    # Final save
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done! Results:")
    print(f"  Found & downloaded: {found}")
    print(f"  Not found on Spotify: {missed}")
    print(f"  API errors: {errors}")
    print(f"  Total in dataset: {len(songs)}")
    print(f"  Hit rate: {found/len(songs)*100:.1f}%")
    print(f"  Manifest saved to: {args.manifest}")
    print(f"  Covers saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
