"""
Step 2: Build a HuggingFace-compatible dataset from fetched covers.

Reads the manifest from Step 1 and produces a metadata.jsonl file
in the HuggingFace ImageFolder format, ready for diffusers training.

Usage:
    python data/build_dataset.py --manifest data/covers_manifest.json

Output:
    dataset/
    ├── metadata.jsonl       # {"file_name": "...", "text": "..."}
    └── images/              # Symlinked or copied cover images
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def build_caption(entry: dict) -> str:
    """
    Build a rich, descriptive caption from song metadata for SD conditioning.
    
    Template: "{mood} {genre} hip-hop album cover art, {cadence} tempo,
               {year}s aesthetic, for '{song}' by {artist}"
    """
    annotations = entry.get('annotations', {})
    mood = annotations.get('mood', 'expressive')
    genre = annotations.get('genre', 'hip-hop')
    cadence = annotations.get('cadence', 'midtempo')
    rhyme = annotations.get('rhyme_scheme', '')
    
    artist = entry.get('artist', 'unknown artist')
    song = entry.get('song', 'untitled')
    year = entry.get('year')

    # Map genre tags to more visual descriptions
    genre_visuals = {
        'trap': 'trap, urban, neon-lit',
        'boom_bap': 'boom bap, classic, gritty street',
        'drill': 'drill, dark, menacing urban',
        'grime': 'grime, UK street, raw',
        'conscious_rap': 'conscious rap, thoughtful, artistic',
        'melodic_rap': 'melodic rap, vibrant, atmospheric',
        'experimental': 'experimental, avant-garde, abstract',
        'underground': 'underground, indie, lo-fi',
        'mainstream': 'mainstream, polished, commercial',
        'alternative_hiphop': 'alternative hip-hop, creative, eclectic',
    }
    genre_desc = genre_visuals.get(genre, genre)

    # Map mood to visual tone
    mood_visuals = {
        'aggressive': 'aggressive, intense, fiery',
        'confident': 'confident, bold, powerful',
        'melancholic': 'melancholic, moody, somber',
        'introspective': 'introspective, contemplative, deep',
        'motivational': 'motivational, uplifting, triumphant',
        'hype': 'hype, energetic, electric',
        'dark': 'dark, ominous, shadowy',
        'romantic': 'romantic, warm, soulful',
        'vulnerable': 'vulnerable, raw, emotional',
        'rebellious': 'rebellious, defiant, edgy',
        'playful': 'playful, fun, colorful',
        'emotional': 'emotional, heartfelt, passionate',
        'storytelling': 'narrative, cinematic, storytelling',
    }
    mood_desc = mood_visuals.get(mood, mood)

    # Build year description
    year_desc = ''
    if year and isinstance(year, int) and 1980 <= year <= 2030:
        decade = (year // 10) * 10
        year_desc = f', {decade}s aesthetic'

    caption = (
        f"{mood_desc}, {genre_desc} hip-hop album cover art, "
        f"{cadence} tempo{year_desc}, "
        f"professional music album artwork, high quality, detailed"
    )
    return caption


def main():
    parser = argparse.ArgumentParser(description='Build HuggingFace dataset')
    parser.add_argument('--manifest', type=str, default='data/covers_manifest.json',
                        help='Path to covers manifest')
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help='Output directory for HF dataset')
    parser.add_argument('--copy_images', action='store_true', default=True,
                        help='Copy images into dataset dir (default: True)')
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    print(f"Loaded {len(manifest)} entries from manifest")

    # Create output structure
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    metadata_path = os.path.join(args.output_dir, 'metadata.jsonl')
    
    # Track caption stats
    caption_lengths = []
    mood_counts = {}
    genre_counts = {}

    with open(metadata_path, 'w') as f:
        for entry in tqdm(manifest, desc="Building dataset"):
            src_path = entry['image_path']
            filename = entry['image_filename']
            
            if not os.path.exists(src_path):
                print(f"  Warning: image not found: {src_path}")
                continue

            # Copy image into dataset directory
            dst_path = os.path.join(images_dir, filename)
            if args.copy_images and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

            # Build caption
            caption = build_caption(entry)
            caption_lengths.append(len(caption))

            # Track stats
            annotations = entry.get('annotations', {})
            mood = annotations.get('mood', 'unknown')
            genre = annotations.get('genre', 'unknown')
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

            # Write metadata line
            meta_line = {
                'file_name': os.path.join('images', filename),
                'text': caption,
                'artist': entry['artist'],
                'song': entry['song'],
            }
            f.write(json.dumps(meta_line) + '\n')

    print(f"\n{'='*50}")
    print(f"Dataset built successfully!")
    print(f"  Total images: {len(manifest)}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Images dir: {images_dir}")
    print(f"  Avg caption length: {sum(caption_lengths)/max(len(caption_lengths),1):.0f} chars")
    print(f"\nMood distribution:")
    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        print(f"  {mood}: {count}")
    print(f"\nGenre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  {genre}: {count}")


if __name__ == '__main__':
    main()
