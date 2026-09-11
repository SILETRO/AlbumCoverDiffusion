import json
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def build_caption(entry: dict) -> str:
    annotations = entry.get('annotations', {})
    mood = annotations.get('mood', 'expressive')
    genre = annotations.get('genre', 'hip-hop')
    cadence = annotations.get('cadence', 'midtempo')
    rhyme = annotations.get('rhyme_scheme', '')
    
    artist = entry.get('artist', 'unknown artist')
    song = entry.get('song', 'untitled')
    year = entry.get('year')

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, default='data/covers_manifest.json')
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--copy_images', action='store_true', default=True)
    args = parser.parse_args()

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    metadata_path = os.path.join(args.output_dir, 'metadata.jsonl')
    
    with open(metadata_path, 'w') as f:
        for entry in tqdm(manifest, desc="Building dataset"):
            src_path = entry['image_path']
            filename = entry['image_filename']
            
            if not os.path.exists(src_path):
                continue

            dst_path = os.path.join(images_dir, filename)
            if args.copy_images and not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

            caption = build_caption(entry)

            meta_line = {
                'file_name': os.path.join('images', filename),
                'text': caption,
                'artist': entry['artist'],
                'song': entry['song'],
            }
            f.write(json.dumps(meta_line) + '\n')


if __name__ == '__main__':
    main()
