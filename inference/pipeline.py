import os
import sys
import json
import argparse
from pathlib import Path
import torch

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from generate_cover import load_pipeline, generate_cover, format_album_prompt, BASE_MODEL_DEFAULT


ALLOWED_MOODS = [
    "aggressive", "confident", "melancholic", "introspective", "motivational",
    "hype", "dark", "romantic", "vulnerable", "rebellious", "playful",
    "emotional", "storytelling"
]

ALLOWED_GENRES = [
    "trap", "boom_bap", "drill", "grime", "conscious_rap", "melodic_rap",
    "experimental", "underground", "mainstream", "alternative_hiphop"
]

MOOD_VISUALS = {
    "aggressive": "aggressive, intense, fiery, dramatic lighting",
    "confident": "confident, bold, powerful, sharp focus",
    "melancholic": "melancholic, moody, somber, muted colors",
    "introspective": "introspective, contemplative, deep atmosphere",
    "motivational": "motivational, uplifting, triumphant, cinematic",
    "hype": "hype, energetic, electric vibrancy",
    "dark": "dark, ominous, shadowy, high contrast",
    "romantic": "romantic, warm, soulful glow",
    "vulnerable": "vulnerable, raw, emotional texture",
    "rebellious": "rebellious, defiant, gritty street art",
    "playful": "playful, fun, colorful, surreal elements",
    "emotional": "emotional, heartfelt, passionate ambiance",
    "storytelling": "cinematic storytelling, detailed urban narrative",
}


def auto_tag_lyrics(lyrics: str) -> dict:
    if not HAS_GENAI:
        return manual_tags()

    client = genai.Client()
    prompt = (
        "Analyze this rap song and provide tags for album cover art generation.\n"
        f"Select exactly 1 mood from: {ALLOWED_MOODS}\n"
        f"Select exactly 1 genre from: {ALLOWED_GENRES}\n"
        "Respond strictly with valid JSON: {\"mood\": \"...\", \"genre\": \"...\"}\n\n"
        f"Lyrics:\n{lyrics[:2500]}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception:
        return manual_tags()


def manual_tags() -> dict:
    mood = "aggressive"
    genre = "trap"
    return {"mood": mood, "genre": genre}


def main():
    parser = argparse.ArgumentParser(description="End-to-end lyrics to album cover generator")
    parser.add_argument("--lyrics", type=str, default=None, help="Song lyrics string")
    parser.add_argument("--lyrics_file", type=str, default=None, help="Path to lyrics text file")
    parser.add_argument("--mood", type=str, default=None, help="Override mood")
    parser.add_argument("--genre", type=str, default=None, help="Override genre")
    parser.add_argument("--visuals", type=str, default=None, help="Visual style description")
    parser.add_argument("--lora_path", type=str, default="training/best_pix", help="Path to LoRA weights")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_DEFAULT, help="Hugging Face model ID")
    parser.add_argument("--output", type=str, default="outputs/cover.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lyrics = ""
    if args.lyrics_file and os.path.exists(args.lyrics_file):
        with open(args.lyrics_file, "r") as f:
            lyrics = f.read()
    elif args.lyrics:
        lyrics = args.lyrics
    elif not sys.stdin.isatty():
        lyrics = sys.stdin.read()

    mood = args.mood
    genre = args.genre
    if not mood or not genre:
        tags = auto_tag_lyrics(lyrics) if lyrics else manual_tags()
        mood = mood or tags.get("mood", "aggressive")
        genre = genre or tags.get("genre", "trap")

    visuals = args.visuals or MOOD_VISUALS.get(mood, "cinematic album cover artwork")
    lora_path = args.lora_path if args.lora_path and os.path.exists(args.lora_path) else None

    pipe = load_pipeline(base_model=args.base_model, lora_path=lora_path)
    generate_cover(
        pipe=pipe,
        visuals=visuals,
        mood=mood,
        genre=genre,
        lyrics=lyrics if lyrics else None,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        save_path=args.output,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
