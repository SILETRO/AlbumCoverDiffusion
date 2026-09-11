import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from diffusers import PixArtSigmaPipeline
from peft import PeftModel

BASE_MODEL_DEFAULT = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

NEGATIVE_PROMPT = (
    "text, watermark, logo, words, letters, blurry, low quality, ugly, "
    "deformed, disfigured, jpeg artifacts, oversaturated, cropped, "
    "out of frame, duplicate, morbid, mutilated"
)


def get_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, dtype, vram_gb
    if torch.backends.mps.is_available():
        return "mps", torch.float16, None
    return "cpu", torch.float32, None


def load_pipeline(base_model: str = BASE_MODEL_DEFAULT, lora_path: str = None):
    device, dtype, vram_gb = get_device_and_dtype()
    
    pipe = PixArtSigmaPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        clean_caption=False,
    )

    if lora_path and os.path.exists(lora_path):
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path)
        pipe.transformer = pipe.transformer.merge_and_unload()

    if device == "cuda":
        if vram_gb is not None and vram_gb < 20.0:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)

    return pipe


def format_album_prompt(
    visuals: str,
    mood: str = None,
    genre: str = None,
    lyrics: str = None,
    max_lyric_chars: int = 250,
) -> str:
    parts = [f"[Visuals] {visuals.strip()}"]
    annotations = []
    if mood:
        annotations.append(f"Mood: {mood.strip()}")
    if genre:
        annotations.append(f"Genre: {genre.strip()}")
    if annotations:
        parts.append(f"[Annotations] {', '.join(annotations)}")
    if lyrics:
        clean_lyric = lyrics.strip()[:max_lyric_chars]
        parts.append(f"[Lyrics] {clean_lyric}")
    return " ".join(parts)


def generate_cover(
    pipe: PixArtSigmaPipeline = None,
    prompt: str = None,
    visuals: str = None,
    mood: str = None,
    genre: str = None,
    lyrics: str = None,
    seed: int = 42,
    steps: int = 30,
    guidance_scale: float = 4.5,
    save_path: str = None,
    lora_path: str = None,
    base_model: str = BASE_MODEL_DEFAULT,
) -> Image.Image:
    if pipe is None:
        pipe = load_pipeline(base_model=base_model, lora_path=lora_path)

    if prompt is None:
        if visuals is None:
            raise ValueError("Provide either 'prompt' or 'visuals'.")
        prompt = format_album_prompt(visuals=visuals, mood=mood, genre=genre, lyrics=lyrics)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        clean_caption=False,
    ).images[0]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        image.save(save_path)

    return image


def main():
    parser = argparse.ArgumentParser(description="Generate album cover using PixArt-Sigma")
    parser.add_argument("--prompt", type=str, default=None, help="Complete prompt string")
    parser.add_argument("--visuals", type=str, default=None, help="Visual description")
    parser.add_argument("--mood", type=str, default=None, help="Mood annotation")
    parser.add_argument("--genre", type=str, default=None, help="Genre annotation")
    parser.add_argument("--lyrics", type=str, default=None, help="Song lyrics snippet")
    parser.add_argument("--lyrics_file", type=str, default=None, help="Path to file containing lyrics")
    parser.add_argument("--lora_path", type=str, default="training/best_pix", help="Path to LoRA weights directory")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_DEFAULT, help="Hugging Face model ID")
    parser.add_argument("--output", type=str, default="outputs/cover.png", help="Output image file path")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lyrics = args.lyrics
    if args.lyrics_file and os.path.exists(args.lyrics_file):
        with open(args.lyrics_file, "r") as f:
            lyrics = f.read()

    lora_path = args.lora_path if args.lora_path and os.path.exists(args.lora_path) else None

    generate_cover(
        prompt=args.prompt,
        visuals=args.visuals,
        mood=args.mood,
        genre=args.genre,
        lyrics=lyrics,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        save_path=args.output,
        lora_path=lora_path,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
