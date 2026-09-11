import os
import argparse
from generate_cover import load_pipeline, generate_cover, BASE_MODEL_DEFAULT

TEST_CASES = [
    {
        "name": "trap",
        "visuals": "aggressive, intense, fiery, morning, aesthetic, high quality, landscape",
        "mood": "intense",
        "genre": "trap",
        "lyrics": "Yeah, look, it's 3 AM in Toronto. Might need to send you a check on my style.",
        "seed": 0,
        "output": "outputs/example_1_trap.png",
    },
    {
        "name": "boom_bap",
        "visuals": "vintage boom bap album cover art, 90s gritty brooklyn street corner, cinematic warm lighting, grain texture, classic vinyl record aesthetic, high quality",
        "mood": "nostalgic",
        "genre": "boom bap",
        "lyrics": "Walking down these lonely streets, heart beating to a broken beat. Rain falling on the concrete.",
        "seed": 202,
        "output": "outputs/example_2_boom_bap.png",
    },
    {
        "name": "melodic_rap",
        "visuals": "melodic rap album cover art, dreamy surreal clouds and purple sunset, moody lone silhouette, atmospheric emotional vibe, professional music artwork",
        "mood": "melancholic",
        "genre": "melodic rap",
        "lyrics": "Lost inside the memories, fading with the breeze. Wondering if you still think about me.",
        "seed": 303,
        "output": "outputs/example_3_melodic_rap.png",
    },
    {
        "name": "drill",
        "visuals": "dark drill music album cover art, ominous shadowy alley, misty smoke, dramatic high-contrast chiaroscuro lighting, underground hip hop aesthetic",
        "mood": "menacing",
        "genre": "drill",
        "lyrics": "Pull up to the scene, looking so clean, rolling with the team, living out the dream.",
        "seed": 404,
        "output": "outputs/example_4_dark_drill.png",
    },
    {
        "name": "conscious_rap",
        "visuals": "experimental conscious hip-hop album cover art, surreal floating geometric shapes, symbolic artwork, colorful distorted textures, modern artistic album sleeve",
        "mood": "thoughtful",
        "genre": "conscious rap",
        "lyrics": "Life is a journey, a path we walk alone. Seeking the truth in a world made of stone.",
        "seed": 505,
        "output": "outputs/example_5_conscious_rap.png",
    },
]


def run_tests(lora_path: str = None, base_model: str = BASE_MODEL_DEFAULT, specific_case: str = None):
    pipe = load_pipeline(base_model=base_model, lora_path=lora_path)
    cases = [c for c in TEST_CASES if specific_case is None or c["name"] == specific_case]

    for item in cases:
        generate_cover(
            pipe=pipe,
            visuals=item["visuals"],
            mood=item["mood"],
            genre=item["genre"],
            lyrics=item["lyrics"],
            seed=item["seed"],
            save_path=item["output"],
            base_model=base_model,
        )


def main():
    parser = argparse.ArgumentParser(description="Test album cover generation across styles")
    parser.add_argument("--lora_path", type=str, default="training/best_pix", help="Path to LoRA weights")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_DEFAULT, help="Base model ID")
    parser.add_argument("--case", type=str, default=None, choices=["trap", "boom_bap", "melodic_rap", "drill", "conscious_rap"])
    args = parser.parse_args()

    lora_path = args.lora_path if args.lora_path and os.path.exists(args.lora_path) else None
    run_tests(lora_path=lora_path, base_model=args.base_model, specific_case=args.case)


if __name__ == "__main__":
    main()
