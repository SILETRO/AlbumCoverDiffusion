import os
import gc
import re
import math
import json
import time
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

from transformers import T5EncoderModel, AutoTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    text = re.sub(r"-+", "-", text)
    return text[:80]


def truncate_lyrics(lyrics: str, max_chars: int = 250) -> str:
    if len(lyrics) <= max_chars:
        return lyrics
    truncated = lyrics[:max_chars]
    last_newline = truncated.rfind("\n")
    last_period = truncated.rfind(".")
    break_at = max(last_newline, last_period)
    if break_at > max_chars // 2:
        return truncated[: break_at + 1].strip()
    return truncated.strip()


class PreEncodedDataset(Dataset):
    def __init__(self, latents, embeddings, attention_masks):
        self.latents = latents
        self.embeddings = embeddings
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "encoder_hidden_states": self.embeddings[idx],
            "attention_mask": self.attention_masks[idx],
        }


def compute_snr(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    snr = (alpha / sigma) ** 2
    return snr


def load_and_pair_dataset(dataset_path: str, max_lyric_chars: int = 250):
    meta_path = os.path.join(dataset_path, "metadata.jsonl")
    with open(meta_path, "r") as f:
        meta_lines = f.readlines()

    meta_dict = {}
    for line in meta_lines:
        d = json.loads(line)
        key = slugify(f"{d['artist']}_{d['song']}")
        meta_dict[key] = d

    lyrics_path = os.path.join(dataset_path, "cleaned_combined.json")
    with open(lyrics_path, "r") as f:
        rap_data = json.load(f)

    combined_data = []
    for r in rap_data:
        key = slugify(f"{r['artist']}_{r['song']}")
        if key not in meta_dict:
            continue
        meta = meta_dict[key]
        visual_text = meta["text"]
        img_path = os.path.join(dataset_path, meta["file_name"])
        if not os.path.exists(img_path):
            continue

        ann = r.get("annotations", {})
        mood = ann.get("mood", "N/A")
        genre = ann.get("genre", "N/A")
        rhyme = ann.get("rhyme_scheme", "N/A")
        cadence = ann.get("cadence", "N/A")
        lyrics = truncate_lyrics(r.get("lyrics", ""), max_chars=max_lyric_chars)

        prompt = (
            f"[Visuals] {visual_text} "
            f"[Annotations] Mood: {mood}, Genre: {genre}, "
            f"Rhyme Scheme: {rhyme}, Cadence: {cadence} "
            f"[Lyrics] {lyrics}"
        )
        combined_data.append({"image": img_path, "text": prompt})

    return combined_data


def pre_encode_dataset(
    combined_data,
    vae,
    text_encoder,
    tokenizer,
    device0,
    device1,
    resolution=512,
    max_token_length=304,
    batch_size=8,
):
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    all_latents = []
    all_embeddings = []
    all_attention_masks = []

    for start_idx in range(0, len(combined_data), batch_size):
        batch_items = combined_data[start_idx : start_idx + batch_size]
        batch_images = []
        batch_texts = []

        for item in batch_items:
            try:
                img = Image.open(item["image"]).convert("RGB")
                img_tensor = train_transforms(img)
                batch_images.append(img_tensor)
                batch_texts.append(item["text"])
            except Exception:
                continue

        if not batch_images:
            continue

        pixel_values = torch.stack(batch_images).to(device0, dtype=torch.float16)

        with torch.no_grad():
            latent_dist = vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample() * vae.config.scaling_factor

        text_inputs = tokenizer(
            batch_texts,
            max_length=max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(device1)
        attention_mask = text_inputs.attention_mask.to(device1)

        with torch.no_grad():
            encoder_output = text_encoder(input_ids, attention_mask=attention_mask)[0]

        for j in range(len(batch_images)):
            all_latents.append(latents[j].cpu())
            all_embeddings.append(encoder_output[j].cpu())
            all_attention_masks.append(attention_mask[j].cpu())

        del pixel_values, latents, latent_dist, input_ids, attention_mask, encoder_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return PreEncodedDataset(all_latents, all_embeddings, all_attention_masks)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device0)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.base_model, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device1)
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float16
    ).to(device0)
    transformer = Transformer2DModel.from_pretrained(
        args.base_model, subfolder="transformer", torch_dtype=torch.float16
    ).to(device0)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        init_lora_weights="gaussian",
    )

    if args.resume_from and os.path.exists(args.resume_from):
        transformer = PeftModel.from_pretrained(transformer, args.resume_from, is_trainable=True)
    else:
        transformer = get_peft_model(transformer, lora_config)

    transformer.enable_gradient_checkpointing()
    if hasattr(transformer, "enable_input_require_grads"):
        transformer.enable_input_require_grads()

    for p in transformer.parameters():
        if p.requires_grad:
            p.data = p.data.to(device0, dtype=torch.float32)

    combined_data = load_and_pair_dataset(args.dataset_path, max_lyric_chars=args.lyric_max_chars)
    dataset = pre_encode_dataset(
        combined_data,
        vae,
        text_encoder,
        tokenizer,
        device0,
        device1,
        resolution=args.resolution,
        max_token_length=args.max_token_length,
    )

    del text_encoder, vae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device0.type == "cuda"))
    global_step = 0
    loss_history = []
    best_loss = float("inf")
    ema_loss = None
    ema_decay = 0.99

    resolution_cond = torch.tensor([args.resolution, args.resolution]).repeat(args.batch_size, 1).to(device0)
    aspect_ratio_cond = torch.tensor([1.0]).repeat(args.batch_size, 1).to(device0)
    added_cond_kwargs = {"resolution": resolution_cond, "aspect_ratio": aspect_ratio_cond}

    transformer.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            latents = batch["latents"].to(device0, dtype=torch.float16)
            encoder_hidden_states = batch["encoder_hidden_states"].to(device0, dtype=torch.float16)
            encoder_attention_mask = batch["attention_mask"].to(device0)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device0,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device0.type == "cuda")):
                noise_pred = transformer(
                    noisy_latents,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timesteps,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
                noise_pred = noise_pred.chunk(2, dim=1)[0]

            if args.snr_gamma is not None:
                snr = compute_snr(timesteps, noise_scheduler)
                mse = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                mse = mse.mean(dim=list(range(1, len(mse.shape))))
                weight = torch.clamp(snr, max=args.snr_gamma) / snr
                loss = (weight * mse).mean()
            else:
                loss = F.mse_loss(noise_pred.float(), noise.float())

            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in transformer.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.checkpoint_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    transformer.save_pretrained(ckpt_path)

            cur_loss = loss.item() * args.grad_accum
            epoch_loss += cur_loss
            num_batches += 1
            if ema_loss is None:
                ema_loss = cur_loss
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * cur_loss

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            transformer.save_pretrained(best_path)

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    transformer.save_pretrained(final_path)

    with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
        json.dump(
            {
                "epoch_losses": loss_history,
                "final_ema_loss": ema_loss,
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Train PixArt-Sigma LoRA for Album Covers")
    parser.add_argument("--dataset_path", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="training/lora_album_cover")
    parser.add_argument("--base_model", type=str, default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--snr_gamma", type=float, default=5.0)
    parser.add_argument("--max_token_length", type=int, default=304)
    parser.add_argument("--lyric_max_chars", type=int, default=250)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
