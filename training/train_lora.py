import os
import math
import yaml
import json
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision import transforms
from torch.utils.data import DataLoader


def compute_snr(timesteps, noise_scheduler):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    alpha = sqrt_alphas_cumprod[timesteps]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps]
    snr = (alpha / sigma) ** 2
    return snr


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/train_config.yaml")
    parser.add_argument("--dataset_path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_path:
        config["dataset_path"] = args.dataset_path

    resume_from = config.get("resume_from", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(
        config["base_model"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        config["base_model"], subfolder="text_encoder"
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        config["base_model"], subfolder="vae"
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        config["base_model"], subfolder="unet"
    ).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["base_model"], subfolder="scheduler"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config["lora_target_modules"],
        init_lora_weights="gaussian",
    )

    if resume_from and os.path.exists(resume_from):
        unet = PeftModel.from_pretrained(unet, resume_from, is_trainable=True)
    else:
        unet = get_peft_model(unet, lora_config)

    if config.get("gradient_checkpointing", False):
        unet.enable_gradient_checkpointing()

    if config.get("enable_xformers", False):
        try:
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        except ImportError:
            pass

    dataset = load_dataset("imagefolder", data_dir=config["dataset_path"], split="train")

    resolution = config["resolution"]
    train_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = [train_transforms(img.convert("RGB")) for img in examples["image"]]
        captions = examples["text"]
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": torch.stack(images),
            "input_ids": inputs.input_ids,
        }

    dataset.set_transform(preprocess)

    dataloader = DataLoader(
        dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if config.get("use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optimizer_cls(
        trainable_params,
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
    )

    steps_per_epoch = math.ceil(
        len(dataloader) / config["gradient_accumulation_steps"]
    )
    num_update_steps = steps_per_epoch * config["num_train_epochs"]

    lr_scheduler = get_scheduler(
        config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["lr_warmup_steps"],
        num_training_steps=num_update_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if config["mixed_precision"] == "fp16" else None
    autocast_dtype = torch.float16 if config["mixed_precision"] == "fp16" else torch.float32

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    snr_gamma = config.get("snr_gamma", None)
    global_step = 0
    best_loss = float("inf")
    loss_history = []
    ema_loss = None
    ema_decay = 0.99
    start_epoch = 0
    resume_step_in_epoch = 0

    if resume_from and os.path.exists(resume_from):
        state_path = os.path.join(resume_from, "training_state.pt")
        if os.path.exists(state_path):
            ckpt_state = torch.load(state_path, map_location=device)
            global_step = ckpt_state["global_step"]
            best_loss = ckpt_state.get("best_loss", float("inf"))
            loss_history = ckpt_state.get("loss_history", [])
            ema_loss = ckpt_state.get("ema_loss", None)
            optimizer.load_state_dict(ckpt_state["optimizer"])
            lr_scheduler.load_state_dict(ckpt_state["lr_scheduler"])
            if scaler is not None and "scaler" in ckpt_state:
                scaler.load_state_dict(ckpt_state["scaler"])
            start_epoch = global_step // steps_per_epoch
            resume_step_in_epoch = (
                (global_step % steps_per_epoch) * config["gradient_accumulation_steps"]
            )
        else:
            try:
                ckpt_name = os.path.basename(resume_from.rstrip("/"))
                global_step = int(ckpt_name.split("-")[-1])
                start_epoch = global_step // steps_per_epoch
                resume_step_in_epoch = (
                    (global_step % steps_per_epoch) * config["gradient_accumulation_steps"]
                )
                for _ in range(global_step):
                    lr_scheduler.step()
            except (ValueError, IndexError):
                pass

    unet.train()

    for epoch in range(start_epoch, config["num_train_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{config['num_train_epochs']}",
            leave=False,
        )

        for step, batch in enumerate(progress_bar):
            if epoch == start_epoch and step < resume_step_in_epoch:
                continue

            pixel_values = batch["pixel_values"].to(device, dtype=autocast_dtype)
            input_ids = batch["input_ids"].to(device)

            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                if snr_gamma is not None:
                    snr = compute_snr(timesteps, noise_scheduler)
                    mse_loss = F.mse_loss(
                        noise_pred.float(), noise.float(), reduction="none"
                    )
                    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))
                    snr_weight = torch.clamp(snr, max=snr_gamma) / snr
                    loss = (snr_weight * mse_loss).mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                loss = loss / config["gradient_accumulation_steps"]

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config["gradient_accumulation_steps"] == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["max_grad_norm"])
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step > 0 and global_step % config["checkpointing_steps"] == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True)
                    unet.save_pretrained(save_path)
                    state_dict = {
                        "global_step": global_step,
                        "best_loss": best_loss,
                        "loss_history": loss_history,
                        "ema_loss": ema_loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }
                    if scaler is not None:
                        state_dict["scaler"] = scaler.state_dict()
                    torch.save(state_dict, save_path / "training_state.pt")

            cur_loss = loss.item() * config["gradient_accumulation_steps"]
            epoch_loss += cur_loss
            num_batches += 1

            if ema_loss is None:
                ema_loss = cur_loss
            else:
                ema_loss = ema_decay * ema_loss + (1 - ema_decay) * cur_loss

            progress_bar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "ema": f"{ema_loss:.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best"
            best_path.mkdir(exist_ok=True)
            unet.save_pretrained(best_path)

        if (epoch + 1) % config.get("validation_epochs", 25) == 0:
            unet.eval()
            unet.half()
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                config["base_model"],
                unet=unet,
                torch_dtype=torch.float16,
            ).to(device)
            pipeline.safety_checker = None
            
            val_prompt = config.get(
                "validation_prompt",
                "aggressive trap hip-hop album cover art, professional",
            )
            
            with torch.no_grad():
                images = pipeline(
                    val_prompt,
                    num_images_per_prompt=config.get("num_validation_images", 4),
                    num_inference_steps=30,
                    generator=torch.Generator(device).manual_seed(config["seed"]),
                ).images
            
            val_dir = output_dir / "validation"
            val_dir.mkdir(exist_ok=True)
            for idx, img in enumerate(images):
                img.save(val_dir / f"epoch_{epoch+1}_{idx}.png")
            
            del pipeline
            torch.cuda.empty_cache()
            unet.float()
            unet.train()

    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    unet.save_pretrained(final_path)

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump({"epoch_losses": loss_history, "final_ema_loss": ema_loss}, f, indent=2)


if __name__ == "__main__":
    main()
