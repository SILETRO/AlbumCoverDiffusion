# AlbumCoverDiffusion

Multi-signal album cover art generation conditioned on song lyrics, genre annotations, mood, and visual descriptions using a fine-tuned PixArt-Sigma Diffusion Transformer (DiT).

---

## Architecture Overview

The system adapts **PixArt-Sigma** (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`) using Low-Rank Adaptation (LoRA) for album art generation conditioned on multi-modal music signals.

```
Song Metadata (Lyrics, Mood, Genre, Visuals)
               │
               ▼
   [Multi-Signal Prompt Formatting]
               │
               ▼
  T5-XXL Text Encoder (304 tokens) ──┐
                                     │
    AutoencoderKL (VAE Latents) ─────┼──► Transformer2DModel (LoRA) ──► Latent Decoding ──► Cover Art
                                     │
   Noise Scheduler (DDPMScheduler) ──┘
```

### Key Technical Specifications

- **Base Model**: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (Diffusion Transformer)
- **Text Encoder**: T5-XXL (`torch.float16`, context length expanded to 304 tokens)
- **VAE**: AutoencoderKL
- **Fine-Tuning Strategy**: PEFT LoRA on Transformer cross-attention and projection layers
- **Target Modules**: `to_q`, `to_k`, `to_v`, `to_out.0`, `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2`
- **LoRA Hyperparameters**: Rank $r = 8$, Alpha $\alpha = 16$, Dropout $= 0.05$
- **Trainable Parameters**: 6,718,720 (~1.088% of the 617.5M transformer parameters)
- **Loss Formulation**: Min-SNR gamma weighting ($\gamma = 5.0$) on MSE loss for balanced noise level gradient steps

### Multi-Signal Conditioning

Prompts are structured with explicit segmented tokens matching the fine-tuning regime:

```
[Visuals] {aesthetic style, composition, lighting} [Annotations] Mood: {mood}, Genre: {genre}, Rhyme Scheme: {rhyme}, Cadence: {cadence} [Lyrics] {truncated lyrics excerpt}
```

---

## Training Details & Loss Metrics

The model was trained on 2,156 curated pairs of hip-hop album covers, track lyrics, and detailed musical annotations.

### Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Hardware | 2x NVIDIA Tesla T4 (16GB VRAM each) |
| Resolution | 512x512 |
| Batch Size | 2 per device (effective batch size 8 with gradient accumulation) |
| Epochs | 4 |
| Learning Rate | 1e-4 with Cosine Annealing schedule |
| Warmup Steps | 50 |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 0.01$) |
| Precision | Mixed Precision (FP16 autocast + FP32 gradient scaling) |

### Training Loss Progression

| Epoch | Loss | EMA Loss | Step Time | Cumulative Time |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.06769 | 0.06233 | 25.3 min | 25.3 min |
| 2 | 0.06752 | 0.06448 | 25.4 min | 50.7 min |
| 3 | 0.06636 | 0.06465 | 25.5 min | 76.7 min |
| 4 | 0.06430 | 0.05600 | 25.4 min | 102.2 min |

---

## Installation

```bash
git clone https://github.com/SILETRO/AlbumCoverDiffusion.git
cd AlbumCoverDiffusion

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

### 1. Generating a Single Album Cover

Generate an album cover with multi-signal inputs:

```bash
python inference/generate_cover.py \
    --visuals "vintage boom bap album cover art, 90s gritty brooklyn street corner, cinematic warm lighting" \
    --mood "nostalgic" \
    --genre "boom bap" \
    --lyrics "Walking down these lonely streets, heart beating to a broken beat." \
    --output outputs/boom_bap_cover.png
```

### 2. Testing Predefined Genre Styles

Run generations across standard rap subgenres (trap, boom bap, drill, melodic, conscious):

```bash
python inference/test_album_cover.py
```

To test a specific subgenre:

```bash
python inference/test_album_cover.py --case trap
```

### 3. End-to-End Pipeline from Lyrics

Generate cover art directly from lyrics with automatic mood and genre inference:

```bash
python inference/pipeline.py \
    --lyrics "Lost inside the memories, fading with the breeze. Wondering if you still think about me." \
    --output outputs/melodic_cover.png
```

### 4. Training

To train on your own dataset in HuggingFace format:

```bash
python training/train_pixart.py \
    --dataset_path dataset \
    --output_dir training/lora_album_cover \
    --epochs 4 \
    --batch_size 2 \
    --grad_accum 4 \
    --learning_rate 1e-4
```

---

## Project Structure

```
AlbumCoverDiffusion/
├── data/
│   ├── fetch_covers.py       # Spotify album art downloader
│   ├── build_dataset.py      # HuggingFace dataset formatter
│   └── covers_manifest.json  # Metadata manifest
├── training/
│   ├── train_pixart.py       # PixArt-Sigma LoRA training script
│   ├── train_lora.py         # SD 1.5 training pipeline
│   ├── train_config.yaml     # Training hyperparameters
│   └── best_pix/             # Trained LoRA adapter configuration
├── inference/
│   ├── generate_cover.py     # PixArt-Sigma inference module & CLI
│   ├── test_album_cover.py   # Style test suite
│   └── pipeline.py           # End-to-end lyrics-to-cover generator
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── README.md
```
