# MultimodalBBDM
Code for ”Multimodal Brownian Bridge Diffusion Model for Controllable Synthetic Medical Image Generation“

---

## TODO 
- [ ] Release training & inference code (MBBDM UNet + ACBN modules)
- [ ] Upload pretrained checkpoints for Kvasir-SEG / ISIC 2016
- [ ] Publish structured text annotations and prompt template
- [ ] Add `scripts/reproduce.sh` with exact commands + seeds
- [ ] Provide Dockerfile and environment lockfile
- [ ] Release example masks & sample outputs for quick demo
- [ ] Add qualitative figures and result tables to `assets/`

---

## Overview
**Problem.** Medical imaging often lacks large, diverse, labeled data; pure mask-guided synthesis misses fine semantic details (color/texture/subtype).  
**Method.** We propose **MBBDM**, a dual-guided Brownian-Bridge diffusion model that conditions on **segmentation masks** (anatomy/geometry) and **structured text** (appearance semantics). A new **Attention Conditional Batch Normalization (ACBN)** injects text features at every scale of the UNet denoiser to modulate lesion color, texture, and stage while preserving the mask geometry. :contentReference[oaicite:27]{index=27} :contentReference[oaicite:28]{index=28}  
**Contributions.** (i) Multimodal (mask+text) controllable medical image synthesis, (ii) **ACBN** for fine-grained text-aware modulation, (iii) structured text dataset generated via a multimodal LLM and released with the code, (iv) consistent gains on **Kvasir-SEG** and **ISIC 2016** for downstream segmentation (ResUNet++ / DDANet). :contentReference[oaicite:29]{index=29}

---


---

## Environment & Installation
**Requirements**
- OS: Ubuntu 22.04 
- Python ≥3.10; PyTorch 2.6.0 (CUDA 12.4)


**Setup**
Todo

Data & Ethics
Datasets. Kvasir-SEG (1,000 polyp images + masks), ISIC 2016 (~900 dermoscopy). Splits: Kvasir 7:1:2 (train/val/synth), ISIC 7:1:1. Synthetic set pairs real masks with generated images and keeps real GT images for comparison. 

Structured text. Context-aware, controlled-vocab descriptions (location, count, size, color, morphology) generated programmatically via a multimodal LLM with strict composition rules and manual verification. 
 

Ethics. De-identified data; follow original dataset licenses.

Directory:
data/
├─ Kvasir-SEG/
│  ├─ images/
│  ├─ masks/
│  └─ texts/    # structured descriptions (to be released)
└─ ISIC2016/
   ├─ images/
   ├─ masks/
   └─ texts/

Method (Brief)
Backbone. UNet denoiser; standard ResBlocks per scale; ACBN replaces all BN layers to inject text at every stage. 

ACBN. Projects features to text space, fuses with token embeddings via MHA, pools to produce per-channel κ/ρ to modulate BN output; handles long token sequences and stabilizes training. 
 

Objective. Brownian-Bridge diffusion loss with mask as endpoint, plus DDIM-style fast sampling. 


Pretrained Weights
Todo

Evaluation
Generation metrics. SSIM ↑, MSE ↓, PSNR ↑, IS ↑. 

Segmentation metrics. IoU/Jaccard ↑, Dice/F1 ↑, Recall ↑, Precision ↑. 

Results (Paper)
Image quality vs baselines (Kvasir-SEG / ISIC 2016). Our method achieves lowest MSE and highest PSNR on both datasets; BBDM shows higher SSIM in some cases but with worse MSE/PSNR. 

Key table excerpt:
Kvasir-SEG — Ours: MSE 0.0120, PSNR 19.55, SSIM 0.8388, IS 1.0314
ISIC 2016 — Ours: MSE 0.0074, PSNR 22.91, SSIM 0.9333, IS 1.0536. 

Downstream segmentation (add synthetic data). Gains on ResUNet++ and DDANet across IoU/Dice/Recall/Precision on both datasets. 

Example (Kvasir-SEG): ResUNet++ Dice 0.7718 (vs 0.7338 baseline). DDANet Precision 0.9044. 

Joint training benefit. Training one model on combined datasets reduces error dramatically (e.g., Kvasir MSE 0.1163 → 0.0120, PSNR +10.2 dB; ISIC MSE 0.0361 → 0.0074, PSNR +8.1 dB). 
 

Ablation. ACBN + CLIP > ACBN + MedBERT and > CBN + CLIP on PSNR/MSE/PCC. (PSNR 21.23 dB; MSE 0.0097; PCC 0.6688). 

Model Card
Intended Use. Research/benchmarking; not for clinical diagnosis.
Inputs. Binary mask + structured text prompt; images (RGB / dermoscopy / endoscopy).
Outputs. Synthetic images that follow mask geometry and text semantics.
Training Data. Kvasir-SEG / ISIC 2016 with structured text prompts. 

Performance. Best PSNR/MSE across datasets; consistent downstream segmentation gains. 
 

Fairness & Bias. Domain shift (device/site/population) may degrade performance.
Safety. Outputs are synthetic; verify against clinical standards before any downstream use.
