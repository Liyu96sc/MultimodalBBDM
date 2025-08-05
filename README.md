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
**Problem.** 

Medical imaging often lacks large, diverse, labeled data; pure mask-guided synthesis misses fine semantic details (color/texture/subtype).  

**Method.** 

We propose **MBBDM**, a dual-guided Brownian-Bridge diffusion model that conditions on **segmentation masks** (anatomy/geometry) and **structured text** (appearance semantics). A new **Attention Conditional Batch Normalization (ACBN)** injects text features at every scale of the UNet denoiser to modulate lesion color, texture, and stage while preserving the mask geometry. 

**Contributions.** 
(i) Multimodal (mask+text) controllable medical image synthesis, 
(ii) **ACBN** for fine-grained text-aware modulation, 
(iii) structured text dataset generated via a multimodal LLM and released with the code, (iv) consistent gains on **Kvasir-SEG** and **ISIC 2016** for downstream segmentation (ResUNet++ / DDANet). 

---

## Environment & Installation
**Requirements**
- OS: Ubuntu 22.04 
- Python ≥3.10; PyTorch 2.6.0 (CUDA 12.4)


**Setup**

Todo

**Data & Ethics**
Datasets. Kvasir-SEG (1,000 polyp images + masks), ISIC 2016 (~900 dermoscopy). Splits: Kvasir 7:1:2 (train/val/synth), ISIC 7:1:1. Synthetic set pairs real masks with generated images and keeps real GT images for comparison. 

Structured text. Context-aware, controlled-vocab descriptions (location, count, size, color, morphology) generated programmatically via a multimodal LLM with strict composition rules and manual verification. 
 



## Method (Brief)
Backbone. UNet denoiser; standard ResBlocks per scale; ACBN replaces all BN layers to inject text at every stage. 

ACBN. Projects features to text space, fuses with token embeddings via MHA, pools to produce per-channel κ/ρ to modulate BN output; handles long token sequences and stabilizes training. 
 

Objective. Brownian-Bridge diffusion loss with mask as endpoint, plus DDIM-style fast sampling. 


## Pretrained Weights

Todo

