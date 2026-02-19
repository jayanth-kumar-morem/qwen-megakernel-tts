# Finding and Setting Up the RTX 5090

## The GPU Hunt

The megakernel is tuned specifically for the RTX 5090's Blackwell architecture (sm_120a). No other GPU will work — the kernel uses CUDA 12.8+ features and is hand-optimized for the 5090's 170 SMs, 32 GB GDDR7, and 96 MB L2 cache. So the first challenge was finding one to rent.

I went to [vast.ai](https://vast.ai) and started filtering. The critical constraint was **CUDA 12.8+** — without it, the kernel won't compile. Here's what I looked for:

- **GPU**: RTX 5090, single GPU
- **Min CUDA**: 12.8 (non-negotiable — sm_120a requires it)
- **CPU RAM**: >= 32 GB (model loading needs headroom)
- **CPU Cores**: >= 8 (for tokenization, code predictor PyTorch ops)
- **Internet**: >= 100 Mbps download (model weights are ~1.2 GB)
- **Reliability**: >= 97% (didn't want the machine dying mid-compilation)
- **Price**: <= $0.50/hr

I found a machine in South Korea (m:27920) at $0.352/hr — 64 GB RAM, 12th Gen Intel, CUDA 12.8, 99.8% reliability. The price-performance was hard to beat: about $2.80 for an 8-hour session.

## The Docker Image Trap

This is where I almost lost an hour. When you rent on vast.ai, it asks for a Docker image. My first instinct was to search for "PyTorch CUDA 12.8" images. Most community images are stuck on CUDA 11.x or 12.x < 12.8. The RTX 5090 is too new — Blackwell support only landed in CUDA 12.8 and PyTorch 2.7+.

I ended up using the machine's pre-configured template that came with CUDA 12.8 and Python 3.12. Then I installed PyTorch manually:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Environment Verification

Before touching any project code, I verified the basics:

```bash
nvidia-smi                    # RTX 5090, CUDA 12.8 ✓
nvcc --version                # 12.8 ✓
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# NVIDIA GeForce RTX 5090 ✓
```

Then I installed the project dependencies and uploaded my code via rsync:

```bash
pip install transformers huggingface_hub safetensors ninja numpy soundfile qwen-tts pipecat-ai
rsync -avz ./qwen_megakernel/ vast-machine:~/e3/qwen_megakernel/
```

## First Smoke Test: The Vanilla Megakernel

Before making any changes, I ran the original megakernel benchmark to confirm the baseline works:

```bash
python3 -m qwen_megakernel.bench
```

This JIT-compiles the CUDA kernel (takes about 60 seconds the first time) and runs Qwen3-0.6B text generation. I saw ~1,036 tok/s — matching AlpinDale's published numbers. The machine was ready.

## SSH Workflow

Throughout the project, I worked locally on my MacBook M3 (no CUDA) and synced to the GPU machine via SSH:

```bash
ssh -i ~/.ssh/id_personal -p 42240 root@125.136.64.90
rsync -avz -e "ssh -i ~/.ssh/id_personal -p 42240" \
    qwen_megakernel/ root@125.136.64.90:~/e3/qwen_megakernel/
```

This let me use my local editor and tools while running everything on the GPU. The round-trip for a sync + test was about 5 seconds, which kept the iteration loop tight.
