# Getting a GPU and Making Things Work

When I first read the assignment, two things jumped out at me: "RTX 5090" and "CUDA megakernel." I didn't really know what a megakernel was, so the blog post by AlpinDale was my first stop. Reading through it, I started to understand the basic idea — instead of having PyTorch fire off hundreds of tiny GPU operations (one for each matrix multiply, one for each normalization, etc.), a megakernel packs the entire transformer forward pass into a single GPU program. Fewer launches, less overhead, faster inference. It made intuitive sense even before I understood the details.

The assignment also said I'd need an RTX 5090 specifically. The kernel is tuned for this chip's architecture, so no other GPU would work. I'd never rented a GPU before, so I went to vast.ai.

The search page has a ton of filters and I didn't know which ones mattered at first. I worked backwards from the project requirements. The kernel code referenced something called `sm_120a`, which I Googled and learned is NVIDIA's architecture code for Blackwell — the RTX 5090's chip family. This requires CUDA 12.8 or newer, so that became my non-negotiable filter. I also set a minimum of 32 GB RAM (the model needs to load into CPU memory before it moves to the GPU), 100+ Mbps download speed (so downloading the 1.2 GB model weights wouldn't take forever), and 97%+ reliability (I didn't want the machine dying in the middle of compiling).

The one thing that almost tripped me up was the Docker image. When you rent on vast.ai, it asks you to pick a Docker image and pre-installed software for your machine. Most of the available images had older CUDA versions that wouldn't work with the RTX 5090. I ended up using the machine's default template with CUDA 12.8 and Python 3.12, then installed PyTorch myself with `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

Before touching any project code, I wanted to make sure the basics worked. I ran `nvidia-smi` to confirm the GPU was visible, checked that PyTorch could see it, and then ran AlpinDale's original benchmark: `python3 -m qwen_megakernel.bench`. It compiled the kernel (about 60 seconds the first time) and reported ~1,036 tok/s, matching what the blog post claimed. The machine was ready.

My workflow for the rest of the project was to write and edit code on my MacBook (more comfortable, better tools), then sync changes to the GPU machine over SSH to test. Each edit-sync-test cycle took about 5 seconds, which kept things moving fast. Nothing fancy — just rsync and SSH.
