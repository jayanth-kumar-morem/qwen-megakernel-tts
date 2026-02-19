# What I Learned Along the Way

Looking back, a few moments stand out where something clicked and changed how I approached the problem. These aren't things I knew going in — they're things I figured out by reading code, running experiments, and paying attention when something didn't behave the way I expected.

## Reusing the kernel for the code predictor

This was easily the most impactful thing I stumbled onto. The code predictor — the component that takes each audio code and expands it into 15 more codebook groups — is a 5-layer transformer. When I first got it working with standard PyTorch, it took 179ms per audio frame. I did the math and realized that alone blew past the RTF target by a factor of 7. The code predictor was the bottleneck, not the talker decoder.

I'd been thinking of the megakernel as "the thing that runs the 28-layer talker." But while reading the kernel code, I noticed the layer count isn't hardcoded — it's just a number you pass in. The kernel loops from layer 0 to `num_layers`, doing the same thing each time. I started wondering: what if I pass 5 instead of 28?

It took some work to get the code predictor's weights into the exact format the kernel expects, and I had to allocate a separate, smaller KV cache for it. But once I got it running through the megakernel, the time dropped from 179ms to 10.9ms. The whole RTF problem vanished. Zero kernel code changes — just calling the same compiled program with different parameters and different weights.

## The embedding sentinel trick

In normal text generation, the kernel receives a token ID and looks up its embedding from a table. But in TTS, the input at each step is a combination of 16 different embeddings plus a text embedding. There's no single token ID for that.

Rather than launching a separate GPU operation to handle this, I added a tiny check to the kernel: if the token ID is negative, skip the table lookup and read from a pre-filled buffer instead. Python writes the combined embedding into this buffer and passes -1 as the token ID. The kernel sees -1, reads the buffer, and proceeds as normal. It's only 3 lines of CUDA code, but it avoids an extra GPU launch on every single frame.

## Warmup is more complicated than I expected

I kept running into a pattern where the first call to something was dramatically slower than subsequent calls. The vocoder's first decode took 834ms (subsequent: 38ms). The code predictor's first sampling call took 107ms (subsequent: 13ms). Even simple operations like `torch.multinomial` had a noticeable first-call penalty.

What I eventually understood is that CUDA is deeply lazy. Almost everything — memory allocation, kernel compilation, internal buffer setup — is deferred until it's actually needed. The first time you call an operation, you're paying for all that setup. The second time, it's already done.

The tricky part was that different code paths have independent warmup needs. Warming up the argmax path (deterministic token selection) doesn't warm up the sampling path (random selection from top candidates), because they use different underlying operations. I had to warm up both explicitly. Same for the vocoder — I ran dummy decode calls with different input sizes to make sure all the internal buffers got pre-allocated.

I ended up with a warmup routine that runs 5 predict cycles (2 argmax + 3 sampling) plus 3 vocoder decode calls of different sizes. It adds about 2 seconds to startup, but every subsequent request starts fast.

## Keeping everything on the GPU

There was a subtle performance issue in my code predictor loop. I was writing something like `token = logits.argmax().item()` — the `.item()` call pulls the result from the GPU back to the CPU as a Python integer. Then I'd create a new tensor from that integer to pass back to the GPU. This round-trip forced the CPU to wait for the GPU to finish (a "synchronization point"), and it happened 15 times per frame.

The fix was to keep everything as GPU tensors. Instead of `.item()`, I used `.argmax(keepdim=True)` which returns a tensor that stays on the GPU. The embedding lookup can take a tensor directly. No round-trip, no synchronization. It saved about 1.5ms per frame, which adds up across a full utterance.

## The prefill format is specific and undocumented

I spent a while getting poor audio quality and couldn't figure out why. The kernel was producing correct transformer outputs — I'd validated that. The issue turned out to be the prefill format.

The talker decoder expects a very specific 8-step conditioning sequence before it starts generating audio. Three of those steps use "thinking tokens" — special IDs (2155, 2156, 2157) that represent a compressed thinking phase. These aren't mentioned in the model card, the config file, or any documentation I could find. I only discovered them by reading through the model's 1,800-line generation script and tracing the exact construction.

My initial implementation used generic padding tokens in those positions. The model still generated audio, but with noticeably worse quality. Once I matched the exact official format, things improved immediately. The lesson: when working with a specific model's expected input format, the source code is the only reliable documentation.

I had a similar experience with the "trailing text" — the text tokens that feed into the generation loop alongside the audio codes. The official code strips the last 5 tokens from the text before feeding it in, because those tokens are just chat template formatting (`<|im_end|>\n<|im_start|>assistant\n`). Without stripping them, the model tries to "speak" those formatting characters, which produces garbage audio after the actual content finishes.

## Precomputing what doesn't change

Every utterance starts with the same role tokens (`<|im_start|>assistant\n`), the same special TTS tokens, and the same codec thinking markers. I was computing these fresh every time, which meant running the text projection network and the embedding lookups on identical inputs over and over.

Moving these computations to the initialization step — compute once, cache as tensor attributes, reuse forever — saved about 6ms per utterance. That doesn't sound like much, but when your TTFC target is 90ms, 6ms is almost 7% of your budget.

## Word count is a better estimator than character count

Because the M-RoPE limitation means the model doesn't reliably signal when it's done speaking, I needed a heuristic to estimate how many audio frames to generate. My first attempt used character count — something like "3 characters per second." This was wildly off. A 300-character paragraph would get 100 seconds of estimated audio, most of which was silence.

Word count turned out to be much more stable. English speech averages about 150 words per minute, or 2.5 words per second. At 12.5 codec frames per second, each word is about 5 frames. I add a 2x safety margin so the model has room to finish naturally. With this formula, "Hello, how are you today?" (5 words) gets about 50 frames (4 seconds of audio), and a 52-word paragraph gets about 520 frames (42 seconds). The durations match what you'd expect from natural speech.

## What I'd explore next

If I had more time, two things would make the biggest difference. First, implementing M-RoPE in the kernel — this would fix the EOS detection issue, eliminate the frame limit heuristic, and likely improve audio quality for longer utterances. The change is surgical (modify the RoPE rotation to split head dimensions into three groups), but it touches the most performance-sensitive part of the kernel, so it needs careful testing. Second, adding token suppression — the official implementation blocks certain token IDs during generation to prevent the model from emitting meaningless special tokens. Adding this in Python (zeroing out logits before the argmax) would be straightforward and would improve output quality.
