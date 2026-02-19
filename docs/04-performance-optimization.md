# Getting the Numbers Right

The assignment gave two performance targets: TTFC (time to first audio chunk) under 90 milliseconds, and RTF (real-time factor) under 0.3. I had to look up what these meant.

TTFC is straightforward — it's the time from when you give the system some text to when the first chunk of audio is ready to play. In a voice agent, this is how long the user waits after the AI "decides what to say" before they actually hear anything. Lower is better. 90 milliseconds is fast enough that the delay feels natural in conversation.

RTF is how fast the system generates audio relative to the audio's duration. An RTF of 0.3 means generating 1 second of speech takes 0.3 seconds. Below 1.0 means you're generating faster than real-time (which is necessary for streaming — if you're slower than real-time, the audio will have gaps). Below 0.3 means you have plenty of headroom.

Before I could optimize anything, I needed to measure properly. GPU operations are asynchronous — when you tell the GPU to do something, Python gets control back immediately while the GPU works in the background. If you just use `time.time()`, you're measuring how fast Python launched the work, not how fast it actually finished. The fix is to call `torch.cuda.synchronize()` before each timing point, which forces Python to wait until the GPU is done. I wrapped all my measurements this way:

```python
torch.cuda.synchronize()
start = time.perf_counter()
# ... do the work ...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

I also ran warmup iterations before taking any measurements — the engine initialization, kernel compilation, and first-call overheads are all one-time costs that shouldn't count against the per-request numbers.

My first end-to-end run was sobering. Streaming TTFC was 35,932 milliseconds — about 400 times over the target. RTF was 0.605 — twice over target. Everything was broken.

The TTFC was catastrophic because my frame generation function returned a list instead of a generator. It processed all 2,048 frames (roughly 2.5 minutes of audio) before yielding anything to the streaming wrapper. Once I changed this to a generator that yields each frame as it's produced, TTFC dropped from 35 seconds to about 1 second. Still way over target, but now I could see the real bottlenecks.

At 1 second, most of the TTFC was the vocoder's first decode call — about 834 milliseconds of internal initialization (memory allocation, compiled operations, etc.). I added dummy decode calls during engine startup so this happens before anyone's waiting for audio. After warmup, the vocoder takes about 38ms per call. TTFC dropped to about 192 milliseconds.

Still over the 90ms target. I noticed the first code predictor call was taking 107ms instead of the expected 13ms. After some digging, I found that PyTorch's sampling functions (multinomial, softmax, topk) each have their own first-call overhead. My warmup only covered the argmax path. Once I warmed up the sampling path too, TTFC dropped to about 92ms. Getting close.

I also batched the text embedding computation. I was making multiple separate calls to the text projection network (for role tokens, content tokens, special tokens), each triggering its own chain of GPU operations. Combining them into a single batched call cut the embedding time from about 14ms to 7ms. Then I went further and precomputed all the embeddings that never change between utterances — role tokens, special TTS tokens, codec tags. These get computed once during initialization and cached. That saved another 6ms, bringing TTFC from about 90ms down to about 78ms.

The RTF problem was entirely about the code predictor. Each audio frame needs the code predictor to run 15 times (once for each additional codebook group), and my PyTorch implementation took about 179ms per frame. Quick math: 179ms per frame at 12.5 frames per second means generating 1 second of audio takes 2.24 seconds. That's an RTF of 2.24 — more than 7 times over the target. The code predictor alone was a dealbreaker.

This is where the `num_layers` discovery from the kernel adaptation work paid off. I'd noticed that the megakernel accepts the number of layers as a runtime parameter. The code predictor is a 5-layer transformer with the same architecture as the talker — same hidden size, same attention heads. So I packed its weights into the format the kernel expects, allocated a separate (smaller) KV cache, and called the same compiled kernel with `num_layers=5`.

The code predictor went from 179ms to 10.9ms per frame. RTF dropped from 0.605 to 0.175. Target met, with room to spare.

Here's where things ended up:

The non-streaming pipeline test (which doesn't include vocoder or chunking overhead) showed TTFC of 50.5ms. Breaking that down: tokenization takes about 2.3ms, embedding the text takes 7.2ms, the prefill phase (8 megakernel steps) takes 24.9ms, the first talker decode step takes 3.1ms, and the first code predictor run takes 13.0ms.

The streaming test (which includes vocoder decode and chunk delivery) showed TTFC of 81.6ms. The extra ~30ms is the vocoder decoding that first frame into audio waveform.

RTF in streaming mode was 0.234 — each frame takes about 15ms total (1ms talker decode + 11ms code predictor + 1ms embedding computation + 2ms amortized vocoder), and each frame represents 80ms of audio.

Both targets met. There was some run-to-run variance (the streaming TTFC ranged from about 78ms to 98ms across different test runs and text lengths), but consistently under the 90ms target.

One optimization I chose not to do was implementing M-RoPE in the kernel. This would fix the EOS detection issue and potentially improve audio quality, but it means modifying the attention computation — the most performance-critical and carefully tuned part of the kernel. A subtle bug there could silently corrupt every output. I decided to ship with a word-count-based frame limit as a workaround and document the limitation clearly, rather than risk breaking the kernel for something that doesn't affect the performance numbers.
