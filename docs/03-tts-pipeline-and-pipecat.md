# Building the TTS Pipeline and Pipecat Integration

I came into this not really knowing how text-to-speech systems work. I knew that text goes in and audio comes out, but the actual mechanics were a mystery. So before writing any pipeline code, I sat down with the Qwen3-TTS source code — specifically `modeling_qwen3_tts.py`, which ships with the model — and traced through what happens when you give it a sentence.

The picture that emerged was surprisingly logical once I saw it. First, the text gets tokenized (turned into numbers) and embedded (turned into vectors), with a small neural network resizing those vectors to match the decoder's expectations. Then a "prefill" step primes the decoder with some special tokens — kind of like clearing its throat before it starts speaking. Then the actual generation loop begins: each step, the decoder produces one audio code and some internal state. A second, smaller model (the "code predictor") takes that internal state and expands it into 15 more codes. Together, those 16 codes form one "frame" of audio. The model generates about 12.5 frames per second. Finally, a vocoder — another neural network — takes the accumulated codes and turns them into the actual waveform you can hear, at 24,000 samples per second.

Understanding this pipeline was the foundation for everything else.

The trickiest part was getting the prefill format right. The decoder expects a very specific sequence of tokens before it starts generating audio. I initially guessed at this based on the model's tokenizer documentation, but the audio came out distorted.

I went back to the source code and traced through the exact construction, around line 2136. What I found was an 8-step sequence that includes "thinking tokens" — special markers (with IDs 2155, 2156, 2157) that represent a compressed thinking phase. These aren't documented in the model card or the config file. I only found them by reading the actual generation code line by line. My first attempt had used generic padding tokens in those positions, which gave the model a completely different conditioning signal. Once I matched the exact official format, audio quality improved noticeably.

I also discovered something about how the text feeds into the generation loop. It's not just the previous audio codes that go in as input each step — there's also a text embedding that advances one token per frame. The model is essentially "reading along" with the text as it generates speech. Getting this wrong was subtle — if you include the wrong tokens (like chat template formatting characters at the end of the text), the model tries to "speak" those characters, which produces garbage after the actual content. I figured out from the source code that you need to strip the first token (already consumed during prefill) and the last five (which are just `<|im_end|>\n<|im_start|>assistant\n` formatting).

With the prefill and trailing text sorted out, the model was generating reasonable audio. But it wasn't streaming yet.

The assignment was very clear about streaming: audio chunks should be pushed as they're generated, not buffered until the entire utterance is done. This matters a lot for a voice agent — you want the user to start hearing the response immediately, not wait 5 seconds for the whole sentence to render.

My first version collected all the audio frames into a list and returned them all at once. The streaming wrapper then iterated over this list. The problem was obvious once I looked at the timing: the model generated all 2,048 frames before any audio went out. Streaming TTFC was 35 seconds. Not exactly real-time.

The fix was almost embarrassingly simple. I changed the frame generation function from building a list (`frames.append(code); return frames`) to a Python generator (`yield code`). A generator hands each frame to the consumer the moment it's produced, without waiting for the rest. This single change — literally replacing `append` + `return` with `yield` — dropped TTFC from 35 seconds to about 1 second.

Still too slow, though. The remaining bottleneck was the vocoder's first decode call, which took about 834 milliseconds due to internal lazy initialization (memory allocation, compiled operations, etc.). I added warmup calls during engine startup — a few dummy decode operations that force all this initialization to happen upfront, when nobody's waiting for audio. After warmup, each vocoder call takes about 38ms.

I also changed the first chunk size. Instead of waiting for 10 frames to fill up a chunk (about 800ms of audio), I send the very first frame by itself as soon as it's ready. It's only 80ms of audio — barely a syllable — but it means the user hears something almost instantly. After that first quick chunk, subsequent chunks batch 10 frames for efficiency.

One more thing caught me off guard. I'd warmed up the kernel and the vocoder, but the first code predictor call was still taking 107ms instead of the expected 13ms. After some investigation, I realized that PyTorch's sampling operations — the functions that randomly pick a token from the probability distribution instead of just taking the most likely one — have their own first-call overhead. My warmup only ran the deterministic (argmax) path. Once I added warmup calls that also exercise the sampling path (torch.multinomial, torch.softmax, torch.topk), the first-call penalty disappeared.

All three of these — vocoder warmup, first-frame-first chunking, and sampling path warmup — brought the streaming TTFC from about 1 second down to about 80 milliseconds.

With the engine working and streaming, the Pipecat integration was the cleanest part of the whole project. Pipecat is a framework for building voice agents — it connects speech-to-text, language models, and text-to-speech into a pipeline where data flows through each component. I read through their docs and examples and found that the TTS service interface is well-designed: you subclass `TTSService`, implement a `run_tts` method that yields audio frames, and the framework handles everything else — routing, turn management, interruptions.

My implementation yields a "started" frame, then streams audio chunks as the megakernel engine generates them (converting from the vocoder's float32 output to the 16-bit PCM that Pipecat expects), then yields a "stopped" frame. The engine initialization happens lazily on first use, so there's no startup delay when the pipeline is constructed.

For the full voice agent demo, I wired it together with Deepgram for speech-to-text and OpenAI for the language model response generation. The pipeline flows like a conversation: the user speaks, Deepgram transcribes it, OpenAI generates a text response, and our megakernel TTS speaks it back. There's also a text-only mode that lets you type text and hear it spoken, which is useful for testing the TTS in isolation without needing a microphone or API keys.
