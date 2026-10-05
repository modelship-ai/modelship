# Example configs

Example `models.yaml` configs; the images also carry them under `/modelship/config/examples/`. The files named after a loader show the different ways to declare a model for it (a Hub reference, a single file, a local path), so take the entries you need instead of deploying the file whole. `full-stack.yaml` and `mini-pc.yaml` are complete stacks: pass either with `--config` as it is.

| File | What it runs | Hardware |
|---|---|---|
| [llama-server.yaml](llama-server.yaml) | Quantized GGUF chat (concurrent), vision, sharded GGUF, embeddings via a llama-server subprocess | CPU or NVIDIA GPU |
| [vllm-cpu.yaml](vllm-cpu.yaml) | Quantized (AWQ/GPTQ) chat via vLLM's CPU backend | CPU |
| [vllm.yaml](vllm.yaml) | High-throughput chat with tool calling, embeddings, Whisper | NVIDIA GPU |
| [vllm-vision.yaml](vllm-vision.yaml) | Vision-language chat (image_url content parts) | NVIDIA GPU |
| [diffusers.yaml](diffusers.yaml) | SDXL Turbo image generation | NVIDIA GPU |
| [stable-diffusion-cpp.yaml](stable-diffusion-cpp.yaml) | GGUF-quantized image generation (SD 2.1, Flux-schnell) | CPU |
| [sherpa-onnx.yaml](sherpa-onnx.yaml) | Kokoro TTS via sherpa-onnx | CPU only |
| [whispercpp.yaml](whispercpp.yaml) | Speech-to-text via pywhispercpp | CPU (Metal on macOS) |
| [full-stack.yaml](full-stack.yaml) | LLM + TTS + STT + embeddings on one GPU | NVIDIA GPU |
| [mini-pc.yaml](mini-pc.yaml) | Low-resource stack: llama-server chat + Kokoro TTS + whisper.cpp STT | CPU (e.g. Intel N100) |

Example:

```bash
docker run --rm --shm-size=8g \
  -v ./models-cache:/.cache \
  -p 8000:8000 \
  ghcr.io/modelship-ai/modelship:latest-cpu start --config /modelship/config/examples/mini-pc.yaml
```

See [../../docs/model-configuration.md](../../docs/model-configuration.md) for the full field reference.
