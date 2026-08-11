# iPhone Long-Context Model Research

Status: research document, 2026-08-02

This document compares permissively licensed 3B–4B language models that could be
ported to iPhone with Core AI. It separates models that already work in this
repository from models that still need a Core AI authoring/export adapter.

## Executive conclusion

The best long-context architecture to investigate is **Qwen3.5-4B**. It is
Apache 2.0, advertises a 262K context, and uses a hybrid architecture with
linear-attention layers that do not grow a conventional KV cache for every
token. It is not compatible with the current Qwen3 adapter, however.

The best model that works with the current repository is **Qwen3-4B**. It is
Apache 2.0—not the Qwen2.5 research license—and its existing iOS export should
be kept as the quality-first option. Its regular attention cache makes 8K a
reasonable iPhone target; 32K is expensive.

If a new adapter is acceptable, the shortlist is:

1. **Qwen3.5-4B** — best long-context architecture.
2. **SmolLM3-3B** — best Apache-licensed conventional 3B research target.
3. **Ministral 3 3B** — strongest advertised context, but multimodal and a
   larger integration project.
4. **Granite 3.3 2B** — conservative Apache-licensed 128K alternative.

## Candidate comparison

| Model | License | Size | Published context | Attention/cache profile | Repository status | iPhone assessment |
| --- | --- | ---: | ---: | --- | --- | --- |
| Qwen3-4B | Apache 2.0 | 4B | 40,960 | Standard attention, 8 KV heads | Existing iOS adapter and asset | **Use now; 4K–8K recommended** |
| Qwen3.5-4B | Apache 2.0 | 4B | 262,144 | Hybrid linear/full attention, 4 KV heads in the text config | No adapter | **Best research target for long context** |
| Phi-4-mini-instruct | MIT | ~3.8B | 131,072 | GQA, 8 KV heads; Phi-3 architecture family | Phi adapter exists, but iOS path rejects >4K LongRoPE | Good minimal-port candidate, but Phi is excluded from the preferred shortlist |
| SmolLM3-3B | Apache 2.0 | 3B | 65,536 native; 131,072 with YaRN | GQA, 4 KV heads; NoPE/layer-specific behavior | No adapter | **Best clean 3B adapter project** |
| Ministral 3 3B | Apache 2.0 | 3.4B text + 0.4B vision | 262,144 | GQA, 8 KV heads; multimodal | No adapter | Strong edge model, but high integration effort |
| Granite 3.3-2B-Instruct | Apache 2.0 | 2B | 131,072 | GQA, 8 KV heads | No adapter | Reasonable lower-capability fallback |
| Qwen2.5-1.5B-Instruct | Qwen research license | 1.5B | 32,768 | GQA, 2 KV heads | Existing iOS adapter | Technically excellent for memory, but excluded by licensing |

Primary model references: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B),
[Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct),
[SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B),
[Ministral 3 3B](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16),
and [Granite 3.3-2B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct).

## KV-cache estimates

For a conventional FP16 KV cache, an approximate per-token cost is:

```text
layers × 2 (key and value) × KV heads × head dimension × 2 bytes
```

These figures include only the conventional KV cache. They exclude model
weights, activations, recurrent state, tokenizer memory, Core AI workspace, and
the app itself.

| Model | Approx. cache per token | 8K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 144 KiB | 1.13 GiB | 2.25 GiB | 4.50 GiB |
| Phi-4-mini | 128 KiB | 1.00 GiB | 2.00 GiB | 4.00 GiB |
| SmolLM3-3B | 72 KiB | 0.56 GiB | 1.13 GiB | 2.25 GiB |
| Ministral 3 3B | ~104 KiB | ~0.81 GiB | ~1.63 GiB | ~3.25 GiB |
| Granite 3.3-2B | ~80 KiB | ~0.63 GiB | ~1.25 GiB | ~2.50 GiB |

Qwen3.5-4B is not represented by one simple number because most of its layers
use linear attention and only periodic full-attention layers retain a normal
growing KV cache. Its recurrent state and full-attention state must be measured
after authoring; that is the reason it is promising, not a reason to assume a
free 262K iPhone context.

## Repository findings

The current registry has only two iOS LLM presets:

- [Qwen2.5-1.5B](/Users/msnabiel/Desktop/coreai-models/python/src/coreai_models/model_registry.py:156)
- [Qwen3-4B](/Users/msnabiel/Desktop/coreai-models/python/src/coreai_models/model_registry.py:166)

There are no `experimental=True` iOS LLM entries in the registry. The
`--experimental` flag allows an unregistered model to enter the generic export
path, but it does not create an architecture adapter automatically. The model
still needs a matching implementation under
`python/src/coreai_models/models/ios/`.

The current iOS cache primitive uses a model-shaped cache with the layout
`[layers, batch, KV-heads × head-dim, 1, max-sequence]`:
[iOS cache primitive](/Users/msnabiel/Desktop/coreai-models/python/src/coreai_models/primitives/ios/cache.py:15).
The iOS export path is therefore sensitive to the chosen context length and
cache layout.

## Porting difficulty

### Qwen3.5-4B

Qwen3.5 is not a larger ordinary Qwen3 block. Its config contains
`linear_attention` and `full_attention` layer types, separate linear key/value
dimensions, recurrent-state parameters, and a multimodal wrapper. The current
`qwen3.py` adapter cannot consume those layers as-is.

Required work includes:

- Implementing or reusing Core AI-compatible linear-attention/recurrent
  primitives.
- Authoring the periodic full-attention blocks and their cache.
- Handling the Qwen3.5 RoPE and partial rotary configuration.
- Exporting the text-only path first; vision should be a separate milestone.
- Comparing PyTorch, exported, and compiled outputs layer-by-layer.

This is the best architecture, but not a quick registry-only change.

### SmolLM3-3B

SmolLM3 is a cleaner text-only target than Qwen3.5, but it has NoPE/layer
patterns that a plain Llama/Qwen adapter should not silently ignore. It is a
good next model if the goal is a genuinely open Apache 3B model with 16K–32K
context rather than the absolute minimum porting effort.

### Ministral 3 3B

Ministral 3 is explicitly positioned as an edge model and has a 256K context,
but the checkpoint combines a text model with a vision encoder. A text-only
Core AI port should be scoped first; importing the entire multimodal model
would add unnecessary iPhone memory and export complexity.

### Granite 3.3-2B

Granite is a conventional decoder architecture and Apache 2.0, but there is no
existing Granite adapter in this repository. It is a sensible fallback for a
smaller general model, not the strongest quality-per-engineering-hour choice.

## Recommended product lineup

### Ship immediately

Keep the existing **Qwen3-4B iOS** asset as the quality-first model:

- 4K default context.
- 8K only as a high-memory experimental variant.
- Apache 2.0 licensing.
- Existing Core AI export, tokenizer, upload, and FlowKit integration.

### Next research asset

Prototype **Qwen3.5-4B text-only at 8K**, then 16K. Do not start at 262K.
First prove:

1. Export correctness against the source model.
2. Stable state updates across prefill and decode.
3. Memory use on an A17 Pro or newer iPhone.
4. Prefill latency and generated tokens per second.
5. Long-context retrieval accuracy at 8K and 16K.

### If the Qwen3.5 port is too large

Use **SmolLM3-3B at 8K or 16K** as the clean Apache-licensed fallback. Use
Ministral 3 only if its multimodal capability is important; otherwise its
vision component is extra work and memory.

## Core AI and iPhone constraints

Core AI can convert PyTorch models into `.aimodel` assets, expose mutable model
state, and support custom lowerings and Metal kernels. This makes a new model
architecture possible, but does not make unsupported attention/state patterns
automatic. See [coreai-torch](https://apple.github.io/coreai-torch/) and
[Apple's Core AI overview](https://developer.apple.com/core-ai/).

Ahead-of-time compilation can reduce first-load specialization work, but it
does not eliminate model weights, attention state, recurrent state, or decode
compute. See [Core AI AOT compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time).

The correct success criterion is not the model card's maximum context. It is a
real-device test that measures memory pressure, prefill latency, decode speed,
quality, and app stability at the context lengths FlowKit actually promises.

## Final ranking

| Rank | Model | Why |
| ---: | --- | --- |
| 1 | Qwen3.5-4B | Best long-context architecture and Apache 2.0; requires a serious new adapter |
| 2 | Qwen3-4B | Already works in this repo and is Apache 2.0; best immediate quality option |
| 3 | SmolLM3-3B | Apache 2.0, long context, smaller model; moderate new-adapter work |
| 4 | Ministral 3 3B | Apache 2.0 and 256K, but multimodal integration is substantial |
| 5 | Granite 3.3-2B | Apache 2.0 and 128K, but lower capability and no adapter |

