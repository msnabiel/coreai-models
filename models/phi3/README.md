# Phi-3

Microsoft Phi-3 causal language models exported to Core AI through the native
`coreai-torch` pipeline. Hugging Face supplies the checkpoint and tokenizer;
the export uses a Core AI-authored Phi-3 graph for fused QKV attention and the
fused gate/up MLP layout.

## Export

```bash
uv run coreai.llm.export phi-3.5-mini-instruct
```

For a smoke test without compression:

```bash
uv run coreai.llm.export microsoft/Phi-3-mini-4k-instruct \
  --platform macOS --compression none --compute-precision float16 \
  --num-layers 1 --max-context-length 512 --experimental
```

The registered 4096-token Phi-3.5 export uses the checkpoint’s `short_factor`
LongRoPE branch. The adapter intentionally rejects exports beyond the original
4096-token context until dynamic LongRoPE selection is implemented and tested.
