# llama.cpp comparison

Run the normal comparison with an optional llama.cpp column:

```bash
.venv/bin/python benches/benchmark.py \
  --model HuggingFaceTB/SmolLM3-3B \
  --llama-gguf benches/SmolLM3-3B-F16.gguf
```

PyTorch runs in a child process that exits before the custom engine starts,
ensuring its GPU allocations are released. The script starts
`build/llama-hip/bin/llama-server` after PyTorch and the custom
engine release their GPU allocations, and stops it after benchmarking. Use
`--llama-server PATH` to select another binary. The server uses one request slot,
all layers on GPU, and FP16 K/V cache. Check `benches/llama_server.log` for the
actual backend and offload count.
Combined results are saved to `benches/comparison.json`.

All engines receive the same prompt token IDs, generate the same fixed number
of tokens, and use greedy sampling. llama.cpp disables prompt reuse between
requests and ignores EOS. Its reported timings are client wall clock: time to
the first streamed token, then time through completion. They include HTTP and
server overhead, unlike a kernel-only `llama-bench` result. Token differences
are reported; speed alone does not establish numerical correctness. llama.cpp
suppresses EOS when ignoring it, whereas the other loops continue after EOS;
if EOS would be selected, the generated paths may differ.

## Rebuild and convert

The tested source revision is `1623d8ce47bc0bee9757068780c23ca7abd45160`.
The distribution installation currently fails with the missing symbol
`ggml_rope_set_offset`; this separate build keeps llama.cpp and ggml together.

```bash
git clone https://github.com/ggml-org/llama.cpp.git build/llama-source
git -C build/llama-source checkout 1623d8ce47bc0bee9757068780c23ca7abd45160
cmake -S build/llama-source -B build/llama-hip \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build/llama-hip --target llama-server llama-bench -j 8
.venv/bin/pip install sentencepiece
.venv/bin/python build/llama-source/convert_hf_to_gguf.py \
  /home/marc/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B/snapshots/a07cc9a04f16550a088caea529712d1d335b0ac1 \
  --outfile benches/SmolLM3-3B-F16.gguf --outtype f16
```

GGUF conversion uses the cached BF16 snapshot and writes FP16 matrix weights
(some small tensors remain FP32). No model download or quantization is needed.
The generated GGUF is about 6.2 GB and is ignored by Git.
