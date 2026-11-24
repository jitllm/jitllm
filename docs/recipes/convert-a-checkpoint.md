# Choose a supported checkpoint and convert it

Find out whether jitllm runs a model before you download it, then convert it to
the `.jlm` container the engine reads.

**Prerequisites:** `jitllm` on your `PATH`, `curl`, and disk for the model.

## 1. Is it supported?

Support is decided by the architecture, not the model's name. Three sources say
which architectures convert, from the same lists in the converter:

- [docs/models.md](../models.md): every GGUF architecture, Hugging Face class
  and vision projector, and the models each covers (generated from the code).
- `jitllm capabilities` and `docs/capabilities.json`, where the build has them:
  the same, machine-readable, with the API and backend facts beside it.
- `jitllm library`: the catalog `convert` fetches by name, each one known to
  convert.

```sh
jitllm library
```
<!-- test: expect qwen3-0.6b -->

A GGUF names its architecture in `general.architecture`; for a Hugging Face
checkpoint it is `architectures` in `config.json`. `jitllm info` reads a GGUF's
header without loading it:

<!-- test: instead cp "$JITLLM_REPO/testdata/models/stories260K.gguf" . -->
```sh
curl -fLo stories260K.gguf https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf
```

```sh
jitllm info stories260K.gguf
```
<!-- test: expect arch      llama -->

```text
gguf      v3   48 tensors   19 kv   alignment 32
arch      llama "llama"
          L=5  d=64  ffn=172  heads=8/4  head_dim=8  n_rot=8
types
          F32        48       1,171,200  100.00%
```

Look the architecture up in `docs/models.md`. Quantization types matter too:
F32, F16, BF16, Q4_0, Q5_0, Q5_1, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K and MXFP4 load;
a GGUF in another type (IQ-quants, Q2_K, ...) is refused at conversion.

## 2. Convert

A local GGUF, with an explicit output path:

```sh
mkdir -p models
jitllm convert stories260K.gguf models/stories260K.jlm
```
<!-- test: expect models/stories260K.jlm -->

A catalog name downloads and converts in one step (`-o` is where it lands):

<!-- test: skip downloads 640 MB from Hugging Face -->
```sh
jitllm convert -o models qwen3-0.6b
```

Other sources (each downloads):

```text
jitllm convert model.gguf mmproj.gguf out.jlm        # a GGUF and its vision projector: one container
jitllm convert ./hf-checkpoint-dir out.jlm            # a Hugging Face directory (safetensors)
jitllm convert -q8 ./hf-checkpoint-dir out.jlm        # the same, every matrix stored as Q8_0
jitllm convert hf://Qwen/Qwen3-0.6B out.jlm           # a repository, read by range, no full download
jitllm convert -plan hf://Qwen/Qwen3-0.6B             # what that would write, and what it would refuse
jitllm convert -chat-template tokenizer_config.json model.gguf out.jlm   # the publisher's chat template
```

Check the container runs:

```sh
jitllm run -devices cpu -n 16 -temp 0 models/stories260K.jlm "Once upon a time"
```
<!-- test: expect there was a little girl named Lily -->

## 3. What an unsupported model looks like

An architecture is not code: it is a statement in `convert/models/*.json` of
its graph over the engine's ops. A file whose architecture no statement names is
refused at conversion, naming it and every architecture that is stated:

```text
convert: GGUF architecture "ollmo": its graph is not stated over ops in convert/models (stated: apertus, arcee, ..., starcoder2): no library file states it
```

Nothing partial is written, and no other command will run it: adding a
statement is a decision the project makes (AGENTS.md, RULE 7). An unknown
pre-tokenizer is refused the same way, at load
(`pre-tokenizer "X" is not implemented`).

The engine refuses a GGUF at run time and prints the command that converts it:

<!-- test: exit 1 -->
```sh
jitllm run stories260K.gguf "Once upon a time"
```
<!-- test: expect jitllm convert stories260K.gguf -->

## When it does not work

| You see | Cause | Do this |
|---|---|---|
| `GGUF architecture "X": its graph is not stated` | no library file states the architecture | choose a checkpoint of a stated architecture |
| `missing block_count/embedding_length/head_count` | the GGUF's metadata is incomplete or for another architecture | re-download; a hand-edited GGUF is not a model |
| `ErrNotImplemented` naming a projector | a vision projector outside the list | convert the text model alone |
| a key spelled two ways that disagree is refused | the config contradicts itself | fix the config, or use the publisher's GGUF |
| `no model directory here, so it lands beside the source` | a local file with no output path | give the output path or `-o DIR`, or set `JITLLM_MODELS` |
| the download stops | network or Hugging Face rate limits | rerun the same command; it resumes |
