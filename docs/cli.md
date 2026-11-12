# CLI and model conversion

[Back to the README](../README.md)

| Command | What it does |
|---|---|
| `jitllm hardware` | the CPU tier, the GPUs each backend sees, and their memory |
| `jitllm doctor [-json] [model.jlm]` | can this machine and build run (and run this container); exit 0 ok, 2 missing, 1 error |
| `jitllm capabilities [-json]` | what this build converts, runs and serves ([capabilities.md](capabilities.md)) |
| `jitllm library` | the models `convert` can fetch by name |
| `jitllm convert` | GGUF, a Hugging Face directory, a URL or a catalog name to `.jlm` |
| `jitllm run` | generate, with `-chat`, `-image`, sampling, devices and placement; with no prompt an interactive chat, or the prompt from piped stdin |
| `jitllm embed` | an embedding model's normalised vector |
| `jitllm decide` | a decision model's answers to a `/v1/systemone` request |
| `jitllm tokenize` / `info` | the token ids of a text / a GGUF's metadata |
| `jitllm speed` / `batch` | prompt and decode rates, and batched decode |
| `jitllm verify` | compare a device run with the host, position by position |
| `jitllm bench` | one-core decode as a fraction of a memory read rate measured in the same process |

## The models folder

jitllm, jitllmd, the desktop app and the terminal app share one models
folder:

| System | Folder |
|---|---|
| Linux | `$XDG_DATA_HOME/jitllm/models` (`~/.local/share/jitllm/models`) |
| macOS | `~/Library/Application Support/jitllm/models` |
| Windows | `%LOCALAPPDATA%\jitllm\models` |

`JITLLM_MODELS` replaces it, and `jitllmd serve -models DIR` replaces it for
one server. `jitllm convert NAME` with no output path writes the container
there, creating the folder. A `models` folder in the working directory, where
earlier releases looked, is still searched by name, after the folder above;
nothing in it is moved.

## Convert a model

```sh
# A model from the catalog (jitllm library lists them), into the models
# folder, where it also answers to the catalog name: jitllm run llama-3.2-1b-instruct
./jitllm convert llama-3.2-1b-instruct

# A GGUF you already have.
./jitllm convert path/to/model.gguf models/model.jlm

# A Hugging Face directory, optionally quantized to Q8_0.
./jitllm convert -q8 path/to/hf-model-dir models/model-q8.jlm

# A Hugging Face repository, read by range without downloading every shard.
./jitllm convert hf://HuggingFaceTB/SmolLM2-360M-Instruct models/smollm2-hf.jlm

# Replace the source's chat template: a tokenizer_config.json, a
# chat_template.json, a .jinja file or a model directory.
./jitllm convert -chat-template path/to/tokenizer_config.json model.gguf models/model.jlm
```

| Flag | Meaning |
|---|---|
| `-o DIR` | where the container and a downloaded source land; the output path is optional |
| `-q8` | safetensors only: store weight matrices as Q8_0 |
| `-chat-template FILE` | store this file's chat template(s) in place of the source's |
| `-config FILE` | GGUF only: the publisher's config.json (or its directory), for what the GGUF dropped: a LongRoPE's factors (MiniCPM-V's) |
| `-plan` | safetensors only: print the container the conversion would write, from the shards' headers, without converting |

The container carries the configuration, tokenizer, stop tokens, chat templates
and any vision tower, in the layout the kernels read, so a page-in is a copy and
not a repack. Set `HF_TOKEN` for gated repositories.

Use the model's own chat template with `-chat`. Generation stops at any of the
model's stop tokens, which the container takes from its source (a GGUF's
end-of-turn ids, a Hugging Face model's `generation_config` list), so a chat
ends at the end of its turn:

```sh
./jitllm run -chat -system "Explain things with simple examples." \
  -temp 0.7 -top-p 0.9 -n 256 models/smollm2.jlm "What is recursion?"
```

With no prompt and a terminal on stdin, `run` holds a conversation through the
model's chat template: `/help`, `/clear`, `/bye` (or Ctrl+D) to leave, Ctrl+C
to stop a reply, a trailing `\` or a `"""` block for several lines. A pipe or
file on stdin is read whole as the prompt instead.

## Run by name, through the service

`jitllm run NAME` (and `jitllm embed NAME`) asks the local service first. When
a jitllmd answers on `localhost:4242` (`-addr` names another), the service
resolves NAME -- a file name in its models folder, an alias, a loaded id --
loads it if it is not loaded, and streams the answer; no second engine starts.
With no prompt and a terminal, it is the same interactive chat as in-process.
When nothing answers, NAME is looked up in the models folder (and a `models`
folder in the working directory) by file name or alias and run in this
process. A path always runs in this process, and so does a run with
`-embedded` or with any flag about this process's engine (`-devices`,
`-placement`, `-kv-cache`, `-image`, ...). One line on stderr says which:

```text
jitllm: using the service at http://localhost:4242 (jitllmd v0.4.0) for qwen3
jitllm: running /home/me/.local/share/jitllm/models/Qwen3-8B-Q4_K_M.jlm in this process (no service answers at localhost:4242)
```

A service started with an API key is used with `-api-key KEY` (default
`JITLLM_API_KEY`) and refused by name without it. When nothing answers on
`localhost:4242`, a service on the old default port `127.0.0.1:8080` is used
and one more line says so (`jitllmd service install` moves it); an `-addr` you
type is never swapped for it.

A service of another API version is refused with a message rather than
bypassed. The measuring commands (`speed`, `bench`, `batch`, `verify`) and
`decide` always run in this process: they measure or drive this process's
engine.
