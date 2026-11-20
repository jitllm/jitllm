"""llama.cpp oracles for the gemma3, internvl and pixtral vision gates.

Writes, under $JITLLM_MODELS/vlm/oracle/:

  <family>-rainbow.txt  llama-mtmd-debug's graph dump of the tower on its raw
                        "rainbow" picture: every node's corner values and sum
  <family>-cli.json     llama-mtmd-cli's greedy answer on a picture already at
                        the tower's size (so no resize stands between the two
                        engines' pixels), with the prompt as the chunks it built:
                        {"chunks": [{"text": ...} | {"image": true}], "out": ...}

Run scripts/visiongold.py first: it writes the square pictures.

Usage: vlmgold.py gemma3|internvl|pixtral
"""
import json
import os
import re
import subprocess
import sys

import refenv  # where things are: environment variables only (docs/testing.md)

MODELS = refenv.models()
VLM = os.path.join(MODELS, "vlm")
LLAMA = refenv.need("JITLLM_LLAMACPP_CPU", "a CPU build directory of llama.cpp")
PROMPT = "Describe this image in one sentence."
N = 32

FAMILIES = {
    "gemma3": (os.path.join(MODELS, "gemma-3-4b-it-Q4_K_M.gguf"),
               os.path.join(VLM, "mmproj-gemma-3-4b-it-f16.gguf"), 896),
    "internvl": (os.path.join(VLM, "InternVL3-1B-Instruct-Q8_0.gguf"),
                 os.path.join(VLM, "mmproj-InternVL3-1B-Instruct-Q8_0.gguf"), 448),
    "pixtral": (os.path.join(MODELS, "pixtral", "pixtral-12b-Q4_K_M.gguf"),
                os.path.join(MODELS, "pixtral", "mmproj-pixtral-12b-Q8_0.gguf"), 448),
}

STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ")


def run(args):
    env = dict(os.environ, LD_LIBRARY_PATH=LLAMA)
    return subprocess.run([os.path.join(LLAMA, args[0])] + args[1:], env=env,
                          capture_output=True, text=True, errors="replace", check=True)


def chunks_of(log):
    """The add_text/add_media chunks of a verbose llama-mtmd-cli log, in order."""
    out, cur = [], None
    for line in log.splitlines(keepends=True):
        if STAMP.match(line):
            if cur is not None:
                out.append({"text": cur})
                cur = None
            body = STAMP.sub("", line)
            if body.startswith("add_text: "):
                cur = body[len("add_text: "):]
            elif body.startswith("add_media: "):
                out.append({"image": True})
            continue
        if cur is not None:
            cur += line
    if cur is not None:
        out.append({"text": cur})
    # The logger ends every record with a newline of its own.
    for c in out:
        if "text" in c and c["text"].endswith("\n"):
            c["text"] = c["text"][:-1]
    return out


def main():
    fam = sys.argv[1]
    text, mmproj, size = FAMILIES[fam]
    out = os.path.join(VLM, "oracle")
    os.makedirs(out, exist_ok=True)
    common = ["-m", text, "--mmproj", mmproj, "-t", "6"]
    r = run(["llama-mtmd-debug"] + common + ["-p", "encode", "--image", "rainbow", "-n", str(size)])
    with open(os.path.join(out, fam + "-rainbow.txt"), "w") as f:
        f.write(r.stdout + r.stderr)
    pic = os.path.join(VLM, "goldens", "photo-%d.png" % size)
    r = run(["llama-mtmd-cli"] + common + ["--image", pic, "-p", PROMPT, "-n", str(N),
                                           "--temp", "0", "--top-k", "1", "--jinja", "-v"])
    ch = chunks_of(r.stderr)
    # How many positions the prompt took there, image rows included: the
    # check that both engines built the same one (a doubled BOS shows here).
    m = re.search(r"prompt eval time =.*?/\s*(\d+) tokens", r.stderr)
    with open(os.path.join(out, fam + "-cli.json"), "w") as f:
        json.dump({"picture": os.path.basename(pic), "chunks": ch, "out": r.stdout.strip("\n"),
                   "tokens": int(m.group(1)) if m else 0}, f, indent=1)
    print(fam, ch, repr(r.stdout))


if __name__ == "__main__":
    main()
