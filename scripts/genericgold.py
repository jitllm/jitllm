#!/usr/bin/env python3
"""genericgold.py LLAMA_SERVER MODEL.gguf TAG DIR: records what llama.cpp's generic tool format does with each
DIR/*.request.json on the model: the prompt its /apply-template renders (DIR/NAME.TAG.prompt.txt) and the
message its chat completion answers with (DIR/NAME.TAG.response.json).

LLAMA_SERVER is a llama-server of the era that still had COMMON_CHAT_FORMAT_GENERIC and minja's tool
polyfills (b4689, commit 90e4dba4): later builds read a template that names no tools as plain content.
The server is run with --jinja, on the cores it is given, and stopped at the end."""
import json
import os
import subprocess
import sys
import time
import urllib.request

server, gguf, tag, d = sys.argv[1:5]
port = 8099
p = subprocess.Popen([server, "-m", gguf, "--jinja", "--port", str(port), "-c", "4096", "-t", "6", "--no-webui",
                      "--log-disable"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


try:
    for _ in range(600):
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    for f in sorted(os.listdir(d)):
        if not f.endswith(".request.json"):
            continue
        name = f[:-len(".request.json")]
        req = json.load(open(os.path.join(d, f)))
        prompt = post("/apply-template", req)["prompt"]
        open(os.path.join(d, "%s.%s.prompt.txt" % (name, tag)), "w").write(prompt)
        msg = post("/v1/chat/completions", req)["choices"][0]["message"]
        calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
                 for c in msg.get("tool_calls") or []]
        json.dump({"content": msg.get("content") or "", "tool_calls": calls},
                  open(os.path.join(d, "%s.%s.response.json" % (name, tag)), "w"), indent=1)
        print(name, tag, json.dumps(calls) if calls else repr(msg.get("content")))
finally:
    p.terminate()
    p.wait()
