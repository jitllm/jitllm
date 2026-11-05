#!/usr/bin/env bash
# Gate library entries: download, convert and generate with each, through the
# real CLI, and log one line per entry.
#
#   scripts/library-gate.sh LOG [NAME...]     (no NAME: every entry, smallest first)
#
# An entry passes when `jitllm convert NAME` writes a container and `jitllm run`
# generates from it on the CPU -- and, for a vision model, from an image too.
# The line records the repository commit that was fetched, which is what the
# entry is then pinned to in convert/library/models.go.
#
# ★ IT DELETES ONLY WHAT IT DOWNLOADED, and only after a pass: a GGUF that was
# already on the disk is the user's, and a failure keeps its input to look at.
set -u
LOG=${1:?usage: library-gate.sh LOG [NAME...]}; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${JITLLM_MODELS:?set JITLLM_MODELS to the model directory}
# ★ NOT jitllm-*: scripts/cap takes that name for a MEASUREMENT and holds its
# exclusive lock for the whole conversion -- download included -- which stalled
# every build on the box for as long as a model took to arrive.
BIN=${JITLLM_GATE_BIN:-$ROOT/.library-gate-cli}
PIC=$ROOT/engine/testdata/quad-and-disc.png

(cd "$ROOT" && ./scripts/cap 8G -- go build -o "$BIN" ./cmd/jitllm) || exit 1

mapfile -t rows < <("$BIN" library -refs | sort -k2,2n)
for row in "${rows[@]}"; do
  read -r name bytes ref tower <<<"$row"
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
  # A restart does not redo what passed: its download is gone by then.
  if [ -e "$LOG" ] && grep -q "^$name OK " "$LOG"; then continue; fi

  file=${ref##*/}
  gguf=$DIR/$file
  jlm=$DIR/${file%.gguf}.jlm
  had_gguf=0; [ -e "$gguf" ] && had_gguf=1
  had_tower=1; twr=""
  if [ -n "${tower:-}" ]; then twr=$DIR/${tower##*/}; had_tower=0; [ -e "$twr" ] && had_tower=1; fi

  # The GGUF, the .part and the container the part replaces can all be on the
  # disk at once.
  free=$(df --output=avail -B1 "$DIR" | tail -1)
  if [ "$free" -lt $((bytes * 5 / 2)) ]; then
    echo "$name SKIP disk: $((free / 1000000000)) GB free for $((bytes / 1000000000)) GB" | tee -a "$LOG"
    continue
  fi

  t0=$(date +%s)
  status=OK; why=""
  if ! (cd "$ROOT" && ./scripts/cap 24G -- "$BIN" convert -o "$DIR" "$name" >"$LOG.$name.convert" 2>&1); then
    status=FAIL; why="convert: $(tail -1 "$LOG.$name.convert")"
  fi
  said=""
  if [ $status = OK ]; then
    if ! said=$(cd "$ROOT" && ./scripts/cap 24G -- "$BIN" run -devices cpu -n 8 "$jlm" "The capital of France is" 2>"$LOG.$name.run"); then
      status=FAIL; why="run: $(tail -1 "$LOG.$name.run")"
    elif [ -z "$(echo "$said" | tr -d '[:space:]')" ]; then
      status=FAIL; why="run: generated nothing"
    fi
  fi
  if [ $status = OK ] && [ -n "${tower:-}" ]; then
    if ! seen=$(cd "$ROOT" && ./scripts/cap 24G -- "$BIN" run -devices cpu -n 12 -image "$PIC" "$jlm" "Describe this image." 2>>"$LOG.$name.run"); then
      status=FAIL; why="image: $(tail -1 "$LOG.$name.run")"
    else
      said="$said | image: $seen"
    fi
  fi

  repo=${ref#hf://}; repo=${repo%%/*}/$(echo "${ref#hf://}" | cut -d/ -f2); repo=${repo%%@*}
  rev=$(curl -sI -m 30 "https://huggingface.co/$repo/resolve/main/$file" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-repo-commit"{print $2}')
  trev=""
  if [ -n "${tower:-}" ]; then
    trepo=${tower#hf://}; trepo=$(echo "$trepo" | cut -d/ -f1)/$(echo "$trepo" | cut -d/ -f2); trepo=${trepo%%@*}
    trev=$(curl -sI -m 30 "https://huggingface.co/$trepo/resolve/main/${tower##*/}" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-repo-commit"{print $2}')
  fi
  echo "$name $status rev=$rev towerrev=$trev secs=$(( $(date +%s) - t0 )) ${why:-said: $(echo "$said" | tr '\n' ' ' | cut -c1-160)}" | tee -a "$LOG"

  if [ $status = OK ]; then
    [ $had_gguf = 0 ] && rm -f "$gguf"
    [ -n "$twr" ] && [ $had_tower = 0 ] && rm -f "$twr"
  fi
done
