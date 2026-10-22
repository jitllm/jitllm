#!/usr/bin/env bash
# Compare jitllm against llama.cpp on one model and one tier, PAIRED and
# INTERLEAVED, and report the median of per-round ratios with its IQR.
#
#   scripts/vs-llamacpp.sh cpu models/gemma-2b.gguf
#   scripts/vs-llamacpp.sh gpu models/llama-2-7b.Q4_K_M.gguf 8
#
# ★ WHY THIS IS A SCRIPT AND NOT A COMMAND LINE. Every previous version of this
# comparison was typed by hand, and three separate things went wrong that way:
# a baseline was quoted from a llama.cpp version measured weeks earlier (RULE
# 1a), a jitllm-on-CUDA number was compared against llama.cpp-on-VULKAN (RULE 1d),
# and the two engines were run A-then-B while the box drifted (RULE 2). All
# three are decisions about HOW to invoke, so they belong next to the invocation.
#
# ★ IT IS THE ONE MEASUREMENT jitllm CANNOT DO IN ONE PROCESS. RULE 6 records that
# a cross-process A/B came back at IQR/median 0.123 and was rejected -- but the
# other engine is another program, so there is no in-process option here and the
# answer is to interleave hard and report the dispersion honestly rather than to
# pretend. ABBA rather than ABAB for the same reason the tuner uses it (RULE 5e):
# a fixed order lets any drift accumulate on one side.
set -euo pipefail

# ★ THE WHOLE COMPARISON HOLDS THE MEASUREMENT LOCK, AND RUNS UNDER NO MEMORY CGROUP.
#
# A comparison is ATOMIC: locking each sample separately would let an
# interloper take the lock between two samples of the same ABBA round. So the
# script takes scripts/cap's lock file EXCLUSIVE once and holds it to the end;
# builds still take it shared through scripts/cap and wait.
#
# It used to get that lock by re-execing itself under scripts/cap, which also
# put both engines inside an 8G memory cgroup on every host. That cgroup charges
# llama.cpp's mmap page cache and throttles it at MemoryHigh, while jitllm reads
# with O_DIRECT, so a large model's row measured the cap as well as the engines.
# scripts/cap protects the developer's own machine (AGENTS.md RULE 3); a bench
# host runs both engines bare. On the laptop the hook still wants the outer
# command wrapped in scripts/cap, which classifies this script as a measurement
# and already holds the lock EXCLUSIVE on fd 9 when it starts. A second open of
# the file would be a second lock and wait on its own parent forever, so when
# fd 9 is that file the script re-takes the lock on the SAME descriptor, which
# is immediate. JITLLM_NO_LOCK=1 opts out, as it does in scripts/cap.
if [ "${JITLLM_NO_LOCK:-0}" != "1" ] && command -v flock >/dev/null 2>&1; then
  vsLock="${TMPDIR:-/tmp}/jitllm-cap.lock"
  if [ "$(readlink -f /proc/$$/fd/9 2>/dev/null)" != "$(readlink -f "$vsLock" 2>/dev/null)" ]; then
    exec 9>>"$vsLock"
  fi
  if ! flock -n -x 9; then
    echo ">> waiting for the measurement lock $vsLock (another measurement or build holds it)" >&2
    flock -w "${JITLLM_LOCK_WAIT:-1800}" -x 9 || { echo ">> REFUSING: the box stayed busy" >&2; exit 75; }
  fi
fi

TIER="${1:?usage: vs-llamacpp.sh <cpu|gpu> <model.gguf> [rounds]}"
MODEL="${2:?usage: vs-llamacpp.sh <cpu|gpu> <model.gguf> [rounds]}"
# ★ THE TWO ARMS READ DIFFERENT FILES, AND THAT IS THE COMPARISON, NOT A FLAW.
# jitllm runs the jlm container -- it reads nothing else -- and llama.cpp
# runs the GGUF, because that is the format it has. Converting is offline and
# one-time; measuring jitllm on a GGUF would measure a path that no longer
# exists. JITLLM_PATH overrides the derived name.
#
# ★ AND IT RECONVERTS WHEN THE BINARY IS NEWER, NOT ONLY WHEN THE FILE IS
# MISSING. The container has a version and there is no backward compatibility on
# purpose, so a bump leaves a file that exists and cannot be read -- which is a
# measurement that dies rather than one that lies, but it dies in the middle of
# a round. The same "it is there, so it is fine" rule left the VLM gates red at
# "container is version 4 and this build reads 5".
# JITLLM_BIN is what the measurement below runs (see the JITLLM= line further
# down); converting with a DIFFERENT binary is how a stale container survives a
# version bump, so both come from the same variable.
JLMBIN="${JITLLM_BIN:-${JITLLM:-./jitllm}}"
JITLLM_PATH="${JITLLM_PATH:-${MODEL%.gguf}.jlm}"
# ★ mtime IS A PROXY FOR "STALE" AND IT IS WRONG WHENEVER ANOTHER BINARY WROTE
# THE CONTAINER. A container written by an OLDER tree has a NEWER mtime, so
# `-ot "$JLMBIN"` says fresh and the run then dies on "container is version 6
# and this build reads 15". It happened here: a `go test ./engine/model -run
# TestFusionCeiling` in a v6 worktree converts into the shared models directory
# and clobbers the v15 container with a v6 one. The failure was SILENT because
# jitllm_run discards the engine's stdout and lcpp_run discards stderr, so
# `set -e` killed the script mid-soak with an empty error and exit 1.
#
# The version is in the header -- 8-byte magic then a little-endian uint32 at
# offset 8 -- so read it and compare against the source constant instead of
# guessing from a timestamp. Either signal saying stale triggers a reconvert;
# reconverting unnecessarily costs time, and NOT reconverting costs the run.
jlm_file_version() {
  [ -f "$1" ] || { echo -1; return; }
  od -A n -t u4 -j 8 -N 4 "$1" 2>/dev/null | tr -d ' ' || echo -1
}
# Off the tree (a GPU server holds binaries and these scripts, never the
# source) the constant is not there to read: JITLLM_JLM_VERSION states it, and
# without either the check is skipped rather than killing the run under set -e.
JLM_WANT="${JITLLM_JLM_VERSION:-$(sed -n 's/^const Version = \([0-9]*\).*/\1/p' "$(dirname "$0")/../format/jlm/format.go" 2>/dev/null || true)}"
JLM_HAVE=$(jlm_file_version "$JITLLM_PATH")
if [ -n "$JLM_WANT" ] && [ -n "$JLM_HAVE" ] && [ "$JLM_HAVE" != "$JLM_WANT" ] && [ -f "$JITLLM_PATH" ]; then
  echo ">> $(basename "$JITLLM_PATH") is container v$JLM_HAVE and this tree reads v$JLM_WANT -- reconverting"
fi
if [ ! -f "$JITLLM_PATH" ] || [ "$JITLLM_PATH" -ot "$JLMBIN" ] \
   || { [ -n "$JLM_WANT" ] && [ "$JLM_HAVE" != "$JLM_WANT" ]; }; then
  echo ">> converting $(basename "$MODEL") -> $(basename "$JITLLM_PATH")"
  # Bare, like the runs: on the laptop the outer scripts/cap that the hook
  # requires covers it, and a bench host has no desktop to protect. The child
  # inherits fd 9, so the lock stays held through the conversion.
  "$JLMBIN" convert "$MODEL" "$JITLLM_PATH" || exit 1
fi
ROUNDS="${3:-6}"
N="${JITLLM_VS_N:-320}"
# ★ DEPTH: decode measured with this many tokens ALREADY in the KV cache. Every
# other number in this project is a short-context number, and short context
# measures the matvecs and almost nothing else -- attention grows with position
# and the weight traffic does not. llama-bench spells it -d; jitllm run spells it
# -depth; they mean the same thing.
DEPTH="${JITLLM_DEPTH:-0}"   # long enough that jitllm run's un-warmed first tokens are a small fraction; see RULE 1d-quater
# ★ MODE: decode or prefill. An engine does two things and this script measured
# ONE of them -- every row on RULE 1d's scoreboard is a decode row, and the
# prefill rows in AGENTS.md came from ad-hoc scripts typed by hand, which is
# exactly what RULE 12q says not to do: "a scoreboard whose rows come from two
# different tools is not a scoreboard". The soak, the ABBA, the PSI gate, the
# clamp gate and the IQR gate are all mode-independent, so prefill only needs
# its two run functions. Two cold rows sit beside them: load (process start to
# model ready) and ttft (model ready to the first token of a fresh process's
# first prompt), each engine's boundary read from its own output; see "THE
# COLD COSTS" below.
MODE="${JITLLM_MODE:-decode}"
# Prompt for the prefill mode: repeated so its length is set by a count rather
# than by prose, and long enough that jitllm's fixed startup is a small share.
#
# ★ 46 REPS = 507 TOKENS, AND THE NUMBER IS CHOSEN TO SIT JUST UNDER 512.
# llama.cpp's default n_ubatch IS 512, so a prompt of 573 runs as one full
# micro-batch plus a ragged 61-token tail and llama.cpp reads 146-151 tok/s
# where the same box at 512 reads 157-158. A row measured at 573 is therefore
# quoting llama.cpp at a length its batching handles badly, which flatters jitllm
# by ~5% for a reason that has nothing to do with either kernel. Landing both
# engines inside ONE micro-batch removes the artifact; the row is length
# sensitive either way, so the length is printed with it.
PPREPS="${JITLLM_PP_REPS:-46}"
CORES="${JITLLM_VS_CORES:-0,2,4,6,8,10}"
# ★ THE DEFAULTS ARE PER-HOST BECAUSE THE COMPARISON IS PER-HOST. RULE 1c: the
# Mac is the second target, not an afterthought, and a harness that only runs on
# one box guarantees the other one gets measured by hand and quoted from memory.
case "$(uname -s)" in
  # ★ THE DEFAULT PROBES BOTH NAMES BECAUSE THE RENAME DID NOT MOVE THE DISK.
  # jpt -> jitllm renamed the project; ~/jpt-arm on the Mac and jpt-models on the
  # Linux box kept their names, so a default pinned to the new one silently finds
  # no llama.cpp and the Darwin rows cannot be taken at all.
  Darwin)
    LCPP="${JITLLM_LCPP:-}"
    if [ -z "$LCPP" ]; then
      for d in "$HOME/jitllm-arm/lcpp/llama-b10825" "$HOME/jpt-arm/lcpp/llama-b10825"; do
        [ -x "$d/llama-bench" ] && LCPP="$d" && break
      done
      : "${LCPP:=$HOME/jpt-arm/lcpp/llama-b10825}"
    fi
    ;;
  *)      LCPP="${JITLLM_LCPP:-}" ;;
esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JITLLM="${JITLLM_BIN:-}"

# JITLLM_VS_B names a second jitllm binary to run in llama.cpp's arm: two
# builds of jitllm compared through every gate, the order and the per-round
# ratio a jitllm-vs-llama.cpp row goes through. llama.cpp is not run then,
# nor under JITLLM_VS_AA.
if [ -z "${JITLLM_VS_B:-}" ] && [ -z "${JITLLM_VS_AA:-}" ]; then
  [ -n "$LCPP" ] || { echo "set JITLLM_LCPP to a llama.cpp build directory (the one holding llama-bench)" >&2; exit 1; }
  [ -x "$LCPP/llama-bench" ] || { echo "no llama-bench at $LCPP (JITLLM_LCPP)" >&2; exit 1; }
fi

# ★ GATE ON PSI, NOT loadavg. This box reads loadavg 4.4 while idle; see
# bench.Guard and RULE 2b.
if [ -r /proc/pressure/cpu ]; then
  P=$(awk '/^some/{for(i=1;i<=NF;i++) if($i ~ /^avg10=/){sub("avg10=","",$i); print $i}}' /proc/pressure/cpu)
  if [ "$(echo "$P > 25" | bc)" = "1" ]; then
    echo "refusing: CPU pressure is ${P}% over the last 10s, want < 25" >&2
    exit 1
  fi
fi

# ★ AND PSI ALONE IS BLIND TO THE TENANT THAT ACTUALLY RUINS A ROW, WHICH IS
# WHY THIS SECOND GATE EXISTS. PSI "some" measures time tasks spend STALLED
# waiting for a CPU. This box has 20 logical CPUs, so a foreign job using nine
# of them stalls almost nothing: measured here with test262.test at 913% and a
# Go `compile` at 386% -- 1372% of 2000% capacity, 69% of the machine -- PSI
# read 7.52% and this script's own guard said the box was quiet enough to
# measure. Worse, that tenant sat on psr 0, which is the FIRST core in RULE 5's
# taskset -c 0,2,4,6,8,10, so the row would have been pinned on top of it.
#
# The honest question is not "is the machine stalled" but "are the cores I am
# about to pin to already busy", and /proc/stat answers it directly: sample the
# per-cpu jiffies for exactly those CPUs twice and take the busy fraction. No
# process attribution, no guessing, and it sees a tenant whether it is pinned,
# unpinned, another project's test, or another session's compile.
#
# JITLLM_VS_BUSY overrides the threshold; 20% allows ordinary desktop noise and
# refuses anything that is really running.
if [ -r /proc/stat ] && [ "$(uname -s)" = "Linux" ]; then
  BUSY=$("$(dirname "$0")/corebusy.py" "$CORES" 2>/dev/null)
  BUSY_MAX="${JITLLM_VS_BUSY:-20}"
  if [ -n "$BUSY" ] && [ "$(echo "$BUSY > $BUSY_MAX" | bc)" = "1" ]; then
    echo "refusing: cores $CORES are already ${BUSY}% busy before this run starts," >&2
    echo "  want < ${BUSY_MAX}%. Something else is on the cores this row pins to --" >&2
    echo "  PSI cannot see it when the box has spare CPUs elsewhere (RULE 3.3)." >&2
    ps -eo pcpu,args --no-headers --sort=-pcpu | awk '$1>20' | head -3 | sed 's/^/  /' >&2
    exit 1
  fi
  # Temperature is REPORTED, never a refusal: RULE 2a soaks to steady state on
  # purpose, so a warm package is the correct state to measure in and a gate on
  # it would refuse every legitimate row. It is printed so a clamped row can be
  # read against it afterwards.
  for z in /sys/class/thermal/thermal_zone*; do
    [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ] || continue
    echo "  cores $CORES ${BUSY}% busy, package $(( $(cat "$z/temp") / 1000 ))C at start"
    break
  done
fi

if [ -z "$JITLLM" ]; then
  JITLLM="$(mktemp -d)/jitllm"
  GOFLAGS=-p=4 GOMAXPROCS=6 go build -o "$JITLLM" "$REPO/cmd/jitllm"
fi
# ${PIN[@]} under `set -u` on bash 3.2 (which is what macOS ships) errors on an
# empty array, so every expansion of it is guarded.

# taskset is Linux-only. macOS has no CPU affinity API for user processes at all,
# and it does not need one here: RULE 1c measured JITLLM_CORES on the M4 and the
# rate saturates at the four P-cores, which is where the scheduler puts a
# compute-bound thread anyway.
# ★ THE THREAD COUNT FOLLOWS $CORES, AND HARDCODING IT INFLATED EVERY ROW TAKEN
# OFF THE LAPTOP. It read `LCPP_THREADS=6` -- this laptop's six P-cores -- while
# jitllm sizes its pool from the taskset it is given. So on the Xeon, where
# JITLLM_VS_CORES is 0,1,2,3, llama.cpp was told -t 6 and then CONFINED to four
# cores: six threads oversubscribing four. Measured on the same box, same
# binary, same model, same session:
#
#     inside the harness, -t 6 on 4 cores      24.7 tok/s
#     standalone,         -t 4 on 4 cores      32.08 tok/s
#
# A 23% under-measurement of the BASELINE arm, and it biases the ratio UP --
# jitllm's own arm is unaffected because it reads the taskset. The recorded
# Xeon row's llama.cpp figure is 31.99, which is the standalone number, so this
# fault is newer than that row or that row set the coreset differently; either
# way a hardcoded constant cannot follow the box and must not try.
#
# It counts the entries in $CORES rather than asking nproc, because the taskset
# is the thing both arms are actually confined to.
LCPP_THREADS=$(awk -F, '{print NF}' <<<"$CORES")
PIN=(taskset -c "$CORES")
if [ "$(uname -s)" = Darwin ]; then PIN=(); LCPP_THREADS=4; fi

case "$TIER" in
  cpu) JITLLM_DEV=cpu; LCPP_ARGS=(-dev none -t "$LCPP_THREADS"); JITLLM_EXTRA=() ;;
  # ★ -ngl IS OVERRIDABLE BECAUSE 99 IS NOT ALWAYS THE FAIR SETTING. On a 4 GB
  # card a 7B Q4_K_M does not fit, and llama.cpp does not degrade -- it REFUSES
  # to load. jitllm splits the blocks between device and host automatically and
  # runs, so the honest comparison is against llama.cpp at the largest -ngl that
  # loads, which has to be found by hand. JITLLM_NGL passes it.
  # ★ JITLLM_DEVICES NAMES THE CARDS, AND BOTH ARMS SEE THE SAME ONES. A
  # multi-card row is like for like only when the two engines hold the same
  # cards, so the card set is chosen with CUDA_VISIBLE_DEVICES, which both
  # inherit; JITLLM_DEVICES is jitllm's -devices within it (cuda, and the
  # default gpu, take every visible card, as -ngl 99 does; cuda:0 pins one) and
  # JITLLM_LCPP_EXTRA passes llama-bench anything more (a
  # split mode, -ts). JITLLM_VRAM=0 lets jitllm ask each card what it has free.
  gpu) JITLLM_DEV="${JITLLM_DEVICES:-gpu}"; LCPP_ARGS=(-ngl "${JITLLM_NGL:-99}"); PIN=(); JITLLM_EXTRA=(-vram "${JITLLM_VRAM:-3400000000}")
       if [ -n "${JITLLM_LCPP_EXTRA:-}" ]; then read -r -a _lx <<<"$JITLLM_LCPP_EXTRA"; LCPP_ARGS+=("${_lx[@]}"); fi
       # ★ A HYBRID ROW MUST PIN, A FULL-OFFLOAD ROW MUST NOT. With every block
       # on the card the host is idle and pinning it changes nothing; with the
       # blocks SPLIT the host runs most of the token, and RULE 5 is then in
       # force -- unpinned it spreads onto E-cores and the spin barrier goes
       # 620 ns to 11.9 us. Measured on a 9-of-16 split: 0.6848 unpinned.
       # JITLLM_PIN=1 pins both arms and gives llama.cpp the same thread count.
       if [ -n "${JITLLM_PIN:-}" ]; then
         PIN=(taskset -c "$CORES"); LCPP_ARGS+=(-t "$LCPP_THREADS")
       fi ;;
  *)   echo "tier must be cpu or gpu" >&2; exit 1 ;;
esac

jitllm_run()  { ${PIN[@]+"${PIN[@]}"} "$JITLLM" run -devices "$JITLLM_DEV" ${JITLLM_EXTRA[@]+"${JITLLM_EXTRA[@]}"} -depth "$DEPTH" -n "$N" "$JITLLM_PATH" "Once upon a time in a distant land there" 2>&1 >/dev/null \
               | sed -n 's|.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok/s).*|\1|p'; }  # BSD grep has no -P
lcpp_run() { ${PIN[@]+"${PIN[@]}"} "$LCPP/llama-bench" -m "$MODEL" "${LCPP_ARGS[@]}" -p 0 -n "$N" -d "$DEPTH" -r 1 2>/dev/null \
               | awk -F'|' '/tg'"$N"'/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"±"); print a[1]}'; }

# ★ PREFILL, AND THE TWO ENGINES ARE MADE TO PROCESS THE SAME NUMBER OF TOKENS.
#
# llama-bench's -p takes an exact token count; jitllm takes TEXT, so its count is
# whatever its tokenizer produces. The hand-written comparisons this replaces
# ran jitllm on a ~573-token prompt against llama.cpp's pp512 and called it a row.
# That is not a fair pairing in a knowable direction: prefill rate RISES with
# length, because the fixed per-call costs amortize, so the longer side is
# flattered. Here jitllm is run once first, its actual token count is read off its
# own output, and llama-bench is then told -p <that count>.
PP=""
PPTOK=""
jitllm_pp()  { ${PIN[@]+"${PIN[@]}"} "$JITLLM" run -devices "$JITLLM_DEV" ${JITLLM_EXTRA[@]+"${JITLLM_EXTRA[@]}"} -n 1 "$JITLLM_PATH" "$PP" 2>&1 >/dev/null \
              | sed -n 's|.*prompt [0-9]* tok in [^(]*(\([0-9.]*\) tok/s).*|\1|p'; }
jitllm_pp_ntok() { ${PIN[@]+"${PIN[@]}"} "$JITLLM" run -devices "$JITLLM_DEV" ${JITLLM_EXTRA[@]+"${JITLLM_EXTRA[@]}"} -n 1 "$JITLLM_PATH" "$PP" 2>&1 >/dev/null \
              | sed -n 's|.*prompt \([0-9]*\) tok in .*|\1|p'; }
lcpp_pp() { ${PIN[@]+"${PIN[@]}"} "$LCPP/llama-bench" -m "$MODEL" "${LCPP_ARGS[@]}" -p "$PPTOK" -n 0 -r 1 2>/dev/null \
              | awk -F'|' '/pp'"$PPTOK"'/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"±"); print a[1]}'; }

# ★ AND THE PREFILL ARM IS WARM, AS llama-bench's IS. jitllm_pp times the first
# prompt of a fresh `jitllm run` process: its KV pages are allocated, its pinned
# staging grown and its first chunk's buffers sized inside the timed window,
# all of which llama-bench does before its warm-up run and never times (on an
# RTX 3050 Ti, Llama-3.2-1B, the first prompt of a process spent 43 ms outside
# its kernels, 30 of them growing the KV pool). The faster the card, the larger
# that fixed share of the row; the V100 board, taken with `jitllm speed`, is
# the warm figure. `jitllm speed -p N -n 0 -r 1` is
# llama-bench's pp: one untimed warm-up of the same prompt length on a fresh
# State, then the timed prompt on another. JITLLM_PP_COLD=1 keeps the
# first-prompt-of-a-process figure, which is a time-to-first-token measurement
# and is labelled as one.
jitllm_pp_warm() { ${PIN[@]+"${PIN[@]}"} "$JITLLM" speed -devices "$JITLLM_DEV" ${JITLLM_EXTRA[@]+"${JITLLM_EXTRA[@]}"} -p "$PPTOK" -n 0 -r 1 "$JITLLM_PATH" 2>/dev/null \
              | sed -n 's|^pp[0-9]* \([0-9.]*\).*|\1|p'; }

# THE COLD COSTS ARE ROWS OF THEIR OWN, NOT HIDDEN IN THE WARM ONE. Two rows,
# each engine's boundary read from its own output, never guessed:
#
# JITLLM_MODE=load: process start to model ready. jitllm: `speed -ttft`'s
#   "load" (open + device init and kernel compile + attach: placement, upload,
#   the tuners, the KV pool and the first prompt's kernels), its breakdown
#   printed in the header; "start" is process start (Go's package
#   initialisation) to the command's own first line. llama.cpp: the
#   --log-timestamps stamp of llama-completion's "generate:" line, the last it
#   logs before the prompt's first decode, counted like jitllm's from its
#   logger's start at process start; its breakdown (backend init, model and
#   context loaded, ready) is printed in the header too. Not its "load time",
#   which leaves out the backend's initialisation.
# JITLLM_MODE=ttft: model ready to the first token of the process's first
#   prompt, tokenizing excluded. jitllm: `speed -ttft`'s "ttft-loaded", which
#   refuses the run if any kernel compiled or any tuner trial ran after ready
#   (readyHonest). llama.cpp: llama-completion --no-warmup's "prompt eval time"
#   (its first decode after load; the sampled token is one argmax more).
#
# Both print a per-second figure so the ratio reads as the other rows do:
# loads a second, and prompt tokens a second of the ttft span. The file is in
# the page cache after the token-count probe, which serves llama.cpp's mmap;
# jitllm reads the container with O_DIRECT and gets no help from it. jitllm's
# tuning cache is whatever this host holds (tier.WithTuneCache); the header
# says how many tuner trials the probe run timed.
lcpp_comp() { ${PIN[@]+"${PIN[@]}"} "$LCPP/llama-completion" -m "$MODEL" "${LCPP_ARGS[@]}" -n 1 -no-cnv --no-warmup --log-timestamps --log-prefix -p "$PP" </dev/null 2>&1; }
# lcpp_stamp PATTERN: seconds of the first log line matching PATTERN; llama.cpp
# stamps lines M.SS.mmm.uuu from its logger's start.
lcpp_stamp() { python3 -c '
import re, sys
pat = re.compile(sys.argv[1])
for l in sys.stdin:
    m = re.match(r"(\d+)\.(\d+)\.(\d+)\.(\d+) ", l)
    if m and pat.search(l):
        a, b, c, d = map(int, m.groups())
        print("%.1f" % ((a * 60 + b) * 1e3 + c + d / 1e3))
        break
' "$1"; }
jitllm_cold() { ${PIN[@]+"${PIN[@]}"} "$JITLLM" speed -devices "$JITLLM_DEV" ${JITLLM_EXTRA[@]+"${JITLLM_EXTRA[@]}"} -ttft -p "$PPTOK" -r 1 "$JITLLM_PATH" 2>&1; }
per_s() { python3 -c 'import sys; v=sys.stdin.read().split(); print("%.3f" % (float(sys.argv[1])*1e3/float(v[0]))) if v else None' "$1"; }

if [ "$MODE" = load ] || [ "$MODE" = ttft ]; then
  [ -x "$LCPP/llama-completion" ] || { echo "no llama-completion in $LCPP" >&2; exit 1; }
  PP="$(python3 -c 'import sys; print(" ".join(["the quick brown fox jumps over the lazy dog"]*int(sys.argv[1])))' "$PPREPS")"
  PPTOK="$(jitllm_pp_ntok)"
  [ -n "$PPTOK" ] || { echo "could not determine jitllm's prompt token count" >&2; exit 1; }
  probe="$(jitllm_cold)"
  echo "$probe" | grep -q '^load ' || { echo "jitllm speed -ttft gave no load line:" >&2; echo "$probe" >&2; exit 1; }
  echo "  jitllm probe: $(echo "$probe" | grep '^load ')"
  lprobe="$(lcpp_comp)"
  echo "  llama.cpp probe: backend init $(echo "$lprobe" | lcpp_stamp 'llama backend init') ms, model and context loaded $(echo "$lprobe" | lcpp_stamp 'threadpool init') ms, ready $(echo "$lprobe" | lcpp_stamp ' generate: ') ms (from its logger's start)"
  echo "  caches: model file in the page cache (llama.cpp mmaps it; jitllm reads O_DIRECT), jitllm tuning cache as found on this host"
  if [ "$MODE" = load ]; then
    echo "  load: process start to model ready, as loads/s (jitllm speed -ttft 'load' vs llama-completion's 'generate:' log stamp)"
    jitllm_run() { jitllm_cold | sed -n 's|^load \([0-9.]*\) ms.*|\1|p' | per_s 1; }
    lcpp_run() { lcpp_comp | lcpp_stamp ' generate: ' | per_s 1; }
  else
    echo "  ttft (model loaded), pp$PPTOK: ready to the first token of a fresh process's first prompt, as prompt tok/s (jitllm speed -ttft 'ttft-loaded' vs llama-completion --no-warmup 'prompt eval time')"
    jitllm_run() { jitllm_cold | sed -n 's|.*ttft-loaded \([0-9.]*\) ms.*|\1|p' | per_s "$PPTOK"; }
    lcpp_run() { lcpp_comp | sed -n 's|.*prompt eval time = *\([0-9.]*\) ms.*|\1|p' | per_s "$PPTOK"; }
  fi
  DEPTH=0
  N="$PPTOK"
elif [ "$MODE" = prefill ]; then
  PP="$(python3 -c 'import sys; print(" ".join(["the quick brown fox jumps over the lazy dog"]*int(sys.argv[1])))' "$PPREPS")"
  PPTOK="$(jitllm_pp_ntok)"
  [ -n "$PPTOK" ] || { echo "could not determine jitllm's prompt token count" >&2; exit 1; }
  if [ -n "${JITLLM_PP_COLD:-}" ]; then
    echo "  prefill: jitllm's first prompt of a fresh process (cold, JITLLM_PP_COLD) against llama-bench's warm pp"
    jitllm_run() { jitllm_pp; }
  else
    echo "  warm prefill, pp$PPTOK, no prefix reuse: each sample 1 untimed warm-up then 1 timed prompt on a fresh State (jitllm speed -n 0 -r 1, llama-bench -n 0 -r 1), every sample used"
    jitllm_run() { jitllm_pp_warm; }
  fi
  lcpp_run() { lcpp_pp; }
  DEPTH=0
  N="$PPTOK"
elif [ "$MODE" != decode ]; then
  echo "JITLLM_MODE must be decode, prefill, load or ttft" >&2; exit 1
fi

# ★ THE A/A SELF-CONTROL RUNS ON THIS HARNESS, NOT BESIDE IT. JITLLM_VS_AA=1
# puts jitllm in both arms, so every gate, the order and the per-round ratio
# are the ones the real row goes through, and the median should read 1.0
# inside its IQR. RULE 2: reproducing a number proves a harness consistent; only
# a self-control proves it discriminates, and it does not transfer between
# harnesses or hosts.
if [ -n "${JITLLM_VS_AA:-}" ]; then
  lcpp_run() { jitllm_run; }
  echo "  A/A self-control: jitllm in both arms"
fi
# JITLLM_VS_B_ENV is environment for the B arm alone, space-separated
# NAME=VALUE pairs: JITLLM_PLAN_DEVICE=ptx:0 runs the B build's composed plan
# against the A build's own path.
if [ -n "${JITLLM_VS_B:-}" ]; then
  [ -x "$JITLLM_VS_B" ] || { echo "JITLLM_VS_B=$JITLLM_VS_B is not an executable" >&2; exit 1; }
  lcpp_run() { local JITLLM="$JITLLM_VS_B"; ( if [ -n "${JITLLM_VS_B_ENV:-}" ]; then export ${JITLLM_VS_B_ENV}; fi; jitllm_run ); }
  echo "  B arm: $JITLLM_VS_B (jitllm${JITLLM_VS_B_ENV:+, $JITLLM_VS_B_ENV}), not llama.cpp"
fi

# ★ THE CLAMP GATE. This laptop's sustained-power limit (PL1) intermittently
# pins the package to ~900 MHz for a whole run, against 2900-3600 normally, and
# the run then reads 25 tok/s where its neighbours read 65. It is NOT the thermal
# drift RULE 2c is about, and the giveaway is that the die gets COOLER while the
# rate collapses -- 84 C on a clamped run against 99 C on the fast ones -- because
# a package running at 900 MHz stops producing heat. Measured, rate against the
# median of the frequency samples taken during that same run:
#
#   3599 MHz -> 74.68      3100 MHz -> 64.56      1500 MHz -> 37.95
#   3500 MHz -> 73.00      3100 MHz -> 63.54       900 MHz -> 25.18
#
# It lands on whichever arm happens to be running, so it does not cancel in an
# ABBA and it is not dispersion the IQR gate can see -- one clamped sample in
# twelve moves the median and passes. Two consecutive passes of the same
# comparison read 0.9607 (IQR 0.045, PASSED) and 0.9446 (IQR 0.526, REJECTED),
# which is RULE 2c's "run it twice" catching what one pass could not.
#
# So every sample carries the median CPU frequency observed while it ran, and a
# sample below CLAMP_FRAC of the session median is DISCARDED rather than
# averaged in. Discarding is the honest move: the clamped run measures the
# machine's power budget, not the engine.
CLAMP_FRAC="${JITLLM_CLAMP_FRAC:-0.75}"
# ★ THE COHERENCE GATE, AND IT IS A DIFFERENT QUESTION FROM THE CLAMP GATE.
# The clamp gate asks "was this sample taken while the package was pinned?" and
# judges each sample against the SESSION. It cannot see the failure that
# actually moves a row here: the two arms of one PAIR running at different
# clocks. A real round from one pass --
#
#   jitllm 46.85 37.11   llama.cpp 49.09 42.40   [MHz 2175 1569 1788 1345]
#
# -- spans 1345-2175 MHz INSIDE one round, a 1.62x range, and not one of those
# four samples is below 0.75 of the session median, so all four survive. The
# ratio of that pair is measuring the frequency difference.
#
# And the size is not marginal. This file's own rate-against-frequency table
# (see the clamp gate above) reads 3100 MHz -> 64.05 and 3599 -> 73.84, so rate
# tracks clock very nearly linearly in this range: a 10% clock mismatch between
# the arms is a 10% error in the ratio, against effects of a few percent. RULE 1
# says a ratio is measured in the same pass; a pair whose halves ran at
# different clocks is not the same pass in the way that matters.
#
# A pair whose two frequencies differ by more than this factor is DISCARDED.
# Tighten it on a box that can hold a clock; loosen it only with a reason.
COHERE="${JITLLM_COHERE:-1.05}"
# ★ A GPU ROW WITH EVERY BLOCK ON THE CARDS IS NOT A FUNCTION OF THE HOST'S
# CLOCK, AND BOTH CPU GATES THEN REFUSE IT FOR NOTHING. Unpinned, the host is
# idle while the cards run the token, so its cores sit at whatever the idle
# governor picks: on the V100 box a Llama-3.3-70B row over four cards read
# jitllm 16.96-17.06 and llama.cpp 17.02-17.03 tok/s, every round, while the
# sampled clocks wandered 1200-1550 MHz and the coherence gate discarded five
# of six pairs. The clock samples stay in the output; only the defaults move.
# JITLLM_PIN=1 (a hybrid row, where the host runs blocks) keeps both gates.
if [ "$TIER" = gpu ] && [ -z "${JITLLM_PIN:-}" ]; then
  COHERE="${JITLLM_COHERE:-1}"
  CLAMP_FRAC="${JITLLM_CLAMP_FRAC:-0}"
fi
MHZTMP="$(mktemp)"; trap 'rm -f "$MHZTMP"' EXIT
# Mean over all CPUs, not one: the clamp is a package-wide event.
# ★ SAMPLE THE CORES THE ROW RUNS ON, NOT THE PACKAGE, AND THE OLD FORM BIASED
# THE COHERENCE GATE. Averaging /proc/cpuinfo over ALL cpus mixes the measured
# cores with idle ones, and the two engines do not leave the same cores idle:
# jitllm's pool SPINS for 250 us (this file records 60e9 instructions of workers
# WAITING) which holds its cores' clocks up, where llama.cpp's threads park. The
# result was a SYSTEMATIC separation -- jitllm's sampled MHz exceeded
# llama.cpp's in 27 of 32 pairs, median ratio 1.080 -- against a coherence
# threshold of 1.05. A threshold under the systematic separation does not remove
# outliers; it removes the MAJORITY. It discarded 23 of 32 pairs on one pass and
# 15 of 20 on an olmoe pass, which is what rejected that row at n=5.
#
# CORES is the taskset this row pins to, so these are the only clocks the ratio
# is about. Falling back to the package average keeps darwin and any host
# without cpufreq working exactly as before.
mhz_now() {
  local s n f
  s=0; n=0
  for c in ${CORES//,/ }; do
    f=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null) || continue
    s=$((s + f)); n=$((n + 1))
  done
  if [ "$n" -gt 0 ]; then echo $((s / n / 1000)); return; fi
  # ★ NO cpufreq SYSFS (the Xeon D-2123IT is like this) -- take the same cpus
  # out of /proc/cpuinfo instead, which carries a per-PROCESSOR "cpu MHz". The
  # fallback that averaged the whole file is what biased the coherence gate, so
  # it is not the fallback any more: only a host with neither source lands there.
  awk -v want=",${CORES}," '
    /^processor/ { p = $NF }
    /^cpu MHz/   { if (index(want, "," p ",")) { s += $4; n++ } }
    END { if (n) printf "%.0f", s/n }
  ' /proc/cpuinfo 2>/dev/null
}
# Darwin exposes no such file; there the gate is a no-op, and RULE 1c's Mac IQRs
# of 0.003-0.010 say it is not needed -- the M4 does not thermally thrash.
HAVE_MHZ=1; [ -z "$(mhz_now)" ] && HAVE_MHZ=0

# sample <fn> -> "<rate> <median MHz during it>"
sample() {
  if [ "$HAVE_MHZ" = 0 ]; then echo "$("$1") 0"; return; fi
  : > "$MHZTMP"
  ( while :; do mhz_now >> "$MHZTMP"; echo >> "$MHZTMP"; sleep 0.4; done ) &
  _sp=$!
  _rate=$("$1")
  kill "$_sp" 2>/dev/null; wait "$_sp" 2>/dev/null
  _m=$(grep -v '^$' "$MHZTMP" | sort -n | awk '{v[NR]=$1} END{if(NR) print v[int((NR+1)/2)]; else print 0}')
  echo "$_rate ${_m:-0}"
}

echo "== $(basename "$MODEL")  $TIER  $MODE  n=$N  depth=$DEPTH  rounds=$ROUNDS"
# ★ SOAK TO THERMAL STEADY STATE FIRST, WHICH IS THE OPPOSITE OF LETTING IT COOL.
# This laptop saturates within seconds of six-core AVX2 and settles at 100 C with
# the clocks pinned; on the way there it passes through every frequency in
# between. Measured on tinyllama, the SAME A/B three ways:
#
#   cold start, back to back      two passes read 0.9733 and 0.8530  -- 14% apart
#   6 s cooldown between samples  IQR/median 0.154                   -- REJECTED
#   soaked first, then measured   IQR/median 0.067, reproducible     -- usable
#
# Cooling between samples is the intuitive fix and it is the wrong one: it puts
# every sample at a DIFFERENT point on the ramp. Saturating first puts them all
# at the same point, which happens to be the slowest one -- and the slowest one
# is stable, which is what a ratio needs. Absolute numbers from a soaked run are
# ~20% under this box's peak and are not comparable with a cold single run; the
# RATIO is the measurement and it is the only thing quoted.
#
# It also serves the other two warm-ups for free: jitllm's tuner converges over the
# first tokens (RULE 2a) and both engines fault in the weight file.
echo "  soaking to thermal steady state..."
for _ in $(seq 1 "${JITLLM_SOAK:-6}"); do jitllm_run >/dev/null; lcpp_run >/dev/null; done

SAMPLES=""
for r in $(seq 1 "$ROUNDS"); do
  sa1=$(sample jitllm_run); sb1=$(sample lcpp_run); sb2=$(sample lcpp_run); sa2=$(sample jitllm_run)  # ABBA
  a1=${sa1% *}; f1=${sa1##* }; b1=${sb1% *}; g1=${sb1##* }
  b2=${sb2% *}; g2=${sb2##* }; a2=${sa2% *}; f2=${sa2##* }
  # ★ AN EMPTY SAMPLE IS A FAILED ENGINE, NOT A ZERO. llama.cpp prints its error
  # and exits 0 with an empty results table when a model will not fit, and with
  # `set -e` plus a silent `tail` that presented as the script hanging in the
  # soak. Say which side produced nothing.
  for v in "a1=$a1" "b1=$b1" "b2=$b2" "a2=$a2"; do
    case "$v" in *=) echo "  ${v%=} produced no sample -- the engine failed to run this config" >&2; exit 1 ;; esac
  done
  printf "  round %d: jitllm %s %s   llama.cpp %s %s   [MHz %s %s %s %s]\n" \
    "$r" "$a1" "$a2" "$b1" "$b2" "$f1" "$f2" "$g1" "$g2"
  SAMPLES="$SAMPLES$a1 $b1 $f1 $g1
$a2 $b2 $f2 $g2
"
done

echo "$SAMPLES" | CLAMP_FRAC="$CLAMP_FRAC" COHERE="$COHERE" python3 -c '
import os, sys, statistics
rows = []
for ln in sys.stdin:
    f = ln.split()
    if len(f) == 4:
        a, b, fa, fb = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        if a > 0 and b > 0:
            rows.append((a, b, fa, fb))
if not rows:
    print("  no samples"); sys.exit(1)

# ★ DISCARD SAMPLES TAKEN WHILE THE PACKAGE WAS POWER-CLAMPED. A clamped run
# measures the sustained-power budget of the box, not the engine, and because the
# clamp lands on ONE arm it does not cancel in the ABBA. Judge each sample
# against the session median frequency rather than an absolute threshold, so the
# gate travels to a box with different clocks.
#
# ★ A SAMPLE WITH NO FREQUENCY READING IS UNJUDGEABLE, NOT CLAMPED, AND CALLING
# IT CLAMPED CORRUPTS THE ONE NUMBER THIS GATE EXISTS TO PRODUCE. The read can
# come back 0 for a sample on a host that reports frequencies for the others,
# and `0 >= frac*fmed` is false, so such a sample was dropped AND counted in
# "N sample(s) discarded: CPU power-clamped". That number is what gets read as
# "the box is thermally sick" -- the Llama-3.2-1B entry in AGENTS.md turns on
# how many samples were clamped -- so a failed read was being promoted into
# thermal evidence. The coherence gate below already had this right: it guards
# `fa > 0 and fb > 0` before judging a pair. This one now matches it, and the
# unjudged count is reported SEPARATELY rather than folded in.
frac = float(os.environ.get("CLAMP_FRAC", "0.75"))
freqs = [f for r in rows for f in (r[2], r[3]) if f > 0]
dropped = 0
nofreq = 0
if freqs:
    fmed = statistics.median(freqs)
    judged = [r for r in rows if r[2] > 0 and r[3] > 0]
    nofreq = len(rows) - len(judged)
    keep = [r for r in judged if r[2] >= frac * fmed and r[3] >= frac * fmed]
    dropped = len(judged) - len(keep)
    # Unjudged samples ride through: the gate has nothing to say about them, and
    # silently deleting a sample it cannot assess is how n shrinks for a reason
    # nobody can see afterwards.
    keep += [r for r in rows if not (r[2] > 0 and r[3] > 0)]
    if keep:
        rows = keep

# ★ AND DISCARD PAIRS WHOSE TWO ARMS DID NOT RUN AT THE SAME CLOCK. Separate
# from the clamp gate above, which judges each sample against the session and
# therefore cannot see a pair that is internally incoherent. A frequency of 0
# means the host exposes none (darwin), where this is a no-op.
cohere = float(os.environ.get("COHERE", "1.05"))
incoh = 0
if cohere > 1:
    keep = []
    for r in rows:
        fa, fb = r[2], r[3]
        if fa > 0 and fb > 0 and max(fa, fb) / min(fa, fb) > cohere:
            incoh += 1
        else:
            keep.append(r)
    if keep:
        rows = keep
    else:
        print("  every pair ran its two arms at different clocks (> %.2fx): "
              "this box cannot produce a row right now" % cohere)
        sys.exit(1)

rs = sorted(a / b for a, b, _, _ in rows)
med = statistics.median(rs)
iqr = rs[len(rs)*3//4] - rs[len(rs)//4]
verdict = "jitllm WINS" if med > 1 else "jitllm loses"
gate = "" if iqr/med <= 0.10 else "   REJECTED: IQR/median %.3f > 0.10" % (iqr/med)
# ★ n IS REPORTED AFTER BOTH GATES AND A THIN ROW IS REFUSED, because a median
# of three survivors is not a measurement however tight its IQR looks.
if len(rs) < 6:
    gate = "   REJECTED: only %d pair(s) survived the gates" % len(rs)
# ★ AND THE RATIO ITSELF MAY BE A FUNCTION OF THE CLOCK, WHICH PAIR-COHERENCE
# CANNOT SEE. The gate above makes each pair internally fair; it says nothing
# about whether a fair pair at 1500 MHz gives the same ratio as a fair pair at
# 2100. This file already records that the two engines do not shed rate to a
# clamp equally (the M4 prefill row: jitllm -7.4%, llama.cpp -4.2%), so they
# need not. If they do not, a session that wanders across 600 MHz cannot produce
# a row however many pairs survive, and the IQR is the symptom rather than the
# cause.
#
# So the correlation is REPORTED, every run, rather than left to be rediscovered
# by hand -- and it is reported only at n >= 6, because a Pearson r over two
# points is 1.000 by construction and reads as a finding.
trend = ""
if len(rows) >= 6:
    xs = [(r[2] + r[3]) / 2 for r in rows if r[2] > 0 and r[3] > 0]
    ys = [r[0] / r[1] for r in rows if r[2] > 0 and r[3] > 0]
    if len(xs) >= 6:
        n_ = len(xs)
        mx, my = sum(xs) / n_, sum(ys) / n_
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs)
        dy = sum((y - my) ** 2 for y in ys)
        if dx > 0 and dy > 0:
            r_ = num / (dx * dy) ** 0.5
            per100 = 100 * num / dx
            if abs(r_) >= 0.5:
                # ★ A CORRELATION IS A SUSPICION, NOT A VERDICT, AND THE FIRST
                # VERSION OF THIS MESSAGE GAVE ORDERS. It read "Pin the
                # frequency ... this row is a reading of the clock", and on
                # olmoe it fired at r=+0.54 over 13 points for a line that is
                # not in the data: subtracting it took IQR/median from 0.013 to
                # 0.064. The test of a trend is whether removing it TIGHTENS the
                # spread, which needs the residual, which is scripts/clockadj.py.
                # So this points at the check rather than prescribing the cure.
                trend = ("\n  ratio may track the clock: r=%+.2f, %+.4f per 100 MHz "
                         "over %.0f-%.0f MHz.\n    Check it with `scripts/clockadj.py "
                         "-ref <MHz>` on this output: if removing the\n    clock term does "
                         "not TIGHTEN the spread, the trend is not in the data and the\n"
                         "    median above stands." %
                         (r_, per100, min(xs), max(xs)))
            else:
                trend = "\n  ratio vs clock: r=%+.2f over %.0f-%.0f MHz (no trend)" % (
                    r_, min(xs), max(xs))
note = "" if not dropped else "  [%d sample(s) discarded: CPU power-clamped]" % dropped
if nofreq:
    note += ("  [%d sample(s) KEPT UNJUDGED: the host reported no frequency for "
             "them, which is not evidence of a clamp]" % nofreq)
if incoh:
    note += "  [%d pair(s) discarded: arms ran at different clocks, > %.2fx]" % (incoh, cohere)
print("  ratio %.4f  (IQR/median %.3f, n=%d)   %s%s%s%s" % (med, iqr/med, len(rs), verdict, gate, note, trend))
'
