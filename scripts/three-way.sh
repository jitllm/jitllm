#!/usr/bin/env bash
# Prove that ONE model runs with its blocks split across the HOST, a CUDA device
# and a SECOND PHYSICAL device reached through Vulkan -- all three carrying real
# work in a single run -- and that the tokens are the ones the host alone
# produces.
#
# ★ IT IS A SCRIPT AND NOT A TRANSCRIPT BECAUSE A TRANSCRIPT CANNOT BE RE-RUN.
# The claim it makes is the kind that rots silently: a router change that sends
# every block to the first device, a budget change that starves the second, a
# kernel that quietly computes garbage on the card nobody benchmarks -- each of
# those leaves a run that still prints a rate and still answers in English.
# Every one of them fails an assertion here.
#
# ★ WHAT IS ASSERTED, AND WHY EACH ONE EXISTS (--violate NAME breaks exactly one
# of them on purpose; a gate that has never been seen to fail is not a gate):
#
#   ids        the three-way run's token ids equal `-devices cpu`'s, same prompt,
#              same greedy sampling. THE LOAD-BEARING ONE: a device that computes
#              garbage, or that declines every block while reporting success,
#              gives a run that looks exactly like a working one.
#              violation: --violate ids (compare against a doctored reference)
#   devices    both devices RAN blocks, not merely held them. gpu.Stats.Blocks
#              counts submissions that returned and whose output was copied back,
#              per device, so "placed 6, run 0" is visible.
#              violation: --violate one-device
#   distinct   the two devices are different PHYSICAL devices. On this laptop the
#              discrete card answers to BOTH cuda:0 and vulkan:0, so a two-entry
#              spec that looks like a split can be one card twice -- the single
#              most likely way this ships broken.
#              violation: --violate same-card
#   host       the host still ran blocks. Two devices taking the whole model is a
#              different configuration wearing the same output.
#              violation: --violate host
#   repeat     the second run agrees with the first, so the placement is not an
#              ordering accident.
#   negative   with one device budgeted out, the block counts MOVE and the ids
#              STILL match. If the ids matched no matter what, nothing on the
#              devices would be reaching the tokens.
#
# ★ THE SECOND DEVICE USED TO NEED JITLLM_GPU_SOFTMAX=scalar HERE, AND THAT
# WORKAROUND IS GONE. The Iris Xe reports minSubgroupSize 8 /
# maxSubgroupSize 32 and chooses a width per shader, so a 32-lane butterfly
# reduction was undefined behaviour on it -- correct under an on-device probe,
# NaN logits inside a real submission of two or more blocks. The width is now
# PINNED at pipeline creation (VK_EXT_subgroup_size_control) on every device that
# will promise it, and the scalar twin is selected by construction on every
# device that will not, so this script sets nothing. `--violate subgroup-width`
# puts the bug back -- JITLLM_VK_SUBGROUP=force assumes the width instead of
# pinning it -- and the `ids` assertion must then fail.
#
# ★ THE MODEL HAS TO BE ONE THE DEVICE TIER ALREADY REPRODUCES, AND THAT IS NOT
# EVERY MODEL. This compares whole greedy CHAINS, so one flipped argmax anywhere
# makes every later id differ (RULE 12i) -- and a model the device gets wrong on
# ONE card is wrong on two. tinyllama q3_K_M is clean on both devices here --
# the free-running greedy chains this script compares matched `-devices cpu`
# exactly in every pass and both negative controls. Its teacher-forced diff is
# 0/12 on four runs of six and 1/12 on the other two, ALWAYS at position 4, whose
# HOST-side argmax margin is 0.000162: a coin-flip the f32 seam decides either
# way, not a device fault. The same harness on the NVIDIA card reports
# max|dlogit| 0.599-0.682 against the iGPU's 0.605-0.717, so the band is the
# seam's, not this device's. Qwen2-1.5B is NOT clean, at
# 7/12 and max|dlogit| 19 on one CUDA card with no split at all, at this commit
# and at the commit before any of this work. The script says so rather than
# blaming the router: see the single-device note it prints on a failure.
#
# usage:  scripts/three-way.sh [-m MODEL] [-n TOKENS] [-p PROMPT] [--violate NAME]
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${JITLLM_TW_MODEL:-${JITLLM_MODELS:?set JITLLM_MODELS to the model directory}/tinyllama-1.1b-q3_K_M.gguf}
NTOK=12
PROMPT="The capital of France is"
VIOLATE=""
# 200 MB apiece is chosen so that NEITHER device can hold the model: the seam has
# to land on both cards and still leave blocks for the host, which is the whole
# configuration under test. A budget that fits the model proves nothing.
BUDGET=200M
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL=$2; shift 2 ;;
    -n) NTOK=$2; shift 2 ;;
    -p) PROMPT=$2; shift 2 ;;
    --violate) VIOLATE=$2; shift 2 ;;
    -h|--help) sed -n '1,/^# usage:/p' "$0"; exit 0 ;;
    *) echo "three-way: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -r "$MODEL" ] || { echo "three-way: no model at $MODEL (-m MODEL, or JITLLM_TW_MODEL)" >&2; exit 2; }

# ★ THE TUNER IS PINNED, AND NOT PINNING IT IS WHY THIS SCRIPT FAILED ~1 RUN IN
# 30 FOR MONTHS. Every assertion below is TOKEN-ID EQUALITY against a host run,
# and tuneSplit measures at load -- so the split, and the f32 reduction order
# with it, varies PER PROCESS. AGENTS.md already said so in the entry on
# CPU-vs-device equality: "Run-to-run it is the SPLIT TUNER ... JITLLM_GPU_TUNE=0
# makes three runs identical. Pin it before comparing anything." This script did
# not pin it, so it was asserting bit-stability across a knob that legitimately
# moves.
#
# Attributed with a deterministic reproducer. Llama-3.2-1B-Instruct
# Q4_K_M on cuda:0, the SHIPPED path, one process per split:
#
#     split  1  2  4     correct
#     split  8  16       degenerate repetition -- it replays an earlier span
#     split 32           correct
#
# and `jitllm verify` names the mechanism at the flip: "pos 5: cpu 35490
# (margin 0.00167)". A 0.0017-logit margin is inside the reduction-order band
# this file documents at 0.2-0.94, so the argmax is a coin flip and greedy
# decoding amplifies it into the repetition. tinyllama and gemma-2b do not move
# at any split; it takes a knife-edge margin to show.
export JITLLM_GPU_TUNE=0
CAP="$ROOT/scripts/cap 8G -- taskset -c 0,2,4,6,8,10"
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
BIN=$ROOT/jitllm
fails=0
note() { printf '%s\n' "$*"; }
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fails=$((fails+1)); }

$ROOT/scripts/cap 8G -- go build -o "$BIN" ./cmd/jitllm || exit 1

# ★ AND IT RUNS THE CONTAINER, NOT THE GGUF. `model.Open` refuses anything but a
# .jlm, so passing $MODEL straight to `run` made every arm of this proof print
# the convert command and the script exit 1 -- which it had been doing since the
# container landed, because nothing here reconverts. Same rule as
# vs-llamacpp.sh: reconvert when the file is MISSING or OLDER THAN THE BINARY,
# since a container version bump leaves a file that exists and cannot be read.
JLM="${MODEL%.gguf}.jlm"
if [ ! -f "$JLM" ] || [ "$JLM" -ot "$BIN" ]; then
  note ">> converting $(basename "$MODEL") -> $(basename "$JLM")"
  $ROOT/scripts/cap 26G -- "$BIN" convert "$MODEL" "$JLM" >/dev/null || exit 1
fi

# ---------------------------------------------------------------- device choice
# ★ THE DEVICES ARE DISCOVERED, NOT HARDCODED. `vulkan:1` is the Iris Xe on this
# laptop and is something else on the next one; the property that matters is
# "reached through a different API AND a different piece of silicon", and
# `jitllm hardware` is where the engine says which is which.
$CAP "$BIN" hardware >"$OUT/hw" 2>&1 || { cat "$OUT/hw"; exit 1; }
CUDA_ORD=$(awk '/ ptx +#[0-9]+/{ match($0, /#[0-9]+/); print substr($0, RSTART+1, RLENGTH-1); exit }' "$OUT/hw")
CUDA_NAME=$(sed -n 's/^.* ptx  *#[0-9][0-9]*  *\(.*\)$/\1/p' "$OUT/hw" | head -1 \
            | sed 's/  *(sm_[^)]*).*//; s/  *slots .*//; s/ *$//')
# The Vulkan device to take is the one whose memory is SHARED with the host --
# an integrated GPU is by construction not the discrete card. Failing that, any
# Vulkan device whose NAME differs from the CUDA one and that is not a software
# rasteriser.
VK_IDX=$(awk -v c="$CUDA_NAME" '
  /^ +vulkan:[0-9]+/ {
    line=$0; match(line, /vulkan:[0-9]+/); idx=substr(line, RSTART+7, RLENGTH-7)
    name=line; sub(/^ +vulkan:[0-9]+ +/, "", name); sub(/ +(memory SHARED|software rasteriser).*/, "", name)
    gsub(/ +$/, "", name)
    if (line ~ /software rasteriser/) next
    if (line ~ /memory SHARED with the host/) { print idx; exit }
    if (c != "" && name != c && alt == "") alt = idx
  }
  END { if (alt != "") print alt }' "$OUT/hw")
VK_NAME=$(awk -v i="$VK_IDX" '$0 ~ ("^ +vulkan:" i " ") { sub(/^ +vulkan:[0-9]+ +/, ""); sub(/ +(memory SHARED|software rasteriser).*/, ""); sub(/ +$/, ""); print; exit }' "$OUT/hw")
# The Vulkan index of the CUDA card itself, if it has one: that is the --violate
# same-card spec, and it is discovered rather than assumed to be vulkan:0.
VK_SAME=$(awk -v c="$CUDA_NAME" '
  /^ +vulkan:[0-9]+/ {
    line=$0; match(line, /vulkan:[0-9]+/); idx=substr(line, RSTART+7, RLENGTH-7)
    name=line; sub(/^ +vulkan:[0-9]+ +/, "", name); sub(/ +(memory SHARED|software rasteriser).*/, "", name)
    gsub(/ +$/, "", name)
    if (c != "" && name == c) { print idx; exit }
  }' "$OUT/hw")
if [ -z "${CUDA_ORD:-}" ] || [ -z "${VK_IDX:-}" ]; then
  note "three-way: this host cannot demonstrate a two-PHYSICAL-device split."
  note "           CUDA ordinal: ${CUDA_ORD:-none}   second Vulkan device: ${VK_IDX:-none}"
  note "           jitllm hardware says:"; sed -n '/^gpu/,$p' "$OUT/hw"
  exit 2
fi
note "host      $(uname -sm)"
note "cuda:$CUDA_ORD    $CUDA_NAME"
note "vulkan:$VK_IDX  $VK_NAME"
note "model     $MODEL"
note "prompt    \"$PROMPT\"   -n $NTOK   greedy (-temp 0)"
note ""

SPEC3="cuda:$CUDA_ORD=$BUDGET,vulkan:$VK_IDX=$BUDGET"
# ★ NOTHING IS SET HERE ANY MORE, AND THAT IS THE CLAIM. Every kernel whose
# arithmetic needs a 32-lane subgroup is created with the width pinned or is not
# created at all, so the second device needs no help from the harness. It is
# still an array, and still printed with the pass, because --violate fills it.
SMENV=()

case "$VIOLATE" in
  same-card)  [ -n "$VK_SAME" ] || { echo "three-way: no Vulkan index names the CUDA card on this host; the same-card violation is not available" >&2; exit 2; }
              SPEC3="cuda:$CUDA_ORD=$BUDGET,vulkan:$VK_SAME=$BUDGET" ;; # both entries = ONE card
  one-device) SPEC3="cuda:$CUDA_ORD=$BUDGET" ;;
  host)       SPEC3="cuda:$CUDA_ORD=4G,vulkan:$VK_IDX=$BUDGET" ;;  # one card swallows the model
  subgroup-width) SMENV=(JITLLM_VK_SUBGROUP=force) ;;  # assume the width instead of pinning it
  ids|"")     ;;
  *) echo "three-way: unknown violation $VIOLATE (ids|one-device|same-card|host|subgroup-width)" >&2; exit 2 ;;
esac
[ -n "$VIOLATE" ] && note "★ VIOLATION MODE \"$VIOLATE\": one assertion below MUST fail, or the gate is not a gate."

# run TAG SPEC [env...] -- writes $OUT/TAG, echoes nothing.
run() {
  local tag=$1 spec=$2; shift 2
  $CAP env "$@" "$BIN" run -devices "$spec" -n "$NTOK" -temp 0 "$JLM" "$PROMPT" >"$OUT/$tag.stdout" 2>"$OUT/$tag"
}
ids()      { sed -n 's/^ids \(\[.*\]\)$/\1/p' "$OUT/$1" | tail -1; }
# placed/run/matvecs for the device whose label matches $2
placed()   { sed -n "s/^device .*$2.* \([0-9][0-9]*\) block(s) placed, .*/\1/p" "$OUT/$1" | tail -1; }
ran()      { sed -n "s/^device .*$2.* [0-9][0-9]* block(s) placed, \([0-9][0-9]*\) run, .*/\1/p" "$OUT/$1" | tail -1; }
# host blocks come from the engine's own seam line: "<devices>: 13/22 blocks"
hostblocks() { awk '/^device .*: [0-9]+\/[0-9]+ blocks/ { match($0, /[0-9]+\/[0-9]+ blocks/); split(substr($0, RSTART, RLENGTH-7), a, "/"); print a[2]-a[1]; exit }' "$OUT/$1"; }
devlines() { grep -E '^device .*block\(s\) (placed|of)' "$OUT/$1"; }

# ------------------------------------------------------------------- reference
note "== reference: the host alone =="
run cpu cpu NOTHING=1 || { sed -n '$p' "$OUT/cpu"; exit 1; }
REF=$(ids cpu)
[ -n "$REF" ] || { echo "three-way: the cpu arm printed no ids"; cat "$OUT/cpu"; exit 1; }
note "  -devices cpu   $REF"
if [ "$VIOLATE" = "ids" ]; then
  # Doctor the reference: the comparator itself has to be able to fail.
  REF=$(printf '%s' "$REF" | sed 's/\[1 /[999 /')
  note "  ★ doctored reference: $REF"
fi
note ""

# ------------------------------------------------------- the three-way run x2
for pass in 1 2; do
  note "== pass $pass: -devices $SPEC3 ${SMENV[*]:-} =="
  run "t$pass" "$SPEC3" "${SMENV[@]:-NOTHING=1}" || { tail -3 "$OUT/t$pass"; bad "pass $pass did not run"; continue; }
  devlines "t$pass" | sed 's/^/  /'
  note "  ids $(ids "t$pass")"

  [ "$(ids "t$pass")" = "$REF" ] && ok "ids: identical to the host-only run" \
                                 || bad "ids: pass $pass differs from the host-only run"

  cran=$(ran "t$pass" "\\[ptx\\]"); vran=$(ran "t$pass" "\\[spirv\\]")
  cpl=$(placed "t$pass" "\\[ptx\\]"); vpl=$(placed "t$pass" "\\[spirv\\]")
  [ "${cran:-0}" -gt 0 ] 2>/dev/null && ok "devices: the CUDA device EXECUTED ${cran} block-calls (${cpl:-0} blocks resident)" \
                                     || bad "devices: the CUDA device executed ${cran:-0} block-calls (${cpl:-0} blocks resident)"
  [ "${vran:-0}" -gt 0 ] 2>/dev/null && ok "devices: the Vulkan device EXECUTED ${vran} block-calls (${vpl:-0} blocks resident)" \
                                     || bad "devices: the Vulkan device executed ${vran:-0} block-calls (${vpl:-0} blocks resident)"

  # distinct: the spirv device must not be the card CUDA is on.
  if devlines "t$pass" | grep -q '\[spirv\]'; then
    vlabel=$(devlines "t$pass" | grep '\[spirv\]' | sed 's/^device [0-9]*://; s/ [0-9]* block(s).*//')
    if [ "$vlabel" = "$CUDA_NAME [spirv]" ]; then
      bad "distinct: the Vulkan device IS the CUDA card ($vlabel) -- one card, two APIs, no split"
    else
      ok "distinct: $vlabel is not $CUDA_NAME"
    fi
  else
    bad "distinct: no Vulkan device in the run at all"
  fi

  hb=$(hostblocks "t$pass")
  [ "${hb:-0}" -gt 0 ] 2>/dev/null && ok "host: the host ran ${hb} blocks" \
                                   || bad "host: the host ran ${hb:-0} blocks -- the devices took everything"
  note ""
done
if [ -s "$OUT/t1" ] && [ -s "$OUT/t2" ]; then
  [ "$(ids t1)" = "$(ids t2)" ] && ok "repeat: both passes produced the same ids" \
                                || bad "repeat: the two passes disagree"
  note ""
fi

# ---------------------------------------------------------- negative controls
# ★ THE COUNTS MUST MOVE AND THE IDS MUST NOT. A device budgeted to one byte
# declines every block, so its share goes to the next device or to the host; if
# the ids were the same whatever the placement, nothing on the cards would be
# reaching the tokens.
note "== negative controls: budget one device out =="
run nocuda "cuda:$CUDA_ORD=1,vulkan:$VK_IDX=$BUDGET" "${SMENV[@]:-NOTHING=1}"
run novk   "cuda:$CUDA_ORD=$BUDGET,vulkan:$VK_IDX=1" "${SMENV[@]:-NOTHING=1}"
for tag in nocuda novk; do
  devlines "$tag" | sed 's/^/  /'
  [ "$(ids "$tag")" = "$REF" ] && ok "$tag: ids still identical to the host-only run" \
                               || bad "$tag: ids changed when a device was budgeted out"
done
zc=$(ran nocuda "\\[ptx\\]"); zv=$(ran novk "\\[spirv\\]")
{ [ "${zc:-1}" = "0" ] && [ "${zv:-1}" = "0" ]; } \
  && ok "negative: the budgeted-out device ran 0 blocks in each control (cuda $zc, vulkan $zv)" \
  || bad "negative: a device budgeted to 1 byte still ran blocks (cuda ${zc:-?}, vulkan ${zv:-?})"
t1c=$(ran t1 "\\[ptx\\]"); t1v=$(ran t1 "\\[spirv\\]")
{ [ "${zc:-0}" != "${t1c:-0}" ] && [ "${zv:-0}" != "${t1v:-0}" ]; } \
  && ok "negative: the block counts MOVED (cuda ${t1c:-0}->${zc:-?}, vulkan ${t1v:-0}->${zv:-?})" \
  || bad "negative: the block counts did not move, so the budget is not what places blocks"
note ""

# ★ AND IF THE IDS DISAGREE, SAY WHETHER THE SPLIT IS THE VARIABLE. A model the
# device tier gets wrong on ONE device is wrong on two, and reading that as a
# multi-device failure sends the next person to the router. Qwen2-1.5B is such a
# model on this box TODAY: `jitllm verify -devices cuda:0 -vram 400000000` reports
# 7/12 positions apart at max|dlogit| 19 with one card and no split at all, and it
# reports the same at the commit before any of this work.
if [ "$fails" -gt 0 ] && { [ "$(ids nocuda)" != "$REF" ] || [ "$(ids novk)" != "$REF" ]; }; then
  note "★ a SINGLE-device control also disagrees with the host, so the SPLIT is not the"
  note "  variable: this model does not reproduce the host on one device either."
  note "  Localise it with the teacher-forced diff, which reports the margin:"
  note "      jitllm verify -devices cuda:$CUDA_ORD -vram 400000000 -n $NTOK $MODEL"
  note "      jitllm verify -devices vulkan:$VK_IDX -vram 400000000 -n $NTOK $MODEL"
  note ""
fi

if [ -n "$VIOLATE" ]; then
  if [ "$fails" -gt 0 ]; then
    note "VIOLATION \"$VIOLATE\": $fails assertion(s) failed, which is the expected result."
    exit 0
  fi
  note "VIOLATION \"$VIOLATE\": NOTHING FAILED. The assertion it targets is not a gate."
  exit 1
fi
if [ "$fails" -eq 0 ]; then
  note "PROVED on this host: one model, $((${cpl:-0} + ${vpl:-0} + ${hb:-0})) blocks --"
  note "  ${cpl:-0} on $CUDA_NAME [ptx]"
  note "  ${vpl:-0} on $VK_NAME [spirv]"
  note "  ${hb:-0} on the host"
  note "  token ids identical to -devices cpu in both passes and in both negative controls."
  exit 0
fi
note "$fails assertion(s) failed"
exit 1
