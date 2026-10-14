#!/usr/bin/env bash
# Compare jitllm against Strata (Niko1221/Strata, the engine built for
# Qwen3.8-Flash-Next) on one model and one device, PAIRED and INTERLEAVED, and
# report the median of per-round ratios with its IQR (AGENTS.md RULE 1, 2).
#
#   JITLLM_PATH=/m/q.jlm STRATA=~/strata-src/build-v100/strata \
#     STRATA_PACK=/m/strata/pack STRATA_SHARD1=/m/q-00001-of-00004.gguf \
#     scripts/vs-strata.sh 20
#
# It is vs-llamacpp.sh's method with Strata in the B arm: the other engine is
# another program, so each sample is a fresh process of one engine, the rounds
# alternate ABBA so drift cannot accumulate on one side, and each engine's own
# decode rate is read off its own output (load and prompt excluded on both).
# STRATA_VS_AA=1 puts jitllm in both arms: the self-control that shows the
# harness discriminates, run first and quoted beside the row.
#
# Both engines run on one GPU and one socket: CUDA_VISIBLE_DEVICES=$GPU, and
# numactl binds the CPU and the memory to $NODE, since both run routed experts
# on the host. Speculation is off in both arms: jitllm runs no -spec, and
# Strata (whose native pack requires verify windows, --spec 2) is given no
# drafter -- no --mtp and --suffix-draft 0 -- so no draft is ever accepted; its
# report of accepted drafts is checked to be zero on every sample.
set -euo pipefail

if [ "${JITLLM_NO_LOCK:-0}" != "1" ] && command -v flock >/dev/null 2>&1; then
  lock="${TMPDIR:-/tmp}/jitllm-cap.lock"
  exec 9>>"$lock"
  flock -w "${JITLLM_LOCK_WAIT:-1800}" -x 9 || { echo ">> REFUSING: the box stayed busy" >&2; exit 75; }
fi

ROUNDS="${1:-20}"
JITLLM="${JITLLM:-./jitllm}"
: "${JITLLM_PATH:?the .jlm container}" "${STRATA:?the strata binary}" "${STRATA_PACK:?the Strata pack}"
: "${STRATA_SHARD1:?the first GGUF shard, where Strata reads the experts}"
GPU="${GPU:-0}"
NODE="${NODE:-0}"
N="${STRATA_VS_N:-128}"
PROMPT="${STRATA_VS_PROMPT:-The capital of France is}"
JITLLM_ENV="${JITLLM_ENV:-JITLLM_EXPERTS=host}"
STRATA_ARGS="${STRATA_ARGS:---resident-budget-gib 100 --expert-cache auto --max-context 4096 --prefill auto}"
STRATA_PROFILE="${STRATA_PROFILE:-$(dirname "$STRATA")/../data/expert-profile.bin}"
PIN=(numactl --cpunodebind="$NODE" --membind="$NODE")
LOG="${STRATA_VS_LOG:-vs-strata.$(date +%s).log}"

# The prompt as ids, from jitllm's own tokenizer, so both engines read the
# same tokens.
IDS="$("$JITLLM" tokenize "$JITLLM_PATH" "$PROMPT" 2>&1 | sed -n 's/^[0-9]* tokens: \[\(.*\)\]$/\1/p' | tr ' ' ',')"
[ -n "$IDS" ] || { echo "could not tokenize the prompt" >&2; exit 1; }

psi() { awk '/^some/{split($2,a,"="); print a[2]}' /proc/pressure/cpu 2>/dev/null || echo 0; }
# How busy the CPUs are before a sample (corebusy.py): PSI cannot see a
# tenant on a box with CPUs to spare (RULE 2), and every GPU must be empty.
# Every CPU, not only the node's: a tenant elsewhere on the box still
# shares the memory controllers and the PCIe root, and on cato other
# sessions' gates ran on every core and on this run's GPU.
CPUS="$(seq -s, 0 $(($(nproc --all) - 1)))"
BUSYMAX="${STRATA_VS_BUSY:-10}"
busy() { "$(dirname "$0")/corebusy.py" "$CPUS" 2>/dev/null || echo 0; }
# Waits for the node to go quiet (STRATA_VS_WAIT seconds at most), so a
# tenant that started first is outlasted rather than measured.
quiet() {
  local waited=0 b g
  while :; do
    b="$(busy)"
    g="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')"
    awk -v b="$b" -v m="$BUSYMAX" -v g="${g:-0}" 'BEGIN{exit !(b <= m && g < 600)}' && return 0
    if [ "$waited" -ge "${STRATA_VS_WAIT:-0}" ]; then
      echo "the box is ${b}% busy and a GPU holds ${g} MiB before a sample: refusing" >&2
      return 1
    fi
    sleep 30
    waited=$((waited + 30))
  done
}
gpuclk() { nvidia-smi -i "$GPU" --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

jitllm_run() {
  env CUDA_VISIBLE_DEVICES="$GPU" $JITLLM_ENV "${PIN[@]}" "$JITLLM" run -devices cuda:0 -n "$N" "$JITLLM_PATH" "$PROMPT" \
    2>>"$LOG" >/dev/null || return 1
  tail -n 40 "$LOG" | sed -n 's|.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok/s).*|\1|p' | tail -1
}
strata_run() {
  local out
  out="$("${PIN[@]}" "$STRATA" --pack "$STRATA_PACK" --native "$STRATA_SHARD1" --expert-profile "$STRATA_PROFILE" \
    $STRATA_ARGS --tokens "$IDS" --max-new "$N" --gpu "$GPU" --spec 2 --suffix-draft 0 2>&1)" || { echo "$out" >>"$LOG"; return 1; }
  echo "$out" >>"$LOG"
  local acc
  acc="$(echo "$out" | sed -n 's|.*drafts accepted \([0-9]*\) of.*|\1|p' | tail -1)"
  if [ "${acc:-1}" != "0" ]; then
    echo "Strata accepted ${acc:-?} drafts: this sample speculated" >&2
    return 1
  fi
  echo "$out" | sed -n 's|^decode .* -> *\([0-9.]*\) tok/s.*|\1|p' | tail -1
}
A=jitllm_run
B=strata_run
NAME_B=strata
if [ -n "${STRATA_VS_AA:-}" ]; then
  B=jitllm_run
  NAME_B=jitllm
fi

echo "jitllm $(basename "$JITLLM_PATH") ($JITLLM_ENV) vs $NAME_B on GPU $GPU, node $NODE: decode $N tokens after the prompt [$IDS], $ROUNDS ABBA rounds; log $LOG"
nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader
# The soak: one sample of each, unused.
quiet || exit 75
echo ">> soak"
$A >/dev/null || true
$B >/dev/null || true
ratios=()
for r in $(seq 1 "$ROUNDS"); do
  quiet || exit 75
  p="$(psi)"
  if awk -v p="$p" 'BEGIN{exit !(p > 25)}'; then
    echo "round $r: PSI some avg10 $p > 25: the box is busy, refusing" >&2
    exit 75
  fi
  if [ $((r % 2)) -eq 1 ]; then
    a="$($A)"; ca="$(gpuclk)"; b="$($B)"; cb="$(gpuclk)"
  else
    b="$($B)"; cb="$(gpuclk)"; a="$($A)"; ca="$(gpuclk)"
  fi
  [ -n "$a" ] && [ -n "$b" ] || { echo "round $r: a sample gave no rate (see $LOG)" >&2; exit 1; }
  ratio="$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.4f", a/b}')"
  ratios+=("$ratio")
  echo "round $r: jitllm $a tok/s (sm,temp $ca)  $NAME_B $b tok/s (sm,temp $cb)  ratio $ratio  psi $p"
done
printf '%s\n' "${ratios[@]}" | sort -n | awk '
  { v[NR]=$1 }
  END {
    n=NR
    m = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
    lo = v[int((n+3)/4)]; hi = v[int((3*n+1)/4)]
    iqr = hi-lo
    printf "median ratio %.4f  IQR %.4f  IQR/median %.4f  over %d rounds\n", m, iqr, iqr/m, n
    if (iqr/m > 0.10) { print "REJECT: IQR/median above 0.10"; exit 3 }
  }'
