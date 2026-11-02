#!/usr/bin/env bash
# The 70B claim, made re-runnable: a 4 GiB card runs EVERY block of a 39.59 GiB
# model instead of two of them, and answers the same tokens.
#
# ★ WHY A SECOND SCRIPT AND NOT A FLAG ON scripts/paging.sh. That one proves the
# MECHANISM on a model small enough to iterate on -- it can run its seven arms in
# a minute and it is the gate to run after touching page.go. This one is the
# EVIDENCE for the configuration the mechanism was built for, and it is a
# different kind of job: one arm loads 42.5 GB through a 4 GiB card, no arm
# finishes in under a minute, and the model does not fit in host RAM either, so
# every number here is also a statement about NVMe. Folding it into paging.sh
# would make the fast gate slow and the slow evidence look like a unit test.
#
# ★ THE BEFORE IS RE-MEASURED, NOT QUOTED (RULE 1). "2/80 blocks, 1.22 GiB" is on
# record and a recorded baseline is exactly what RULE 1 refuses:
# the before arm here runs in the same pass, on the same binary, at the same
# pinned budget as the after arm, so the two numbers are comparable by
# construction. They differ from the record because the record took whatever the
# card had free that day and this pins -devices cuda:0=$BUDGET.
#
# ★ WHAT IS ASSERTED, AND THE --violate NAME THAT BREAKS EXACTLY ONE. A gate that
# has never been seen to fail is not a gate (RULE 10b).
#
#   every-block  paging ON places ALL NLayer blocks on the device where the SAME
#                budget with paging off places a handful.
#                violation: --violate no-paging
#   paged        that run's page-in counter is NON-ZERO. Without it "every block
#                placed" is also true of a tier that accepted every block and
#                quietly kept six.
#                violation: --violate no-paging (same arm, it reads 0)
#   arithmetic   the page-ins are what the slot arithmetic predicts, and the
#                arithmetic is page.go's own: slots = BUDGET / the widest block.
#                Under MRU the resident set is a pinned prefix plus the cursor,
#                so exactly NLayer-C blocks miss on EVERY pass and
#                page-ins = (NLayer-C) x passes for one C. The check is that C
#                agrees with budget/bytes-per-page-in, and that the streaming is
#                uniform, and that 1 <= C < NLayer.
#
#                ★ IT IS CHECKED AGAINST THE BUDGET AND NOT AGAINST Stats.Slots,
#                AND THAT IS A CORRECTION. Slots is the tier's own estimate,
#                computed at the end of LOAD from limit-minus-permanent, and it
#                is neither an upper nor a lower bound on what the run achieves:
#                measured here, tinyllama at 400M reported 13 and kept 16, and
#                the 70B at 2G reported 3 and kept 2.4. Gating on it made this
#                assertion a test of an estimate. The budget is the number the
#                user set and the one page.go divides, so it is the one to
#                divide here.
#
#                ★ AND THE SLACK IS 1.5 BLOCKS BECAUSE THE PERMANENT PARTS COME
#                OUT OF THE SAME BUDGET -- the KV cache, the norms, the biases,
#                the block scratch and the output projection are all charged to
#                it and none of them is counted in bytes-per-page-in. It is a
#                loose band on a 70B, where the whole budget buys three blocks;
#                what it still separates is the failure that matters, an LRU
#                policy on a cyclic scan, which keeps NOTHING and reads C ~ 0.
#                violation: --violate no-paging or --violate declines
#   executed     ★ THE ONE THE OTHER THREE CANNOT COVER. A pager that swapped
#                every block in and then DECLINED every compute places NLayer of
#                NLayer, prints every page counter, runs on the host and answers
#                in English. So this reads gpu.Stats.Blocks -- incremented after
#                a submission RETURNED and its output was copied back -- and
#                requires NLayer per forward pass, not merely more than zero.
#                violation: --violate declines (shrink the budget below the slot
#                floor mid-generation: placement stays NLayer/NLayer, the ids
#                stay identical, and the executed count collapses)
#   ids          the paged run's ids equal `-devices cpu`'s. Necessary and NOT
#                sufficient, which is what `executed` is for.
#                violation: --violate ids (doctor the reference)
#   unpacked     ★ WHERE THE BITS WERE EXTRACTED, WHICH THE FOUR ABOVE CANNOT
#                SEE. Every one of them is equally true of the engine at
#                60e530b, which paged the same eighty blocks and spent 69% of
#                the token repacking them on six host cores at 0.90 GB/s.
#                Stats.Unpacks counts tensors whose RAW GGUF bytes were uploaded
#                and turned into the device layout by a kernel, so it is zero
#                there and non-zero only when a card did the extraction.
#                violation: --violate no-unpack (JITLLM_GPU_UNPACK=0, which IS
#                the 60e530b engine: same placement, same ids, same page-ins)
#   accounting   the four terms of a page-in -- host pack, upload, device
#                unpack, block compute -- fit inside the seconds that elapsed,
#                and the host pack is no longer the largest of them. Both halves
#                are SUBTRACTED from the run's own counters rather than
#                estimated: cmd/jitllm prints pack/upload/unpack in the load
#                banner and again at the end, so (end - load) is what the tokens
#                paid. The 60e530b decomposition multiplied a load-time RATE by
#                a per-token byte count, and the error in that estimate landed in
#                the unaccounted remainder where nothing could argue with it.
#                violation: --violate accounting (doctor the wall, as --violate
#                ids doctors the reference: the terms then exceed it)
#
# ★ AND `accounting` NEEDS A MODEL WHOSE PAGE-IN COSTS SECONDS, WHICH IS A
# SMALLER CLAIM THAN THE ONE THAT STOOD HERE. That text said a q3_K_M model has
# no covered tensor; it is wrong, measured -- tinyllama at 200M pages 374 Q4_K
# tensors and `unpacked` is green on it, because q3_K_M is a MIXTURE and its
# attention and gate matrices are Q4_K. What tinyllama cannot do is decide the
# decomposition: its four terms are tenths of a second over a pass and each is
# the difference of two numbers the load banner rounded to the millisecond, so a
# verdict there is arithmetic on noise. The script prints NOT DECIDED rather
# than a green line (RULE 10a), which also means --violate accounting has to run
# on the 70B. Llama-3.3-70B-Instruct-Q4_K_M is 84.2% Q4_K by element, its blocks
# are 0.517 GiB, and it is the model both of these are about.
#   budget       ★ THE BUDGET IS THE CONTROL, NOT THE MODEL. A second, SMALLER
#                budget must give a smaller slot count, and the run must be worse
#                in the way the arithmetic predicts -- which is one of TWO
#                outcomes and the script says which it got:
#                  above the slot floor: every block is still placed and there
#                    are strictly MORE page-ins (tinyllama, 200M -> 120M);
#                  below it: there is nowhere to swap through, so the tier
#                    declines exactly as it did before paging existed and places
#                    FEWER blocks (Llama-3.3-70B, 2G -> 1G, 80/80 -> 1/80).
#                The second is not a weaker result than the first: minSlots is 2
#                and a 4 GiB card holds three blocks of a 70B, so the floor is a
#                real edge of this card and the budget is what decides which side
#                of it the run lands on.
#                violation: --violate budget (make the second budget equal to the
#                first; the slot count does not move)
#
# ★ AND THE FITS ARM, WHICH IS RULE 6's OBLIGATION AND IS MEASURED, NOT ASSERTED.
# A model that fits must pay NOTHING for the machinery being present: zero
# page-ins, zero page-outs, and a rate indistinguishable from the same budget
# with paging off. That one is cheap enough to do properly -- ABBA interleaved,
# A/A self-control first, median of per-round ratios with its IQR -- so it is,
# on tinyllama, in --rate mode.
#
# ★ WHAT IS NOT ASSERTED: THE 70B's RATE. It is REPORTED, by --rate, and the
# report is a measurement rather than a gate because the answer is not known in
# advance and a gate on it would be a gate on the box's NVMe. See the doc.
#
# ★ RUN THE VIOLATIONS ON A SMALL MODEL WHERE THE ASSERTION ALLOWS IT, AND THE
# TWO NEW ONES DO NOT. ids, no-paging, budget and declines read counters rather
# than weights, so tinyllama proves they can fail in a minute:
#
#   scripts/paging-70b.sh -m .../tinyllama-1.1b-q3_K_M.gguf -b 200M -B 120M \
#                         -n 6 --violate ids
#
# ★ no-unpack AND accounting ARE PROVED ON THE 70B ITSELF, and the reason is
# worth stating because the obvious shortcut is wrong twice over. A q3_K_M model
# has no covered tensor, so `unpacked` fails there whatever the engine does --
# and a violation run that "fails" for a reason the violation did not cause
# proves nothing (RULE 10b). The Q4_K models small enough to be quick then fail
# `arithmetic` for an unrelated reason: its slack is 1.5 blocks, sized on a 70B
# whose output projection is about one block, and a 1B's head is FOUR of its
# blocks. Llama-3.2-1B at 200M keeps 3.4 blocks where the budget buys 5.1, which
# is the head, correctly, and not a pager fault.
#
# usage:  scripts/paging-70b.sh [-m MODEL] [-b BUDGET] [-B SMALLBUDGET] [-n TOKENS]
#                               [--rate] [--rate-only]
#                               [--violate ids|no-paging|budget|declines|no-unpack|accounting]
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
MODEL=${JITLLM_70B_MODEL:-${JITLLM_MODELS:?set JITLLM_MODELS to the model directory}/Llama-3.3-70B-Instruct-Q4_K_M.gguf}
# The model that FITS, for the RULE 6 arm. Small on purpose: the claim there is
# that nothing moves and the rate does not change, and that is the same claim on
# any model whose weights fit the budget.
FITS=${JITLLM_FITS_MODEL:-${JITLLM_MODELS:?set JITLLM_MODELS to the model directory}/tinyllama-1.1b-q3_K_M.gguf}
FITS_BUDGET=${JITLLM_FITS_BUDGET:-700M}
NTOK=${JITLLM_70B_N:-4}
PROMPT="The capital of France is"
DEV=${JITLLM_70B_DEV:-cuda:0}
# ★ THE BUDGET IS PINNED AND THE SECOND ONE IS EXACTLY HALF. Free VRAM moves with
# whatever the desktop is drawing, so an unpinned budget makes the block count
# unreproducible -- and 2G is the budget at which this card reproduces the
# recorded "2/80 blocks, 1.22 GiB" exactly, which is what makes the before arm
# comparable with the record as well as with the after arm. Halving rather than
# doubling because 2G is already most of what the card has free.
BUDGET=${JITLLM_70B_BUDGET:-2G}
BIG=${JITLLM_70B_BIG:-1G}
CAPMEM=${JITLLM_70B_CAP:-24G}
RATE=0; RATEONLY=0; VIOLATE=""
ROUNDS=${JITLLM_70B_ROUNDS:-3}
FITROUNDS=${JITLLM_FIT_ROUNDS:-6}
FITN=${JITLLM_FIT_N:-64}
while [ $# -gt 0 ]; do
  case "$1" in
    -m) MODEL=$2; shift 2 ;;
    -b) BUDGET=$2; shift 2 ;;
    -B) BIG=$2; shift 2 ;;
    -n) NTOK=$2; shift 2 ;;
    -d) DEV=$2; shift 2 ;;
    --rate) RATE=1; shift ;;
    --rate-only) RATE=1; RATEONLY=1; shift ;;
    --violate) VIOLATE=$2; shift 2 ;;
    -h|--help) sed -n '1,/^# usage:/p' "$0"; exit 0 ;;
    *) echo "paging-70b: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -r "$MODEL" ] || { echo "paging-70b: no model at $MODEL (-m MODEL, or JITLLM_70B_MODEL)" >&2; exit 2; }

# ★ RULE 3: every run below is capped, and 24G because a model over ~16 GB has
# its mmap'd page cache charged to the cgroup. RULE 2b's PSI gate is here too:
# this box reads loadavg 4.4 while genuinely idle, so the load average is not the
# thing to look at.
CAP="$ROOT/scripts/cap $CAPMEM -- taskset -c 0,2,4,6,8,10"
if [ -r /proc/pressure/cpu ]; then
  P=$(awk '/^some/{for(i=1;i<=NF;i++) if($i ~ /^avg10=/){sub("avg10=","",$i); print $i}}' /proc/pressure/cpu)
  if [ "$(echo "$P > 25" | bc)" = "1" ]; then
    echo "refusing: CPU pressure is ${P}% over the last 10s, want < 25 (RULE 2b)" >&2
    exit 1
  fi
fi

OUT=${JITLLM_70B_OUT:-$(mktemp -d)}
mkdir -p "$OUT"
BIN=$ROOT/jitllm
# ★ THE NAME OF EVERY ASSERTION THAT WENT RED, NOT JUST HOW MANY. A violation
# run used to pass on fails > 0, which is RULE 10b's own trap wearing the
# harness's clothes: --violate ids on a q3_K_M model "succeeds" because
# `unpacked` failed for an unrelated reason and nothing checked WHICH assertion
# the violation was supposed to break. The name is the text before the first
# colon, which is why every ok/bad string starts with one.
fails=0; redlist=""
note() { printf '%s\n' "$*"; }
ok()   { printf '  PASS  %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; fails=$((fails+1)); redlist="$redlist ${1%%:*}"; }

$ROOT/scripts/cap 8G -- go build -o "$BIN" ./cmd/jitllm || exit 1

# run TAG MODEL SPEC NTOK env... -- stderr to $OUT/TAG
# STREAM=1 among the environment words is not a variable: it streams every
# block through spec's first device (-placement '*=DEV~'), since streaming is a
# placement and no switch turns it on.
run() {
  local tag=$1 mdl=$2 spec=$3 nt=$4; shift 4
  local dv=(-devices "$spec") env=() place=()
  for w in "$@"; do
    if [ "$w" = STREAM=1 ]; then local first=${spec%%,*}; place=(-placement "*=${first%%=*}~"); else env+=("$w"); fi
  done
  $CAP env "${env[@]}" "$BIN" run "${dv[@]}" "${place[@]}" -n "$nt" -temp 0 "$mdl" "$PROMPT" \
      >"$OUT/$tag.stdout" 2>"$OUT/$tag"
}
ids()   { sed -n 's/^ids \(\[.*\]\)$/\1/p' "$OUT/$1" | tail -1; }
seam()  { sed -n 's/^device .*: \([0-9][0-9]*\)\/\([0-9][0-9]*\) blocks.*/\1 \2/p' "$OUT/$1" | tail -1; }
# ★ FROM gpu.Stats, NOT FROM A PLACEMENT LINE. "N block(s) placed, R run" is
# cmd/jitllm printing Stats.Blocks, which is incremented after the submission
# returned and its output was read back -- the only counter that separates a
# block held on a card from a block computed on it.
execd() { sed -n 's/^device .*[0-9] block(s) placed, \([0-9][0-9]*\) run, .*/\1/p' "$OUT/$1" | tail -1; }
pagefield() { sed -n "s/^paging \([0-9]*\) block slot(s), \([0-9]*\) page-in(s), \([0-9]*\) page-out(s), \([0-9.]*\) GiB moved.*/\\$2/p" "$OUT/$1" | tail -1; }
slots()   { pagefield "$1" 1; }
pagein()  { pagefield "$1" 2; }
pageout() { pagefield "$1" 3; }
moved()   { pagefield "$1" 4; }
repacks() { sed -n 's/^paging .*and \([0-9][0-9]*\) repacked.*/\1/p' "$OUT/$1" | tail -1; }
# ★ THE UNPACK COUNTERS, WHICH ARE THE ONES THIS PATH IS FALSIFIED BY. A tier
# that quietly kept packing on the host places the same blocks, computes the same
# ids and leaves every other number here looking healthy -- Stats.Unpacks is zero
# there and non-zero only when a card really did the bit extraction.
unpacks()  { sed -n 's/^paging .*; \([0-9][0-9]*\) tensor(s) uploaded RAW and unpacked on the device.*/\1/p' "$OUT/$1" | tail -1; }
unpackgib(){ sed -n 's/^paging .*uploaded RAW and unpacked on the device, \([0-9.]*\) GiB in .*/\1/p' "$OUT/$1" | tail -1; }
staging()  { sed -n 's/^paging .*, staging \([0-9.]*\) MiB.*/\1/p' "$OUT/$1" | tail -1; }
# prep is the LOAD-time pack/upload/unpack, printed by the banner before a token
# runs; cost is the same three summed over the whole run, plus the device's own
# block compute. The DIFFERENCE is what the tokens paid, and it is the only way
# to get a per-token pack out of counters that also cover load.
prep()  { sed -n 's/.*prep \([0-9.]*\) ms pack + \([0-9.]*\) ms upload + \([0-9.]*\) ms device unpack.*/\1 \2 \3/p' "$OUT/$1" | tail -1; }
cost()  { sed -n 's/^device cost: pack \([0-9.]*\) s, upload \([0-9.]*\) s, device unpack \([0-9.]*\) s, block compute \([0-9.]*\) s.*/\1 \2 \3 \4/p' "$OUT/$1" | tail -1; }
# secs TAG FIELD -> the prefill (1) or decode (2) wall in seconds.
#
# ★ FROM THE DURATION, NEVER FROM THE tok/s FIELD, for scripts/paging-rate.sh's
# reason: cmd/jitllm prints tok/s as %.2f and a paged 70B reads "0.02", which is
# one significant figure. The duration beside it is millisecond-rounded.
secs() {
  sed -n 's/^prompt [0-9]* tok in \([^ ]*\) .*decode [0-9]* tok in \([^ ]*\) .*/\1 \2/p' "$OUT/$1" | tail -1 \
  | awk -v f="$2" '{
      d = $f; s = 0
      # Sub-second units FIRST: "372ms" otherwise parses as 372 MINUTES.
      if      (d ~ /[0-9]ms$/) { sub(/ms$/, "", d); s = d / 1e3 }
      else if (d ~ /s$/ && d !~ /[0-9](h|m)/) { sub(/s$/, "", d); s = d + 0 }
      else {
        if (match(d, /[0-9.]+h/)) { s += substr(d, RSTART, RLENGTH-1) * 3600; d = substr(d, RSTART+RLENGTH) }
        if (match(d, /[0-9.]+m/)) { s += substr(d, RSTART, RLENGTH-1) * 60;   d = substr(d, RSTART+RLENGTH) }
        if (match(d, /[0-9.]+s/)) { s += substr(d, RSTART, RLENGTH-1) }
      }
      printf "%.3f", s }'
}
rate()    { sed -n 's/.*decode [0-9]* tok in [^(]*(\([0-9.]*\) tok\/s).*/\1/p' "$OUT/$1" | tail -1; }
resident(){ sed -n 's/^device .*blocks[^,]*, \([0-9.]*\) GiB resident.*/\1/p' "$OUT/$1" | tail -1; }

PAGE_ENV=(STREAM=1)
SHRINK_ENV=()
UNPACK_ENV=()
case "$VIOLATE" in
  no-paging)  PAGE_ENV=() ;;
  budget)     BIG=$BUDGET ;;
  declines)   SHRINK_ENV=(JITLLM_PAGE_SHRINK=1) ;;
  # ★ no-unpack IS THE 60e530b ENGINE, NOT A BROKEN ONE, and that is what makes
  # it the right violation for the two unpack assertions: every tensor goes back
  # through the host packer, the placement is unchanged, the ids are unchanged,
  # and exactly the two claims about WHERE the extraction happened go red.
  no-unpack)  UNPACK_ENV=(JITLLM_GPU_UNPACK=0) ;;
  accounting) ;;   # handled where the breakdown is computed
  ids|"")     ;;
  *) echo "paging-70b: unknown violation $VIOLATE (ids|no-paging|budget|declines|no-unpack|accounting)" >&2; exit 2 ;;
esac
[ -n "$VIOLATE" ] && note "★ VIOLATION MODE \"$VIOLATE\": one assertion below MUST fail, or the gate is not a gate."

note "host      $(uname -sm)   cap $CAPMEM"
note "device    $DEV   budgets $BUDGET, then $BIG as the control"
note "model     $MODEL"
note "          $(stat -Lc %s "$MODEL" | awk '{printf "%.2f GiB on disk", $1/1073741824}')"
note "prompt    \"$PROMPT\"   -n $NTOK   greedy (-temp 0)"
note ""

if [ "$RATEONLY" = 0 ]; then
# ------------------------------------------------------------------- reference
note "== reference: the host alone =="
run cpu "$MODEL" cpu "$NTOK" || { tail -3 "$OUT/cpu"; exit 1; }
REF=$(ids cpu)
[ -n "$REF" ] || { echo "paging-70b: the cpu arm printed no ids"; tail -20 "$OUT/cpu"; exit 1; }
note "  -devices cpu   $REF   ($(rate cpu) tok/s)"
if [ "$VIOLATE" = "ids" ]; then
  REF=$(printf '%s' "$REF" | sed 's/\[\([0-9]*\) /[999 /')
  note "  ★ doctored reference: $REF"
fi
note ""

# ------------------------------------------------------------- before / after
note "== the same card, the same budget, paging off then on =="
run before "$MODEL" "$DEV=$BUDGET" "$NTOK" || { tail -3 "$OUT/before"; bad "the paging-off arm did not finish"; }
run after  "$MODEL" "$DEV=$BUDGET" "$NTOK" "${PAGE_ENV[@]}" "${SHRINK_ENV[@]}" "${UNPACK_ENV[@]}" || { tail -3 "$OUT/after"; bad "the paging-on arm did not finish"; }
read -r bpl btot <<<"$(seam before)"
read -r apl atot <<<"$(seam after)"
note "  paging off   ${bpl:-?}/${btot:-?} blocks, $(resident before) GiB resident, $(execd before) block-call(s) run, $(rate before) tok/s"
note "  paging on    ${apl:-?}/${atot:-?} blocks, $(resident after) GiB resident, $(execd after) block-call(s) run, $(rate after) tok/s"
note "  $(grep -m1 '^paging ' "$OUT/after")"
note "  ids off $(ids before)"
note "  ids on  $(ids after)"
note ""

PASSES=$((NTOK + 1))   # one prefill pass over every block, then one per token
[ "${apl:-0}" = "${atot:-0}" ] && [ "${atot:-0}" -gt 0 ] 2>/dev/null \
  && ok "every-block: the card holds ALL $atot blocks (the same budget with paging off holds ${bpl:-0})" \
  || bad "every-block: the card holds ${apl:-0} of ${atot:-?} (paging off holds ${bpl:-0})"
[ "${bpl:-0}" -lt "${atot:-1}" ] 2>/dev/null \
  && ok "control: without paging the same budget holds only ${bpl:-0}/${atot:-?} -- the budget is what changed, not the model" \
  || bad "control: paging off already holds ${bpl:-0}/${atot:-?}; this budget does not exercise the pager"
[ "$(pagein after)" -gt 0 ] 2>/dev/null \
  && ok "paged: $(pagein after) page-in(s), $(pageout after) page-out(s), $(moved after) GiB moved -- blocks really left the card and came back" \
  || bad "paged: $(pagein after) page-ins. NOTHING PAGED, so every count above proves nothing"

# ★ WHERE THE BITS WERE EXTRACTED, WHICH IS A DIFFERENT QUESTION FROM WHETHER
# THE BLOCKS MOVED. Every assertion above is equally true of the 60e530b engine,
# which paged the same eighty blocks and spent 69% of the token repacking them on
# six host cores at 0.90 GB/s. Stats.Unpacks is the counter that separates the
# two: it counts tensors whose RAW GGUF bytes were uploaded and turned into the
# device layout by a kernel, so it is zero on that engine and non-zero only when
# a card really did the work.
#
# ★ AND A MODEL WITH NO COVERED TENSOR FAILS THIS RATHER THAN SKIPPING IT. Q4_K
# is the one format with a device unpack, so a q3_K_M model reports zero here and
# the host packer running is CORRECT -- but a green line on a run that could not
# have shown the claim either way is RULE 10a's most expensive habit, so the
# script says which model it needs instead of passing.
if [ "$(unpacks after)" -gt 0 ] 2>/dev/null; then
  ok "unpacked: $(unpacks after) tensor(s) uploaded RAW and unpacked ON THE CARD, $(unpackgib after) GiB, staging $(staging after) MiB of VRAM -- the host packer was never asked for them"
elif [ "$(repacks after)" -gt 0 ] 2>/dev/null && [ -z "${UNPACK_ENV[*]}" ]; then
  bad "unpacked: 0 of $(repacks after) packed tensor(s) were unpacked on the device. Either this model has no Q4_K tensor -- in which case it cannot exercise the claim and the gate needs a Q4_K_M model, not a verdict -- or the page-ins went through the host packer, which is the engine this work replaced. The run said: $(sed -n 's/^paging .*had no room: //p' "$OUT/after" | tail -1 | sed 's/)$//')"
else
  bad "unpacked: 0 tensors were unpacked on the device. Every page-in above went through the host packer, which is the engine this work replaced"
fi

# ★ THE SLOT ARITHMETIC, CHECKED AGAINST THE BUDGET RATHER THAN AGAINST THE
# TIER'S OWN ESTIMATE. page.go divides the budget by the WIDEST block; this
# divides the budget by the bytes a page-in actually moved, which is the same
# quantity arrived at from the other end, and compares it with the residency the
# page-in count implies. See the header for why Stats.Slots is printed here and
# not asserted on.
python3 - "$OUT/after" "${atot:-0}" "$PASSES" "$(slots after)" "$(pagein after)" "$(moved after)" "$BUDGET" <<'PY' > "$OUT/arith"
import sys
_, _f, tot, passes, slots, pin, moved, budget = sys.argv
tot, passes, slots, pin, moved = int(tot), int(passes), int(slots or 0), int(pin or 0), float(moved or 0)
mult = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
b = budget.strip()
bytes_ = float(b[:-1]) * mult[b[-1].upper()] if b and b[-1].upper() in mult else float(b)
bud = bytes_ / (1 << 30)                      # GiB, the unit the report prints
if tot == 0 or passes == 0 or pin == 0:
    print("FAIL|no blocks, passes or page-ins to check"); raise SystemExit
per = pin / passes
C = tot - per                                 # blocks the card kept resident
gib = moved / pin                             # one block's packed weights
want = bud / gib                              # what the budget buys, before the permanent parts
ok, why = True, []
if abs(per - round(per)) > 1.0:
    ok = False; why.append("the streaming is not uniform: %.2f page-ins per pass" % per)
if abs(C - want) > 1.5:
    ok = False; why.append("the budget of %.2f GiB buys %.1f blocks of %.3f GiB and the run kept %.1f"
                           % (bud, want, gib, C))
if C >= tot:
    ok = False; why.append("the implied residency %.1f is the whole model: nothing streamed" % C)
if C < 1:
    ok = False; why.append("the implied residency %.1f is under one block" % C)
print("%s|%d page-in(s) = %.1f per pass over %d pass(es), %.3f GiB each: the card kept %.1f of %d "
      "blocks and streamed the other %.1f every pass. The %.2f GiB budget buys %.1f blocks before "
      "the KV, the norms and the head come out of it; the tier called it %d slot(s)"
      % ("PASS" if ok else "FAIL", pin, per, passes, gib, C, tot, per, bud, want, slots)
      + ("" if ok else "  [%s]" % "; ".join(why)))
PY
read -r verdict msg <<<"$(sed 's/|/ /' "$OUT/arith")"
[ "$verdict" = PASS ] && ok "arithmetic: $msg" || bad "arithmetic: $msg"

# ★ EXECUTION, FROM gpu.Stats.Blocks. NLayer per forward pass is the claim: the
# card ran every block of every token, not "some blocks sometimes".
WANT=$(( ${atot:-0} * PASSES ))
AE=$(execd after)
[ "${AE:-0}" = "$WANT" ] \
  && ok "executed: the card RAN $AE block-calls = $atot blocks x $PASSES pass(es); a pager that declined every compute reads far less" \
  || bad "executed: the card ran ${AE:-0} block-calls, want $WANT ($atot blocks x $PASSES pass(es))"
BE=$(execd before)
[ "${AE:-0}" -gt "${BE:-0}" ] 2>/dev/null \
  && ok "executed: $AE against ${BE:-0} without paging -- $(python3 -c "print('%.1fx'%(${AE:-0}/max(${BE:-1},1)))") the work on the same card" \
  || bad "executed: paging did not increase the work the card did (${AE:-0} against ${BE:-0})"

[ "$(ids after)" = "$REF" ] && ok "ids: the paged run is identical to the host-only run" \
                            || bad "ids: the paged run differs from the host-only run"
[ "$(ids before)" = "$REF" ] && ok "ids: the paging-OFF run is identical too, so the seam is not the variable" \
                             || bad "ids: the paging-off run already differs from the host-only run"
note ""

# ------------------------------------------------------- what a page-in COSTS
# ★ THE DECOMPOSITION IS SUBTRACTED, NOT ESTIMATED, AND THAT IS A CORRECTION.
# The 60e530b breakdown took a RATE from the load-time counters (49.4 s to pack
# 41.4 GiB = 0.90 GB/s) and multiplied it by a per-token byte count. That is an
# estimate of a term, and its error lands in the unaccounted remainder where it
# cannot be argued with. cmd/jitllm prints the same four counters twice now --
# once in the load banner, once at the end -- so every term here is
# (end - load) / passes, with nothing modelled.
#
# PASSES, not tokens: the prefill streams all eighty blocks exactly as a decode
# step does, so the unit of paging cost is a PASS over the model. The per-token
# decode figure is printed beside it because that is the one the ratio uses.
note "== what one pass over 80 blocks costs, from the run's own counters =="
python3 - "$(prep after)" "$(cost after)" "$(secs after 1)" "$(secs after 2)" "$NTOK" \
         "$(pagein after)" "$(moved after)" "$(unpacks after)" "$(unpackgib after)" \
         "$(repacks after)" "$(secs before 2)" "$(execd after)" "$VIOLATE" <<'PY' > "$OUT/breakdown"
import sys
prep, cost, pre, dec, ntok, pin, moved, unp, ugib, rep, hostdec, ran, violate = sys.argv[1:14]
pk0, up0, un0 = [float(x)/1000 for x in prep.split()] if prep.strip() else (0., 0., 0.)
pk1, up1, un1, cmp1 = [float(x) for x in cost.split()] if cost.strip() else (0., 0., 0., 0.)
pre, dec, hostdec = float(pre or 0), float(dec or 0), float(hostdec or 0)
ntok, pin, ran = int(ntok), int(pin or 0), int(ran or 0)
moved, ugib = float(moved or 0), float(ugib or 0)
unp, rep = int(unp or 0), int(rep or 0)
passes = ntok + 1
obs = pre + dec
# ★ --violate accounting DOCTORS THE WALL, the way --violate ids doctors the
# reference: the gate below says the four terms cannot add up to more than the
# seconds that elapsed, and the only way to see it fail on a healthy engine is to
# hand it a wall the terms do not fit in.
if violate == "accounting":
    obs = obs / 10.0
    print("  ★ doctored wall: %.1f s instead of %.1f" % (obs, pre + dec))
# The load banner prints milliseconds with no decimal, so a term whose whole cost
# was the load subtracts to a small negative. Clamp it: the resolution is the
# banner's, and a -0.00 in a table reads as a finding rather than as rounding.
pk, up, un, cm = max(0.0, pk1 - pk0), max(0.0, up1 - up0), max(0.0, un1 - un0), cmp1
acc = pk + up + un + cm
per = lambda v: v / passes
rate = lambda gib, sec: (gib * 1.073741824 / sec) if sec > 0 else 0.0   # GiB -> GB/s
print("  per pass over %d block(s), %d pass(es) (1 prefill + %d decode):" % (ran // passes if passes else 0, passes, ntok))
print("    page-ins        %6.1f   %6.2f GiB of block weights landed on the card" % (pin / passes, per(moved)))
print("    host pack       %6.2f s  %d tensor(s) the host still packs: the formats with no unpack kernel" % (per(pk), rep))
print("    upload          %6.2f s  %.2f GB/s against the bytes that LANDED (a lower bound: the raw"
      % (per(up), rate(per(moved), per(up))))
print("                             bytes a covered tensor uploads are not the bytes it occupies)")
# ★ THIS TERM IS AN ENQUEUE, NOT A DEVICE TIME, AND SAYING SO IS THE DIFFERENCE
# BETWEEN A BREAKDOWN AND A STORY. cuLaunchKernel on the legacy stream returns
# before the kernel runs, so TUnpack is what the HOST spent handing the launch
# over; the extraction itself runs while the next memcpy on the same stream
# waits, which puts it inside the upload term. The bound is the per-tensor A/B
# in kernels/arena.go -- 47.4 GB/s read+written, i.e. under a second for the
# 40 GiB a pass moves -- so it cannot be the row either way, and the upload term
# below is an upper bound that carries it.
print("    device unpack   %6.2f s  %d launch(es) ENQUEUED, %.2f GiB raw over the whole run; the"
      % (per(un), unp, ugib))
print("                             launch is async, so the kernel's own time is inside the upload")
print("    block compute   %6.2f s  the card running the blocks and reading their output back" % per(cm))
print("    ---------------------------")
print("    accounted       %6.2f s" % per(acc))
print("    observed        %6.2f s  (%.1f s prefill + %.1f s decode over %d passes)" % (per(obs), pre, dec, passes))
print("    unaccounted     %6.2f s  %.0f%% -- the GGUF faulting in from NVMe under the terms above"
      % (per(obs - acc), 100 * (obs - acc) / obs if obs else 0))
if hostdec > 0:
    print("    the host arm it replaces: %.2f s/token against this arm's %.2f" % (hostdec / ntok, dec / ntok))
# ★ A DECOMPOSITION OF TENTHS OF A SECOND DECIDES NOTHING, AND A GREEN LINE FOR
# IT IS RULE 10a's most expensive habit. Both clauses below are about a page-in
# that costs SECONDS: "the terms fit inside the wall" is trivially true when the
# terms are rounding, and "the host pack is not the largest" compares two numbers
# that are each one millisecond of load. Say so, and let the caller take it to a
# model that pages real bytes.
if acc / passes < 1.0:
    print("VERDICT|SKIP|the four terms come to %.2f s a pass on this model, which is the "
          "resolution of the load banner they are subtracted from -- this gate decides nothing "
          "here. Run it on a model whose page-in costs seconds (Llama-3.3-70B at 2G: 40 s a pass)"
          % (acc / passes))
    raise SystemExit
why = []
if acc > obs:
    why.append("the four terms add to %.1f s and only %.1f s elapsed: above the roofline is a harness bug" % (acc, obs))
if pk >= up + un:
    why.append("the host pack is %.1f s against %.1f s of upload+unpack -- it is still the row" % (pk, up + un))
print("VERDICT|%s|%s" % ("PASS" if not why else "FAIL", "; ".join(why) if why else
      "the four terms fit inside the wall, and the host pack (%.1f s) is no longer the largest of them (upload %.1f s + unpack %.1f s)" % (pk, up, un)))
PY
sed '/^VERDICT|/d' "$OUT/breakdown"
BD=$(sed -n 's/^VERDICT|//p' "$OUT/breakdown")
case "${BD%%|*}" in
  PASS) ok   "accounting: ${BD#*|}" ;;
  SKIP) note "  ----  accounting: NOT DECIDED -- ${BD#*|}" ;;
  *)    bad  "accounting: ${BD#*|}" ;;
esac
note ""

# ---------------------------------------------------- the budget is the control
note "== the same model on a budget of $BIG, which is the CONTROL =="
run big "$MODEL" "$DEV=$BIG" "$NTOK" "${PAGE_ENV[@]}" "${UNPACK_ENV[@]}" || { tail -3 "$OUT/big"; bad "the larger-budget arm did not finish"; }
read -r gpl gtot <<<"$(seam big)"
note "  ${gpl:-?}/${gtot:-?} blocks, $(resident big) GiB resident, $(execd big) block-call(s) run, $(rate big) tok/s"
note "  $(grep -m1 '^paging ' "$OUT/big")"
note "  ids     $(ids big)"
if [ "$(slots big)" -ge "$(slots after)" ] 2>/dev/null; then
  bad "budget: $BUDGET gave $(slots after) slot(s) and $BIG gave $(slots big); the slot count did not move with the budget"
elif [ "${gpl:-0}" -lt "${apl:-0}" ] 2>/dev/null; then
  ok "budget: $BUDGET -> $BIG took the slot count $(slots after) -> $(slots big), through the floor of 2 a swap needs -- so the tier DECLINED and the placement fell ${apl}/${atot} -> ${gpl}/${gtot}. The budget decides whether the card runs the model at all"
elif [ "$(pagein big)" -gt "$(pagein after)" ] 2>/dev/null; then
  ok "budget: $BUDGET -> $BIG took the slot count $(slots after) -> $(slots big) and the page-ins $(pagein after) -> $(pagein big); the model still runs whole and the BUDGET sets the slot count"
else
  bad "budget: the slot count fell $(slots after) -> $(slots big) but neither the placement (${apl}/${atot} -> ${gpl}/${gtot}) nor the page-ins ($(pagein after) -> $(pagein big)) moved with it"
fi
[ "$(ids big)" = "$REF" ] && ok "budget: ids unchanged across the budget change" || bad "budget: the ids moved with the budget"
note ""
fi   # RATEONLY

# ------------------------------------------------------------------ the rates
if [ "$RATE" = 1 ]; then
  note "== RULE 6: a model that FITS pays nothing =="
  note "   $(basename "$FITS") at $DEV=$FITS_BUDGET, paging ON against paging OFF,"
  note "   ABBA interleaved in one pass, A/A self-control FIRST, $FITROUNDS round(s) of 4, n=$FITN"
  "$ROOT/scripts/paging-rate.sh" "$FITS" "$DEV=$FITS_BUDGET" "$FITN" "$FITROUNDS" "$CAPMEM" | tee "$OUT/fitrate"
  FR=$(sed -n 's/^  paging ratio \([0-9.]*\) .*/\1/p' "$OUT/fitrate" | tail -1)
  FZ=$(sed -n 's/^  paging page-ins: A \([0-9]*\)\.\.\([0-9]*\) .*/\1 \2/p' "$OUT/fitrate" | tail -1)
  if [ -z "$FR" ]; then bad "fits-rate: the paired run produced no ratio"
  elif grep -q 'REJECTED' "$OUT/fitrate"; then bad "fits-rate: the spread is too wide to decide (RULE 2's IQR gate)"
  elif [ "$FZ" != "0 0" ]; then bad "fits-rate: a model that FITS paged ($FZ page-in(s)); this arm is measuring the wrong thing"
  else
    python3 -c "import sys; r=float('$FR'); sys.exit(0 if 0.95 <= r <= 1.05 else 1)" \
      && ok "fits: ZERO page-ins with paging ON, and the rate is $FR of the same budget with paging off (RULE 6: a model that fits pays nothing)" \
      || bad "fits: zero page-ins, but the rate moved to $FR -- the machinery is not free on a model that fits"
  fi
  note ""
  note "== the 70B rate: paging ON against paging OFF, same binary, same budget =="
  note "   ★ REPORTED, NOT GATED, because the answer is a property of this box NVMe as much"
  note "   as of the engine. Re-taken with the device unpack, -n 8: 0.4251"
  note "   (IQR/median 0.061, n=4, self-control 1.0020) -- paging the 70B is 2.35x SLOWER"
  note "   than leaving 78 of its 80 blocks on the host. The same script at -n 4 in the same"
  note "   session read a median of 0.4614 and the dispersion gate REJECTED it at 0.106,"
  note "   all of the spread in the HOST arm; that is why this row is taken at 8."
  note ""
  note "   ★ AND THE COST IS NO LONGER THE REPACK. It was 0.2988 at 60e530b with a 0.9533"
  note "   self-control, and --violate no-unpack still reproduces that engine here: 58.74 s"
  note "   of host pack a pass against 11.68 now, because Q4_K -- 84.2% of this model by"
  note "   element -- is uploaded RAW and extracted by a kernel. What is left is the NVMe"
  note "   READ, which both arms pay and which the upload term now carries: 40.10 GiB a pass"
  note "   at 1.61 GB/s against the host arm own 39.04 GiB/token at ~2.16. The link was"
  note "   never the competitor -- the read is in SERIES with it. See the breakdown above"
  note "   and docs/design/block-paging.md."
  "$ROOT/scripts/paging-rate.sh" "$MODEL" "$DEV=$BUDGET" "$NTOK" "$ROUNDS" "$CAPMEM" | tee "$OUT/rate70"
  note ""
fi

if [ -n "$VIOLATE" ]; then
  # Which assertion each violation is AIMED at. A violation that leaves its
  # target green is not a gate, and a violation whose target is green while
  # something ELSE went red is worse: it looks like a pass.
  case "$VIOLATE" in
    ids)        want="ids" ;;
    no-paging)  want="every-block" ;;
    budget)     want="budget" ;;
    declines)   want="executed" ;;
    no-unpack)  want="unpacked" ;;
    accounting) want="accounting" ;;
  esac
  hit=0; other=""
  for n in $redlist; do
    if [ "$n" = "$want" ]; then hit=1; else other="$other $n"; fi
  done
  if [ "$hit" = 1 ]; then
    note "VIOLATION \"$VIOLATE\": \"$want\" went RED, which is the assertion it targets."
    [ -n "$other" ] && note "  and these failed too, which the violation did NOT target -- read them:$other"
    exit 0
  fi
  if [ "$want" = accounting ] && grep -q '^VERDICT|SKIP' "$OUT/breakdown" 2>/dev/null; then
    # Not green: NOT DECIDED. Saying "not a gate" here would be a false claim
    # about the gate rather than about the model it was pointed at.
    note "VIOLATION \"$VIOLATE\": the assertion it targets was NOT DECIDED on this model, so"
    note "  this run cannot prove it either way. $(sed -n 's/^VERDICT|SKIP|//p' "$OUT/breakdown")"
    exit 1
  fi
  note "VIOLATION \"$VIOLATE\": \"$want\" stayed GREEN. The assertion it targets is not a gate."
  [ -n "$other" ] && note "  (these failed instead, for reasons the violation did not cause:$other)"
  exit 1
fi
if [ "$fails" -eq 0 ]; then
  note "PROVED on this host, in one pass, one binary:"
  if [ "$RATEONLY" = 0 ]; then
    note "  $(basename "$MODEL") -- $atot blocks, $(stat -Lc %s "$MODEL" | awk '{printf "%.2f", $1/1073741824}') GiB on disk -- on $DEV at $BUDGET:"
    note "    paging off  ${bpl}/${atot} blocks, $(execd before) block-calls run"
    note "    paging on   ${apl}/${atot} blocks, $(execd after) block-calls run, $(pagein after) page-in(s), $(moved after) GiB moved"
    note "    $(unpacks after) tensor(s) uploaded RAW and unpacked ON THE CARD, $(unpackgib after) GiB, staging $(staging after) MiB"
    note "    ids identical to -devices cpu in every arm."
    sed '/^VERDICT|/d' "$OUT/breakdown" | sed 's/^/  /'
  fi
  exit 0
fi
note "$fails assertion(s) failed"
exit 1
