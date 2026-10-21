#!/usr/bin/env bash
# Run every llava gate against a deliberate VIOLATION.
#
# ★ RULE 10: a gate that has never fired is not a gate. Each case below breaks
# exactly one thing the llava work added, runs the gates that should see it, and
# restores the tree. Every one of these violations produces FINITE, VARIED,
# PLAUSIBLE embeddings -- that is the whole reason they need gates rather than
# an assertion about NaN.
#
# Usage: scripts/llava-violations.sh          (all cases)
#        scripts/llava-violations.sh cls      (one case)
set -uo pipefail
cd "$(dirname "$0")/.."

CAP="./scripts/cap 24G -- taskset -c 0,2,4,6,8,10"
only="${1:-}"

restore() { git checkout -- "$@"; }

run() { # run <name> <test-regex> <package>
  echo "=== VIOLATION: $1"
  $CAP go test "$3" -run "$2" -count=1 2>&1 | grep -E "FAIL|ok |---|_test.go:" | head -20
  echo
}

# 1. The activation: quick-GELU becomes GELU-tanh, which is what reading only
#    clip.use_gelu and defaulting false would do.
case_act() {
  sed -i 's/\t\tv.Flags |= jlm.FlagQuickGELU/\t\tv.Flags |= jlm.FlagGELU/' convert/config.go
  rm -f ../models/llava-phi-3-mini-int4-vlm.jlm "${JITLLM_MODELS:-../models}/llava-phi-3-mini-int4-vlm.jlm"
  run "the tower's activation is GELU-tanh, not quick-GELU" \
      'TestLlavaTowerLoads|TestLlavaTowerIsDeclinedByTheDevice' ./engine/model
  restore convert/config.go
  rm -f ../models/llava-phi-3-mini-int4-vlm.jlm "${JITLLM_MODELS:-../models}/llava-phi-3-mini-int4-vlm.jlm"
}

# 2. The class row: the projector takes rows 0..575 instead of 1..576, so it
#    projects the class token and drops the last patch.
case_cls() {
  sed -i 's/\tpatches := s.x\[first\*c.NEmbd:\]/\tpatches := s.x[0:]/' engine/model/vision.go
  run "the projector reads from row 0, so every output row is shifted by one patch" \
      'TestLlavaOutputRowsAlignWithPatches' ./engine/model
  restore engine/model/vision.go
}

# 3. The pre-loop LayerNorm is skipped.
case_preln() {
  sed -i 's/\tif t.preLnW != nil {/\tif false \&\& t.preLnW != nil {/' engine/model/vision.go
  run "v.pre_ln is loaded and never applied" \
      'TestLlavaPreLoopNormIsApplied' ./engine/model
  restore engine/model/vision.go
}

# 4. The position table is not paired with its rows: every row reads row 0.
#    ★ THIS IS THE VIOLATION THE FIRST VERSION OF THE GATE PASSED AGAINST. It
#    rotated the whole table, which a tower reading row 0 for everything still
#    sees; only perturbing ONE row separates the two.
case_pos() {
  sed -i 's|pos := t.posW\[p\*c.NEmbd : (p+1)\*c.NEmbd\]|pos := t.posW[0:c.NEmbd]|' engine/model/vision.go
  run "every row reads position row 0" \
      'TestLlavaPositionsArePairedWithTheirRows' ./engine/model
  restore engine/model/vision.go
}

# 5. The tower's weights are left verbatim, so the GGUF row-major family is
#    reachable again.
case_gguf() {
  sed -i 's/^\t\tif jlm.Packed(t.Type) {/\t\tif true || jlm.Packed(t.Type) {/' convert/vision.go
  rm -f ../models/llava-phi-3-mini-int4-vlm.jlm "${JITLLM_MODELS:-../models}/llava-phi-3-mini-int4-vlm.jlm"
  run "the converter leaves the tower's matrices verbatim" \
      'TestLlavaTowerTouchesNoRowMajorQuant|TestLlavaTowerLoads' ./engine/model
  restore convert/vision.go
  rm -f ../models/llava-phi-3-mini-int4-vlm.jlm "${JITLLM_MODELS:-../models}/llava-phi-3-mini-int4-vlm.jlm"
}

# 6. The device stops declining quick-GELU and runs GELU-tanh instead.
case_decline() {
  sed -i 's/\tcase p.Act != nn.ActSiLU \&\& p.Act != nn.ActGELU:/\tcase false:/' jit_old/gpu/tier/layer.go
  run "the device accepts a quick-GELU block and bakes GELU-tanh" \
      'TestQuickGELUIsDeclinedByTheDevice|TestDeclineReasonReachesLastErr' ./jit_old/gpu/tier
  run "  and the model-level gate sees it too" \
      'TestLlavaTowerIsDeclinedByTheDevice' ./engine/model
  restore jit_old/gpu/tier/layer.go
}

# 7. The quick-GELU kernel emits SiLU.
case_kernel() {
  sed -i 's/^\tcase ActQuickGELU:$/\tcase ActQuickGELU + 100:/' jit_old/cpu/act.go
  run "EmitAct(ActQuickGELU) falls through to SiLU" \
      'TestQuickGELUMatchesItsDefinition|TestUngatedActivationsAreAllDistinct' ./jit_old/cpu
  restore jit_old/cpu/act.go
}

for c in act cls preln pos gguf decline kernel; do
  if [ -z "$only" ] || [ "$only" = "$c" ]; then
    "case_$c"
  fi
done
