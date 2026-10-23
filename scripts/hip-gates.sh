#!/usr/bin/env bash
# The AMD backend's gates on a host with an AMD card and ROCm (docs/design/amd-hip.md).
#
#   JITLLM_MODELS=/path/to/models ./scripts/hip-gates.sh
#
# JITLLM_ROCM names the ROCm library directory when it is not in a default
# place (/opt/rocm/lib, /opt/rocm-*/lib). Every step runs under scripts/cap
# (AGENTS.md RULE 3). The run fails if any HIP gate skipped for want of a
# device or of comgr: on this host a skip is a gate that did not run.
set -euo pipefail
cd "$(dirname "$0")/.."
export CGO_ENABLED=0
log=$(mktemp)
trap 'rm -f "$log"' EXIT

step() {
	echo "== $*"
	"$@" 2>&1 | tee -a "$log"
}

step ./scripts/cap 8G -- go test ./backend/hip -count=1 -v -run 'TestRealROCmLoadsOrSaysWhy'
step ./scripts/cap 8G -- go test ./jit/amdgpu ./backend/hip -count=1
step ./scripts/cap 8G -- go test ./jit/gpu/lowertest -count=1 -v -run 'AMDGPU'
# Every backend kernel gate runs on every device backend.Open returns, the
# HIP device among them; TestHIP* are its own.
step ./scripts/cap 16G -- go test ./backend -count=1 -v
step ./scripts/cap 8G -- go test ./jit/gpu/tier -count=1
# The model gates. gpu:0 is the first device backend.Open returns, which on a
# host with no CUDA is the HIP device.
: "${JITLLM_MODELS:?set JITLLM_MODELS to the model directory (docs/testing.md)}"
step ./scripts/cap 24G -- go test ./engine/model -count=1 -v -timeout 3h -run \
	'TestEveryModelRunsGenerated|TestGreedyMatchesLlamaCpp|TestBatchMatchesForward|TestDecodeDoesNotAllocate|TestLayerRelocatesToAnotherDevice|TestRelocatedBlocksComeBack|TestKVPagesRelocateWithTheLayer|TestPagingDoesNotChangeTheAnswer|TestOpenHonoursAPageBudget|TestForwardBatchFaultsItsBlocksIn|TestSlidingWindowRunsOnTheDevice|TestDeviceMatchesHostOnABiasedModel'

if grep -E 'NO AMD DEVICE|NO COMGR|NO ROCM' "$log"; then
	echo "FAIL: a HIP gate skipped on the host meant to run it" >&2
	exit 1
fi
if ! grep -q 'amdgcn' "$log"; then
	echo "FAIL: no gate named the amdgcn backend; the HIP device did not run" >&2
	exit 1
fi
echo "HIP gates passed"
