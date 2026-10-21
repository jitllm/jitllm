package cpu

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// codesSlot is where PackedScratch keeps a code table, on every architecture.
const codesSlot = 384

// putCodes writes MXFP4's code table into a packed scratch block, once per
// 128-bit lane: VPSHUFB looks up within a lane, so the sixteen entries are
// written twice, and one instruction (TBL on arm64) turns thirty-two e2m1 codes
// into the unsigned integers the dot takes.
func putCodes(b []byte) {
	c := kernels.Codes(kernels.MXFP4)
	copy(b[codesSlot:], c[:])
	copy(b[codesSlot+16:], c[:])
}

// hasCodes reports whether q reads its codes through a table, and refuses a
// table the scratch does not hold -- there is one slot, and it is MXFP4's.
func hasCodes(q kernels.Quant) (bool, error) {
	c := kernels.Codes(q)
	switch {
	case c == nil:
		return false, nil
	case c != kernels.Codes(kernels.MXFP4):
		return false, fmt.Errorf("jit: %s has a code table the packed scratch does not hold", q)
	}
	return true, nil
}
