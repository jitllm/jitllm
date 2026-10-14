package nn

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit/cpu"
)

// The IQ4_NL row decode (jit/cpu/iq4nl_x86.go has the contract): one row of
// Qwen3.8-Flash-Next's n-gram table into floats, generated.

var (
	iq4nlK     [cpu.NumTiers]*cpu.Code
	iq4nlOnce  tierOnce
	iq4nlConst = cpu.IQ4NLConsts(quant.IQ4NLValues)
)

// RowIQ4NL32JIT widens raw, whole IQ4_NL blocks, into dst (32 floats a
// block), with 32 bytes of the caller's scratch.
func RowIQ4NL32JIT(dst []float32, raw, scratch []byte) {
	nb := len(raw) / 18
	if len(raw)%18 != 0 || nb == 0 || len(dst) < 32*nb || len(scratch) < 32 {
		panic(fmt.Sprintf("jit: iq4nl row of %d bytes into %d floats with %d bytes of scratch",
			len(raw), len(dst), len(scratch)))
	}
	tier := cpu.HostTier()
	iq4nlOnce.do(tier, func() {
		iq4nlK[tier] = mustEmit("row_iq4nl")(cpu.EmittersFor(tier).RowIQ4NL())
	})
	iq4nlK[tier].Call(&cpu.Args{
		Out: &dst[0], W: (*byte)(unsafe.Pointer(&raw[0])), K: int64(nb),
		Scr: &iq4nlConst[0], Scratch: &scratch[0],
	})
}
