//go:build amd64 || arm64

package nn

import (
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The greedy sampler's argmax, generated. It is a package-level kernel rather
// than a JIT's, because model.Greedy takes only the logits and has no session.
var (
	argmaxCode   [cpu.NumTiers]*cpu.Code
	argmaxOnce   tierOnce
	argmaxConsts = cpu.ArgmaxConsts()
)

func argmaxFor() *cpu.Code {
	t := cpu.HostTier()
	argmaxOnce.do(t, func() {
		b, err := cpu.EmittersFor(t).Argmax()
		if err != nil {
			return // owed on this tier; the caller keeps its scan
		}
		if c, err := cpu.MapNamed(b, "argmax"); err == nil {
			argmaxCode[t] = c
		}
	})
	return argmaxCode[t]
}

// Argmax32JIT returns the index of the largest element of x, with the LOWEST
// index winning a tie -- what model.Greedy's `v > best` keeps and what
// llama.cpp's top-k=1 does.
//
// It serves every length (the kernel takes its own ragged tail); the only
// decline is a tier with no kernel. Args.K is the element count, not a vector
// count.
func Argmax32JIT(x []float32) (int32, bool) {
	c := argmaxFor()
	if c == nil || len(x) == 0 {
		return 0, false
	}
	var idx int32
	var val float32
	args := cpu.Args{
		Q32:  &x[0],
		K:    int64(len(x)),
		Out:  &val,
		ASum: &idx,
		Scr:  (*byte)(unsafe.Pointer(&argmaxConsts[0])),
	}
	c.Call(&args)
	return idx, true
}
