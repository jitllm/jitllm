//go:build amd64 || arm64

package nn

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// DeepSeek V4's two small kernels (jit_old/cpu/hc_const.go has the contracts):
// the hyper-connection mixer and the compressor's pool over positions. Both
// are package-level like the router: they are not a layer's, and a model asks
// for one shape of each for its whole life.

type hcMixKey struct {
	t     cpu.Tier
	iters int
	head  bool
}

// hcMixes caches the mixer per (tier, rounds, head), hcConsts its constant
// block per eps (by bits), and colPools the pool per tier.
var hcMixes, hcConsts, colPools sync.Map

// HCStreams is the stream count the mixer is built for.
const HCStreams = cpu.HCStreams

// HCMix32JIT turns one row's mixes into the hyper-connection's weights with
// generated code: out = pre (4), post (4) and the Sinkhorn-normalised mixer
// (16, row-major), from mix, scale and base (24 each), at eps over iters
// rounds (0 stops at the softmax). With head set it is the head's collapse
// weights alone: out, mix, scale and base are then 8 long, the last four
// padding, and out's first four are pre.
func HCMix32JIT(out, mix, scale, base []float32, eps float32, iters int, head bool) {
	n := (2 + HCStreams) * HCStreams
	if head {
		n = 2 * HCStreams
	}
	if iters < 0 || len(out) < n || len(mix) < n || len(scale) < n || len(base) < n {
		panic(fmt.Sprintf("jit: hcmix over %d, %d, %d and %d floats wants %d each, %d rounds",
			len(out), len(mix), len(scale), len(base), n, iters))
	}
	k := hcMixKey{cpu.HostTier(), iters, head}
	c, ok := hcMixes.Load(k)
	if !ok {
		c, _ = hcMixes.LoadOrStore(k, mustEmit("hcmix")(cpu.EmittersFor(k.t).HCMix(iters, head)))
	}
	kc, ok := hcConsts.Load(math.Float32bits(eps))
	if !ok {
		kc, _ = hcConsts.LoadOrStore(math.Float32bits(eps), cpu.HCMixConsts(eps))
	}
	kons := kc.([]float32)
	c.(*cpu.Code).Call(&cpu.Args{
		Out:    &out[0],
		Q32:    &mix[0],
		AScale: &scale[0],
		Q2:     &base[0],
		Scr:    (*byte)(unsafe.Pointer(&kons[0])),
	})
}

// ColPool32JIT is the compressor's pool with generated code: for each of the
// len(out) channels, the softmax over the slots of gate times kv, summed. kv
// and gate are slots rows of len(out).
func ColPool32JIT(out, kv, gate []float32, slots int) {
	w := len(out)
	if w == 0 {
		return
	}
	if slots <= 0 || len(kv) < slots*w || len(gate) < slots*w {
		panic(fmt.Sprintf("jit: colpool of %d slots of %d over %d kv and %d gate floats", slots, w,
			len(kv), len(gate)))
	}
	t := cpu.HostTier()
	c, ok := colPools.Load(t)
	if !ok {
		c, _ = colPools.LoadOrStore(t, mustEmit("colpool")(cpu.EmittersFor(t).ColPool()))
	}
	c.(*cpu.Code).Call(&cpu.Args{
		Out:    &out[0],
		Q32:    &kv[0],
		Q2:     &gate[0],
		Scr:    (*byte)(unsafe.Pointer(&elemConsts[0])),
		K:      int64(w / cpu.ElemLanes),
		Rows:   int64(w % cpu.ElemLanes),
		Cols:   int64(slots),
		RowStr: int64(4 * w),
	})
}
