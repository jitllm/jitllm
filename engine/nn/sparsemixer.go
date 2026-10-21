//go:build amd64 || arm64

package nn

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// Phi-3.5-MoE's sparsemixer router on the host: the plain top-k kernel over
// the raw logits at Norm off, then cpu's sparsemixer weights kernel. Both are
// generated; there is no Go arithmetic (MoEGate.SparseMixer has the contract).

// sparseMixerKey keys the weights kernel by tier and expert count.
type sparseMixerKey struct {
	t cpu.Tier
	n int
}

var sparseMixers struct {
	mu   sync.Mutex
	snap atomic.Pointer[map[sparseMixerKey]*cpu.Code]
}

func init() {
	m := map[sparseMixerKey]*cpu.Code{}
	sparseMixers.snap.Store(&m)
}

// sparseMixerFor returns the weights kernel for n experts on this host, or nil
// when the tier has none. A failure is cached as nil, as moeTopKFill's is.
func sparseMixerFor(n int) *cpu.Code {
	key := sparseMixerKey{cpu.HostTier(), n}
	if c, ok := (*sparseMixers.snap.Load())[key]; ok {
		return c
	}
	sparseMixers.mu.Lock()
	defer sparseMixers.mu.Unlock()
	old := *sparseMixers.snap.Load()
	if c, ok := old[key]; ok {
		return c
	}
	var c *cpu.Code
	if b, err := cpu.EmittersFor(key.t).SparseMixer(n); err == nil {
		if mapped, err := cpu.MapNamed(b, "sparsemixer"); err == nil {
			c = mapped
		}
	}
	next := make(map[sparseMixerKey]*cpu.Code, len(old)+1)
	for kk, vv := range old {
		next[kk] = vv
	}
	next[key] = c
	sparseMixers.snap.Store(&next)
	return c
}

// sparseMixerRoute is MoERouteJIT for a sparsemixer gate. The top-k at Norm
// off leaves the two selected logits in r.Wt (x/1 is x), which is where the
// weights kernel reads them and writes the weights over them; r.Sum stays the
// 1 the top-k wrote.
func sparseMixerRoute(logits []float32, g MoEGate, r *MoERoute) bool {
	k, n := len(r.Sel), len(logits)
	if k != 2 || g.Sigmoid || g.Bias != nil || g.NGroup > 1 || g.Norm || g.scale() != 1 {
		panic(fmt.Sprintf("jit: a sparsemixer gate is a plain top-2 (k=%d sigmoid=%v bias=%v "+
			"groups=%d norm=%v scale=%g)", k, g.Sigmoid, g.Bias != nil, g.NGroup, g.Norm, g.scale()))
	}
	if r.mixEps != g.SparseMixer || len(r.Mix) == 0 {
		panic(fmt.Sprintf("jit: a mixture route built for sparsemixer eps %g called with %g -- "+
			"NewMoERouteFor(n, 2, gate) is the constructor", r.mixEps, g.SparseMixer))
	}
	route := moeRouteFor(n, k, cpu.MoEGate{})
	mix := sparseMixerFor(n)
	if route == nil || mix == nil {
		return false
	}
	// A memmove, not arithmetic: the kernels read the route's own copy.
	copy(r.Sc[:n], logits)
	route.Call(&cpu.Args{
		Q32:      &r.Sc[0],
		Scr:      (*byte)(unsafe.Pointer(&r.Kons[0])),
		Scratch:  (*byte)(unsafe.Pointer(&r.Scr[0])),
		ASum:     &r.Sel[0],
		AHalfSum: &r.Ord[0],
		Out:      &r.Wt[0],
		Out2:     &r.OWt[0],
		AScale:   &r.Sum[0],
	})
	mix.Call(&cpu.Args{
		Q32:      &r.Sc[0],
		Scr:      (*byte)(unsafe.Pointer(&r.Mix[0])),
		ASum:     &r.Sel[0],
		AHalfSum: &r.Ord[0],
		Out:      &r.Wt[0],
		Out2:     &r.OWt[0],
	})
	return true
}
