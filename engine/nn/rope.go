package nn

import (
	"fmt"
	"sync"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

type ropeKey struct {
	t        cpu.Tier // see elemSet: the tier is in every package-level key
	hd, nrot int
	neox     bool
}

// ropes caches the rotary kernels per (head dim, rotated dims, layout): a model
// uses one, or two when its rotary is partial on some layers. A lost
// LoadOrStore race emits one kernel twice and keeps one; the other is a few
// hundred bytes of executable memory held for the process.
var ropes sync.Map

// RoPE32JIT rotates every head of hd floats in x by one position's table cs
// ({cos, sin} per pair, as Rope.Table fills it), with generated code. The
// dimensions of each head past len(cs) are left alone -- partial rotary.
func RoPE32JIT(x []float32, hd int, cs []float32, neox bool) {
	if len(x) == 0 {
		return
	}
	nrot := len(cs)
	if hd <= 0 || len(x)%hd != 0 {
		panic(fmt.Sprintf("jit: rope over %d floats is not a whole number of %d-wide heads", len(x), hd))
	}
	k := ropeKey{cpu.HostTier(), hd, nrot, neox}
	c, ok := ropes.Load(k)
	if !ok {
		c, _ = ropes.LoadOrStore(k, mustEmit("rope")(cpu.EmittersFor(k.t).RoPE(hd, nrot, neox)))
	}
	c.(*cpu.Code).Call(&cpu.Args{Out: &x[0], AScale: &cs[0], Rows: int64(len(x) / hd)})
}

type ropeSplitKey struct {
	t        cpu.Tier
	hd, nrot int
}

var ropeSplits sync.Map

// RoPESplit32JIT is XD-RoPE's rotation (HunyuanVL) of every head of hd floats
// in x: NEOX pairs whose first halves turn by table A and second halves by
// table B, cs holding A then B ({cos, sin} per pair, nrot floats each), with
// generated code (cpu.EmitRoPESplit). With A equal to B it is RoPE32JIT's NEOX
// rotation to the bit.
func RoPESplit32JIT(x []float32, hd int, cs []float32) {
	if len(x) == 0 {
		return
	}
	if len(cs)%4 != 0 || hd <= 0 || len(x)%hd != 0 {
		panic(fmt.Sprintf("jit: split rope over %d floats of %d-wide heads with %d table floats",
			len(x), hd, len(cs)))
	}
	k := ropeSplitKey{cpu.HostTier(), hd, len(cs) / 2}
	c, ok := ropeSplits.Load(k)
	if !ok {
		c, _ = ropeSplits.LoadOrStore(k, mustEmit("rope split")(cpu.EmittersFor(k.t).RoPESplit(hd, k.nrot)))
	}
	c.(*cpu.Code).Call(&cpu.Args{Out: &x[0], AScale: &cs[0], Rows: int64(len(x) / hd)})
}
