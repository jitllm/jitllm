package kernels

import (
	"fmt"

	"github.com/jitllm/jitllm/jit/gpu/ir"
)

// Wide amax windows. A window whose words do not fit one Quantize group --
// more than quantGroup threads, i.e. wider than 512 elements -- cannot meet its
// maximum in one group's shared memory, so the quantizers loop over it
// instead. The window a model asks for this way is a whole row: BitNet
// quantizes every projection input per token (nn.WithActWindow), and the host
// clamps every window at its row's end, so a device kernel is handed
// ClampWindow(window, k).

// WideWindow reports whether window is too wide for Quantize's shared-memory
// tile and takes the looping form (quantizeWide, quantPhase's row mode).
func WideWindow(window int) bool { return max(window/32, 1)*8 > quantGroup }

// ClampWindow is the amax window a kernel over rows of k elements quantizes
// with: the session's window, clamped to the row as the host clamps it
// (cpu.QuantizeQ8Window's window ends at the row). A window that ran past a
// row would take the next token's maximum into this one's scale.
func ClampWindow(window, k int) int {
	if window > k {
		return k
	}
	return window
}

// quantizeWide is quantize for a wide window, with Quantize's launch: n/4
// threads in groups of quantGroup. The groups share the windows out, group g
// taking windows g, g+G, ... (G groups); a group finds its window's maximum
// by a strided scan and a tree in shared memory, then quantizes it a word a
// thread. A surplus group or thread re-does the last window or word and
// writes the same bits; every shared slot a thread reads belongs to a real
// owner (RULE 13).
func quantizeWide(n, window int, kind *ActKind) (*ir.Kernel, error) {
	if window%32 != 0 || n%window != 0 {
		return nil, fmt.Errorf("kernels: Quantize: window %d does not tile %d elements", window, n)
	}
	const g = quantGroup
	nb := n / 32
	nWin := n / window
	wWords := window / 4
	groups := (n/4 + g - 1) / g
	perGroup := (nWin + groups - 1) / groups
	chunks := (wWords + g - 1) / g

	name := "quantizewide"
	if kind != nil {
		name = "quantizeactwide"
	}
	b := ir.New(name, [3]int{g, 1, 1})
	var pX, pG, pU, pOut ir.Value
	if kind == nil {
		pX = b.Param("pX", ir.F32)
	} else {
		pG, pU, pOut = b.Param("pG", ir.F32), b.Param("pU", ir.F32), b.Param("pOut", ir.F32)
	}
	pA := b.Param("pA", ir.U32)
	pAX := b.Param("pAX", ir.F32)
	shMax := b.Shared("amax", ir.F32, g)
	shSum := b.Shared("sum", ir.F32, g)

	c := func(v int) ir.Value { return b.Const(ir.U32, int64(v)) }
	tid := b.TID()
	gid := b.CTAID()
	value := func(at ir.Value, store bool) ir.Value {
		if kind == nil {
			return b.Load(ir.F32, pX, at, 0)
		}
		v := actMulValue(b, *kind, b.Load(ir.F32, pG, at, 0), func() ir.Value { return b.Load(ir.F32, pU, at, 0) })
		if store {
			b.Store(pOut, at, v, 0)
		}
		return v
	}
	words := func(word ir.Value, store bool) [4]ir.Value {
		var v [4]ir.Value
		base := b.Mul(ir.U32, word, c(4))
		for i := range v {
			v[i] = value(b.Add(ir.U32, base, c(i)), store)
		}
		return v
	}
	half := b.Const(ir.U32, 0x3F000000)
	signMask := b.Const(ir.U32, 0x80000000)
	lastWin, lastWord := c(nWin-1), c(wWords-1)
	for wi := 0; wi < perGroup; wi++ {
		win := b.Min(ir.U32, b.Add(ir.U32, gid, c(wi*groups)), lastWin)
		wbase := b.Mul(ir.U32, win, c(wWords))
		// The window's maximum: a strided scan, then a tree.
		acc := b.ConstF32(0)
		for j := 0; j < chunks; j++ {
			wl := b.Min(ir.U32, b.Add(ir.U32, tid, c(j*g)), lastWord)
			v := words(b.Add(ir.U32, wbase, wl), false)
			acc = b.Max(ir.F32, acc, amaxOf(b, v[:]))
		}
		amax := treeMax(b, shMax, tid, acc, g)
		d, inv := quantScales(b, amax)
		for j := 0; j < chunks; j++ {
			wl := b.Min(ir.U32, b.Add(ir.U32, tid, c(j*g)), lastWord)
			// This word's slot in the chunk: a surplus thread reads the real
			// owners' sums, never its own.
			local := b.Sub(ir.U32, wl, c(j*g))
			word := b.Add(ir.U32, wbase, wl)
			v := words(word, true)
			sum := b.Const(ir.I32, 0)
			var packed ir.Value
			for i := range v {
				f := b.Mul(ir.F32, v[i], inv)
				sgn := b.And(ir.U32, b.Bitcast(ir.U32, f), signMask)
				rnd := b.Bitcast(ir.F32, b.Add(ir.U32, sgn, half))
				q := b.CvtI32(b.Add(ir.F32, f, rnd))
				sum = b.Add(ir.I32, sum, q)
				byteI := b.And(ir.U32, q, c(0xFF))
				if i == 0 {
					packed = byteI
				} else {
					packed = b.Add(ir.U32, packed, b.Shl(ir.U32, byteI, c(8*i)))
				}
			}
			b.Store(pA, word, packed, 0)
			b.Store(shSum, tid, b.CvtF32(sum), 0)
			b.Barrier()
			qbase := b.Mul(ir.U32, b.Div(ir.U32, local, c(4)), c(4))
			s16 := b.Load(ir.F32, shSum, qbase, 0)
			for i := 1; i < 4; i++ {
				s16 = b.Add(ir.F32, s16, b.Load(ir.F32, shSum, qbase, int64(i)))
			}
			b.Store(pAX, b.Div(ir.U32, word, c(8)), d, 0)
			b.Store(pAX, b.Div(ir.U32, word, c(4)), s16, int64(nb))
			b.Barrier()
		}
	}
	return b.Done(), nil
}

// treeMax reduces every thread's x to the group's maximum through sh, a
// g-wide shared array, and returns it to every thread. The tree needs no
// conditional (RULE 13): every thread takes its partner's partial, clamped
// into the group, and a barrier separates the reads from the writes. It ends
// on a barrier, so sh is free for the caller's next writes.
func treeMax(b *ir.Builder, sh, tid, x ir.Value, g int) ir.Value {
	b.Store(sh, tid, x, 0)
	b.Barrier()
	top := b.Const(ir.U32, int64(g-1))
	for s := g / 2; s >= 1; s /= 2 {
		mine := b.Load(ir.F32, sh, tid, 0)
		other := b.Load(ir.F32, sh, b.Min(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, int64(s))), top), 0)
		m := b.Max(ir.F32, mine, other)
		b.Barrier()
		b.Store(sh, tid, m, 0)
		b.Barrier()
	}
	out := b.Load(ir.F32, sh, b.Const(ir.U32, 0), 0)
	b.Barrier()
	return out
}
