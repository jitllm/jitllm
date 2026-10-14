package ir

import (
	"strings"
	"testing"
)

// sumKernel is out[i] = sum of in[0..n): every workgroup reads every element,
// through a loop-carried index the span analysis cannot bound.
func sumKernel(width, n int) *Kernel {
	b := New("sum", [3]int{width, 1, 1})
	in := b.Param("in", F32)
	out := b.Param("out", F32)
	i := b.Add(U32, b.Mul(U32, b.CTAID(), b.NTID()), b.TID())
	acc0 := b.ConstF32(0)
	j0 := b.Const(U32, 0)
	b.Loop(int64(n))
	j := b.Phi(U32, j0)
	acc := b.Phi(F32, acc0)
	b.SetPhi(acc, b.Fma(b.Load(F32, in, j, 0), b.ConstF32(1), acc))
	b.SetPhi(j, b.Add(U32, j, b.Const(U32, 1)))
	b.EndLoop()
	b.Store(out, i, acc, 0)
	return b.Done()
}

// TestSpansBoundTheElementwiseKernel: scale touches ctaid*w + [0, w).
func TestSpansBoundTheElementwiseKernel(t *testing.T) {
	sp := scaleKernel(128).spans()
	for p := 0; p < 2; p++ {
		s := sp[p]
		if !s.ok || s.a != 128 || s.lo != 0 || s.hi != 127 {
			t.Fatalf("param %d: %+v, want ctaid*128 + [0,127]", p, s)
		}
	}
	if s := sumKernel(128, 256).spans()[0]; s.ok {
		t.Fatalf("a loop-carried read was bounded: %+v", s)
	}
	// A clamp to the last element keeps the span for the groups whose range
	// starts at or below it.
	b := New("clamped", [3]int{128, 1, 1})
	in := b.Param("in", F32)
	out := b.Param("out", F32)
	i := b.Min(U32, b.Add(U32, b.Mul(U32, b.CTAID(), b.NTID()), b.TID()), b.Const(U32, 299))
	b.Store(out, i, b.Load(F32, in, i, 0), 0)
	s := b.Done().spans()[0]
	if !s.holds(3) || s.holds(4) {
		t.Fatalf("clamp at 299: %+v holds for 3 groups %v, for 4 %v", s, s.holds(3), s.holds(4))
	}
}

// TestPersistDepsEdges composes a chain and checks which waits it emits:
// elementwise to elementwise is a tile wait, a loop-carried read a whole one,
// and independent parts wait on nothing.
func TestPersistDepsEdges(t *testing.T) {
	sc, sm := scaleKernel(128), sumKernel(128, 384)
	for _, c := range []struct {
		name        string
		parts       []Part
		whole, tile int
	}{
		{"chain of elementwise", []Part{{K: sc, Groups: 3, Bind: []int{0, 1}}, {K: sc, Groups: 3, Bind: []int{1, 2}}}, 0, 1},
		{"independent", []Part{{K: sc, Groups: 3, Bind: []int{0, 1}}, {K: sc, Groups: 3, Bind: []int{0, 2}}}, 0, 0},
		{"all-to-all", []Part{{K: sc, Groups: 3, Bind: []int{0, 1}}, {K: sm, Groups: 3, Bind: []int{1, 2}}}, 1, 0},
		{"write after read", []Part{{K: sm, Groups: 3, Bind: []int{0, 1}}, {K: sc, Groups: 3, Bind: []int{2, 0}}}, 1, 0},
		// The third part reads what the second wrote, and the second waited
		// on all of the first, so the third's wait on the first is implied.
		{"implied", []Part{{K: sc, Groups: 3, Bind: []int{0, 1}}, {K: sm, Groups: 3, Bind: []int{1, 2}},
			{K: sm, Groups: 3, Bind: []int{2, 1}}}, 2, 0},
	} {
		p, err := PersistDeps("t", c.parts, 128, 2, 4)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		whole, tile := 0, 0
		for _, o := range p.K.Ops {
			if o.Kind == OpDepWait {
				if o.Imm&DepTile != 0 {
					tile++
				} else {
					whole++
				}
			}
			if o.Kind == OpGridBarrier {
				t.Fatalf("%s: a grid barrier in a dep kernel", c.name)
			}
		}
		if whole != c.whole || tile != c.tile {
			t.Fatalf("%s: %d whole and %d tile waits, want %d and %d", c.name, whole, tile, c.whole, c.tile)
		}
		if !p.K.GridSync() || p.K.GridSyncParam() != len(p.K.Params)-1 {
			t.Fatalf("%s: the dep buffer is not the co-resident launch's last parameter", c.name)
		}
		if p.Words < 2 {
			t.Fatalf("%s: a region of %d words", c.name, p.Words)
		}
	}
}

// TestValidateHoldsTheProtocol: RULE 13's in-place refusal survives the
// counters -- a task reading what an earlier one wrote without a wait naming
// it is refused, and so is any other use of the dep buffer.
func TestValidateHoldsTheProtocol(t *testing.T) {
	build := func(wait bool, touch bool) *Kernel {
		b := New("v", [3]int{128, 1, 1})
		x := b.Param("x", F32)
		y := b.Param("y", F32)
		dep := b.Param("dep", U32)
		e := b.DepEpoch(dep, 0)
		i := b.TID()
		b.DepTask(0)
		b.Store(x, i, b.ConstF32(1), 0)
		b.DepSignal(dep, b.Const(U32, 2), 0)
		b.DepTask(1)
		if wait {
			b.DepWait(dep, b.Const(U32, 2), e, 0, false)
		}
		if touch {
			b.Store(dep, i, b.Const(U32, 0), 0)
		}
		b.Store(y, i, b.Load(F32, x, i, 0), 0)
		b.DepExit(dep, 0)
		return b.Done()
	}
	if err := build(true, false).Validate(); err != nil {
		t.Fatalf("a waited read: %v", err)
	}
	if err := build(false, false).Validate(); err == nil || !strings.Contains(err.Error(), "without waiting") {
		t.Fatalf("an unwaited read passed: %v", err)
	}
	if err := build(true, true).Validate(); err == nil || !strings.Contains(err.Error(), "dep buffer") {
		t.Fatalf("a store to the dep buffer passed: %v", err)
	}
}
