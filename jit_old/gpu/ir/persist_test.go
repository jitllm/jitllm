package ir

import (
	"strings"
	"testing"
)

// scale is out[i] = in[i] * 2 over n elements, one thread an element.
func scaleKernel(width int) *Kernel {
	b := New("scale", [3]int{width, 1, 1})
	in := b.Param("in", F32)
	out := b.Param("out", F32)
	i := b.Add(U32, b.Mul(U32, b.CTAID(), b.NTID()), b.TID())
	b.Store(out, i, b.Fma(b.Load(F32, in, i, 0), b.ConstF32(2), b.ConstF32(0)), 0)
	return b.Done()
}

// TestPersistPhasesFollowTheBuffers composes launches and checks where the
// grid barriers fall: a launch reading what an earlier one in the phase wrote
// starts a phase, one writing what the phase read starts a phase, and
// independent launches share one.
func TestPersistPhasesFollowTheBuffers(t *testing.T) {
	k := scaleKernel(128)
	for _, c := range []struct {
		name   string
		binds  [][]int
		phases int
	}{
		{"chain a->b->c", [][]int{{0, 1}, {1, 2}}, 2},
		{"fan out a->b, a->c", [][]int{{0, 1}, {0, 2}}, 1},
		{"write after read a->b, c->a", [][]int{{0, 1}, {2, 0}}, 2},
		{"two writers of b", [][]int{{0, 1}, {2, 1}}, 2},
		{"fan out then join", [][]int{{0, 1}, {0, 2}, {1, 3}, {2, 4}}, 2},
	} {
		var parts []Part
		for _, bd := range c.binds {
			parts = append(parts, Part{K: k, Groups: 3, Bind: bd})
		}
		p, err := Persist("t", parts, 128, 2)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if p.Phases != c.phases {
			t.Fatalf("%s: %d phases, want %d", c.name, p.Phases, c.phases)
		}
		n := 0
		for _, o := range p.K.Ops {
			if o.Kind == OpGridBarrier {
				n++
			}
		}
		if n != c.phases-1 {
			t.Fatalf("%s: %d grid barriers for %d phases", c.name, n, c.phases)
		}
		if got := len(p.K.Params); got != len(p.Bufs)+1 {
			t.Fatalf("%s: %d params for %d buffers and the sync", c.name, got, len(p.Bufs))
		}
	}
}

// TestPersistRefusesWhatItCannotCompose holds PersistJoins to its rules.
func TestPersistRefusesWhatItCannotCompose(t *testing.T) {
	if err := PersistJoins(scaleKernel(64), 128); err != nil {
		t.Fatalf("a narrower kernel without workgroup memory: %v", err)
	}
	if err := PersistJoins(scaleKernel(256), 128); err == nil {
		t.Fatal("a wider kernel joined")
	}
	b := New("shared", [3]int{64, 1, 1})
	sh := b.Shared("s", F32, 64)
	out := b.Param("out", F32)
	b.Store(sh, b.TID(), b.ConstF32(1), 0)
	b.Barrier()
	b.Store(out, b.TID(), b.Load(F32, sh, b.TID(), 0), 0)
	if err := PersistJoins(b.Done(), 128); err == nil || !strings.Contains(err.Error(), "workgroup memory") {
		t.Fatalf("a narrower kernel with a barrier joined: %v", err)
	}
	// The composed kernel is valid IR even though it reads what it wrote: a
	// grid barrier separates them.
	p, err := Persist("t", []Part{{K: scaleKernel(128), Groups: 2, Bind: []int{0, 1}},
		{K: scaleKernel(128), Groups: 2, Bind: []int{1, 0}}}, 128, 2)
	if err != nil {
		t.Fatal(err)
	}
	// With a workgroup barrier in its place the in-place check fires.
	bad := *p.K
	bad.Ops = append([]Op(nil), p.K.Ops...)
	for i, o := range bad.Ops {
		if o.Kind == OpGridBarrier {
			bad.Ops[i] = Op{Kind: OpBarrier}
		}
	}
	if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "both read and written") {
		t.Fatalf("the composition with its grid barrier made a workgroup barrier: %v", err)
	}
}
