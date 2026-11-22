package ir

import (
	"strings"
	"testing"
)

// scale is out[i] = in[i] * 2 over n elements, one thread an element.
func scaleKernel(t testing.TB, width int) *Kernel {
	b := New("scale", [3]int{width, 1, 1})
	in := b.Param("in", F32)
	out := b.Param("out", F32)
	i := b.Add(U32, b.Mul(U32, b.CTAID(), b.NTID()), b.TID())
	b.Store(out, i, b.Fma(b.Load(F32, in, i, 0), b.ConstF32(2), b.ConstF32(0)), 0)
	return built(t, b)
}

// TestPersistPhasesFollowTheBuffers composes launches and checks where the
// grid barriers fall: a launch reading what an earlier one in the phase wrote
// starts a phase, one writing what the phase read starts a phase, and
// independent launches share one.
func TestPersistPhasesFollowTheBuffers(t *testing.T) {
	k := scaleKernel(t, 128)
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
	if err := PersistJoins(scaleKernel(t, 64), 128); err != nil {
		t.Fatalf("a narrower kernel without workgroup memory: %v", err)
	}
	if err := PersistJoins(scaleKernel(t, 256), 128); err == nil {
		t.Fatal("a wider kernel joined")
	}
	b := New("shared", [3]int{64, 1, 1})
	sh := b.Shared("s", F32, 64)
	out := b.Param("out", F32)
	b.Store(sh, b.TID(), b.ConstF32(1), 0)
	b.Barrier()
	b.Store(out, b.TID(), b.Load(F32, sh, b.TID(), 0), 0)
	if err := PersistJoins(built(t, b), 128); err == nil || !strings.Contains(err.Error(), "workgroup memory") {
		t.Fatalf("a narrower kernel with a barrier joined: %v", err)
	}
	// The composed kernel is valid IR even though it reads what it wrote: a
	// grid barrier separates them.
	p, err := Persist("t", []Part{{K: scaleKernel(t, 128), Groups: 2, Bind: []int{0, 1}},
		{K: scaleKernel(t, 128), Groups: 2, Bind: []int{1, 0}}}, 128, 2)
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

// TestPersistRunsCutWhereALaunchCannotJoin holds the plan's cuts: a run
// takes its first launch's width and every launch that joins it; an
// in-place launch and one with no IR go alone, saying why.
func TestPersistRunsCutWhereALaunchCannotJoin(t *testing.T) {
	k128, k256 := scaleKernel(t, 128), scaleKernel(t, 256)
	parts := []Part{
		{K: k128, Groups: 3, Width: 128, Bind: []int{0, 1}},
		{K: k128, Groups: 3, Width: 128, Bind: []int{1, 2}},
		{K: k128, Groups: 3, Width: 128, Bind: []int{3, 3}},
		{K: k256, Groups: 2, Width: 256, Bind: []int{2, 4}},
		{K: k128, Groups: 5, Width: 128, Bind: []int{4, 5}},
		{Groups: 1, Width: 128},
	}
	want := []struct {
		lo, hi, w int
		why       string
	}{{0, 2, 128, ""}, {2, 3, 128, "input and an output"}, {3, 5, 256, ""}, {5, 6, 128, "no IR"}}
	runs := PersistRuns(parts)
	if len(runs) != len(want) {
		t.Fatalf("%d runs %+v, want %d", len(runs), runs, len(want))
	}
	for i, w := range want {
		r := runs[i]
		if r.Lo != w.lo || r.Hi != w.hi || r.W != w.w {
			t.Errorf("run %d: [%d, %d) width %d, want [%d, %d) width %d", i, r.Lo, r.Hi, r.W, w.lo, w.hi, w.w)
		}
		if (w.why == "") != (r.Why == nil) || (r.Why != nil && !strings.Contains(r.Why.Error(), w.why)) {
			t.Errorf("run %d: why %v, want %q", i, r.Why, w.why)
		}
		if r.Composed() != (w.hi-w.lo >= 2) {
			t.Errorf("run %d: composed %v", i, r.Composed())
		}
	}
	if g := PersistGrid(parts[3:5], 0); g != 5 {
		t.Errorf("grid %d, want the widest launch's 5", g)
	}
	if g := PersistGrid(parts[3:5], 4); g != 4 {
		t.Errorf("grid %d under a limit of 4", g)
	}
}

// TestPersistAtFallsBackWhenTheCountersDoNotFit holds PersistAt's choice:
// dependency counters in a region that holds them, grid barriers in one a
// word short of them and in the empty region.
func TestPersistAtFallsBackWhenTheCountersDoNotFit(t *testing.T) {
	k := scaleKernel(t, 128)
	parts := []Part{{K: k, Groups: 4, Width: 128, Bind: []int{0, 1}}, {K: k, Groups: 4, Width: 128, Bind: []int{1, 2}}}
	const base = 16
	d, err := PersistDeps("probe", parts, 128, 2, base)
	if err != nil {
		t.Fatal(err)
	}
	if d.Words == 0 {
		t.Fatal("a dependent pair takes no counter words: the fallback cannot be told apart")
	}
	for _, c := range []struct {
		name string
		dep  DepRegion
		want bool
	}{
		{"a region that fits", DepRegion{base, base + d.Words}, true},
		{"a word short", DepRegion{base, base + d.Words - 1}, false},
		{"no region", DepRegion{}, false},
	} {
		p, err := PersistAt(parts, 128, 2, c.dep)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		barriers := 0
		for _, o := range p.K.Ops {
			if o.Kind == OpGridBarrier {
				barriers++
			}
		}
		if p.Dep != c.want || (barriers > 0) == c.want {
			t.Errorf("%s: dep %v with %d grid barriers, want dep %v", c.name, p.Dep, barriers, c.want)
		}
		if !p.Dep && (p.Words != 0 || p.Tiles != 0) {
			t.Errorf("%s: grid barriers with %d words, %d tiles", c.name, p.Words, p.Tiles)
		}
	}
}
