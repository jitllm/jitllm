package ir_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// TestShuffleBounds pins the two things that make one shuffle lowering correct
// on three targets at once.
//
// Neither is checkable anywhere else: a mask of 32 or more assembles and reads
// a lane in another subgroup on some devices, and a workgroup that is not
// whole warps breaks lane == tid mod 32. Both are wrong answers without a
// fault.
func TestShuffleBounds(t *testing.T) {
	build := func(group [3]int, mask int64) error {
		b := ir.New("k", group)
		p := b.Param("pOut", ir.F32)
		v := b.ConstF32(1)
		b.Store(p, b.TID(), b.ShuffleXor(ir.F32, v, mask), 0)
		return mustBuild(t, b).Validate()
	}
	if err := build([3]int{32, 1, 1}, 16); err != nil {
		t.Fatalf("a 32-wide group with mask 16 is the shipping shape: %v", err)
	}
	if err := build([3]int{128, 1, 1}, 1); err != nil {
		t.Fatalf("a 128-wide group is four whole warps: %v", err)
	}
	for _, m := range []int64{0, 32, 64} {
		if err := build([3]int{32, 1, 1}, m); err == nil {
			t.Fatalf("mask %d was accepted; it can reach outside the 32-lane block", m)
		} else if !strings.Contains(err.Error(), "shuffle mask") {
			t.Fatalf("mask %d: %v", m, err)
		}
	}
	if err := build([3]int{48, 1, 1}, 1); err == nil {
		t.Fatal("a 48-wide group was accepted; it is one and a half warps")
	}
	if err := build([3]int{32, 2, 1}, 1); err == nil {
		t.Fatal("a 2-D group was accepted; lane == tid.x only holds in 1-D")
	}
}

// TestWidthIsDeclaredNotRemembered: a kernel whose answer depends on the
// subgroup width must SAY so, and a kernel that says so must mean it.
//
// It is the gate under backend.GuaranteedLanes: a kernel needing W lanes runs
// only where W is guaranteed, which rests on it declaring W. An undeclared
// shuffle computes the right answer on any 32-wide device, so no correctness
// test can see it.
func TestWidthIsDeclaredNotRemembered(t *testing.T) {
	build := func() *ir.Kernel {
		b := ir.New("k", [3]int{32, 1, 1})
		p := b.Param("p", ir.F32)
		q := b.Param("q", ir.F32)
		v := b.Load(ir.F32, p, b.TID(), 0)
		b.Store(q, b.TID(), b.Add(ir.F32, v, b.ShuffleXor(ir.F32, v, 16)), 0)
		return mustBuild(t, b)
	}
	k := build()
	if k.Lanes != ir.SubgroupLanes {
		t.Fatalf("emitting a shuffle left Lanes=%d, want %d", k.Lanes, ir.SubgroupLanes)
	}
	if err := k.Validate(); err != nil {
		t.Fatalf("a declared shuffling kernel was refused: %v", err)
	}

	// The violation: the same kernel with the declaration dropped, which is
	// what a hand-built Kernel or a Builder that forgot would produce.
	k = build()
	k.Lanes = 0
	err := k.Validate()
	if err == nil {
		t.Fatal("Validate accepted a shuffling kernel that declares no subgroup width; " +
			"it would be offered to a device that guarantees nothing")
	}
	t.Logf("undeclared: %v", err)

	// And the other direction: a declaration nothing uses, which costs a device
	// that could have run the kernel.
	b := ir.New("k", [3]int{32, 1, 1})
	p := b.Param("p", ir.F32)
	q := b.Param("q", ir.F32)
	b.Store(q, b.TID(), b.Load(ir.F32, p, b.TID(), 0), 0)
	k = mustBuild(t, b)
	k.Lanes = ir.SubgroupLanes
	if err := k.Validate(); err == nil {
		t.Fatal("Validate accepted a stale subgroup declaration on a kernel with no cross-lane op")
	} else {
		t.Logf("stale: %v", err)
	}
}
