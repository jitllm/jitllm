package ir

// The subgroup-width class, asserted from inside the package because the bound
// it sweeps to (nKind) is not exported.

import "testing"

// TestWidthDependentIsTheWholeClass pins exactly which op kinds can make a
// kernel's answer depend on which lanes share a subgroup. All four tile ops
// are in it because a cooperative matrix is distributed across the subgroup;
// even tile.splat, whose value does not depend on the width, builds such an
// object. Shared memory and Barrier are workgroup-scoped. A new cross-lane op
// fails this until it is classified, which keeps the kernel sweep complete.
func TestWidthDependentIsTheWholeClass(t *testing.T) {
	var got []Kind
	for k := Kind(0); k < nKind; k++ {
		if WidthDependent(k) {
			got = append(got, k)
		}
	}
	want := []Kind{OpShuffleXor, OpMMA, OpTileLoad, OpTileSplat, OpTileMMA, OpTileStore,
		OpSubgroupReduce, OpSubgroupBroadcast, OpShuffle, OpLaneID}
	if len(got) != len(want) {
		t.Fatalf("width-dependent ops are %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("width-dependent ops are %v, want %v", got, want)
		}
	}
	// OpMMAGet reads a fragment the MMA already produced in this lane, so it
	// is not itself cross-lane.
	// OpSubgroupSize reports the width rather than depending on it.
	for _, k := range []Kind{OpShared, OpBarrier, OpMMAGet, OpLoad, OpStore, OpSubgroupSize} {
		if WidthDependent(k) {
			t.Errorf("%s is classified as width-dependent", k)
		}
	}
}

// TestKindNamesLineUp: every Kind has a name, no name is used twice, and three
// spot ordinals are the op they claim to be.
//
// A missing name in the middle shifts every later name, so Validate's
// diagnostics would name the wrong op; the length check alone cannot see it.
func TestKindNamesLineUp(t *testing.T) {
	seen := map[string]Kind{}
	for k := Kind(0); k < nKind; k++ {
		n := k.String()
		if n == "" {
			t.Errorf("kind %d has no name; every name after the gap is now wrong", k)
			continue
		}
		if prev, ok := seen[n]; ok {
			t.Errorf("kinds %d and %d are both named %q", prev, k, n)
		}
		seen[n] = k
	}
	for _, c := range []struct {
		k    Kind
		want string
	}{
		{OpXor, "xor"}, {OpShuffleXor, "shuffle.xor"}, {OpShared, "shared"},
		{OpMMA, "mma"}, {OpMMAGet, "mma.get"},
	} {
		if got := c.k.String(); got != c.want {
			t.Errorf("kind %d prints as %q, want %q", c.k, got, c.want)
		}
	}
}
