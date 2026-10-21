package tier

import "testing"

// admitBank is a cache of slots slots over nExpert experts with its heat
// table set, and a token whose selection is sel (all misses).
func admitBank(slots, nExpert int, sel []uint32) *streamBank {
	st := &streamBank{sel: sel, nExpert: nExpert}
	st.initCache(slots, nExpert)
	st.heat = make([]uint32, nExpert)
	st.hit = make([]bool, len(sel))
	st.tick = 1
	return st
}

// fullOf puts a cold expert (heat) in every slot of st.
func fullOf(st *streamBank, heat uint32) {
	for sl := range st.owner {
		st.owner[sl], st.slotOf[sl] = int32(sl), int32(sl)
		st.heat[sl] = heat
	}
}

// TestAdmitFillsAnEmptyCacheAtOnce holds admit's two budgets apart: filling
// empty slots takes every miss the token has, and only a slot changing hands
// is limited by max and the swap budget. Against the old rule (max items
// whatever the slots held) the first assertion fails.
func TestAdmitFillsAnEmptyCacheAtOnce(t *testing.T) {
	const slots, nExpert = 16, 64
	g := newAdmitTier()
	miss := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	st := admitBank(slots, nExpert, miss)
	if err := st.admit(g, nil, nil, nExpert, 2); err != nil {
		t.Fatal(err)
	}
	if len(st.items) != len(miss) {
		t.Fatalf("an empty cache admitted %d of %d misses", len(st.items), len(miss))
	}

	// A full cache of cold experts and a token of hot misses: only max take
	// a slot.
	g = newAdmitTier()
	st = admitBank(slots, nExpert, []uint32{40, 41, 42, 43, 44})
	fullOf(st, 1)
	for _, e := range st.sel {
		st.heat[e] = 100
	}
	if err := st.admit(g, nil, nil, nExpert, 2); err != nil {
		t.Fatal(err)
	}
	if len(st.items) != 2 {
		t.Fatalf("a full cache swapped %d slots with max 2", len(st.items))
	}

	// Misses not clearly hotter than the slots' experts take none.
	g = newAdmitTier()
	st = admitBank(slots, nExpert, []uint32{40, 41})
	fullOf(st, 10)
	st.heat[40], st.heat[41] = 11, 11
	if err := st.admit(g, nil, nil, nExpert, 2); err != nil {
		t.Fatal(err)
	}
	if len(st.items) != 0 {
		t.Fatalf("a full cache admitted %d misses that were not clearly hotter", len(st.items))
	}
}

// newAdmitTier is the parts of a devTier admit reads and writes.
func newAdmitTier() *devTier {
	return &devTier{devShared: &devShared{Config: &Config{}}, devSess: &devSess{}, lane: &lane{}, Stats: &Stats{}}
}
