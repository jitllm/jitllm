package tier

import "testing"

// TestSessionIdsAreUniquePerTier: Attach numbers sessions from the tier's own
// counter, so a second tier starts again and two tiers never share one.
func TestSessionIdsAreUniquePerTier(t *testing.T) {
	slot := func() []Slot { return []Slot{{Dev: &fakeDev{name: "card"}, Bytes: 1 << 30}} }
	a, err := New(slot())
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := New(slot())
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	a1, a2, b1 := a.Attach().(*gpuSession), a.Attach().(*gpuSession), b.Attach().(*gpuSession)
	if a1.sid == a2.sid {
		t.Fatalf("two sessions on one tier share id %d", a1.sid)
	}
	if b1.sid != 1 {
		t.Fatalf("the second tier's first session is %d, want 1: the counter is not the tier's", b1.sid)
	}
}
