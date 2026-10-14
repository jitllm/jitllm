package sched

import "testing"

// TestFuseWhileTheOtherViewIsParked: a session sending its prefill elsewhere
// parks its decode view (Park), which makes the other session's view its
// cores' only user for a fused region; Unpark takes the user back, and a view
// closed while parked leaves the counts as if it had never been one.
func TestFuseWhileTheOtherViewIsParked(t *testing.T) {
	a := Shared([]int{0, 1, 2})
	b := Shared([]int{0, 1, 2})
	defer a.Close()
	defer b.Close()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("Fuse opened beside a view that is not parked: this gate is not on the case it names")
	}
	b.Park()
	if !a.Fuse() {
		t.Fatal("Fuse refused beside a parked view")
	}
	a.Unfuse()
	b.Unpark()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("Fuse opened after the other view was taken back")
	}
	b.Park()
	b.Park() // pairs with one Unpark, not two
	b.Close()
	if !a.Fuse() {
		t.Fatal("Fuse refused after the parked view closed")
	}
	a.Unfuse()
	c := Shared([]int{0, 1, 2})
	defer c.Close()
	if a.Fuse() {
		a.Unfuse()
		t.Fatal("a view opened after a parked one closed is not counted")
	}
}

// TestECrewDoesNotTakeThePCrewsLockBesideTheWideSet: with the P and E prefill
// set open (every session has one), an E-core crew still has a lock of its own
// and is no user of the P-cores' crew, so a decoder there can fuse while
// another session prefills on the E-cores; the wide set takes the E crew's
// lock after its own, so it never drives an E-core beside it.
func TestECrewDoesNotTakeThePCrewsLockBesideTheWideSet(t *testing.T) {
	if !CanSplitE() {
		t.Skip("no E-cores a thread can be kept on here")
	}
	p := Shared(CoreSet("p"))
	wide := SharedBeside(CoreSet("pe"), p)
	e := Shared(CoreSet("e"))
	defer p.Close()
	defer wide.Close()
	defer e.Close()
	if p.crew.mu == e.crew.mu {
		t.Fatal("the E-core crew took the P-cores' lock: a prefill there would wait on a decoder")
	}
	if l := wide.crew.ex.Load(); l == nil || len(*l) != 1 || (*l)[0] != e.crew.mu {
		t.Fatal("the P and E crew does not take the E-core crew's lock")
	}
	if !p.Fuse() {
		t.Fatal("a decoder on the P-cores cannot fuse beside an E-core crew")
	}
	p.Unfuse()
	// A region on the wide set runs with the E crew's lock held after its own.
	wide.Do(64, 1, func(_, _, _ int) {})
}
