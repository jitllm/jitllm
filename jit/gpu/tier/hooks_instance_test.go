//go:build jitllmfault || jitllmtest

package tier

import "testing"

// TestFaultsBelongToTheTierThatArmedThem opens two tiers in one process, arms
// a fault and a violation on one only, and demands the other reads clean, both
// at open (WithFault, WithViolation) and mid-run (SetFault, SetViolation).
func TestFaultsBelongToTheTierThatArmedThem(t *testing.T) {
	slot := func() []Slot { return []Slot{{Dev: &fakeDev{name: "card"}, Bytes: 1 << 30}} }
	armed, err := New(slot(), WithFault("rows", "nocap"), WithViolation("addnorm"))
	if err != nil {
		t.Fatal(err)
	}
	defer armed.Close()
	clean, err := New(slot())
	if err != nil {
		t.Fatal(err)
	}
	defer clean.Close()
	if !armed.devs[0].rowsFaulted(rowsNocap) || !armed.devs[0].addNormBroken() || !armed.devs[0].blockFaulted() {
		t.Fatal("the armed tier does not read its own fault: the gate arms nothing")
	}
	if clean.devs[0].rowsFaulted(rowsNocap) || clean.devs[0].addNormBroken() || clean.devs[0].blockFaulted() {
		t.Fatal("a fault armed on one tier reads on another in the same process")
	}
	clean.SetFault("paged", "alias")
	if !clean.devs[0].pagedFaulted(pagedAlias) || armed.devs[0].pagedFaulted(pagedAlias) {
		t.Fatal("SetFault reached the wrong tier")
	}
	clean.SetFault("paged", "")
	if clean.devs[0].pagedFaulted(pagedAlias) || clean.SetViolation("gateup", true) || !clean.devs[0].gateUpSwapped() || armed.devs[0].gateUpSwapped() {
		t.Fatal("disarming, or a violation toggle, crossed tiers")
	}
}
