package tier

import "testing"

// TestEveryDevKindHasAName holds the kind-name table to the enum: a kind
// added without a name would dump as "".
//
// VIOLATION SIGNATURE. Add a kind before numKinds and leave it out of
// devKindName, and this names it.
func TestEveryDevKindHasAName(t *testing.T) {
	for k := devKind(0); k < numKinds; k++ {
		if devKindName[k] == "" {
			t.Errorf("device-program kind %d has no name", k)
		}
	}
	for r := progReason(1); r < numProgReasons; r++ {
		if progReasonName[r] == "" {
			t.Errorf("program reason %d has no name", r)
		}
	}
	if len(devRewrites) > 64 {
		t.Errorf("%d rewrites: rwSet holds 64", len(devRewrites))
	}
}
