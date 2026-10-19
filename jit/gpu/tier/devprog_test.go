package tier

import "testing"

// TestEveryDevKindHasAName holds the names to the enums: a kind, slot or
// reason added without a name would dump as "".
//
// VIOLATION SIGNATURE. Add a kind before numKinds and leave it out of its
// String's switch, and this names it.
func TestEveryDevKindHasAName(t *testing.T) {
	for k := devKind(0); k < numKinds; k++ {
		if k.String() == "" {
			t.Errorf("device-program kind %d has no name", k)
		}
	}
	for r := progReason(1); r < numProgReasons; r++ {
		if r.String() == "" {
			t.Errorf("program reason %d has no name", r)
		}
	}
	for s := devSlot(0); s < numSlots; s++ {
		if s.String() == "" {
			t.Errorf("slot %d has no name", s)
		}
	}
	if n := len(newDevRewrites()); n > 64 {
		t.Errorf("%d rewrites: rwSet holds 64", n)
	}
}

// TestCompositeKindsOnlyShrink pins the kinds that still run an encoder
// sequence whole (compositeKind): the device program is a compiler of
// primitives, and each of these goes as its mechanism is decomposed. A kind
// added to the set fails here, and so does a list entry no longer composite.
//
// VIOLATION SIGNATURE. Mark a new kind composite, or lower a mechanism to
// a wrapper of mixFFN again, and this names it.
func TestCompositeKindsOnlyShrink(t *testing.T) {
	allowed := map[string]bool{"recurrent": true}
	for k := devKind(0); k < numKinds; k++ {
		name := k.String()
		switch {
		case compositeKind(k) && !allowed[name]:
			t.Errorf("%s is composite and not on the list: lower it to primitives", name)
		case !compositeKind(k) && allowed[name]:
			t.Errorf("%s is on the list and no longer composite: remove it from the list", name)
		}
	}
}
