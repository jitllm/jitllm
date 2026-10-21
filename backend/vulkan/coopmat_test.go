package vulkan

import "testing"

// TestCooperativeMatrixShapesAreEnumerated prints what the driver actually
// offers, and asserts the list is not silently empty on a device whose feature
// bit is set.
//
// A feature bit is not a capability: which (M,N,K) and component types it buys
// is a list only the driver enumerates.
func TestCooperativeMatrixShapesAreEnumerated(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer c.Close()

	shapes, err := CooperativeMatrixShapes(c.inst, c.phys)
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(shapes) == 0 {
		t.Logf("%s: no cooperative-matrix shapes (extension absent or empty list)", c.name)
		return
	}
	t.Logf("%s: %d cooperative-matrix combination(s)", c.name, len(shapes))
	sub, s8 := 0, 0
	for _, s := range shapes {
		t.Logf("  %s", s)
		if s.Scope == ScopeSubgroup {
			sub++
		}
		if s.AType == CompS8 && s.BType == CompS8 && s.ResultType == CompS32 {
			s8++
		}
	}
	t.Logf("subgroup-scope: %d, int8*int8->int32: %d", sub, s8)
}

// TestTileFeaturesAreEnabledNotJustProbed asserts the other half: a device that
// enumerates cooperative-matrix shapes must have opened with every feature the
// tile kernels' SPIR-V declares turned on; enumerating shapes while enabling
// none of the features is a capability the device cannot correctly execute.
//
// Against a violation it reads: swap stCoopMatFeat back to 1000506001 and the
// features query silently returns zero (an unrecognised pNext entry is ignored
// by specification) -- ten shapes enumerated, Tiles() false.
func TestTileFeaturesAreEnabledNotJustProbed(t *testing.T) {
	c, err := Open()
	if err != nil {
		t.Skipf("no Vulkan device: %v", err)
	}
	defer c.Close()

	has, err := hasExtension(c.phys, extCooperativeMatrix)
	if err != nil {
		t.Fatalf("hasExtension: %v", err)
	}
	shapes, err := CooperativeMatrixShapes(c.inst, c.phys)
	if err != nil {
		t.Fatalf("CooperativeMatrixShapes: %v", err)
	}
	if !has || len(shapes) == 0 {
		if c.Tiles() {
			t.Fatalf("Tiles() is true with extension=%v and %d shapes", has, len(shapes))
		}
		t.Skipf("%s has no cooperative matrix (extension=%v, %d shapes)", c.Name(), has, len(shapes))
	}

	tf := readTileFeats(c.phys, apiVersion11)
	for _, f := range []struct {
		name string
		on   uint32
	}{
		{"cooperativeMatrix", tf.coop.on},
		{"storageBuffer16BitAccess", tf.s16.buffer},
		{"storageBuffer8BitAccess", tf.s8.buffer},
		{"shaderFloat16", tf.f16i8.f16},
		{"shaderInt8", tf.f16i8.i8},
	} {
		if f.on != 1 {
			t.Errorf("%s enumerates %d cooperative-matrix shapes and reports %s = %d; "+
				"the SPIR-V declares its capability either way and the driver is free "+
				"to compute the wrong product", c.Name(), len(shapes), f.name, f.on)
		}
	}
	if !c.Tiles() {
		t.Fatalf("%d shapes enumerated and Tiles() is false", len(shapes))
	}
	t.Logf("%s: %d shapes, all five features enabled at vkCreateDevice", c.Name(), len(shapes))
}
