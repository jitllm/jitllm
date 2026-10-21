package cpu

import "testing"

// TestA64BITMatchesClang pins BIT16b against LLVM's AArch64 assembler -- the
// same backend clang uses, and the same treatment TestA64ShiftEncodingsMatchClang
// gives the shift immediates.
//
// BIT shares its encoding with AND, BIC, ORR, ORN, EOR, BSL and BIF and differs
// only in U (bit 29) and size (bits 23:22); a wrong size gives BSL or BIF,
// which disassemble fine and insert under a different rule. The packed matvec
// uses BIT to merge a two-plane weight in every Q5_K and Q6_K dot.
//
// Reference words from:
//
//	llvm-mc -triple=aarch64 -show-encoding
//
// The registers are not arbitrary: v7/v8/v5 and v9/v8/v5 are the pairs
// EmitA64PackedMatVec actually emits for Q6_K at rows=8 (t0, vSec, vHi), and
// v31/v30/v29 and v16/v0/v31 exercise the top of each field.
func TestA64BITMatchesClang(t *testing.T) {
	for _, c := range []struct {
		what       string
		vd, vn, vm VReg
		want       uint32
	}{
		{"bit v7.16b, v2.16b, v4.16b", 7, 2, 4, 0x6ea41c47},
		{"bit v0.16b, v1.16b, v5.16b", 0, 1, 5, 0x6ea51c20},
		{"bit v31.16b, v30.16b, v29.16b", 31, 30, 29, 0x6ebd1fdf},
		{"bit v9.16b, v8.16b, v5.16b", 9, 8, 5, 0x6ea51d09},
		{"bit v7.16b, v8.16b, v5.16b", 7, 8, 5, 0x6ea51d07},
		{"bit v16.16b, v0.16b, v31.16b", 16, 0, 31, 0x6ebf1c10},
	} {
		var a A64
		a.BIT16b(c.vd, c.vn, c.vm)
		b := a.Bytes()
		if len(b) != 4 {
			t.Fatalf("%s: %d bytes", c.what, len(b))
		}
		got := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		if got != c.want {
			t.Errorf("%s: emitted %08x, llvm says %08x", c.what, got, c.want)
		}
	}
}

// TestA64BITIsNotItsNeighbours is the other half of the pin: it asserts BIT is
// distinct from every instruction one field away from it. A table of expected
// words cannot notice that BIT and BSL were swapped if both tables were written
// from the same wrong assumption; these constants come from the encoding's
// structure, so the two checks fail independently.
func TestA64BITIsNotItsNeighbours(t *testing.T) {
	const (
		andW = 0x4e201c00 // U=0 size=00
		bicW = 0x4e601c00 // U=0 size=01
		orrW = 0x4ea01c00 // U=0 size=10
		eorW = 0x6e201c00 // U=1 size=00
		bslW = 0x6e601c00 // U=1 size=01
		bitW = 0x6ea01c00 // U=1 size=10
	)
	var a A64
	a.BIT16b(0, 0, 0)
	got := uint32(a.Bytes()[0]) | uint32(a.Bytes()[1])<<8 |
		uint32(a.Bytes()[2])<<16 | uint32(a.Bytes()[3])<<24
	if got != bitW {
		t.Fatalf("BIT16b(0,0,0) = %08x, want %08x", got, bitW)
	}
	for _, n := range []struct {
		name string
		word uint32
	}{{"AND", andW}, {"BIC", bicW}, {"ORR", orrW}, {"EOR", eorW}, {"BSL", bslW}} {
		if got == n.word {
			t.Errorf("BIT16b emits %s (%08x) -- wrong U or size field", n.name, n.word)
		}
	}
	// And the two fields it is built from are where they are claimed to be.
	if bitW^eorW != 0x00800000 {
		t.Errorf("BIT and EOR should differ only in size (bits 23:22)")
	}
	if bitW^orrW != 0x20000000 {
		t.Errorf("BIT and ORR should differ only in U (bit 29)")
	}

	// The existing AND/ORR emitters must agree with the same table, or the
	// family constant above is fiction.
	var b A64
	b.AND16b(0, 0, 0)
	if w := uint32(b.Bytes()[0]) | uint32(b.Bytes()[1])<<8 |
		uint32(b.Bytes()[2])<<16 | uint32(b.Bytes()[3])<<24; w != andW {
		t.Errorf("AND16b(0,0,0) = %08x, want %08x", w, andW)
	}
	var c A64
	c.ORR16b(0, 0, 0)
	if w := uint32(c.Bytes()[0]) | uint32(c.Bytes()[1])<<8 |
		uint32(c.Bytes()[2])<<16 | uint32(c.Bytes()[3])<<24; w != orrW {
		t.Errorf("ORR16b(0,0,0) = %08x, want %08x", w, orrW)
	}
}
