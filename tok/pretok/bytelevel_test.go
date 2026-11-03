package pretok

import "testing"

// TestByteLevelAlphabetIsGPT2s builds the alphabet the way the GPT-2
// reference does and holds ByteToRune and RuneToByte to it.
func TestByteLevelAlphabetIsGPT2s(t *testing.T) {
	var want [256]rune
	var used [256]bool
	mark := func(lo, hi int) {
		for c := lo; c <= hi; c++ {
			want[c], used[c] = rune(c), true
		}
	}
	mark(0x21, 0x7E)
	mark(0xA1, 0xAC)
	mark(0xAE, 0xFF)
	n := 0
	for c := 0; c < 256; c++ {
		if !used[c] {
			want[c] = rune(256 + n)
			n++
		}
	}
	if n != 68 {
		t.Fatalf("%d bytes outside the printable ranges, want 68", n)
	}
	back := 0
	for c := 0; c < 256; c++ {
		if got := ByteToRune(byte(c)); got != want[c] {
			t.Errorf("ByteToRune(%#x) = %U, want %U", c, got, want[c])
		}
		if b, ok := RuneToByte(want[c]); !ok || int(b) != c {
			t.Errorf("RuneToByte(%U) = %#x, %v, want %#x", want[c], b, ok, c)
		}
		back++
	}
	for r := rune(0); r < 600; r++ {
		b, ok := RuneToByte(r)
		if ok && ByteToRune(b) != r {
			t.Errorf("RuneToByte(%U) = %#x, which maps to %U", r, b, ByteToRune(b))
		}
	}
	if back != 256 {
		t.Fatal("the gate checked no bytes")
	}
}
