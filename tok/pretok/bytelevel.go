package pretok

import "strings"

// byteToRune is the gpt2 byte-level alphabet. Bytes that are printable in
// Latin-1 map to themselves; the other 68 map to U+0100 and up, in byte order.
// That is what makes a BPE vocabulary printable and is why a raw newline
// appears as "Ċ".
var byteToRune = func() (t [256]rune) {
	used := [256]bool{}
	mark := func(lo, hi int) {
		for c := lo; c <= hi; c++ {
			t[c], used[c] = rune(c), true
		}
	}
	mark(0x21, 0x7E)
	mark(0xA1, 0xAC)
	mark(0xAE, 0xFF)
	n := 0
	for c := 0; c < 256; c++ {
		if !used[c] {
			t[c] = rune(256 + n)
			n++
		}
	}
	return t
}()

// ByteToRune is that alphabet as a table, a copy the caller may keep.
func ByteToRune() [256]rune { return byteToRune }

// ByteLevel is s in that alphabet, the form a BPE vocabulary's tokens and
// merges are written in. A converter that starts from raw bytes -- a tiktoken
// vocabulary -- uses it to write what a GGUF would have carried.
func ByteLevel(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := 0; i < len(s); i++ {
		b.WriteRune(byteToRune[s[i]])
	}
	return b.String()
}
