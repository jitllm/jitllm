package pretok

import "strings"

// ByteToRune is the gpt2 byte-level alphabet. Bytes that are printable in
// Latin-1 map to themselves; the other 68 map to U+0100 and up, in byte order.
// That is what makes a BPE vocabulary printable and is why a raw newline
// appears as "Ċ".
func ByteToRune(c byte) rune {
	switch {
	case c >= 0x21 && c <= 0x7E, c >= 0xA1 && c <= 0xAC, c >= 0xAE:
		return rune(c)
	case c <= 0x20:
		return 256 + rune(c)
	case c <= 0xA0:
		return 256 + 33 + rune(c-0x7F)
	}
	return 256 + 67
}

// RuneToByte is ByteToRune's inverse; false for a rune outside the alphabet.
func RuneToByte(r rune) (byte, bool) {
	switch {
	case r >= 0x21 && r <= 0x7E, r >= 0xA1 && r <= 0xAC, r >= 0xAE && r <= 0xFF:
		return byte(r), true
	case r >= 256 && r < 256+33:
		return byte(r - 256), true
	case r >= 256+33 && r < 256+67:
		return byte(r - 256 - 33 + 0x7F), true
	case r == 256+67:
		return 0xAD, true
	}
	return 0, false
}

// ByteLevel is s in that alphabet, the form a BPE vocabulary's tokens and
// merges are written in. A converter that starts from raw bytes -- a tiktoken
// vocabulary -- uses it to write what a GGUF would have carried.
func ByteLevel(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := 0; i < len(s); i++ {
		b.WriteRune(ByteToRune(s[i]))
	}
	return b.String()
}
