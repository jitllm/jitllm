package tok

// The classification the pre-tokenizers split on, from jitllm's own pinned
// tables rather than the Go toolchain's: the unicode package's tables change
// with the compiler, and Go's Unicode 17.0.0 disagrees with the pinned tables
// on thousands of letters, so a `go` upgrade could change token ids.

// The generated tables are const strings of fixed-size records sorted by
// their first word, three bytes little-endian each; ucRanges ends each record
// with a flags byte.

// ucWord is the three-byte word at byte offset o of table t.
func ucWord(t string, o int) uint32 {
	return uint32(t[o]) | uint32(t[o+1])<<8 | uint32(t[o+2])<<16
}

// ucLast is the index of the last record of t, stride bytes each, whose first
// word is <= key, or -1.
func ucLast(t string, stride int, key uint32) int {
	lo, hi := 0, len(t)/stride
	for lo < hi {
		m := int(uint(lo+hi) >> 1)
		if ucWord(t, m*stride) > key {
			hi = m
		} else {
			lo = m + 1
		}
	}
	return lo - 1
}

// ucMap is the second word of the record of t, six bytes each, whose first is r.
func ucMap(t string, r rune) (rune, bool) {
	if r < 0 {
		return 0, false
	}
	u := uint32(r)
	i := ucLast(t, 6, u)
	if i < 0 || ucWord(t, i*6) != u {
		return 0, false
	}
	return rune(ucWord(t, i*6+3)), true
}

func ucFlags(r rune) uint16 {
	if r < 0 {
		return ucFlagUndefined
	}
	i := ucLast(ucRanges, 4, uint32(r))
	if i < 0 {
		return ucFlagUndefined
	}
	return uint16(ucRanges[i*4+3])
}

// ucIsSpace is \s. It is a set, not a category: Unicode's White_Space is not
// \p{Z} (U+0009 is whitespace and not a separator).
func ucIsSpace(r rune) bool {
	if r < 0 {
		return false
	}
	u := uint32(r)
	i := ucLast(ucWhitespace, 3, u)
	return i >= 0 && ucWord(ucWhitespace, i*3) == u
}

// ucToLower is the simple lowercase mapping: one codepoint to one codepoint,
// no locale and no multi-character expansions. That is what the contraction
// rules need.
func ucToLower(r rune) rune {
	if c, ok := ucMap(ucLower, r); ok {
		return c
	}
	return r
}

// ucFold is simple case folding, which is what `(?i:...)` means and what
// ucToLower is not: U+017F LATIN SMALL LETTER LONG S folds to s but is already
// lowercase, so "'ſ" is a contraction under the model's own regex, which an
// expansion of (?i:'s) to '[sS]' would miss.
//
// Only one non-ASCII codepoint folds to any of the eight contraction letters
// (s t r e v m l d), so this is a small table rather than CaseFolding.txt.
func ucFold(r rune) rune {
	if r == 0x017F { // LATIN SMALL LETTER LONG S folds to 's'
		return 's'
	}
	if r < 0x80 {
		if r >= 'A' && r <= 'Z' {
			return r + 32
		}
		return r
	}
	return ucToLower(r)
}
