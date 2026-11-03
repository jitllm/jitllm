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

// ucIsLetter is \p{L}. Note llama.cpp folds every letter subcategory (Lu, Ll,
// Lt, Lm, Lo) into one bit, which is all the pre-tokenizer patterns ask for.
func ucIsLetter(r rune) bool { return ucFlags(r)&ucFlagLetter != 0 }

// ucIsNumber is \p{N} -- decimal, letter and other numbers, as the regex class
// means and as llama.cpp encodes it.
func ucIsNumber(r rune) bool { return ucFlags(r)&ucFlagNumber != 0 }

// ucIsSpace is \s. It is a set, not a category: llama.cpp keeps White_Space
// separate from \p{Z}, and the two are not the same (U+0009 is whitespace and
// not a separator).
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
// rules need and what llama.cpp implements.
func ucToLower(r rune) rune {
	if c, ok := ucMap(ucLower, r); ok {
		return c
	}
	return r
}

// ucFold is simple case folding, which is what `(?i:...)` means and what
// ucToLower is not: U+017F LATIN SMALL LETTER LONG S folds to s but is already
// lowercase, so "'ſ" is a contraction under the model's own regex. llama.cpp
// expands (?i:'s) to '[sS]' and misses it; matching the original regex is a
// deliberate divergence (AGENTS.md RULE 7m).
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

// ucUpperClass and ucLowerClass are o200k's two letter classes -- the shape the
// gpt-4o pre-tokenizer splits a word into and the reason ucUpper is generated.
//
//	ucUpperClass  [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
//	ucLowerClass  [\p{Ll}\p{Lm}\p{Lo}\p{M}]
//
// The general category is not in the flag table (llama.cpp folds Lu/Ll/Lt/Lm/Lo
// into one bit, and the table is generated from that artefact). The two case
// mappings in the same artefact separate the classes exactly:
//
//	Lu, Lt   have a lowercase mapping        -> upper class only
//	Ll       has an uppercase mapping only   -> lower class only
//	Lm, Lo   are letters with neither        -> both classes
//	M        is ucFlagAccentMark             -> both classes
//
// Lt is the case that makes the test "has a lowercase mapping" rather than
// "has no uppercase mapping": U+01C5 ǅ carries both, and it belongs with the
// uppercase letters.
func ucUpperClass(r rune) bool {
	if ucFlags(r)&ucFlagAccentMark != 0 {
		return true
	}
	if !ucIsLetter(r) {
		return false
	}
	if _, down := ucMap(ucLower, r); down {
		return true // Lu or Lt
	}
	_, up := ucMap(ucUpper, r)
	return !up // Lm/Lo have neither mapping; Ll has only the upward one
}

func ucLowerClass(r rune) bool {
	if ucFlags(r)&ucFlagAccentMark != 0 {
		return true
	}
	if !ucIsLetter(r) {
		return false
	}
	_, down := ucMap(ucLower, r)
	return !down // everything but Lu and Lt
}

// ucIsHan is \p{Han}, the script, from the pinned Unicode 15.1.0 table. A
// script cuts across categories, so it is a separate table (from the UCD; see
// scripts/genunicode). A linear scan suffices: only the kimi-k2 splitter asks,
// and Latin text exits on the first range.
func ucIsHan(r rune) bool {
	if r < 0x2E80 {
		return false // below every Han range; the whole of ASCII
	}
	u := uint32(r)
	for o := 0; o < len(ucHan); o += 6 {
		if u < ucWord(ucHan, o) {
			return false
		}
		if u <= ucWord(ucHan, o+3) {
			return true
		}
	}
	return false
}
