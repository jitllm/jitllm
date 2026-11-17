package tok

import (
	"testing"
	"unicode"
)

// TestPinnedUnicodeTables asserts the tokenizer classifies from jitllm's own
// tables and not from the Go toolchain's. It pins codepoints where the two
// disagree, the only place a revert to unicode.IsLetter would be visible.
func TestPinnedUnicodeTables(t *testing.T) {
	if UnicodeVersion == "" {
		t.Fatal("UnicodeVersion is empty; the tables must declare what they encode")
	}
	t.Logf("pinned Unicode %s, Go ships %s", UnicodeVersion, unicode.Version)

	// The table has to be right about the boring things before its disagreements
	// are worth anything.
	for _, c := range []struct {
		r                     rune
		letter, number, space bool
	}{
		{'a', true, false, false}, {'Z', true, false, false},
		{'0', false, true, false}, {'9', false, true, false},
		{' ', false, false, true}, {'\t', false, false, true},
		{'\n', false, false, true}, {'!', false, false, false},
		{'漢', true, false, false}, {'ß', true, false, false},
		{'½', false, true, false}, // No, a "letter number"/"other number"
	} {
		if got := ucIsLetter(c.r); got != c.letter {
			t.Errorf("ucIsLetter(%q) = %v, want %v", c.r, got, c.letter)
		}
		if got := ucIsNumber(c.r); got != c.number {
			t.Errorf("ucIsNumber(%q) = %v, want %v", c.r, got, c.number)
		}
		if got := ucIsSpace(c.r); got != c.space {
			t.Errorf("ucIsSpace(%q) = %v, want %v", c.r, got, c.space)
		}
	}
	if ucToLower('A') != 'a' || ucToLower('Ä') != 'ä' || ucToLower('a') != 'a' {
		t.Error("ucToLower is wrong on the simple cases")
	}

	// The anti-revert pins drive the reference splitter (splitters_test.go),
	// not the table, so a revert in it is caught. U+1C89 is a letter to Go
	// 17.0.0 and undefined in 15.1: under the pinned tables "a<U+1C89>b" is two
	// pieces, under Go's one.
	for _, c := range []struct {
		text string
		want int
		why  string
	}{
		{"a\u1C89b", 2, "U+1C89 CYRILLIC, added after 15.1"},
		{"a\u088Fb", 2, "U+088F ARABIC, added after 15.1"},
		{"a\uA7CBb", 2, "U+A7CB LATIN, added after 15.1"},
		{"ab", 1, "control: plain ASCII letters are one run in either table"},
	} {
		got := len(splitLlama3(c.text, 3))
		if got != c.want {
			t.Errorf("splitLlama3(%q) gave %d pieces, want %d (%s).\n"+
				"  If this reads %d, the pre-tokenizer is classifying with Go's "+
				"Unicode %s tables instead of the pinned %s ones.",
				c.text, got, c.want, c.why, 1, unicode.Version, UnicodeVersion)
		}
	}
	// U+10D40 GARAY DIGIT ZERO is a number to Go and not to 15.1.
	if ucIsNumber(0x10D40) {
		t.Error("U+10D40: pinned tables say NUMBER; it was added after 15.1")
	}
	if !unicode.IsLetter(0x1C89) {
		t.Fatalf("Go %s no longer calls U+1C89 a letter, so these pins cannot tell "+
			"the two tables apart -- re-derive them from a fresh divergence scan",
			unicode.Version)
	}
	t.Logf("anti-revert pins hold: the splitter is reading the pinned %s tables", UnicodeVersion)
}
