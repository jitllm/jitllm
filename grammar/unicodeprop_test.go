package grammar

import (
	"testing"

	"github.com/jitllm/jitllm/internal/uniprop"
	"github.com/jitllm/jitllm/tok"
)

func setSize(s runeSet) int {
	n := 0
	for _, x := range s {
		n += int(x.hi-x.lo) + 1
	}
	return n
}

// TestPropertyEscapesReadTheTokenizersUnicode holds a pattern's \p{...} to
// the Unicode Character Database of the version tok's tables carry: each
// set's size as the 15.1.0 files state it after its ranges
// (DerivedGeneralCategory.txt, Scripts.txt, DerivedCoreProperties.txt,
// PropList.txt, emoji-data.txt), code points 15.1 assigned (CJK Extension I),
// a script extension beside its script, the aliases ECMA-262 takes, and the
// names it does not: a script as a lone name, a property it does not list, a
// name in the wrong case.
func TestPropertyEscapesReadTheTokenizersUnicode(t *testing.T) {
	if uniprop.Version != tok.UnicodeVersion {
		t.Fatalf("the property tables are Unicode %s, the tokenizer's %s", uniprop.Version, tok.UnicodeVersion)
	}
	for body, want := range map[string]int{
		"Lu": 1831, "Ll": 2233, "Lo": 132234, "Nd": 680, "Cn": 824718, "Zs": 17, "Sm": 948,
		"Script=Latin": 1481, "sc=Han": 99030, "Script=Arabic": 1368, "sc=Grek": 518, "Script=Common": 8306,
		"Alphabetic": 138387, "White_Space": 25, "Emoji": 1424, "Extended_Pictographic": 3537, "Hex_Digit": 44,
	} {
		s, err := property(body)
		if err != nil {
			t.Fatalf(`\p{%s}: %v`, body, err)
		}
		if got := setSize(s); got != want {
			t.Errorf(`\p{%s} holds %d code points, the UCD states %d`, body, got, want)
		}
	}
	for _, c := range []struct {
		r       rune
		in, out []string
	}{
		{'A', []string{"Lu", "L", "LC", "Letter", "Uppercase_Letter", "gc=Lu", "General_Category=L", "Script=Latin",
			"sc=Latn", "scx=Latin", "ASCII", "Alphabetic", "Alpha", "Uppercase", "Hex_Digit", "AHex", "Any", "Assigned"},
			[]string{"Ll", "N", "Script=Greek", "Emoji_Presentation"}},
		{0x0660, []string{"Nd", "digit", "N", "sc=Arabic", "scx=Arabic", "scx=Thaana", "Script_Extensions=Yezi"},
			[]string{"sc=Thaana", "ASCII"}},
		{0x2EBF0, []string{"Lo", "Script=Han", "Ideographic", "Unified_Ideograph", "Assigned"}, []string{"Cn"}},
		{0x3000, []string{"Zs", "Z", "Separator", "White_Space", "space", "WSpace"}, []string{"Cc"}},
		{0x1F600, []string{"So", "Emoji", "Emoji_Presentation", "EPres", "Extended_Pictographic", "ExtPict"},
			[]string{"Sm"}},
		{0x0378, []string{"Cn", "C", "sc=Unknown", "sc=Zzzz"}, []string{"Assigned"}},
	} {
		for _, want := range []bool{true, false} {
			bodies := c.in
			if !want {
				bodies = c.out
			}
			for _, body := range bodies {
				s, err := property(body)
				if err != nil {
					t.Fatalf(`\p{%s}: %v`, body, err)
				}
				if s.has(c.r) != want {
					t.Errorf(`\p{%s} holds U+%04X: %v, want %v`, body, c.r, !want, want)
				}
			}
		}
	}
	for _, body := range []string{"Foo", "lu", "Latin", "Script=Lu", "General_Category=Latin", "Block=Basic_Latin",
		"Script_Extensions=", "=Lu", "IDS_Unary_Operator"} {
		if _, err := property(body); err == nil {
			t.Errorf(`\p{%s} reads as a property`, body)
		}
	}
}
