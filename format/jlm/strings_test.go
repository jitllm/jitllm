package jlm

import (
	"slices"
	"testing"
)

// TestStringsAreTheStringsTheyWereMadeOf builds tables of strings of every
// length from none to a few hundred bytes, empty ones among them, and reads
// each back, from NewStrings and from a section encoded and decoded.
func TestStringsAreTheStringsTheyWereMadeOf(t *testing.T) {
	var in []string
	for n := range 300 {
		b := make([]byte, n%37)
		for i := range b {
			b[i] = byte('a' + (n+i)%26)
		}
		in = append(in, string(b))
	}
	in = append(in, "", "Ġthe", "🙂", "")
	for name, tab := range map[string]Strings{"new": NewStrings(in), "empty": NewStrings(nil)} {
		want := in
		if name == "empty" {
			want = nil
		}
		if tab.Len() != len(want) {
			t.Fatalf("%s: %d strings, want %d", name, tab.Len(), len(want))
		}
		for i, s := range want {
			if got := tab.At(i); got != s {
				t.Fatalf("%s: string %d is %q, want %q", name, i, got, s)
			}
		}
	}
	v := &Vocab{Kind: VocabBPE, Tokens: NewStrings(in), Merges: NewMerges([]Merge{{"Ġt", "he"}, {"", "x"}, {"with", " space"}}),
		BOS: -1, EOS: -1, Unk: -1, Pad: -1, Sep: -1, Mask: -1}
	got, err := decodeVocab(encodeVocab(v))
	if err != nil {
		t.Fatal(err)
	}
	var back []string
	for i := range got.Tokens.Len() {
		back = append(back, got.Tokens.At(i))
	}
	if !slices.Equal(back, in) {
		t.Fatalf("the decoded table differs from the encoded one")
	}
	if got.Merges.Len() != 3 || got.Merges.At(0) != (Merge{"Ġt", "he"}) || got.Merges.At(1) != (Merge{"", "x"}) || got.Merges.At(2) != (Merge{"with", " space"}) {
		t.Fatalf("merges came back as %d: %v %v %v", got.Merges.Len(), got.Merges.At(0), got.Merges.At(1), got.Merges.At(2))
	}
}
