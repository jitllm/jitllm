package tok

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestIndexFindsAsAMapDoes holds the index to a map[string]int32 built the
// way the tokenizer built it, first id winning, over a vocabulary with
// duplicates and an empty string: every string's id, every string split in
// two through find2, every string as bytes, and strings not in it.
func TestIndexFindsAsAMapDoes(t *testing.T) {
	in := []string{"", "a", "b", "ab", "a", "abc", "Ġthe", "Ġ", "the", "ab", "🙂", "c"}
	for i := range 5000 {
		in = append(in, string(rune('a'+i%26))+string(rune('A'+i%23))+string(rune('0'+i%10)))
	}
	ref := map[string]int32{}
	for i, s := range in {
		if _, dup := ref[s]; !dup {
			ref[s] = int32(i)
		}
	}
	x := newIndex(jlm.NewStrings(in))
	for s, want := range ref {
		if got, ok := x.find(s); !ok || got != want {
			t.Fatalf("find(%q) = %d, %v; want %d", s, got, ok, want)
		}
		if got, ok := x.findBytes([]byte(s)); !ok || got != want {
			t.Fatalf("findBytes(%q) = %d, %v; want %d", s, got, ok, want)
		}
		for k := 0; k <= len(s); k++ {
			if got, ok := x.find2(s[:k], s[k:]); !ok || got != want {
				t.Fatalf("find2(%q, %q) = %d, %v; want %d", s[:k], s[k:], got, ok, want)
			}
		}
	}
	for _, s := range []string{"zz", "abcd", "Ġth", "🙂🙂", "aA"} {
		if _, in := ref[s]; in {
			continue
		}
		if id, ok := x.find(s); ok {
			t.Fatalf("find(%q) = %d; it is not in the table", s, id)
		}
		if id, ok := x.find2(s[:1], s[1:]); ok {
			t.Fatalf("find2 of %q = %d; it is not in the table", s, id)
		}
	}
	var empty index
	if _, ok := empty.find("a"); ok {
		t.Fatal("an empty index found a string")
	}
}
