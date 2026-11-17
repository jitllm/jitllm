package pretok

import (
	"strings"
	"testing"
)

// TestPunctuationReadsItsBehavior holds a Punctuation stage to the behavior
// its tokenizer.json states: the two OpPunctuation runs stay that kind, every
// other becomes the class as a pattern stage with that behavior, and a
// behavior the library does not define is refused at the parse.
func TestPunctuationReadsItsBehavior(t *testing.T) {
	parse := func(behavior string) ([]Op, error) {
		return ParsePreTokenizer(strings.NewReader(`{"pre_tokenizer": {"type": "Punctuation", "behavior": "` + behavior + `"}}`))
	}
	for _, c := range []struct {
		behavior string
		want     Op
	}{
		{"Isolated", Op{Kind: OpPunctuation, Behavior: "Isolated"}},
		{"Contiguous", Op{Kind: OpPunctuation, Behavior: "Contiguous"}},
		{"Removed", Op{Kind: OpPattern, Pattern: PunctuationPattern, Behavior: "Removed"}},
		{"MergedWithPrevious", Op{Kind: OpPattern, Pattern: PunctuationPattern, Behavior: "MergedWithPrevious"}},
		{"MergedWithNext", Op{Kind: OpPattern, Pattern: PunctuationPattern, Behavior: "MergedWithNext"}},
	} {
		ops, err := parse(c.behavior)
		if err != nil || len(ops) != 1 || ops[0] != c.want {
			t.Errorf("%s: %+v, %v; want %+v", c.behavior, ops, err, c.want)
		}
	}
	if ops, err := parse("Splitted"); err == nil {
		t.Errorf("an unknown behavior parsed: %+v", ops)
	}
}
