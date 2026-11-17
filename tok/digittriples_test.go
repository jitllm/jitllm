package tok

import (
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok/pretok"
)

// falconPre is tiiuae/falcon-7b's pre_tokenizer, verbatim.
const falconPre = `{"type": "Sequence", "pretokenizers": [{"type": "Punctuation", "behavior": "Contiguous"}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}, {"type": "Digits", "individual_digits": false}, {"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Isolated", "invert": false}]}`

// TestFalconSplitsDigitTriples holds falcon's pipeline to the pieces
// HuggingFace's own `tokenizers` produces (pre_tokenize_str on the model's
// tokenizer.json), with its byte-level Ġ/Ċ read back as the space and newline
// they stand for: this pipeline splits raw text and maps bytes afterwards.
//
// It fails if "[0-9][0-9][0-9]" is read as anything but its own pattern,
// isolated: read as the llama3 alternation skeleton it ran a second
// letter/digit regex instead of isolating digit triples.
func TestFalconSplitsDigitTriples(t *testing.T) {
	ops, err := pretok.ParsePreTokenizer(strings.NewReader(RealTokenizerJSON(falconPre)))
	if err != nil {
		t.Fatal(err)
	}
	want := pretok.Op{Kind: pretok.OpPattern, Pattern: "[0-9][0-9][0-9]", Behavior: "Isolated"}
	if last := ops[len(ops)-1]; last != want {
		t.Fatalf("falcon's last stage parsed as %+v, want %+v", last, want)
	}
	p, err := buildPipeline(ops)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"x 12345 1234567, a99!", []string{"x", " ", "123", "45", " ", "123", "456", "7", ",", " a", "99", "!"}},
		{"def getUserName(self):\n    return 12345", []string{"def", " getUserName", "(", "self", "):", "\n   ", " return", " ", "123", "45"}},
		// HF's Punctuation is is_ascii_punctuation || \p{P}: the ASCII symbols
		// $+<=>^`|~ are \p{S} and still isolated.
		{"a$b+c<d=e>f^g`h|i~j", []string{"a", "$", "b", "+", "c", "<", "d", "=", "e", ">", "f", "^", "g", "`", "h", "|", "i", "~", "j"}},
		{"x += 1; y->z || w <= $5", []string{"x", " ", "+=", " ", "1", ";", " y", "->", "z", " ", "||", " w", " ", "<=", " ", "$", "5"}},
	} {
		if got := p.apply(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%q:\n got %q\nwant %q", c.in, got, c.want)
		}
	}
}

// TestTheDigitTripleSplitIsReadAsItIsWritten: a Split in another behavior, or
// inverted, is that split -- its behavior and its invert flag are part of the
// stage, never read as the isolated form -- and a behavior the tokenizers
// library does not define is refused at the parse.
func TestTheDigitTripleSplitIsReadAsItIsWritten(t *testing.T) {
	for _, c := range []struct {
		pre  string
		want []string
	}{
		{`{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Removed", "invert": false}`,
			[]string{"a", "45b"}},
		{`{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Isolated", "invert": true}`,
			[]string{"a", "123", "45b"}},
		{`{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "MergedWithNext", "invert": false}`,
			[]string{"a", "12345b"}},
	} {
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(RealTokenizerJSON(c.pre)))
		if err != nil {
			t.Fatalf("%s: %v", c.pre, err)
		}
		p, err := buildPipeline(ops)
		if err != nil {
			t.Fatal(err)
		}
		if got := p.apply("a12345b"); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s:\n got %q\nwant %q", c.pre, got, c.want)
		}
	}
	bad := `{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Splitted", "invert": false}`
	if ops, err := pretok.ParsePreTokenizer(strings.NewReader(RealTokenizerJSON(bad))); err == nil {
		t.Errorf("%s: accepted as %+v", bad, ops)
	}
}

// TestDefaultBPEPipeline holds pretok.DefaultBPE() to its regexes, applied in
// order by Python's `regex` module: each pattern over each piece, the gaps
// kept.
//
// "it's" is the sharpest row: the apostrophe is \p{P}, so the first stage
// takes it before any contraction rule sees it. The backtick row is where this
// class parts from HuggingFace's Punctuation.
func TestDefaultBPEPipeline(t *testing.T) {
	p, err := buildPipeline(pretok.DefaultBPE())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"The capital of France is", []string{"The", " capital", " of", " France", " is"}},
		{"In 1987, 12345 people", []string{"In", " ", "198", "7", ",", " ", "123", "45", " people"}},
		{"x += 1; y->z || w <= $5", []string{"x", " ", "+=", " ", "1", ";", " y", "->", "z", " ", "||", " w", " ", "<=", " ", "$", "5"}},
		{"Use `code` and ``x`` here", []string{"Use", " `", "code", "`", " and", " ``", "x", "``", " here"}},
		{"it's 3.14159!\n\n  done", []string{"it", "'", "s", " ", "3", ".", "141", "59", "!", "\n\n ", " done"}},
	} {
		if got := p.apply(c.in); !reflect.DeepEqual(got, c.want) {
			t.Errorf("%q:\n got %q\nwant %q", c.in, got, c.want)
		}
	}
}
