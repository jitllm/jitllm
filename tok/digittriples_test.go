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
// It fails if "[0-9][0-9][0-9]" is read as the llama3 alternation skeleton,
// which runs a second letter/digit regex instead of isolating digit triples.
func TestFalconSplitsDigitTriples(t *testing.T) {
	ops, err := pretok.ParsePreTokenizer(strings.NewReader(RealTokenizerJSON(falconPre)))
	if err != nil {
		t.Fatal(err)
	}
	if last := ops[len(ops)-1]; last.Kind != pretok.OpDigitTriples {
		t.Fatalf("falcon's last stage parsed as %v, want DigitTriples", last.Kind)
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

// TestTheDigitTripleSplitIsReadOnlyIsolated: the pattern is recognised in the
// one form the artefacts carry, and refused in any other rather than guessed.
//
// The rule is deliberately narrow: refusing every Split with no letter class
// would unresolve names (deepseek-llm among them) whose digit stages do parse
// as the skeleton's digit group.
func TestTheDigitTripleSplitIsReadOnlyIsolated(t *testing.T) {
	for _, pre := range []string{
		`{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Removed", "invert": false}`,
		`{"type": "Split", "pattern": {"Regex": "[0-9][0-9][0-9]"}, "behavior": "Isolated", "invert": true}`,
	} {
		if ops, err := pretok.ParsePreTokenizer(strings.NewReader(RealTokenizerJSON(pre))); err == nil {
			t.Errorf("%s: accepted as %+v", pre, ops)
		}
	}
}

// TestLlamaCppDefaultPipeline holds pretok.LlamaCppDefault() to llama.cpp's own
// default regexes (llama-vocab.cpp, the `default:` arm), applied in order by
// Python's `regex` module the way unicode_regex_split applies them: each
// pattern over each piece, the gaps kept.
//
// "it's" is the sharpest row: the apostrophe is \p{P}, so the first stage
// takes it before any contraction rule sees it. The backtick row is where this
// class parts from HuggingFace's Punctuation.
func TestLlamaCppDefaultPipeline(t *testing.T) {
	p, err := buildPipeline(pretok.LlamaCppDefault())
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
