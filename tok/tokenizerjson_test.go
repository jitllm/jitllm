package tok

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/pretok"
)

// Gates for the runtime tokenizer.json parse and for WithTokenizer. Each was
// run against a violation: a wrong pipeline tokenizes fluently and slightly
// differently.

// realTokenizerJSON returns a tokenizer.json body carrying obj as its
// pre_tokenizer, laid out the way HuggingFace writes one: the key sits between
// other top-level keys and a vocabulary follows it.
func realTokenizerJSON(obj string) string {
	return `{"version":"1.0","truncation":null,"padding":null,` +
		`"added_tokens":[{"id":0,"content":"<|endoftext|>","special":true}],` +
		`"normalizer":null,"pre_tokenizer":` + obj +
		`,"post_processor":null,"decoder":{"type":"ByteLevel"},` +
		`"model":{"type":"BPE","vocab":{"Richard":42,"Regex":43},"merges":["a b"]}}`
}

// TestParsePreTokenizerAgreesWithTheGeneratedTable feeds the runtime parser the
// shapes the generator read from the same models, and requires the same ops.
//
// The generator and the runtime share one parse; this keeps it that way.
func TestParsePreTokenizerAgreesWithTheGeneratedTable(t *testing.T) {
	const llama3Pat = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	const qwen2Pat = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	split := func(pat string) string {
		return `{"type":"Split","pattern":{"Regex":` + jsonString(pat) + `},"behavior":"Isolated","invert":false}`
	}
	seq := func(inner ...string) string {
		return `{"type":"Sequence","pretokenizers":[` + strings.Join(inner, ",") + `]}`
	}
	for _, c := range []struct{ name, obj string }{
		// llama3: one Split, then ByteLevel with its own regex off.
		{"llama-bpe", seq(split(llama3Pat), `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}`)},
		// qwen2: the same skeleton with a bare \p{N} -- one digit at a time.
		{"qwen2", seq(split(qwen2Pat), `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}`)},
		// gpt-2: a bare ByteLevel, the pattern living inside the stage.
		{"gpt-2", `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`},
		// smollm: Digits before byte encoding.
		{"smollm", seq(`{"type":"Digits","individual_digits":true}`, `{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`)},
		// falcon3: Punctuation first and Digits after byte encoding -- three
		// stages, where the legacy family map gave it one.
		{"falcon3", seq(`{"type":"Punctuation","behavior":"Contiguous"}`,
			`{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}`,
			`{"type":"Digits","individual_digits":true}`)},
	} {
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(realTokenizerJSON(c.obj)))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		want, ok := pretok.Table()[c.name]
		if !ok {
			t.Errorf("%s: not in the generated table; this case proved nothing", c.name)
			continue
		}
		if !reflect.DeepEqual(ops, want) {
			t.Errorf("%s: runtime parse and generated table disagree\n  parsed %+v\n  table  %+v", c.name, ops, want)
		}
	}
}

// jsonString is enough of a JSON string encoder for the patterns here: they
// carry backslashes and nothing else that needs escaping. A pattern that did
// would produce invalid JSON and fail loudly at the parse, which is the right
// direction for a test helper to be wrong in.
func jsonString(s string) string {
	return `"` + strings.ReplaceAll(s, `\`, `\\`) + `"`
}

// TestPreTokenizerObjectStopsAtTheObject is the falcon3 "Richard" case.
//
// On a model whose pre_tokenizer has no Split, a scan for "Regex" ran into
// model.vocab and returned a token as the pattern. The bodies built here carry
// "Richard" and "Regex" in the vocabulary on purpose.
func TestPreTokenizerObjectStopsAtTheObject(t *testing.T) {
	body := realTokenizerJSON(`{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}`)
	obj, err := pretok.PreTokenizerObject([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(obj), "Richard") {
		t.Fatalf("the extracted object ran past pre_tokenizer into the vocabulary: %s", obj)
	}
	if got := string(obj); got != `{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}` {
		t.Fatalf("object is %s", got)
	}
	ops, err := pretok.ParsePreTokenizer(strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(ops) != 1 || ops[0].Kind != pretok.OpByteLevel {
		t.Fatalf("parsed %+v, want one ByteLevel", ops)
	}
	// The brace count alone is not enough: real patterns happen to balance, so
	// this one carries an odd brace behind an escaped quote, making the string
	// and escape tracking load-bearing.
	tricky := realTokenizerJSON(`{"type":"Split","pattern":{"Regex":"a\"}b"},"behavior":"Isolated"}`)
	obj, err = pretok.PreTokenizerObject([]byte(tricky))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(string(obj), `"behavior":"Isolated"}`) {
		t.Fatalf("a brace inside a string cut the object short: %s", obj)
	}
	if ops, err := pretok.ParsePreTokenizer(strings.NewReader(tricky)); err != nil || len(ops) != 1 || ops[0].Kind != pretok.OpSplit {
		t.Fatalf("parsed %+v, %v", ops, err)
	}
}

// TestParsePreTokenizerRefuses: every refusal, because a default here is how a
// model gets tokenized by the nearest-looking splitter.
func TestParsePreTokenizerRefuses(t *testing.T) {
	for _, c := range []struct{ name, body, want string }{
		{"no pre_tokenizer key", `{"version":"1.0","model":{"type":"BPE"}}`, "pre_tokenizer"},
		{"truncated object", `{"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Byte`, "truncated"},
		{"unknown stage only", realTokenizerJSON(`{"type":"Metaspace","replacement":"_"}`), "no recognised stages"},
		{"a Split with no readable pattern", realTokenizerJSON(`{"type":"Split","pattern":{"String":"x"}}`), "no recognised stages"},
		{"not JSON", realTokenizerJSON(`{"type":,}`), "does not parse"},
	} {
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(c.body))
		if err == nil {
			t.Errorf("%s: accepted, giving %+v", c.name, ops)
			continue
		}
		if !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %q does not mention %q", c.name, err, c.want)
		}
	}
}

// TestGoSourceRendersTheCheckedInTable renders every row the way scripts/gentok
// does and requires the result to appear verbatim in the checked-in table, so
// a renderer change cannot silently rewrite every row at the next regeneration.
func TestGoSourceRendersTheCheckedInTable(t *testing.T) {
	src := checkedInSource(t, "pretok/table.go")
	if len(pretok.Table()) == 0 {
		t.Fatal("the generated table is empty; this test proved nothing")
	}
	for name, ops := range pretok.Table() {
		parts := make([]string, len(ops))
		for i, op := range ops {
			parts[i] = op.GoSource()
		}
		line := fmt.Sprintf("\t%q: {%s},\n", name, strings.Join(parts, ", "))
		if !strings.Contains(string(src), line) {
			t.Errorf("GoSource does not reproduce the checked-in row for %q:\n  %s", name, strings.TrimSpace(line))
		}
	}
}

// gpt2Model opens the smallest byte-level BPE model in the model directory.
func gpt2Model(t *testing.T) (*src, string) {
	t.Helper()
	for _, stem := range []string{
		testmodels.Path("SmolLM2-360M-Instruct-Q8_0"),
		testmodels.Path("Qwen2-1.5B-Instruct-Q4_K_M"),
		testmodels.Path("Llama-3.2-1B-Instruct-Q4_K_M"),
	} {
		s := openSrc(t, stem)
		if s == nil {
			continue
		}
		if s.kind() == "gpt2" {
			return s, s.name
		}
		s.Close()
	}
	// A missing artefact is a task, not a skip.
	testmodels.Missing(t, "%s", "no byte-level BPE model under "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- fetch one rather than skipping")
	return nil, ""
}

// TestWithTokenizerOverridesTheGGUF: the caller's tokenizer.json wins over the
// generated table, and the default path is untouched when no option is passed.
func TestWithTokenizerOverridesTheGGUF(t *testing.T) {
	f, name := gpt2Model(t)
	defer f.Close()

	base, err := New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	pre := f.pre()
	if base.pipe == nil {
		t.Fatalf("%s (pre=%q) did not compose a pipeline; this test cannot see an override", name, pre)
	}

	// A deliberately different pipeline: llama3's Split, which groups digits in
	// threes, followed by a ByteLevel with its own regex off. The guard below
	// asserts it is not what this model already had, so the comparison cannot
	// quietly become a self-comparison.
	const pat = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	obj := `{"type":"Sequence","pretokenizers":[` +
		`{"type":"Split","pattern":{"Regex":` + jsonString(pat) + `},"behavior":"Isolated","invert":false},` +
		`{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}`
	over, err := New(f.vocab(t), WithTokenizer(strings.NewReader(realTokenizerJSON(obj))))
	if err != nil {
		t.Fatal(err)
	}
	want := []pretok.Op{
		{Kind: pretok.OpSplit, Split: pretok.SplitParams{Contractions: true, CaseFold: true, DigitGroup: 3}},
		{Kind: pretok.OpByteLevel},
	}
	if !reflect.DeepEqual(over.pipe.ops, want) {
		t.Fatalf("override did not reach the vocabulary: %+v", over.pipe.ops)
	}
	if reflect.DeepEqual(base.pipe.ops, over.pipe.ops) {
		t.Fatalf("%s already had the override's pipeline; the comparison proves nothing", name)
	}

	// It has to change the pieces and the ids, not just a struct field: a
	// pipeline wired in but never run is invisible otherwise.
	//
	// The text is a blank line because digits prove nothing here (SmolLM2
	// has no multi-digit token). llama3's `\s*[\r\n]+` takes "\n\n" whole
	// where GPT-2's pattern emits two, and "ĊĊ" is one token.
	const text = "a\n\nb"
	if a, b := base.preSplit(text), over.preSplit(text); reflect.DeepEqual(a, b) {
		t.Errorf("%s: the override did not change the split of %q: %q", name, text, a)
	}
	if a, b := base.Encode(text, false), over.Encode(text, false); reflect.DeepEqual(a, b) {
		t.Errorf("%s: the override changed the pipeline but not the tokenization of %q: %v", name, text, a)
	}

	// The default path is unchanged with no option; asserted, not assumed.
	again, err := New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(base.Encode(text, true), again.Encode(text, true)) {
		t.Error("a plain New(vocabOf(t, f)) changed after the variadic parameter was added")
	}
}

// TestWithTokenizerRescuesARefusedName: a tokenizer.ggml.pre with no entry in
// the generated table and no legacy family is refused, and an asserted
// tokenizer.json makes that model loadable.
func TestWithTokenizerRescuesARefusedName(t *testing.T) {
	f, name := gpt2Model(t)
	defer f.Close()
	// The same input a converter that wrote an unknown name would have
	// produced; see src.renamePre.
	f.renamePre("a-name-nobody-has-catalogued")

	if _, err := New(f.vocab(t)); err == nil {
		t.Fatalf("%s with an unknown pre loaded anyway; the refusal this rescues is gone", name)
	} else if !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("unexpected refusal: %v", err)
	}
	const obj = `{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t)|\\p{N}{1,3}"},"behavior":"Isolated"},` +
		`{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]}`
	v, err := New(f.vocab(t), WithTokenizer(strings.NewReader(realTokenizerJSON(obj))))
	if err != nil {
		t.Fatalf("the asserted tokenizer.json did not rescue the load: %v", err)
	}
	if v.pipe == nil || len(v.pipe.ops) != 2 {
		t.Fatalf("loaded, but on the legacy path: %+v", v.pipe)
	}
	if len(v.Encode("hello 123", false)) == 0 {
		t.Error("the rescued vocabulary encodes nothing")
	}
}

// TestWithTokenizerRefusedOnSPM: an SPM vocabulary has no pre-tokenizer stage
// at all, so accepting one would report an assertion that never ran.
func TestWithTokenizerRefusedOnSPM(t *testing.T) {
	f := openSrc(t, testmodels.Path("tinyllama-1.1b-q3_K_M"))
	if f == nil {
		testmodels.Missing(t, "%s", "RULE 11: "+testmodels.Path("tinyllama-1.1b-q3_K_M")+" is neither a GGUF nor a container here -- fetch it rather than skipping")
	}
	defer f.Close()
	if kind := f.kind(); kind != "llama" {
		t.Fatalf("tokenizer.ggml.model is %q, want llama; this test proved nothing", kind)
	}
	const obj = `{"type":"ByteLevel","add_prefix_space":false,"use_regex":true}`
	if _, err := New(f.vocab(t), WithTokenizer(strings.NewReader(realTokenizerJSON(obj)))); err == nil {
		t.Fatal("an SPM vocabulary accepted a tokenizer.json pre-tokenizer silently")
	} else if !strings.Contains(err.Error(), "byte-level BPE") {
		t.Errorf("error %q does not say which vocabularies take the option", err)
	}
	// And the same file still loads without the option.
	if _, err := New(f.vocab(t)); err != nil {
		t.Fatalf("the SPM default path broke: %v", err)
	}
}
