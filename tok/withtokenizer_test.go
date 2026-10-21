package tok

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/pretok"
)

// Gates for the caller-asserted pre-tokenizer: the table is a cache of a parse,
// the override is selected rather than merely accepted, and a name the table
// refuses is loadable once the caller supplies the artefact. Each was run
// against a deliberate violation: a wrong pre-tokenizer produces fluent,
// slightly different text.

// ---------------------------------------------------------------------------
// Gate 1: the generated table is a cache of a parse of the model's own artefact.
// ---------------------------------------------------------------------------

// TestGeneratedTableMatchesArtefact requires, for every pre_tokenizer object in
// testdata/pre, that pretok.ParsePreTokenizer produce exactly pretok.Table[name]
// field by field, and that Op.GoSource render that parse back into the literal
// checked into tok/pretok/table.go, verbatim. The second half makes the table a
// reproducible cache: a hand-edited row fails even if it builds. The diff is
// per field so a one-field drift is not skimmed past.
func TestGeneratedTableMatchesArtefact(t *testing.T) {
	files, err := filepath.Glob("testdata/pre/*.json")
	if err != nil {
		t.Fatal(err)
	}
	// The glob cannot also be the expectation: an empty directory would pass.
	// testdata/pre/README lists six; fewer means artefacts were removed.
	if len(files) < 6 {
		t.Fatalf("testdata/pre holds %d artefacts, want the 6 its README lists -- "+
			"a shrinking corpus is how this gate stops proving anything", len(files))
	}
	src := checkedInSource(t, "pretok/table.go")
	for _, path := range files {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		b, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		ops, err := pretok.ParsePreTokenizer(strings.NewReader(string(b)))
		if err != nil {
			t.Errorf("%s: the model's own pre_tokenizer does not parse: %v", name, err)
			continue
		}
		row, ok := pretok.Table[name]
		if !ok {
			t.Errorf("%s: the artefact is on disk but the generated table has no row for it; "+
				"either the table lost a name or the file is misnamed", name)
			continue
		}
		if len(ops) != len(row) {
			t.Errorf("%s: the artefact describes %d stages and the table has %d\n  artefact %+v\n  table    %+v",
				name, len(ops), len(row), ops, row)
			continue
		}
		var diffs []string
		for i := range ops {
			if ops[i].Kind != row[i].Kind {
				diffs = append(diffs, fmt.Sprintf("op %d: Kind = %s, table has %s", i, ops[i].Kind, row[i].Kind))
				continue
			}
			diffs = append(diffs, fieldDiffs(fmt.Sprintf("op %d (%s): ", i, ops[i].Kind),
				reflect.ValueOf(ops[i]), reflect.ValueOf(row[i]))...)
		}
		if len(diffs) > 0 {
			t.Errorf("%s: the model's own pre_tokenizer and the generated table disagree:\n  %s",
				name, strings.Join(diffs, "\n  "))
			continue
		}
		parts := make([]string, len(ops))
		for i, op := range ops {
			parts[i] = op.GoSource()
		}
		line := fmt.Sprintf("\t%q: {%s},\n", name, strings.Join(parts, ", "))
		if !strings.Contains(string(src), line) {
			t.Errorf("%s: the artefact does not render back to the checked-in row; tok/pretok/table.go "+
				"is not a cache of this parse:\n  %s", name, strings.TrimSpace(line))
		}
	}
}

// fieldDiffs walks two values of the same struct type and names every field that
// differs, recursing into nested structs so pretok.SplitParams is covered too.
//
// It is reflective so a field added to Op is compared without anyone listing
// it.
func fieldDiffs(prefix string, got, want reflect.Value) []string {
	var out []string
	ty := got.Type()
	for i := 0; i < ty.NumField(); i++ {
		f := ty.Field(i)
		g, w := got.Field(i), want.Field(i)
		if f.Type.Kind() == reflect.Struct {
			out = append(out, fieldDiffs(prefix+f.Name+".", g, w)...)
			continue
		}
		if !reflect.DeepEqual(g.Interface(), w.Interface()) {
			out = append(out, fmt.Sprintf("%s%s = %v, table has %v", prefix, f.Name, g.Interface(), w.Interface()))
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Gate 2: the override is selected, and it moves token ids.
// ---------------------------------------------------------------------------

// qwenBPEModel opens a byte-level BPE model whose tokenizer.ggml.pre is "qwen2".
func qwenBPEModel(t *testing.T) (*src, string) {
	t.Helper()
	for _, stem := range []string{
		testmodels.Path("Qwen2-1.5B-Instruct-Q4_K_M"),
		testmodels.Path("Qwen3-1.7B-Q4_K_M"),
		// 14 MB, a Qwen tokenizer on random weights: a small fixture for testing.
		testmodels.Path("qwen2.5-tiny-random"),
	} {
		s := openSrc(t, stem)
		if s == nil {
			continue
		}
		// By the pipeline it composes, not only by its label: a container
		// converted from safetensors is labelled "tokenizer.json" and carries
		// the stages that file describes.
		if s.pre() == "qwen2" || composes(t, s, pretok.Table["qwen2"]) {
			return s, s.name
		}
		s.Close()
	}
	// A missing artefact is a task, and a skipping test is worse than none.
	t.Skip("LOUD SKIP -- NOT A PASS: no qwen2-pipeline BPE model (Qwen2-1.5B, Qwen3-1.7B or " +
		"qwen2.5-tiny-random, as a GGUF or a container) opened under " + testmodels.Dir() + " (JITLLM_MODELS). They ARE on the Linux " +
		"box, and the 14 MB tiny one is on the M4; this skip means the symlink or the files moved. " +
		"Fetch them; do not record the absence.")
	return nil, ""
}

// composes reports whether s's tokenizer composes exactly these stages.
func composes(t *testing.T, s *src, ops []pretok.Op) bool {
	v, err := New(s.vocab(t))
	return err == nil && v.pipe != nil && reflect.DeepEqual(v.pipe.ops, ops)
}

// loadWith loads f with the pre-tokenizer from a testdata artefact.
func loadWith(t *testing.T, f *src, artefact string) *Vocab {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "pre", artefact+".json"))
	if err != nil {
		t.Fatalf("RULE 11: %v -- the artefact is checked in, fetch it rather than skipping", err)
	}
	v, err := New(f.vocab(t), WithTokenizer(strings.NewReader(string(b))))
	if err != nil {
		t.Fatalf("WithTokenizer(%s.json): %v", artefact, err)
	}
	if v.pipe == nil {
		t.Fatalf("%s.json: loaded with no composed pipeline -- the override did not reach newBPE", artefact)
	}
	return v
}

// TestWithTokenizerIsSelected requires the override to change what the
// tokenizer does, not merely to be accepted; both pipelines are correct, so
// only a difference in output shows which one ran.
//
// The digit probe cannot move ids on this vocabulary (Qwen has no multi-digit
// tokens, so digit grouping never merges), so arm A asserts the split and the
// reason the ids cannot move, and arm B uses a different artefact to assert
// the ids.
func TestWithTokenizerIsSelected(t *testing.T) {
	f, name := qwenBPEModel(t)
	defer f.Close()
	t.Logf("running against %s", name)

	base, err := New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	if base.pipe == nil {
		t.Fatalf("%s composed no pipeline; this gate cannot see an override", name)
	}
	if !reflect.DeepEqual(base.pipe.ops, pretok.Table["qwen2"]) {
		t.Fatalf("%s did not select the generated qwen2 row: %+v", name, base.pipe.ops)
	}
	if g := base.pipe.ops[0].Split.DigitGroup; g != 1 {
		t.Fatalf("qwen2 groups digits in %d; this gate is built on its being 1", g)
	}

	// --- arm A: the split changes, exactly and predictably.
	over := loadWith(t, f, "llama-bpe")
	if !reflect.DeepEqual(over.pipe.ops, pretok.Table["llama-bpe"]) {
		t.Fatalf("the llama-bpe artefact did not reach the vocabulary: %+v", over.pipe.ops)
	}
	if reflect.DeepEqual(base.pipe.ops, over.pipe.ops) {
		t.Fatalf("%s already runs the llama-bpe pipeline; both arms are the same and prove nothing", name)
	}
	if g := over.pipe.ops[0].Split.DigitGroup; g != 3 {
		t.Fatalf("the override's Split groups digits in %d, want 3", g)
	}
	for _, c := range []struct {
		text       string
		want, over []string
	}{
		{"1234", []string{"1", "2", "3", "4"}, []string{"123", "4"}},
		{"2024-06-01",
			[]string{"2", "0", "2", "4", "-", "0", "6", "-", "0", "1"},
			[]string{"202", "4", "-", "06", "-", "01"}},
	} {
		if got := base.preSplit(c.text); !reflect.DeepEqual(got, c.want) {
			t.Errorf("default split of %q is %q, want %q", c.text, got, c.want)
		}
		if got := over.preSplit(c.text); !reflect.DeepEqual(got, c.over) {
			t.Errorf("overridden split of %q is %q, want %q", c.text, got, c.over)
		}
	}

	// Why arm A stops at the split: no merge can span a digit boundary here.
	// A vocabulary that gains a multi-digit token fails here, pointing at the
	// arm that needs re-deriving.
	multi := 0
	for i := range base.text.Len() {
		s := base.text.At(i)
		rs := []rune(s)
		if len(rs) < 2 {
			continue
		}
		allDigits := true
		for _, r := range rs {
			if !ucIsNumber(r) {
				allDigits = false
				break
			}
		}
		if allDigits {
			multi++
		}
	}
	if multi != 0 {
		t.Errorf("%s has %d tokens of 2+ digits; digit grouping CAN move ids on it now, so this "+
			"gate should assert them here rather than in arm B", name, multi)
	}
	if a, b := base.Encode("1234", false), over.Encode("1234", false); !reflect.DeepEqual(a, b) {
		t.Errorf("digit grouping moved the ids of %q after all (%v -> %v) -- welcome, but arm A's "+
			"comment is now wrong and the assertion belongs here", "1234", a, b)
	}

	// --- arm B: the ids move, and they move to named tokens.
	//
	// smollm's pipeline is Digits(individual) -> ByteLevel(use_regex=true), so
	// GPT-2's own pattern runs instead of the llama3/qwen skeleton. Two things
	// change that this vocabulary can show: a blank line stays one piece under
	// `\s*[\r\n]+` and becomes two under GPT-2's, and GPT-2's contraction
	// alternation is case-exact where qwen2's folds.
	smol := loadWith(t, f, "smollm")
	if !reflect.DeepEqual(smol.pipe.ops, pretok.Table["smollm"]) {
		t.Fatalf("the smollm artefact did not reach the vocabulary: %+v", smol.pipe.ops)
	}
	ids := func(pieces ...string) []int32 {
		t.Helper()
		out := make([]int32, len(pieces))
		for i, p := range pieces {
			id, ok := base.ids.find(p)
			if !ok {
				t.Fatalf("%s has no %q token; the expectation cannot be derived from the vocabulary", name, p)
			}
			out[i] = id
		}
		return out
	}
	for _, c := range []struct {
		text       string
		want, over []string
	}{
		{"a\n\nb", []string{"a", "ĊĊ", "b"}, []string{"a", "Ċ", "Ċ", "b"}},
		{"IT'S", []string{"IT", "'S"}, []string{"IT", "'", "S"}},
	} {
		wantBase, wantOver := ids(c.want...), ids(c.over...)
		gotBase, gotOver := base.Encode(c.text, false), smol.Encode(c.text, false)
		if !reflect.DeepEqual(gotBase, wantBase) {
			t.Errorf("default ids for %q are %v, want %v (%q)", c.text, gotBase, wantBase, c.want)
		}
		if !reflect.DeepEqual(gotOver, wantOver) {
			t.Errorf("overridden ids for %q are %v, want %v (%q)", c.text, gotOver, wantOver, c.over)
		}
		if reflect.DeepEqual(gotBase, gotOver) {
			t.Errorf("the override changed the pipeline but not the tokenization of %q: %v", c.text, gotBase)
		}
	}
	// The measured numbers, pinned on the model they were taken from. Deriving
	// the expectation from the vocabulary above keeps the fallback model honest;
	// this keeps the derivation itself from drifting.
	if f.stem() == "Qwen2-1.5B-Instruct-Q4_K_M" {
		for _, c := range []struct {
			text string
			v    *Vocab
			want []int32
		}{
			{"a\n\nb", base, []int32{64, 271, 65}},
			{"a\n\nb", smol, []int32{64, 198, 198, 65}},
			{"IT'S", base, []int32{952, 13272}},
			{"IT'S", smol, []int32{952, 6, 50}},
		} {
			if got := c.v.Encode(c.text, false); !reflect.DeepEqual(got, c.want) {
				t.Errorf("%s: %q encodes to %v, want the measured %v", name, c.text, got, c.want)
			}
		}
	}

	// The default path is unchanged, checked after two overrides loaded from
	// the same *meta.File, so an override that mutated shared state shows here.
	again, err := New(f.vocab(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"1234", "a\n\nb", "IT'S", "Hello, world!", "don't  stop"} {
		if a, b := base.Encode(text, true), again.Encode(text, true); !reflect.DeepEqual(a, b) {
			t.Errorf("a plain New(vocabOf(t, f)) no longer tokenizes %q as it did: %v -> %v", text, a, b)
		}
	}
	if !reflect.DeepEqual(again.pipe.ops, pretok.Table["qwen2"]) {
		t.Errorf("a plain New(vocabOf(t, f)) after two overrides composed %+v", again.pipe.ops)
	}
}

// ---------------------------------------------------------------------------
// Gate 3: a pre the table refuses loads when the caller supplies the artefact.
// ---------------------------------------------------------------------------

// TestWithTokenizerRescuesUnknownPre: an uncatalogued tokenizer.ggml.pre is
// refused, and an asserted tokenizer.json makes that model loadable.
//
// The unknown name is synthesised by rewriting the parsed KV entry, since every
// BPE GGUF on hand resolves; this does not cover the reader surfacing such a
// name from a file.
func TestWithTokenizerRescuesUnknownPre(t *testing.T) {
	const unknown = "a-converter-name-no-table-carries"
	if _, ok := pretok.Table[unknown]; ok {
		t.Fatalf("%q is in the generated table; this gate would rescue nothing", unknown)
	}
	if llama3Family[unknown] || qwen2Family[unknown] || gpt2Family[unknown] || starcoderFamily[unknown] {
		t.Fatalf("%q is in a legacy family map; the refusal this gate rescues would never fire", unknown)
	}
	f := openSrc(t, testmodels.Path("SmolLM2-360M-Instruct-Q8_0"))
	if f == nil {
		testmodels.Missing(t, "%s", "RULE 11: "+testmodels.Path("SmolLM2-360M-Instruct-Q8_0")+" is neither a GGUF nor a container "+
			"here; fetch it rather than accepting a skip.")
	}
	defer f.Close()
	if kind := f.kind(); kind != "gpt2" {
		t.Fatalf("tokenizer.ggml.model is %q, want gpt2; the override is BPE-only and this proved nothing", kind)
	}
	f.renamePre(unknown)

	// The refusal first: a rescue is only a rescue if the load fails without it.
	if v, err := New(f.vocab(t)); err == nil {
		t.Fatalf("an uncatalogued pre loaded anyway, composing %+v -- the refusal is gone", v.pipe)
	} else if !strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("refused for the wrong reason: %v", err)
	}

	v := loadWith(t, f, "llama-bpe")
	if !reflect.DeepEqual(v.pipe.ops, pretok.Table["llama-bpe"]) {
		t.Fatalf("rescued, but running %+v", v.pipe.ops)
	}
	// The asserted pipeline must be the one that runs, not lose to a family
	// splitter: compare with the same stages built independently, and against
	// the model's own pipeline (smollm), which the output must not match.
	ref, err := buildPipeline(pretok.Table["llama-bpe"])
	if err != nil {
		t.Fatal(err)
	}
	native, err := buildPipeline(pretok.Table["smollm"])
	if err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{"1234", "a\n\nb", "IT'S", "don't stop 2024-06-01"} {
		got := v.preSplit(text)
		if !reflect.DeepEqual(got, ref.apply(text)) {
			t.Errorf("%q: the rescued vocabulary splits to %q, the asserted pipeline to %q",
				text, got, ref.apply(text))
		}
		if reflect.DeepEqual(got, native.apply(text)) && !reflect.DeepEqual(ref.apply(text), native.apply(text)) {
			t.Errorf("%q: the rescued vocabulary split as the model's own smollm pipeline, not the asserted one", text)
		}
	}
	if got, want := v.preSplit("1234"), []string{"123", "4"}; !reflect.DeepEqual(got, want) {
		t.Errorf("the rescued vocabulary splits %q to %q, want %q", "1234", got, want)
	}
	if ids := v.Encode("hello 1234", false); len(ids) == 0 {
		t.Error("the rescued vocabulary encodes nothing")
	}

	// The option does not leak: the same file, loaded plainly again, is refused
	// again. An override that stuck would turn a refusal into a silent default
	// for every later caller of the same *meta.File.
	if _, err := New(f.vocab(t)); err == nil {
		t.Error("after an override load, the uncatalogued name stopped being refused")
	}

	// Not duplicated here: the SPM half of the design (a tokenizer.json on a
	// "llama" vocabulary is an error, not a silent no-op) is gated by
	// TestWithTokenizerRefusedOnSPM in tokenizerjson_test.go.
}
