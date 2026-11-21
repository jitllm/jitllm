package engine_test

import (
	"bytes"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/tok"
)

// These gates check that model.WithTokenizer actually reaches tok.New: a
// dropped option compiles just the same, so they open the model with and
// without an asserted tokenizer.json and require the ids to move.
//
// tiny-qwen3moe-f32 is the cheapest file carrying a byte-level BPE vocabulary
// (Qwen's real one); its weights are never run here.
func tokOptModel() string { return testmodels.Path("tiny-qwen3moe-f32.gguf") }

// The asserted pipeline is one model's own pre_tokenizer object, dumped
// verbatim by scripts/gentok (see tok/testdata/pre/README).
const tokOptJSON = "../tok/testdata/pre/gpt-2.json"

// The discriminating input is a case-folded contraction: qwen2's split folds
// case so "'S" is one chunk, while GPT-2's regex is lower-case only and splits
// it. Digit grouping would not work: it moves chunks but not ids, since Qwen's
// merges have no multi-digit entry.
const tokOptText = "ABC'S 100 200"

func tokOptArtefacts(t *testing.T) []byte {
	t.Helper()
	// A missing input fails rather than skips.
	if _, err := os.Stat(tokOptModel()); err != nil {
		testmodels.Missing(t, "%s is the gate's input, not an optional extra: %v", tokOptModel(), err)
	}
	b, err := os.ReadFile(tokOptJSON)
	if err != nil {
		t.Fatalf("%s is the gate's input, not an optional extra: %v", tokOptJSON, err)
	}
	return b
}

// TestTokOptionReachesTokNew drives the option through model.Open and watches
// the token ids move.
func TestTokOptionReachesTokNew(t *testing.T) {
	js := tokOptArtefacts(t)

	base, err := model.Open(jlmOf(t, tokOptModel()))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer base.Close()
	if base.Vocab == nil {
		t.Fatalf("the default load has no vocabulary, so there is nothing to override: %v", base.TokErr)
	}
	def := base.Vocab.Encode(tokOptText, false)

	over, err := model.Open(jlmOf(t, tokOptModel()), model.WithTokenizer(bytes.NewReader(js)))
	if err != nil {
		t.Fatalf("Open with WithTokenizer: %v", err)
	}
	defer over.Close()
	if over.Vocab == nil {
		t.Fatalf("WithTokenizer left no vocabulary: %v", over.TokErr)
	}
	got := over.Vocab.Encode(tokOptText, false)

	if slices.Equal(def, got) {
		t.Fatalf("model.Open dropped the tok.Option: %q tokenizes to %v with and without "+
			"WithTokenizer(%s). The override is not reaching tok.New.", tokOptText, got, tokOptJSON)
	}
	t.Logf("%q: default %v -> asserted %v", tokOptText, def, got)

	// Different is not enough (a corrupted pipeline also differs): the ids must
	// equal what tok.New produces directly with the same option.
	f, err := gguf.Open(tokOptModel())
	if err != nil {
		t.Fatalf("gguf.Open: %v", err)
	}
	defer f.Close()
	vc, err := convert.VocabOf(f)
	if err != nil {
		t.Fatalf("convert.VocabOf: %v", err)
	}
	direct, err := tok.New(vc, tok.WithTokenizer(bytes.NewReader(js)))
	if err != nil {
		t.Fatalf("tok.New with WithTokenizer: %v", err)
	}
	if want := direct.Encode(tokOptText, false); !slices.Equal(got, want) {
		t.Fatalf("model.Open(%s, WithTokenizer) gave %v; tok.New on the same file and the same "+
			"reader gives %v -- the option arrives changed", tokOptModel(), got, want)
	}

	// And the default path is unchanged: no option gives tok.New(vc)'s ids.
	plain, err := tok.New(vc)
	if err != nil {
		t.Fatalf("tok.New: %v", err)
	}
	if want := plain.Encode(tokOptText, false); !slices.Equal(def, want) {
		t.Fatalf("the default path moved: model.Open gives %v, tok.New(vc) gives %v", def, want)
	}
}

// TestTokOptionRefusedOnSPM asks the same reachability question where tok.New
// refuses: SPM has no pre-tokenizer stage, so WithTokenizer must produce a
// TokErr. A dropped option would make the load succeed.
func TestTokOptionRefusedOnSPM(t *testing.T) {
	spm := testmodels.Path("stories260K.gguf")
	if _, err := os.Stat(spm); err != nil {
		testmodels.Missing(t, "%s is the gate's input, not an optional extra: %v", spm, err)
	}
	js := tokOptArtefacts(t)

	// The control: this file tokenizes perfectly well without an override, so a
	// nil Vocab below is the option's doing and not the model's.
	base, err := model.Open(jlmOf(t, spm))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer base.Close()
	if base.Vocab == nil {
		t.Fatalf("%s has no vocabulary even without an override (%v), so this test proves nothing", spm, base.TokErr)
	}

	// model.build keeps the weights and records why the vocabulary is nil, so
	// the refusal shows up in TokErr, not in Open's error.
	over, err := model.Open(jlmOf(t, spm), model.WithTokenizer(bytes.NewReader(js)))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer over.Close()
	if over.Vocab != nil {
		t.Fatalf("WithTokenizer on an SPM vocabulary loaded anyway: the option is being "+
			"discarded on the way to tok.New (%s)", spm)
	}
	if over.TokErr == nil || !strings.Contains(over.TokErr.Error(), "WithTokenizer") {
		t.Fatalf("TokErr = %v; want the WithTokenizer refusal", over.TokErr)
	}
	t.Logf("SPM refusal: %v", over.TokErr)
}
