package tok

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/pretok"
)

// TestPipelineIsComposedNotHandPicked asserts models load through the generated
// pipeline (v.pipe non-nil) rather than the legacy family map, which agrees on
// every model here and so could hide a wiring mistake.
func TestPipelineIsComposedNotHandPicked(t *testing.T) {
	checked := 0
	for _, stem := range modelStems() {
		f := openSrc(t, stem)
		if f == nil {
			continue
		}
		p, kind, pre := f.name, f.kind(), f.pre()
		var jv *jlm.Vocab
		if f.f != nil {
			jv = vocabOfMaybe(t, f.f)
		} else {
			jv = f.vocab(t)
		}
		if jv == nil {
			f.Close()
			continue // a projector GGUF has no tokenizer section
		}
		v, verr := New(jv)
		f.Close()
		if kind != "gpt2" || verr != nil {
			continue
		}
		if _, generated := pretok.Table()[pre]; !generated {
			t.Logf("%-42s pre=%-12q NO generated pipeline (legacy path)", filepath.Base(p), pre)
			continue
		}
		if v.pipe == nil {
			t.Errorf("%s: pre=%q has a generated pipeline but the vocab did not compose it",
				filepath.Base(p), pre)
			continue
		}
		checked++
		t.Logf("%-42s pre=%-12q %d composed stages", filepath.Base(p), pre, len(v.pipe.ops))
	}
	if checked == 0 {
		testmodels.Missing(t, "%s", "no model exercised the composed pipeline; this test proved nothing")
	}
}

// TestGeneratedShapes pins pipelines the legacy family map got wrong.
func TestGeneratedShapes(t *testing.T) {
	for _, c := range []struct {
		name  string
		kinds []pretok.OpKind
		why   string
	}{
		{"falcon3", []pretok.OpKind{pretok.OpPunctuation, pretok.OpByteLevel, pretok.OpDigits},
			"was routed to splitLlama3 (a single Split); its real shape is three stages"},
		{"falcon-h1", []pretok.OpKind{pretok.OpPunctuation, pretok.OpSplit, pretok.OpByteLevel, pretok.OpSplit},
			"was routed to splitLlama3; its real shape is four stages"},
		{"starcoder", []pretok.OpKind{pretok.OpDigits, pretok.OpByteLevel},
			"control: the legacy starcoderFamily already did digits-then-gpt2"},
		{"gpt-2", []pretok.OpKind{pretok.OpByteLevel},
			"ByteLevel(use_regex) alone -- the pattern lives INSIDE the stage"},
	} {
		ops, ok := pretok.Table()[c.name]
		if !ok {
			t.Errorf("%s: no generated pipeline (%s)", c.name, c.why)
			continue
		}
		if len(ops) != len(c.kinds) {
			t.Errorf("%s: %d stages, want %d -- %s", c.name, len(ops), len(c.kinds), c.why)
			continue
		}
		for i := range ops {
			if ops[i].Kind != c.kinds[i] {
				t.Errorf("%s stage %d: %v, want %v", c.name, i, ops[i].Kind, c.kinds[i])
			}
		}
	}
	// The digit rule, from the models' own artefacts.
	for name, want := range map[string]int{"llama-bpe": 3, "smaug-bpe": 3, "qwen2": 1, "qwen35": 1} {
		ops := pretok.Table()[name]
		for _, op := range ops {
			if op.Kind == pretok.OpSplit && op.Split.DigitGroup != want {
				t.Errorf("%s: digit group %d, want %d", name, op.Split.DigitGroup, want)
			}
		}
	}
	// qwen35 is [\p{L}\p{M}]+ where qwen2 is \p{L}+ -- a difference the legacy
	// family map could not express at all, since both were "qwen2Family".
	if !pretok.Table()["qwen35"][0].Split.LetterMarks {
		t.Error("qwen35: LetterMarks should be set; its pattern is [\\p{L}\\p{M}]+")
	}
	if pretok.Table()["qwen2"][0].Split.LetterMarks {
		t.Error("qwen2: LetterMarks should NOT be set; its pattern is \\p{L}+")
	}
}
