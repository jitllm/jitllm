package tok_test

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/tok"

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
		v, verr := tok.New(jv)
		f.Close()
		if kind != "gpt2" || verr != nil {
			continue
		}
		if _, generated := pretok.Table()[pre]; !generated {
			t.Logf("%-42s pre=%-12q NO generated pipeline (legacy path)", filepath.Base(p), pre)
			continue
		}
		if !v.Piped() {
			t.Errorf("%s: pre=%q has a generated pipeline but the vocab did not compose it",
				filepath.Base(p), pre)
			continue
		}
		checked++
		t.Logf("%-42s pre=%-12q %d composed stages", filepath.Base(p), pre, len(v.PipeOps()))
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
		{"falcon-h1", []pretok.OpKind{pretok.OpPunctuation, pretok.OpPattern, pretok.OpByteLevel, pretok.OpPattern},
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
	// The digit rule and the letter run, from the models' own artefacts:
	// qwen35 is [\p{L}\p{M}]+ where qwen2 is \p{L}+ -- a difference the
	// legacy family map could not express at all, since both were
	// "qwen2Family".
	for name, want := range map[string][2]string{
		"llama-bpe": {`\p{N}{1,3}`, `\p{L}+`}, "smaug-bpe": {`\p{N}{1,3}`, `\p{L}+`},
		"qwen2": {`|\p{N}|`, `\p{L}+`}, "qwen35": {`|\p{N}|`, `[\p{L}\p{M}]+`},
	} {
		op := pretok.Table()[name][0]
		for _, w := range want {
			if op.Kind != pretok.OpPattern || !strings.Contains(op.Pattern, w) {
				t.Errorf("%s: stage 0 is %+v, want a pattern holding %s", name, op, w)
			}
		}
	}
	if strings.Contains(pretok.Table()["qwen2"][0].Pattern, `[\p{L}\p{M}]+`) {
		t.Error("qwen2: its letter run is \\p{L}+, not [\\p{L}\\p{M}]+")
	}
}
