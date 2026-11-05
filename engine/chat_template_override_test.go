package engine

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/tok/jinja"
)

// Mistral-7B-Instruct-v0.3's GGUF carries a template with no system role, so a
// system prompt is folded into the first user turn; Mistral's own
// tokenizer_config.json puts it before the LAST user turn with a blank line,
// and only there. Converted with -chat-template, the container renders the
// publisher's prompt and the fold stands aside; converted without, it still
// folds.
//
// Both templates are the real ones -- the GGUF's read out of the GGUF, the
// publisher's out of its repo -- carried on the committed stories260K, whose
// vocabulary spells bos and eos as Mistral's does, so no 4 GB conversion is
// needed to render a prompt.
func TestChatTemplateOptionReplacesTheGGUFs(t *testing.T) {
	ggufPath := testmodels.Path("Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
	cfgPath := testmodels.Path(filepath.Join("Mistral-7B-Instruct-v0.3", "tokenizer_config.json"))
	for _, p := range []string{ggufPath, cfgPath} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (internal/testmodels/fetch.sh mistral) -- this gate proved nothing", err)
		}
	}
	f, err := gguf.Open(ggufPath, gguf.WithoutWarm(true))
	if err != nil {
		t.Fatal(err)
	}
	old, _ := f.KV["tokenizer.chat_template"].String()
	f.Close()
	if old == "" {
		t.Fatal("the Mistral GGUF carries no template; this gate proved nothing")
	}
	dir := t.TempDir()
	oldPath := filepath.Join(dir, "gguf.jinja")
	if err := os.WriteFile(oldPath, []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}

	open := func(tpl string) *model.Model {
		t.Helper()
		dst := filepath.Join(t.TempDir(), "out.jlm")
		if _, err := convert.FromGGUF(testmodels.Path("stories260K.gguf"), dst, jlm.Fingerprint{},
			convert.WithChatTemplate(tpl)); err != nil {
			t.Fatal(err)
		}
		m, err := model.Open(dst)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { m.Close() })
		return m
	}
	sys := []model.ChatMessage{{Role: "system", Content: "be terse"}, {Role: "user", Content: "hi"}}
	turns := []model.ChatMessage{{Role: "system", Content: "be terse"},
		{Role: "user", Content: "one"}, {Role: "assistant", Content: "1"}, {Role: "user", Content: "two"}}
	// probe reports whether the fold leaves msgs alone, through the same
	// context renderChat builds.
	probe := func(m *model.Model, msgs []model.ChatMessage) bool {
		src, _ := m.ChatTemplate("")
		tpl, err := jinja.Compile(src)
		if err != nil {
			t.Fatal(err)
		}
		bos, eos := m.SpecialText()
		ctx := map[string]any{"add_generation_prompt": true, "bos_token": bos, "eos_token": eos}
		parts := model.WantsContentParts(tpl)
		return model.SystemRendered(tpl, ctx, parts) && reflect.DeepEqual(model.FoldSystem(tpl, ctx, parts, msgs), msgs)
	}
	render := func(m *model.Model, msgs []model.ChatMessage) string {
		t.Helper()
		out, err := m.ChatPrompt(msgs, true)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}

	pub := open(cfgPath)
	bos, eos := pub.SpecialText()
	if bos != "<s>" || eos != "</s>" {
		t.Fatalf("the carrier spells bos %q and eos %q, not Mistral's", bos, eos)
	}
	if !probe(pub, sys) {
		t.Error("Mistral's own template: the system prompt was folded")
	}
	if got, want := render(pub, sys), "<s>[INST] be terse\n\nhi[/INST]"; got != want {
		t.Errorf("Mistral's own template:\n got %q\nwant %q", got, want)
	}
	if got, want := render(pub, turns), "<s>[INST] one[/INST] 1</s>[INST] be terse\n\ntwo[/INST]"; got != want {
		t.Errorf("Mistral's own template, two turns:\n got %q\nwant %q", got, want)
	}

	gg := open(oldPath)
	if probe(gg, sys) {
		t.Error("the GGUF's template takes a system role; this gate's control proved nothing")
	}
	if got, want := render(gg, sys), "<s>[INST] be terse\nhi [/INST]"; got != want {
		t.Errorf("the GGUF's template:\n got %q\nwant %q", got, want)
	}
	t.Logf("publisher %q", render(pub, turns))
	t.Logf("gguf      %q", render(gg, turns))
}
