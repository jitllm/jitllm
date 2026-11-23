package server

import (
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/jitllm/jitllm/model"
)

// TestARequestReadsTheHostOnce: a model_id request on a model with its memory
// cache re-divides the host's share once, at its release, and reads the host
// once for it -- through the step loop (a prompt of ids) and alone (a prompt
// of spans). Against a re-division at the generate's end and again at the
// release, each reading the host for every bound it computes, a request
// reads it four times.
func TestARequestReadsTheHostOnce(t *testing.T) {
	path := modelPath(t, deviceModel)
	e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Version: "test", HostBudget: 8 << 30}))
	var reads atomic.Int64
	e.hostAvail = func() uint64 { reads.Add(1); return 4 << 30 }
	t.Cleanup(e.Close)
	lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if lm.loop == nil || lm.ttft.store == nil || lm.ttft.shared == nil || lm.budget == 0 {
		t.Fatal("the model has no step loop, no memory cache or no page budget: the gate cannot see what it is about")
	}
	ids := lm.m.Vocab.EncodeSpecial("Once upon a time", true)
	for _, arm := range []struct {
		name string
		p    Prompt
	}{
		{"loop", idsPrompt(ids)},
		{"alone", Prompt{Kind: PromptSpans, Spans: []model.Span{{Tokens: ids}}}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			generate(t, e, GenerateOptions{ModelID: "m", Prompt: arm.p, MaxTokens: 4, IgnoreEOS: true})
			reads.Store(0)
			s0 := lm.loop.stats.steps.Load()
			generate(t, e, GenerateOptions{ModelID: "m", Prompt: arm.p, MaxTokens: 4, IgnoreEOS: true})
			if stepped := lm.loop.stats.steps.Load() != s0; stepped != (arm.name == "loop") {
				t.Fatalf("the request stepped the loop %v", stepped)
			}
			if n := reads.Load(); n != 1 {
				t.Fatalf("a request read the host %d times, want once", n)
			}
		})
	}
}
