package server

import (
	"context"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// TestTheSamplerSeesThePrompt: a request's sampler sees its prompt's tokens
// before its first draw, as llama-server's accepts them
// (common_sampler_accept). With a top-k of one, a request's first token is
// the most likely; with a presence penalty past any logit as well, it is the
// most likely the prompt's tail does not hold -- of a prompt whose most
// likely continuation the prompt holds, so the two differ. A session in a
// batched row (the default) and one alone (MaxBatchRows 1).
func TestTheSamplerSeesThePrompt(t *testing.T) {
	prompts := []string{
		"Once upon a time, there was a little girl. Once upon a time, there was a little",
		"The cat sat on the mat. The cat sat on the mat. The cat sat on the",
		story + ", there was a girl named Lily. Lily liked to play. Lily",
	}
	for _, arm := range []struct {
		name string
		rows int
	}{{"batched", 0}, {"alone", 1}} {
		t.Run(arm.name, func(t *testing.T) {
			path := modelPath(t, smallModel)
			e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true,
				MaxBatchRows: arm.rows}))
			t.Cleanup(e.Close)
			lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"})
			if err != nil {
				t.Fatalf("LoadModel(%s): %v", path, err)
			}
			if (lm.loop != nil) != (arm.rows == 0) {
				t.Fatalf("MaxBatchRows %d: a step loop %v", arm.rows, lm.loop != nil)
			}
			first := func(prompt string, sm engine.Sampler) int32 {
				tok := int32(-1)
				err := e.Generate(context.Background(), GenerateOptions{ModelID: "small", Prompt: Prompt{Kind: PromptText, Text: prompt},
					MaxTokens: 1, Sampling: &sm}, func(ev Event) error {
					if ev.Kind == EventToken && ev.Token.ID >= 0 && tok < 0 {
						tok = ev.Token.ID
					}
					return nil
				})
				if err != nil {
					t.Fatal(err)
				}
				return tok
			}
			seen := 0
			for _, prompt := range prompts {
				ids := lm.m.Vocab.Encode(prompt, true)
				plain := first(prompt, engine.Sampler{Temp: 1, TopK: 1, Seed: 3})
				if !slices.Contains(ids, plain) {
					continue
				}
				seen++
				got := first(prompt, engine.Sampler{Temp: 1, TopK: 1, PresencePen: 100, RepeatLastN: 64, Seed: 3})
				if slices.Contains(ids[max(0, len(ids)-64):], got) {
					t.Errorf("%q: with the prompt's tokens penalized past any logit the first token is %d, which the prompt holds "+
						"(its most likely, %d): the sampler did not see the prompt", prompt, got, plain)
				}
			}
			if seen == 0 {
				t.Fatal("no prompt's most likely continuation is a token of the prompt: the gate sees nothing")
			}
			t.Logf("%d of %d prompts continue with one of their own tokens, each held off it by the penalty", seen, len(prompts))
		})
	}
}
