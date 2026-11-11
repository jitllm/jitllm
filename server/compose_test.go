package server

import (
	"context"
	"encoding/json"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/grammar"
	"github.com/jitllm/jitllm/internal/schemacheck"
)

// TestGrammarComposesWithLogprobsAndChoices: logprobs on a constrained reply
// are the raw distribution's (vLLM's default), so a token the grammar forces
// keeps the log-probability the model gave it rather than 0; n choices are
// each held to the grammar; and speculation with n gives every choice
// refused by name.
func TestGrammarComposesWithLogprobsAndChoices(t *testing.T) {
	sm := structuredModels()[1]
	e, _, _ := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	prompt := Prompt{Kind: PromptText, Text: sm.prompt}

	// A forced literal: under a masked distribution every token's logprob
	// would be 0.
	var lps []float32
	err := e.Generate(context.Background(), GenerateOptions{ModelID: sm.id, Prompt: prompt, MaxTokens: 16,
		Grammar: `root ::= "zebra quantum lettuce"` + "\n", Logprobs: true}, func(ev Event) error {
		if ev.Kind == EventToken && ev.Token.ID >= 0 {
			if ev.Token.Logprob == nil {
				t.Fatal("a token with no logprob")
			}
			lps = append(lps, ev.Token.Logprob.Logprob)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	low := 0
	for _, lp := range lps {
		if lp > 0 {
			t.Fatalf("a logprob above 0: %v", lps)
		}
		if lp < -1 {
			low++
		}
	}
	if low == 0 {
		t.Fatalf("every forced token's logprob is near 0 (%v): these are the masked distribution's", lps)
	}
	t.Logf("forced tokens' raw logprobs %v", lps)

	// n choices, each constrained.
	src, err := grammar.FromJSONSchema([]byte(sm.schemas[0]))
	if err != nil {
		t.Fatal(err)
	}
	var schema any
	if err := json.Unmarshal([]byte(sm.schemas[0]), &schema); err != nil {
		t.Fatal(err)
	}
	texts := map[int]string{}
	err = e.Generate(context.Background(), GenerateOptions{ModelID: sm.id, Prompt: prompt, MaxTokens: int(sm.maxTok),
		Grammar: src, Sampling: &engine.Sampler{Temp: 1}, Seeds: []int64{11, 12, 13}}, func(ev Event) error {
		if ev.Kind == EventToken {
			texts[ev.Choice] += ev.Token.Text
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(texts) != 3 {
		t.Fatalf("%d choices came back", len(texts))
	}
	for i, out := range texts {
		var doc any
		if err := schemacheck.Parse(out, &doc); err != nil {
			t.Fatalf("choice %d: %q: %v", i, out, err)
		}
		if err := schemacheck.Validate(schema, schema, doc); err != nil {
			t.Fatalf("choice %d: %s: %v", i, out, err)
		}
	}

	// Speculation with n: every choice speculates on its own session, and
	// each seed draws what it draws without speculation.
	choices := func(spec *Speculation) map[int][]int32 {
		got := map[int][]int32{}
		err := e.Generate(context.Background(), GenerateOptions{ModelID: sm.id, Prompt: prompt, MaxTokens: 8, Speculation: spec,
			Seeds: []int64{1, 2}, Sampling: &engine.Sampler{Temp: 1}}, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				got[ev.Choice] = append(got[ev.Choice], ev.Token.ID)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	plain, spec := choices(nil), choices(&Speculation{Enabled: true})
	if len(spec) != 2 || !slices.Equal(spec[0], plain[0]) || !slices.Equal(spec[1], plain[1]) {
		t.Fatalf("speculation with n gave %v, without %v", spec, plain)
	}
}
