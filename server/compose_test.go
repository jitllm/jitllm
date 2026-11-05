package server

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/engine/grammar"
	"github.com/jitllm/jitllm/internal/schemacheck"
)

// TestGrammarComposesWithLogprobsAndChoices: logprobs on a constrained reply
// are the raw distribution's (vLLM's default), so a token the grammar forces
// keeps the log-probability the model gave it rather than 0; n choices are
// each held to the grammar; and speculation with logprobs or with n is
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

	on := &Speculation{Enabled: true}
	for what, o := range map[string]GenerateOptions{
		"logprobs": {ModelID: sm.id, Prompt: prompt, MaxTokens: 4, Speculation: on, Logprobs: true},
		"n":        {ModelID: sm.id, Prompt: prompt, MaxTokens: 4, Speculation: on, Seeds: []int64{1, 2}, Sampling: &engine.Sampler{Temp: 1}},
	} {
		err := e.Generate(context.Background(), o, func(Event) error { return nil })
		if !errors.Is(err, ErrInvalid) {
			t.Fatalf("speculation with %s: %v, want ErrInvalid", what, err)
		}
	}
}
