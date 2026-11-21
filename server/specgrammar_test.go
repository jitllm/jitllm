package server

import (
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// TestSpeculationHoldsToTheGrammar: a grammar-constrained generate with
// speculation on speculates -- it restores nothing from the prompt store,
// where the same generate without speculation, repeated, does -- and answers
// the tokens the generate without speculation answers, greedy.
func TestSpeculationHoldsToTheGrammar(t *testing.T) {
	sm := structuredModels()[1]
	e, _ := ttftEngine(t, sm.file, Config{}, LoadOptions{})
	prompt := Prompt{Kind: PromptText, Text: sm.prompt}
	g := GenerateOptions{ModelID: "m", Prompt: prompt, MaxTokens: 16, Sampling: &engine.Sampler{},
		Grammar: `root ::= ("yes" | "no") ", " [a-z ]+ "."` + "\n"}
	plain := generate(t, e, g)
	if again := generate(t, e, g); again.started == 0 {
		t.Fatalf("the repeated plain generate restored nothing: the gate cannot tell a speculative one")
	}
	g.Speculation = &Speculation{Enabled: true}
	sp := generate(t, e, g)
	if sp.started != 0 {
		t.Fatalf("the speculative grammar generate restored %d positions: it did not speculate", sp.started)
	}
	if !slices.Equal(sp.ids, plain.ids) {
		t.Fatalf("speculating under the grammar answered %v, without speculation %v", sp.ids, plain.ids)
	}
	t.Logf("%d tokens under the grammar, alike with speculation", len(sp.ids))
}
