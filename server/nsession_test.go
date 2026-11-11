package server

import (
	"context"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine"
)

// TestSeveralChoicesContinueANamedSession: n choices on a named session. Its
// choice 0 runs on it and the others on fresh sessions, so greedy choices
// all equal one session run alone; continued, every choice continues the
// named session's history -- the others rebuilding it -- and the session holds
// choice 0's reply after, on stories15M. A choice that ignored the history
// would continue from nothing.
func TestSeveralChoicesContinueANamedSession(t *testing.T) {
	sm := structuredModels()[0]
	e, _, _ := loadedEngine(t, sm.file, sm.id, LoadOptions{})
	gen := func(o GenerateOptions) (map[int][]int32, []int) {
		got, prompts := map[int][]int32{}, []int{}
		o.Sampling, o.MaxTokens = &engine.Sampler{Temp: 0}, 6
		if err := e.Generate(context.Background(), o, func(ev Event) error {
			switch {
			case ev.Kind == EventToken && ev.Token.ID >= 0:
				got[ev.Choice] = append(got[ev.Choice], ev.Token.ID)
			case ev.Kind == EventFinished:
				prompts = append(prompts, ev.Finished.PromptTokens)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		return got, prompts
	}
	open := func() string {
		s, err := e.CreateSession(SessionOptions{ModelID: sm.id})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { e.CloseSession(s.id) })
		return s.id
	}
	text := func(s string) Prompt { return Prompt{Kind: PromptText, Text: s} }
	alone, named := open(), open()
	first, _ := gen(GenerateOptions{SessionID: alone, Prompt: text(sm.prompt)})
	got, _ := gen(GenerateOptions{SessionID: named, Prompt: text(sm.prompt), Seeds: []int64{1, 2}})
	if len(got) != 2 || !slices.Equal(got[0], first[0]) || !slices.Equal(got[1], first[0]) {
		t.Fatalf("two choices on a named session gave %v, the session alone %v", got, first[0])
	}
	want, wantPrompt := gen(GenerateOptions{SessionID: alone, Prompt: text(" and then"), Continue: true})
	got, prompts := gen(GenerateOptions{SessionID: named, Prompt: text(" and then"), Continue: true, Seeds: []int64{3, 4, 5}})
	for i := range 3 {
		if !slices.Equal(got[i], want[0]) || prompts[i] != wantPrompt[0] {
			t.Fatalf("continued choice %d is %v over %d prompt tokens, the session alone %v over %d", i, got[i],
				prompts[i], want[0], wantPrompt[0])
		}
	}
	after, _ := gen(GenerateOptions{SessionID: alone, Prompt: text(" so"), Continue: true})
	if g, _ := gen(GenerateOptions{SessionID: named, Prompt: text(" so"), Continue: true}); !slices.Equal(g[0], after[0]) {
		t.Fatalf("after the choices the named session continues to %v, the session alone to %v", g[0], after[0])
	}
	fresh, _ := gen(GenerateOptions{ModelID: sm.id, Prompt: text(" and then")})
	if slices.Equal(fresh[0], want[0]) {
		t.Fatalf("the prompt alone continues to %v, as the history does: the gate cannot see a lost history", fresh[0])
	}
	t.Logf("3 continued choices of %d tokens each equal the session alone; the prompt alone gives %v", len(want[0]), fresh[0])
}
