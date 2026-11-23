package server

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestAMaxTokensEndLeavesItsLastTokenUnrun: a generate that ends at its
// max_tokens keeps its last token and does not run it -- the engine's history
// is a position short of the Position it reports -- and a continue runs it
// first, so the continued reply and its log-probabilities are a fresh
// session's over the whole history, on the CPU's one-at-a-time loop and a
// card's step loop. A history that lost the token continues differently, or
// the gate could not see it lost.
func TestAMaxTokensEndLeavesItsLastTokenUnrun(t *testing.T) {
	const gen = 6
	for _, arm := range []string{"cpu", "card"} {
		t.Run(arm, func(t *testing.T) {
			devs := []string{"cpu"}
			if arm == "card" {
				if !testdevices.Selected() {
					t.Skip("JITLLM_TEST_DEVICES selects no card: the card arm runs where one is given")
				}
				devs = testdevices.Take(t, 1, "cuda:0")
			}
			e, lm, _ := loadedEngine(t, deviceModel, "m", LoadOptions{DeviceIDs: devs})
			if (arm == "card") != (lm.loop != nil) {
				t.Fatalf("%s: step loop %v: the arm does not run the loop it names", arm, lm.loop != nil)
			}
			steps := func() int64 {
				if lm.loop == nil {
					return 0
				}
				return lm.loop.stats.steps.Load()
			}
			type reply struct {
				ids []int32
				lps []float32
				fin *Finished
			}
			run := func(o GenerateOptions) reply {
				var r reply
				o.Sampling, o.MaxTokens, o.Logprobs = &engine.Sampler{Temp: 0}, gen, true
				if err := e.Generate(context.Background(), o, func(ev Event) error {
					switch {
					case ev.Kind == EventToken && ev.Token.ID >= 0:
						r.ids = append(r.ids, ev.Token.ID)
						r.lps = append(r.lps, ev.Token.Logprob.Logprob)
					case ev.Kind == EventFinished:
						r.fin = ev.Finished
					}
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				if len(r.ids) != gen || r.fin == nil || r.fin.Reason != FinishMaxTokens {
					t.Fatalf("%d tokens, finished %+v: want %d ending at max_tokens", len(r.ids), r.fin, gen)
				}
				return r
			}
			session := func() *Session {
				s, err := e.CreateSession(SessionOptions{ModelID: "m"})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { e.CloseSession(s.id) })
				return s
			}
			idsOf := func(text string) Prompt {
				return Prompt{Kind: PromptIDs, IDs: lm.m.Vocab.EncodeSpecial(text, true)}
			}
			p, c := idsOf("Once upon a time"), idsOf(" and then")

			s := session()
			s0 := steps()
			first := run(GenerateOptions{SessionID: s.id, Prompt: p})
			if want := len(p.IDs) + gen; first.fin.Position != want || s.ms.Pos() != want-1 || int(s.snapPos.Load()) != want {
				t.Fatalf("after %d prompt tokens and %d generated: Position %d, the engine's %d, the session's %d; want %d, %d, %d",
					len(p.IDs), gen, first.fin.Position, s.ms.Pos(), s.snapPos.Load(), want, want-1, want)
			}
			if arm == "card" && steps() == s0 {
				t.Fatal("the generate took no step of the loop: the card arm ran the one-at-a-time path")
			}
			cont := run(GenerateOptions{SessionID: s.id, Prompt: c, Continue: true})
			if cont.fin.PromptTokens != len(c.IDs) {
				t.Fatalf("the continue reports %d prompt tokens, its prompt is %d", cont.fin.PromptTokens, len(c.IDs))
			}

			whole := slices.Concat(p.IDs, first.ids, c.IDs)
			fresh := run(GenerateOptions{SessionID: session().id, Prompt: Prompt{Kind: PromptIDs, IDs: whole}})
			if !slices.Equal(cont.ids, fresh.ids) || cont.fin.Position != fresh.fin.Position {
				t.Fatalf("continued: %v at %d; the whole history fresh: %v at %d", cont.ids, cont.fin.Position,
					fresh.ids, fresh.fin.Position)
			}
			worst := 0.0
			for i := range cont.lps {
				worst = max(worst, math.Abs(float64(cont.lps[i]-fresh.lps[i])))
			}
			if worst > 0.05 {
				t.Fatalf("continued log-probabilities %v, the whole history fresh %v: %.4f apart", cont.lps, fresh.lps, worst)
			}
			lost := slices.Concat(p.IDs, first.ids[:gen-1], c.IDs)
			if l := run(GenerateOptions{SessionID: session().id, Prompt: Prompt{Kind: PromptIDs, IDs: lost}}); slices.Equal(l.ids, cont.ids) {
				t.Fatalf("a history without its last token continues to %v as well: the gate cannot see the token lost", l.ids)
			}
			t.Logf("%s: %v then %v, the whole history fresh alike (log-probabilities %.2g apart)", arm, first.ids, cont.ids, worst)
		})
	}
}
