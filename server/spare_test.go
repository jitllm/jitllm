package server

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestASpareSessionServesTheNextRequest: a model_id request's session is kept,
// reset, and the next model_id request runs on it under an id of its own --
// its reply a fresh session's, whatever the first one's prompt left, its
// sampler seed its own -- on the
// CPU and a card. The model holds no more sessions, open and spare, than it
// was loaded for, two requests at once included, and an unload without force
// closes its spares rather than refusing. Against a server that opens every
// request's session anew, the second request runs on another session.
func TestASpareSessionServesTheNextRequest(t *testing.T) {
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
			spares := func() []*Session {
				e.mu.RLock()
				defer e.mu.RUnlock()
				return slices.Clone(lm.spare)
			}
			type reply struct {
				ids []int32
				on  *Session
			}
			run := func(o GenerateOptions) reply {
				var r reply
				o.Sampling, o.MaxTokens = &engine.Sampler{Temp: 0}, 6
				if err := e.Generate(context.Background(), o, func(ev Event) error {
					switch {
					case ev.Kind == EventStarted:
						s, err := e.Session(ev.Started.SessionID)
						if err != nil {
							t.Fatalf("the request's session %q: %v", ev.Started.SessionID, err)
						}
						r.on = s
					case ev.Kind == EventToken && ev.Token.ID >= 0:
						r.ids = append(r.ids, ev.Token.ID)
					}
					return nil
				}); err != nil {
					t.Fatal(err)
				}
				return r
			}
			text := func(s string) Prompt { return Prompt{Kind: PromptText, Text: s} }
			fresh := func(p Prompt) []int32 {
				s, err := e.CreateSession(SessionOptions{ModelID: "m"})
				if err != nil {
					t.Fatal(err)
				}
				defer e.CloseSession(s.id)
				return run(GenerateOptions{SessionID: s.id, Prompt: p}).ids
			}
			want := fresh(text(" and then the dog"))

			first := run(GenerateOptions{ModelID: "m", Prompt: text("Once upon a time")})
			kept := spares()
			if len(kept) != 1 || kept[0] != first.on {
				t.Fatalf("after a model_id request the model keeps %d spares, the request's session among them %v", len(kept),
					len(kept) == 1 && kept[0] == first.on)
			}
			firstID, firstSeed := first.on.id, first.on.sampling.Seed
			second := run(GenerateOptions{ModelID: "m", Prompt: text(" and then the dog")})
			if second.on != first.on {
				t.Fatal("the second model_id request opened a session of its own beside the spare")
			}
			if second.on.id == firstID {
				t.Fatalf("the spare served the second request under the first one's id %q", firstID)
			}
			if second.on.sampling.Seed == firstSeed {
				t.Fatalf("the spare served the second request with the first one's sampler seed %d", firstSeed)
			}
			if !slices.Equal(second.ids, want) {
				t.Fatalf("on the spare the prompt continues to %v, on a fresh session to %v: the first request's history stayed", second.ids, want)
			}

			var wg sync.WaitGroup
			for range 2 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					run(GenerateOptions{ModelID: "m", Prompt: text("Once upon a time")})
				}()
			}
			wg.Wait()
			e.mu.RLock()
			held := len(lm.sessions) + len(lm.spare)
			e.mu.RUnlock()
			if held > max(lm.open.Sessions, 1) {
				t.Fatalf("after two requests at once the model holds %d sessions, open and spare; it was loaded for %d", held, max(lm.open.Sessions, 1))
			}
			if _, err := e.UnloadModel("m", false); err != nil {
				t.Fatalf("an unload with only spares left: %v", err)
			}
			if s := spares(); len(s) != 0 {
				t.Fatalf("the unload left %d spares", len(s))
			}
			t.Logf("%s: the spare took the second request, %v as a fresh session gives", arm, second.ids)
		})
	}
}
