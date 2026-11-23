package server

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/jitllm/jitllm/engine"
	"github.com/jitllm/jitllm/internal/testdevices"
)

// TestASpareSessionServesTheNextRequest: a model_id request's session is
// kept, reset -- it holds no history while it waits -- and the next model_id
// request runs on it under an id and a sampler seed of its own, its reply a
// fresh session's, on the CPU and a card, with the memory cache and without
// (where a session shares no pages, and only the reset empties it). The
// model holds no more sessions, open and spare, than it was loaded for, two
// requests at once included, and an unload without force closes its spares
// rather than refusing. Against a server that opens every request's session
// anew, the second request runs on another session; against a spare kept
// with its history, it holds the first request's.
func TestASpareSessionServesTheNextRequest(t *testing.T) {
	for _, arm := range []string{"cpu/cache", "cpu/nocache", "card/cache", "card/nocache"} {
		t.Run(arm, func(t *testing.T) {
			devs := []string{"cpu"}
			if strings.HasPrefix(arm, "card") {
				if !testdevices.Selected() {
					t.Skip("JITLLM_TEST_DEVICES selects no card: the card arm runs where one is given")
				}
				devs = testdevices.Take(t, 1, "cuda:0")
			}
			path := modelPath(t, deviceModel)
			e := New(admitConfig(Config{ModelDir: filepath.Dir(path), Probe: oneCardProbe, Version: "test", NoOnDemand: true,
				NoMemCache: strings.HasSuffix(arm, "nocache")}))
			t.Cleanup(e.Close)
			lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "m", DeviceIDs: devs})
			if err != nil {
				t.Fatal(err)
			}
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
			if pos, kv := first.on.ms.Pos(), first.on.snapKV.Load(); pos != 0 || kv != 0 {
				t.Fatalf("a spare holds %d positions of history, %d bytes of it: it is kept reset", pos, kv)
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
