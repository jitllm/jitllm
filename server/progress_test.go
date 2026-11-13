package server

import (
	"context"
	"testing"
)

// TestProgressFollowsThePrompt: a generate that asks for its prompt's
// progress gets an EventProgress when the prompt begins, one at each chunk's
// start after it and one when the prompt is in, all before Started: alone,
// where the CPU device runs a prompt 64 rows a chunk, and as a row of the step
// loop fed 64 tokens a step. Each names the prompt's positions and what the
// memory cache restored, which the first counts done; done rises by at most a
// chunk to all of them. The prompt run again is restored, and says so from
// its first event. A generate that does not ask gets none.
func TestProgressFollowsThePrompt(t *testing.T) {
	const chunk = 64
	for _, c := range []struct {
		name string
		cfg  Config
	}{
		{"alone", Config{MaxBatchRows: 1}},
		{"row", Config{PromptChunk: chunk}},
	} {
		t.Run(c.name, func(t *testing.T) {
			e, lm := ttftEngine(t, smallModel, c.cfg, LoadOptions{})
			p := longPrompt(lm.m)
			run := func(progress bool) ([]Progress, *Started) {
				var got []Progress
				var st *Started
				err := e.Generate(context.Background(), GenerateOptions{ModelID: "m", Prompt: idsPrompt(p), MaxTokens: 4,
					Progress: progress}, func(ev Event) error {
					switch ev.Kind {
					case EventProgress:
						if st != nil {
							t.Errorf("a progress event after Started: %+v", *ev.Progress)
						}
						got = append(got, *ev.Progress)
					case EventStarted:
						st = ev.Started
					}
					return nil
				})
				if err != nil || st == nil {
					t.Fatalf("generate: %v (started %v)", err, st != nil)
				}
				if st.Batched != (c.name == "row") {
					t.Fatalf("the generate ran batched=%v", st.Batched)
				}
				return got, st
			}
			check := func(arm string, got []Progress, st *Started) {
				if len(got) == 0 {
					t.Fatalf("%s: no progress", arm)
				}
				if got[0].Done != got[0].Cached || got[0].Cached != st.Restored {
					t.Fatalf("%s: the first progress %+v; want what the cache restored (%d) done", arm, got[0], st.Restored)
				}
				for i, g := range got {
					if g.Total != st.PromptTokens || g.Total != len(p) || g.Cached != got[0].Cached {
						t.Fatalf("%s: progress %d is %+v; want %d positions, %d cached", arm, i, g, len(p), got[0].Cached)
					}
					if i == 0 {
						continue
					}
					if g.Done <= got[i-1].Done || g.Done-got[i-1].Done > chunk || g.Elapsed < got[i-1].Elapsed {
						t.Fatalf("%s: progress %d moved from %+v to %+v; want a chunk of at most %d at a time", arm, i,
							got[i-1], g, chunk)
					}
				}
				if last := got[len(got)-1]; last.Done != last.Total {
					t.Fatalf("%s: the last progress %+v; want the whole prompt done", arm, last)
				}
			}
			cold, st := run(true)
			check("cold", cold, st)
			if st.Restored != 0 || len(cold) < 1+(len(p)+chunk-1)/chunk {
				t.Fatalf("cold: %d restored, %d progress events for %d positions", st.Restored, len(cold), len(p))
			}
			warm, wst := run(true)
			check("warm", warm, wst)
			if wst.Restored == 0 {
				t.Fatalf("warm: the prompt run again restored nothing")
			}
			if got, _ := run(false); len(got) != 0 {
				t.Fatalf("a generate that did not ask got %d progress events", len(got))
			}
		})
	}
}
