package server

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestPreemptionKeepsEveryAnswer: a KV budget that fits two sessions' histories,
// three sessions generating in turn, each continuing its own sequence three
// times. Every turn past the second parks the least recently stepped other
// session, and every session's tokens equal its solo run's, generated alone
// with no budget. Parks, resumes and pages out and in are counted, the pages
// in equal the pages out, and no goroutine outlives the engine.
func TestPreemptionKeepsEveryAnswer(t *testing.T) {
	preemptionKeepsEveryAnswer(t, Config{}, nil)
}

// TestAParkedSessionResumesFromTheMemCache is the same three sessions with the
// model's memory cache bounded to one byte, so every page it holds unpinned
// goes as soon as another arrives: a parked session's pages survive only
// because the view it parked into pins them (model.MemStore.Pinned), and every
// answer is still its solo run's, token for token. The cache is seen holding
// pinned pages for the sessions still parked and evicting everything else.
func TestAParkedSessionResumesFromTheMemCache(t *testing.T) {
	preemptionKeepsEveryAnswer(t, Config{MemCacheBytes: 1}, func(e *Engine, lm *LoadedModel, parked int) {
		st := e.MemCacheStats(lm).Store
		if st.Evicted == 0 {
			t.Fatalf("the one-byte memory cache evicted nothing (%+v): the bound never bit", st)
		}
		if parked > 0 && st.Pinned == 0 {
			t.Fatalf("%d session(s) parked and the memory cache pins nothing (%+v): they did not park into it", parked, st)
		}
		t.Logf("memory cache: %+v", st)
	})
}

func preemptionKeepsEveryAnswer(t *testing.T, cfg Config, check func(e *Engine, lm *LoadedModel, parked int)) {
	base := runtime.NumGoroutine()
	const rounds, perTurn, maxSeq = 5, 70, 512
	prompts := map[string]string{
		"a": "Once upon a time",
		"b": "The little dog",
		"c": "One day a girl",
	}
	order := []string{"a", "b", "c"}
	ctx := context.Background()

	turn := func(e *Engine, id string, cont bool) []int32 {
		t.Helper()
		var out []int32
		p := Prompt{Kind: PromptText, Text: prompts[id]}
		if cont {
			// A continue feeds one more token, the same in both runs.
			p = Prompt{Kind: PromptIDs, IDs: []int32{13}}
		}
		err := e.Generate(ctx, GenerateOptions{SessionID: id, Prompt: p, MaxTokens: perTurn,
			Continue: cont, IgnoreEOS: true}, func(ev Event) error {
			if ev.Kind == EventToken && ev.Token.ID >= 0 {
				out = append(out, ev.Token.ID)
			}
			return nil
		})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		return out
	}

	run := func(budgeted bool, only string) map[string][]int32 {
		path := modelPath(t, smallModel)
		c := cfg
		c.ModelDir, c.Probe, c.Version = filepath.Dir(path), oneCardProbe, "test"
		e := New(admitConfig(c))
		defer e.Close()
		lm, err := e.LoadModel(LoadOptions{Path: path, ModelID: "small"})
		if err != nil {
			t.Fatal(err)
		}
		got := map[string][]int32{}
		ids := order
		if only != "" {
			ids = []string{only}
		}
		for _, id := range ids {
			if _, err := e.CreateSession(SessionOptions{ModelID: "small", SessionID: id, MaxSeq: maxSeq}); err != nil {
				t.Fatal(err)
			}
		}
		if budgeted {
			// Two sessions' full histories, measured on a probe Instance.
			st := e.sessions["a"].in()
			per := st.PositionBytes()
			if per == 0 {
				t.Fatal("a position costs no bytes: the budget would be nothing")
			}
			if err := e.SetKVBudget("small", 2*uint64(rounds*(perTurn+8))*per); err != nil {
				t.Fatal(err)
			}
		}
		for r := range rounds {
			for _, id := range ids {
				got[id] = append(got[id], turn(e, id, r > 0)...)
			}
		}
		if budgeted {
			parks, resumes, out, in := e.PreemptCounts()
			t.Logf("parks %d, resumes %d, pages out %d, in %d", parks, resumes, out, in)
			if parks == 0 || resumes == 0 || out == 0 || in == 0 {
				t.Fatalf("parks %d, resumes %d, pages out %d, in %d: the budget never made a page go and come back",
					parks, resumes, out, in)
			}
			// The sessions still parked hold the pages not yet back.
			var held int64
			for _, id := range ids {
				s := e.sessions[id]
				if s.Parked() {
					held += s.ms.ParkedOut - s.ms.ParkedIn
				}
			}
			if in+held != out {
				t.Fatalf("%d pages out, %d in and %d still parked", out, in, held)
			}
			if check != nil {
				parked := 0
				for _, id := range ids {
					if e.sessions[id].Parked() {
						parked++
					}
				}
				check(e, lm, parked)
			}
		}
		return got
	}

	got := run(true, "")
	for _, id := range order {
		want := run(false, id)[id]
		if !slices.Equal(got[id], want) {
			t.Fatalf("session %s under preemption: %v\nsolo: %v", id, got[id], want)
		}
		if len(want) < rounds*perTurn/2 {
			t.Fatalf("session %s generated %d tokens: too few to have sealed a page", id, len(want))
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for runtime.NumGoroutine() > base && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if n := runtime.NumGoroutine(); n > base {
		t.Fatalf("%d goroutines after four engines closed, %d before; the stacks left:\n%s", n, base, goroutineStacks())
	}
}

// goroutineStacks is every goroutine's stack but the caller's, each distinct
// one with its count, for a gate that finds goroutines left behind.
func goroutineStacks() string {
	buf := make([]byte, 1<<22)
	buf = buf[:runtime.Stack(buf, true)]
	counts := map[string]int{}
	var order []string
	for _, g := range strings.Split(string(buf), "\n\n")[1:] {
		lines := strings.Split(g, "\n")
		// The header names the goroutine; its frames are what it runs.
		key := strings.Join(lines[1:min(len(lines), 9)], "\n")
		if counts[key] == 0 {
			order = append(order, key)
		}
		counts[key]++
	}
	var b strings.Builder
	for _, k := range order {
		fmt.Fprintf(&b, "%d x\n%s\n\n", counts[k], k)
	}
	return b.String()
}
