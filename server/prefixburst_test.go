package server

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"
)

// burst is a prefix burst held at the step loop's door: the single row is
// taken by a blocker, the requests queue behind it, and releasing the blocker
// with the loop widened admits them all in one pass, as a cold simultaneous
// burst would be.
type burst struct {
	t  *testing.T
	e  *Engine
	lp *stepLoop
	lm *LoadedModel
}

type burstResult struct {
	restored, finished int
	ids                []int32
	started            time.Time
	err                error
}

func (b *burst) run(ctx context.Context, e *Engine, prompt []int32, max int) burstResult {
	var r burstResult
	r.err = e.Generate(ctx, GenerateOptions{ModelID: "dev", Prompt: idsPrompt(prompt), MaxTokens: max, IgnoreEOS: true},
		func(ev Event) error {
			switch ev.Kind {
			case EventStarted:
				r.restored, r.started = ev.Started.Restored, time.Now()
			case EventToken:
				if ev.Token.ID >= 0 {
					r.ids = append(r.ids, ev.Token.ID)
				}
			case EventFinished:
				r.finished++
			}
			return nil
		})
	return r
}

func (b *burst) waitFor(what string, ok func() bool) {
	b.t.Helper()
	for end := time.Now().Add(30 * time.Second); !ok(); time.Sleep(time.Millisecond) {
		if time.Now().After(end) {
			b.t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func (b *burst) queued() int { b.lp.mu.Lock(); defer b.lp.mu.Unlock(); return len(b.lp.waiting) }

// newBurst opens a warm engine and a cold one, with the warm one's loop held
// shut and four rows wide: the burst's requests queue, and release lets the
// loop admit them in one pass.
func newBurst(t *testing.T, chunk int) (b *burst, cold *Engine, release func()) {
	cold, _, _ = hostBatchEngine(t, deviceModel, Config{NoMemCache: true})
	e, lm, _ := hostBatchEngine(t, deviceModel, Config{MemCacheBytes: 64 << 20})
	b = &burst{t: t, e: e, lp: lm.loop, lm: lm}
	b.lp.width, b.lp.promptChunk = 4, chunk
	return b, cold, holdGates(t, e, lm)
}

// TestAColdPrefixBurstHasOneProducer: three requests for one prompt and an
// unrelated one are admitted in the same pass. The first computes the prompt;
// the other two wait for it and restore what it sealed instead of computing it
// again; the unrelated request is not held, and starts before them. Every
// continuation is the cold engine's.
func TestAColdPrefixBurstHasOneProducer(t *testing.T) {
	b, cold, release := newBurst(t, 8)
	p := longPrompt(b.lm.m)
	unrelated := b.lm.m.Vocab.Encode("Ben had a red ball. He threw it over the fence and it rolled away. ", true)

	var wg sync.WaitGroup
	var same [3]burstResult
	var diff burstResult
	n := 0
	queue := func(f func()) {
		n++
		wg.Add(1)
		go func() { defer wg.Done(); f() }()
		b.waitFor("the request queued", func() bool { return b.queued() == n })
	}
	for i := range same {
		queue(func() { same[i] = b.run(context.Background(), b.e, p, 6) })
	}
	queue(func() { diff = b.run(context.Background(), b.e, unrelated, 6) })
	h0 := b.lp.stats.prefixHolds.Load()
	release()
	wg.Wait()

	want := b.run(context.Background(), cold, p, 6)
	wantDiff := b.run(context.Background(), cold, unrelated, 6)
	if want.err != nil || wantDiff.err != nil {
		t.Fatal(want.err, wantDiff.err)
	}
	producers := 0
	for i, r := range same {
		if r.err != nil || r.finished != 1 || !slices.Equal(r.ids, want.ids) {
			t.Fatalf("request %d: err %v, %d Finished, %v against the cold engine's %v", i, r.err, r.finished, r.ids, want.ids)
		}
		if r.restored == 0 {
			producers++
		}
		t.Logf("request %d restored %d of %d prompt positions", i, r.restored, len(p))
	}
	if diff.err != nil || diff.finished != 1 || !slices.Equal(diff.ids, wantDiff.ids) {
		t.Fatalf("the unrelated request: err %v, %v against %v", diff.err, diff.ids, wantDiff.ids)
	}
	if producers != 1 {
		t.Fatalf("%d requests computed the shared prompt, want exactly one", producers)
	}
	if h := b.lp.stats.prefixHolds.Load() - h0; h != 2 {
		t.Fatalf("%d row(s) held for the producer, want 2", h)
	}
	for i, r := range same {
		if r.restored != 0 && !diff.started.Before(r.started) {
			t.Fatalf("the unrelated request started after follower %d: it waited on the burst", i)
		}
	}
}

// TestACancelledProducerPromotesAFollower: the row computing a shared prompt
// is cancelled while it is still prompting; the rows held for it are not
// stranded -- one computes the prompt, and the answers are the cold engine's.
func TestACancelledProducerPromotesAFollower(t *testing.T) {
	b, cold, release := newBurst(t, 1)
	p := longPrompt(b.lm.m)

	var wg sync.WaitGroup
	prodCtx, cancelProd := context.WithCancel(context.Background())
	defer cancelProd()
	var prod burstResult
	var follow [2]burstResult
	wg.Add(1)
	go func() { defer wg.Done(); prod = b.run(prodCtx, b.e, p, 6) }()
	b.waitFor("the producer queued", func() bool { return b.queued() == 1 })
	for i := range follow {
		wg.Add(1)
		go func() { defer wg.Done(); follow[i] = b.run(context.Background(), b.e, p, 6) }()
		b.waitFor("a follower queued", func() bool { return b.queued() == 2+i })
	}
	h0 := b.lp.stats.prefixHolds.Load()
	release()
	b.waitFor("the followers held", func() bool { return b.lp.stats.prefixHolds.Load()-h0 == 2 })
	cancelProd()
	wg.Wait()

	want := b.run(context.Background(), cold, p, 6)
	if want.err != nil {
		t.Fatal(want.err)
	}
	for i, r := range follow {
		if r.err != nil || r.finished != 1 || !slices.Equal(r.ids, want.ids) {
			t.Fatalf("follower %d: err %v, %d Finished, %v against the cold engine's %v", i, r.err, r.finished, r.ids, want.ids)
		}
	}
	t.Logf("producer ended with %v after %d tokens; followers restored %d and %d", prod.err, len(prod.ids),
		follow[0].restored, follow[1].restored)
}
