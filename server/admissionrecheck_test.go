package server

import (
	"context"
	"slices"
	"sync"
	"testing"
)

// TestAdmissionRechecksTheMemCacheForAQueuedColdRow: two requests for the same
// prompt arrive while the step loop's only row cannot be had, so both look the
// memory cache up before they are queued and both miss. The first admitted
// runs the prompt and seals it; the second, admitted after, must find what the
// first left rather than run the whole prompt again, and answer as the cold
// engine does.
func TestAdmissionRechecksTheMemCacheForAQueuedColdRow(t *testing.T) {
	b, cold, release := newBurst(t, 64)
	b.lp.width = 1
	p := longPrompt(b.lm.m)

	var wg sync.WaitGroup
	var pair [2]burstResult
	for i := range pair {
		wg.Add(1)
		go func() { defer wg.Done(); pair[i] = b.run(context.Background(), b.e, p, 6) }()
		b.waitFor("the request queued", func() bool { return b.queued() == i+1 })
	}
	release()
	wg.Wait()

	want := b.run(context.Background(), cold, p, 6)
	if want.err != nil {
		t.Fatal(want.err)
	}
	for i, r := range pair {
		if r.err != nil || r.finished != 1 || !slices.Equal(r.ids, want.ids) {
			t.Fatalf("request %d: err %v, %d Finished, %v against the cold engine's %v", i, r.err, r.finished, r.ids, want.ids)
		}
	}
	lo, hi := min(pair[0].restored, pair[1].restored), max(pair[0].restored, pair[1].restored)
	t.Logf("the pair restored %d and %d of %d prompt positions", pair[0].restored, pair[1].restored, len(p))
	if lo != 0 {
		t.Fatalf("both ran after the cache held the prompt (%d, %d restored): the burst never missed", pair[0].restored, pair[1].restored)
	}
	if hi == 0 {
		t.Fatalf("the second request restored nothing: admission did not look at the cache again")
	}
}
