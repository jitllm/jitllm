package tier

import (
	"sync"
	"testing"
	"time"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestBackgroundTrialWaitsForSubmissionsInFlight: the background split tuner
// times no candidate while a submission is in flight. A trial holds g.mu and
// ends in a context-wide synchronise, which CUDA holds off while a queue is
// recording; a recording submission that takes g.mu for a build (subLock)
// then waits for the trial, and the trial for it. A server's warm-up prefill
// hung there beside the tuner on a T4. A submission's ticket is held here
// across the tuner's quiet period, and every trial must see none in flight;
// the trials must still run once it is given back. The tuner runs as a
// session of its own: the queuing session's view says whether that session's
// submission is running, which is not the tuner's.
func TestBackgroundTrialWaitsForSubmissionsInFlight(t *testing.T) {
	const rows, k = 4096, 4096
	var mu sync.Mutex
	trials, busy := 0, 0
	oldBG, oldID := bgTrial, binaryID
	t.Cleanup(func() { bgTrial, binaryID = oldBG, oldID })
	bgTrial = func(g *devTier, _, _ backend.Kernel, _ *resident, _, _, _ backend.Buf, _, split int, grp bool) (float64, bool) {
		// Called with g.mu held: the tickets are what is in flight.
		mu.Lock()
		defer mu.Unlock()
		trials++
		if len(g.tickets) > 0 {
			busy++
		}
		return float64(split), true
	}
	binaryID = func() string { return "revA" }
	g, err := New([]Slot{{Dev: idDev{&fakeDev{name: "card"}, "aa", "cuda12040"}, Bytes: 1 << 30}},
		WithTuneCache(t.TempDir()), WithLazyTune(true))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer g.Close()
	d := g.devs[0]

	d.mu.Lock()
	ticket := d.beginSub()
	d.tuneSplit(kernels.Q4_0, rows, k, &resident{ok: true})
	running := d.tuneRunning
	// As layersOnce marks its session for the submission it lets g.mu go
	// for: the tuner is not that session, and must not read this as its own.
	d.inSub = true
	d.mu.Unlock()
	if !running {
		t.Fatalf("the shape queued no background tune; the gate cannot see a trial")
	}
	// Well past tuneQuiet: an unguarded tuner has started its trials by now.
	time.Sleep(10 * tuneQuiet)
	mu.Lock()
	early := trials
	mu.Unlock()

	d.mu.Lock()
	d.inSub = false
	d.endSub(ticket)
	d.mu.Unlock()
	d.tuneWG.Wait()

	mu.Lock()
	defer mu.Unlock()
	if busy > 0 || early > 0 {
		t.Fatalf("%d of %d background trials ran with a submission in flight (%d before it completed)", busy, trials, early)
	}
	if trials == 0 {
		t.Fatalf("no background trial ran after the submission completed")
	}
}
