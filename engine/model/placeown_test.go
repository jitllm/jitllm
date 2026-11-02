package model

import (
	"sync"
	"testing"
)

// TestConcurrentPlacementGivesEachStateItsOwnHistory: instances attaching to one
// tier at once must each come away with their own history on every block they
// placed, so each one's first prompt runs where it was placed.
//
// A block takes a session's history when it is offered -- a linear block's
// recurrent state, an attention block's KV -- and the device files it under
// whichever session is current. The offers went through the shared tier
// (Instance.ldCand) rather than the instance's own view, so the current session was
// whichever had made the last call: with two instances placing at once, a block
// found the OTHER session's history already there and gave this one none. The
// placement still counted every block, and the instance's first prompt then
// failed on the device ("block 0 has no recurrent state for session N") and
// demoted every block to the host -- correct tokens, from the host, which is
// how a server creating sessions per request ran most of them there.
//
// The assertion is per instance after its first prompt: every block still on the
// device and no demotion. Against offers made through the shared tier, one of
// each pair of instances demotes in nearly every round.
func TestConcurrentPlacementGivesEachStateItsOwnHistory(t *testing.T) {
	m := openStepHybrid(t)
	defer m.Close()
	ids := m.Vocab.Encode("The capital of France is", true)
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			g := stepTier(t, m, dev, false)
			defer g.Close()
			const rounds, n = 4, 3
			for round := range rounds {
				ss := make([]*Instance, n)
				for i := range ss {
					ss[i] = m.NewInstance(64)
				}
				var wg sync.WaitGroup
				for _, s := range ss {
					wg.Add(1)
					go func() { defer wg.Done(); s.SetDevice(g) }()
				}
				wg.Wait()
				for i, s := range ss {
					placed := s.GPULayers()
					if placed != m.Cfg.NLayer {
						t.Skipf("NO DEVICE ROOM: state %d placed %d of %d blocks (%s)", i, placed, m.Cfg.NLayer, g.Err())
					}
					if _, err := s.Prefill(ids); err != nil {
						t.Fatal(err)
					}
					if s.DeviceDemotions() != 0 || s.GPULayers() != placed {
						t.Fatalf("round %d, state %d: its first prompt demoted it (%d of %d blocks left on the "+
							"device): %s", round, i, s.GPULayers(), placed, g.Err())
					}
				}
				for _, s := range ss {
					s.Close()
				}
			}
			t.Logf("%d rounds of %d instances placed at once kept every block through their first prompt", rounds, n)
		})
	}
}
