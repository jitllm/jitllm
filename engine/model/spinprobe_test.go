//go:build jitllmbench && amd64 && linux

// This file is a measurement instrument, not a gate, so it is opt-in by build
// tag rather than by a skip in every default run:
//
//	go test -tags jitllmbench -run <Name> ./<pkg>
//
// JITLLM_* variables select its parameters.

package model

import (
	"os"
	"testing"
	"time"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/engine/sched"
)

// TestTowerSpinProbe prices one tower encode against the worker spin budget, to
// test whether run-to-run bimodality in encode time is the park knife-edge: a
// region arriving while workers spin costs a sequence load, one arriving after
// they park costs a wake and a scheduler round trip.
func TestTowerSpinProbe(t *testing.T) {
	if _, err := os.Stat(towerPath()); err != nil {
		t.Skip("no mmproj present")
	}
	for _, spin := range []time.Duration{0, 250 * time.Microsecond, 20 * time.Millisecond} {
		for i := 0; i < 3; i++ {
			func() {
				vlm, tw := openTower(t, WithJITOptions(nn.WithSched(sched.WithSpin(spin))))
				defer vlm.Close()
				s := tw.testState()
				defer s.Close()
				px := towerCb(tw.Cfg.ImageSz)
				t0 := time.Now()
				if _, err := s.Encode(px); err != nil {
					t.Fatal(err)
				}
				d := time.Since(t0)
				par, ser := s.Regions()
				t.Logf("spin=%-8v run=%d  %7.2fs  regions parallel=%d serial=%d",
					spin, i, d.Seconds(), par, ser)
			}()
		}
	}
}
