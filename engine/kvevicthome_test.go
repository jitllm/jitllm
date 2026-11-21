package engine

import (
	"math"
	"slices"
	"testing"
)

// TestEvictedHistoryComesHome holds relocation to paging: a block whose
// history on the card is partly at home -- pages the squeeze sent home, or a
// windowed layer's pages released behind its window over a long decode --
// comes home at a live position with every position of it, entries too, and
// the host runs on from there with the host arm's answer.
func TestEvictedHistoryComesHome(t *testing.T) {
	for _, name := range evictModels() {
		t.Run(name, func(t *testing.T) {
			eg := newEvictGateOf(t, name)
			t.Run("released", func(t *testing.T) { evictedHistoryComesHome(t, eg, nil, nil) })
			t.Run("evicted", func(t *testing.T) {
				free := eg.run(t, nil, nil)
				evictedHistoryComesHome(t, eg, &free.squeeze, free.logits)
			})
		})
	}
}

// evictedHistoryComesHome brings every block home at the decode's middle
// step, or squeezed by sq at the first step a page has gone home, free the
// unconstrained card's logits that squeeze was measured on.
func evictedHistoryComesHome(t *testing.T, eg *evictGate, sq *evictSqueeze, free [][]float32) {
	g, st, _, logits := eg.open(t, sq, nil)
	// Before the move the card is in its own band of the host; after it the
	// host runs on the card's history, inside that band or near it.
	band, dev := 0.0, 0.0
	for i, lg := range logits {
		band = max(band, logitNMSE(lg, eg.host[i]))
		if free != nil {
			dev = max(dev, logitNMSE(lg, free[i]))
		}
	}
	moved, worst, at := 0, 0.0, 0
	var after []float64
	for i, id := range eg.ids[eg.mid():] {
		step := eg.mid() + i
		if moved == 0 && (sq == nil || g.Stats().KVEvictions > 0) {
			if n := st.SetGPULayers(0); n != 0 || st.devCount() != 0 {
				t.Fatalf("step %d: %d block(s) stayed on the card (%s)", step, st.devCount(), g.Err())
			}
			moved = step
		}
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatalf("step %d: %v (%s)", step, err, g.Err())
		}
		e := logitNMSE(lg, eg.host[step])
		if moved == 0 {
			band = max(band, e)
			if free != nil {
				dev = max(dev, logitNMSE(lg, free[step]))
			}
		} else {
			after = append(after, e)
			if !(e <= worst) {
				worst, at = e, step
			}
		}
	}
	if moved == 0 {
		t.Fatalf("no page was sent home in %d steps: nothing came home from home", len(eg.ids))
	}
	s := g.Stats()
	t.Logf("every block home at step %d (%d layer page(s) sent home, %d streamed passes): worst logit NMSE "+
		"%.3e against the host at step %d; the card read %.3e before", moved, s.KVEvictions, s.KVStreamPasses, worst, at, band)
	// Squeezed, the history the card made is as far from the host's as the
	// eviction gate holds its evicting arm (hostBound): the unconstrained
	// card's distance from the host and the squeezed card's from it, both
	// measured here -- a mixture's int8 activations move its routing.
	bound := max(evictNMSE, band)
	if free != nil {
		ctl := 0.0
		for i := range min(len(free), len(eg.host)) {
			ctl = max(ctl, logitNMSE(free[i], eg.host[i]))
		}
		bound = max(bound, math.Pow(math.Sqrt(ctl)+math.Sqrt(dev), 2))
	}
	slices.Sort(after)
	if p90 := after[len(after)*9/10]; !evictHeld(eg.m, p90, worst, bound) {
		t.Fatalf("logit NMSE %.3e against the host at step %d after every block came home at step %d, %.3e at the 90th percentile, past %.3e",
			worst, at, moved, p90, bound)
	}
}
