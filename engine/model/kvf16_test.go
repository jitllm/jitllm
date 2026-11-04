package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestKVF16IsSelectedAndRuns checks the binary16 cache end to end: that it is
// actually CHOSEN, that it halves the allocation, and that the logits land in
// the band TestKVPrecisionCost measured for rounding alone.
//
// The first assertion is the one that matters: a cache that quietly stayed f32
// would pass every numeric check. KVIsF16 reports what the state allocated.
func TestKVF16IsSelectedAndRuns(t *testing.T) { kvF16IsSelectedAndRuns(t) }

// kvF16IsSelectedAndRuns is TestKVF16IsSelectedAndRuns with opts on every model it opens.
func kvF16IsSelectedAndRuns(t *testing.T, opts ...Option) {
	paths := modelFiles()
	var ran int
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil || fi.Size() > 512<<20 {
			continue
		}
		dst, ok := sweepJLM(t, p)
		if !ok {
			continue
		}
		m, err := Open(dst, opts...)
		if err != nil {
			continue
		}
		name := filepath.Base(p)
		ran++

		// Sixteen positions, every one compared: a storage bug reaches every
		// position after the one it corrupts, while a router that thresholds
		// (phimoe's sparsemixer masks an expert when its gap to the best
		// passes 2*jitter) can take one decision the other way on a rounding
		// and move that position alone -- synth-phimoe does at position 7 of
		// these ids and synth-llama4-q8's top-1 router at 2 and 13, 1e-3 on
		// either side.
		ids := []int32{7, 19, 33, 51, 64, 12, 88, 3, 5, 9, 41, 77, 2, 63, 18, 30}
		for i := range ids {
			ids[i] %= int32(m.Cfg.NVocab)
		}
		run := func(f16 bool) (*Instance, [][]float32) {
			m.SetKVF16(f16)
			defer m.SetKVF16(false)
			s := m.NewInstance(32)
			var outs [][]float32
			for _, id := range ids {
				out, err := s.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				outs = append(outs, append([]float32(nil), out...))
			}
			return s, outs
		}
		s32, a := run(false)
		s16, b := run(true)

		if s32.KVIsF16() {
			t.Errorf("%s: the f32 state reports an f16 cache", name)
		}
		if !s16.KVIsF16() {
			t.Errorf("%s: asked for f16 and got f32 -- head dim %d has no attention kernel?",
				name, m.Cfg.HeadDim)
		}
		// Half the bytes, checked per page: Bytes() counts what has been
		// committed, which at construction is nothing.
		if got, want := s16.kv.layers[0].pp*4, s32.kv.layers[0].pp*4/2; got != want {
			t.Errorf("%s: f16 page is %d bytes, want %d", name, got, want)
		}
		// Rounding alone measured 0.20 to 0.94 on real models; a storage bug
		// would be orders above, at every position from the one it hit: a
		// stored row is read by every later position. A model can also amplify
		// the rounding at a few positions and pass it no further -- a router's
		// decision, or SmolVLM-256M's layer-11 FFN on the SSE tier, where one
		// neuron's output moves 25% for a 4% input change at positions 8 and 9
		// and position 10 is back at 0.7. So the verdict is persistence: more
		// than a quarter of the positions past the bound, three in a row, or
		// the last one (nothing after it to show whether it carried), fail.
		var d float64
		var over []int
		for p := range a {
			var dp float64
			for i := range a[p] {
				dp = math.Max(dp, math.Abs(float64(a[p][i]-b[p][i])))
			}
			if dp > 3 {
				over = append(over, p)
				continue
			}
			d = math.Max(d, dp)
		}
		switch {
		case carried(over, len(ids)):
			t.Errorf("%s: positions %v past a max|dlogit| of 3 between an f32 and an f16 cache -- far "+
				"past what rounding explains", name, over)
		case len(over) > 0:
			t.Logf("%s: positions %v past a max|dlogit| of 3 and the rest in band: an amplified "+
				"rounding that did not carry, not the cache", name, over)
		}
		t.Logf("%-30s f16 cache selected, %d -> %d bytes, max|dlogit| %.4e",
			name, s32.kv.layers[0].pp*4, s16.kv.layers[0].pp*4, d)

		// And the tuner's own answer, unforced: it takes f16 only when the f16
		// kernel costs no more, so amd64 and arm64 can differ.
		m.ClearKVF16()
		auto := m.NewInstance(32)
		t.Logf("%-30s unforced: KVWidthPaysOff -> f16=%v", name, auto.KVIsF16())
		auto.Close()
		m.SetKVF16(false)
		s32.Close()
		s16.Close()
		m.Close()
	}
	if ran == 0 {
		t.Skip("no model under 512 MiB present")
	}
}
