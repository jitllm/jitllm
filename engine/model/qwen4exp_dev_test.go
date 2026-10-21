package model

import (
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestQwen4ExpOnEveryDevice runs each fixture with every block and the head
// on each device present against the host -- the hyper-connections' mixes
// and placements, the sigmoid-gated delta rule over a state slot that carries
// the n-gram tail, the n-gram block, QSA's pooled-key selection and the
// mixture (tier's q4e.go and qsa.go) -- on decode and on a prefill chunk.
// Then each violation alone, which the device must run rather than decline.
func TestQwen4ExpOnEveryDevice(t *testing.T) {
	for _, fx := range q4eFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, q4eViolations, 1e-5)
		})
	}
}

// TestQwen4ExpDeviceRunsTheChunkWhole is the device gate's selection check: a
// prompt goes to the device as one batched submission -- every mix, the
// n-gram block and both QSA selections -- and a decode token runs them all
// there too. Without it, a block declined or a chunk replayed a row at a time
// would pass TestQwen4ExpOnEveryDevice as well.
func TestQwen4ExpDeviceRunsTheChunkWhole(t *testing.T) {
	m, g := openQ4E(t, "synth-qwen4exp")
	defer m.Close()
	c := m.Cfg
	nl, qsa := c.NLayer, 0
	for li := 0; li < nl; li++ {
		if c.QSAAt(li) {
			qsa++
		}
	}
	ran := 0
	for _, spec := range stepDevices() {
		// A replayed graph issues a token's launches without emitting them,
		// and the counters count emits: graphs off, so every call counts.
		gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true })},
			testTierOpts(t)...)...)
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			st := m.NewState(len(g.IDs) + 2)
			defer st.Close()
			st.SetDeviceLayers(gpu, -1)
			if st.GPULayers() != nl {
				t.Fatalf("placed %d of %d blocks: %v %v", st.GPULayers(), nl, st.DeviceDeclines(), gpu.Err())
			}
			s0 := gpu.Stats()
			lg, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			s1 := gpu.Stats()
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			s2 := gpu.Stats()
			t.Logf("prefill of %d: %d mixes, %d n-gram, %d selections; a decode token: %d, %d, %d",
				len(g.IDs), s1.Q4EMixes-s0.Q4EMixes, s1.Q4ENGram-s0.Q4ENGram, s1.QSASelects-s0.QSASelects,
				s2.Q4EMixes-s1.Q4EMixes, s2.Q4ENGram-s1.Q4ENGram, s2.QSASelects-s1.QSASelects)
			for _, d := range [][2]tier.Stats{{s0, s1}, {s1, s2}} {
				if d[1].Q4EMixes-d[0].Q4EMixes != 2*nl || d[1].Q4ENGram-d[0].Q4ENGram != 1 ||
					d[1].QSASelects-d[0].QSASelects != qsa {
					t.Fatalf("a call did not run every block's mixes, the n-gram block and every selection " +
						"on the device in one submission")
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestQwen4ExpQSATurnsByTheMarks holds the device's QSA to the host where a
// picture has moved the positions after it (M-RoPE's marks): the pooled keys
// of the blocks past the mark turn by their position plus its delta, a turn
// the device composes from two tables. A mark is set on both arms at
// position 6, ahead of the decode that selects (from position 12), once
// moving the positions back and once forward; each must move the host's
// logits, or the comparison could not see the device ignore it.
func TestQwen4ExpQSATurnsByTheMarks(t *testing.T) {
	m, g := openQ4E(t, "synth-qwen4exp")
	defer m.Close()
	const at = 6
	run := func(st *State, delta int) [][]float32 {
		var out [][]float32
		for p, id := range g.IDs {
			if p == at && delta != 0 {
				st.markRope(0, at, delta)
			}
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, append([]float32(nil), lg...))
		}
		return out
	}
	nmse := func(a, b [][]float32) float64 {
		w := 0.0
		for p := range a {
			var num, den float64
			for i := range a[p] {
				d := float64(a[p][i] - b[p][i])
				num, den = num+d*d, den+float64(a[p][i])*float64(a[p][i])
			}
			w = max(w, num/den)
		}
		return w
	}
	host := func(delta int) [][]float32 {
		st := m.NewState(len(g.IDs) + 1)
		defer st.Close()
		return run(st, delta)
	}
	plain := host(0)
	ran := 0
	for _, spec := range stepDevices() {
		gpu, err := openTier(t, append([]tier.Option{tier.WithDevices(spec),
			tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || gpu == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpu.Close()
			for _, delta := range []int{-3, 5} {
				want := host(delta)
				if moved := nmse(plain, want); !(moved > 1e-4) {
					t.Fatalf("delta %d: the mark moves the host's logits by %.3e, which no gate could see", delta, moved)
				}
				st := m.NewState(len(g.IDs) + 1)
				st.SetDeviceLayers(gpu, -1)
				if st.GPULayers() != m.Cfg.NLayer {
					t.Fatalf("placed %d of %d blocks: %v", st.GPULayers(), m.Cfg.NLayer, gpu.Err())
				}
				got := run(st, delta)
				st.Close()
				if w := nmse(want, got); !(w < 1e-6) {
					t.Fatalf("delta %d: worst logit NMSE %.3e against the host", delta, w)
				} else {
					t.Logf("delta %d: worst logit NMSE %.3e", delta, w)
				}
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}

// TestBatchSeamMovesCarryEveryRowQwen4Exp is the batch seam gate on
// Qwen3.8-Flash-Next: every move carries each row's linear blocks' state slot
// (the delta rule's and the n-gram block's history after it), its QSA blocks'
// cached rows with the raw indexer key, and the streams across the seam.
func TestBatchSeamMovesCarryEveryRowQwen4Exp(t *testing.T) {
	defer fusedRunsCheck(t, false)()
	p, ok := existingModel(testmodels.Path("synth-qwen4exp.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-qwen4exp.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run "+q4eGoldScript+" (RULE 11)")
	}
	batchSeamMoves(t, p, 1e-3, true)
}
