package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpudev"
	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestQwen4ExpOnEveryDevice runs each fixture with every block and the head
// on each device present against the host -- the hyper-connections' mixes
// and placements, the sigmoid-gated delta rule over a state slot that carries
// the n-gram tail, the n-gram block, QSA's pooled-key selection and the
// mixture (tier's q4e.go and qsa.go) -- on decode and on a prefill chunk.
// Then each violation alone, which the device must run rather than decline.
func TestQwen4ExpOnEveryDevice(t *testing.T) {
	for _, fx := range q4eFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openQ4E(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceWithin(t, m, g, q4eViolations(), 1e-5)
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
		gpux, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec),
			gpu.WithDeviceTune(gpu.TuneOff), gpu.WithConfig(func(c *gpu.Config) { c.NoGraph = true })},
			testTierOpts(t)...)...)
		if err != nil || gpux == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpux.Close()
			st := m.NewInstance(len(g.IDs) + 2)
			defer st.Close()
			st.SetDeviceLayers(gpux, -1)
			if st.GPULayers() != cardBlocks(m) {
				t.Fatalf("placed %d of %d blocks: %v %v", st.GPULayers(), cardBlocks(m), st.DeviceDeclines(), gpux.Err())
			}
			s0 := gpux.Stats()
			lg, err := st.Prefill(g.IDs)
			if err != nil {
				t.Fatal(err)
			}
			s1 := gpux.Stats()
			if _, err := st.Forward(Greedy(lg)); err != nil {
				t.Fatal(err)
			}
			s2 := gpux.Stats()
			ps0, pd0 := s0.ServedPart(gpu.ComposedPrompt)
			ps1, pd1 := s1.ServedPart(gpu.ComposedPrompt)
			bs1, bd1 := s1.ServedPart(gpu.ComposedBlock)
			bs2, bd2 := s2.ServedPart(gpu.ComposedBlock)
			t.Logf("prefill of %d: %d composed block steps, %d declined; a decode token: %d, %d (%d QSA blocks)",
				len(g.IDs), ps1-ps0, pd1-pd0, bs2-bs1, bd2-bd1, qsa)
			cb := cardBlocks(m)
			if ps1-ps0 < cb || (ps1-ps0)%cb != 0 || pd1 != pd0 || bs2-bs1 != cb || bd2 != bd1 {
				t.Fatalf("the composer did not serve every block of the prompt and of the decode token on the device")
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
	run := func(st *Instance, delta int) [][]float32 {
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
		st := m.NewInstance(len(g.IDs) + 1)
		defer st.Close()
		return run(st, delta)
	}
	plain := host(0)
	ran := 0
	for _, spec := range stepDevices() {
		gpux, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(spec),
			gpu.WithDeviceTune(gpu.TuneOff)}, testTierOpts(t)...)...)
		if err != nil || gpux == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer gpux.Close()
			for _, delta := range []int{-3, 5} {
				want := host(delta)
				if moved := nmse(plain, want); !(moved > 1e-4) {
					t.Fatalf("delta %d: the mark moves the host's logits by %.3e, which no gate could see", delta, moved)
				}
				st := m.NewInstance(len(g.IDs) + 1)
				st.SetDeviceLayers(gpux, -1)
				if st.GPULayers() != cardBlocks(m) {
					t.Fatalf("placed %d of %d blocks: %v", st.GPULayers(), m.Cfg.NLayer, gpux.Err())
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
	w := fusedRunsCheck(t, false)
	defer w.done()
	p, ok := existingModel(testmodels.Path("synth-qwen4exp.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-qwen4exp.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run "+q4eGoldScript+" (RULE 11)")
	}
	batchSeamMoves(t, p, 1e-3, true, w)
}

// cardBlocks is how many of m's blocks a card takes: every one but a block
// gathering a row table's rows, which no card holds (the n-gram table does
// not fit one), so that block runs on the CPU device.
func cardBlocks(m *Model) int {
	n := 0
	for li := range m.Cfg.NLayer {
		if cpudev.TableRowsOf(m.Composing().BlockGraph(li)) < 0 {
			n++
		}
	}
	return n
}
