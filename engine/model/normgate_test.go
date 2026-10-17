//go:build jitllmtest

package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// normGateFixtures are the gated delta rule's three output gates: qwen3next's
// SiLU of a full-rank z, Kimi Delta Attention's sigmoid of a low-rank one,
// and Qwen3.8-Flash-Next's sigmoid under four hyper-connection streams.
var normGateFixtures = []struct {
	name string
	open func(t *testing.T) *Model
	gate graph.Act
}{
	{"hybrid", func(t *testing.T) *Model { return hybridModelOpt(t, hyOpt{ctx: 320}) }, graph.ActSiLU},
	{"synth-kimilinear", func(t *testing.T) *Model {
		path := testmodels.Path("synth-kimilinear.jlm")
		if !fileExists(path) {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS) (RULE 11)", path)
		}
		m, err := Open(jlmOf(t, path), WithKVF16(false))
		if err != nil {
			t.Fatal(err)
		}
		return m
	}, graph.ActSigmoid},
	{"synth-qwen4exp", func(t *testing.T) *Model {
		m, _ := openQ4E(t, "synth-qwen4exp")
		return m
	}, graph.ActSigmoid},
}

// normGateDecode prefills a fixed prompt (a chunk's rows) and decodes toks
// teacher-forced with every block on the device, under recipe f, and returns
// the logits of each step and the fused launches it ran.
func normGateDecode(t *testing.T, m *Model, gpu *tier.GPU, f tier.NormGateForm, ids, toks []int32) ([][]float32, int) {
	t.Helper()
	gpu.NormGate = f
	st := m.NewState(len(ids) + len(toks) + 4)
	defer st.Close()
	if err := st.SetDeviceLayers(gpu, -1); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("placed %d of %d blocks: %v", st.GPULayers(), m.Cfg.NLayer, st.DeviceDeclines())
	}
	s0 := gpu.Stats()
	lg, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := [][]float32{append([]float32(nil), lg...)}
	for _, tok := range toks {
		if lg, err = st.Forward(tok); err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out, gpu.Stats().NormGateRuns - s0.NormGateRuns
}

// TestNormGateMatchesTwoLaunches holds graph.NormGate's recipe (the gated
// delta rule's output norm and its gate in one launch,
// kernels.HeadNormGated) to the two launches it replaces, bit for bit, on
// every device present: every block placed, a prompt chunk and a decode, on
// each of the three gates. Selection is asserted (RULE 10): every gated delta
// block's sublayer graph verified and matched the rewrite, the fused arm ran
// one fused launch per gated delta block per call, and the reference arm
// none. The violation: the fused launch reading the rule's output and z the
// wrong way round must move the logits.
func TestNormGateMatchesTwoLaunches(t *testing.T) {
	ran := 0
	for _, fx := range normGateFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m := fx.open(t)
			defer m.Close()
			ids, toks := make([]int32, 12), make([]int32, 4)
			for i := range ids {
				ids[i] = int32((1 + 37*i) % m.Cfg.NVocab)
			}
			for i := range toks {
				toks[i] = int32((5 + 53*i) % m.Cfg.NVocab)
			}
			if errs := m.DeltaGraphErrors(); len(errs) != 0 {
				t.Fatalf("a sublayer graph did not verify: %v", errs)
			}
			blocks := 0
			for li := 0; li < m.Cfg.NLayer; li++ {
				g := m.DeltaGraph(li)
				if g == nil {
					continue
				}
				ds := graph.Apply(g, graph.NormGate)
				if len(ds) != 1 || ds[0].Err != nil {
					t.Fatalf("block %d: normgate %v", li, ds)
				}
				if a := g.Ops[ds[0].Region.Ops[1]].Attr.Act; a != fx.gate {
					t.Fatalf("block %d: the graph gates by %d, the fixture by %d", li, a, fx.gate)
				}
				blocks++
			}
			if blocks == 0 {
				t.Fatal("no block has a gated delta sublayer graph")
			}
			for _, spec := range stepDevices() {
				gpu, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
					tier.WithDeviceTune(tier.TuneOff), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true })},
					testTierOpts(t)...)...)
				if err != nil || gpu == nil {
					t.Logf("%s: not present (%v)", spec, err)
					continue
				}
				t.Run(spec, func(t *testing.T) {
					defer gpu.Close()
					two, n2 := normGateDecode(t, m, gpu, tier.NormGateTwo, ids, toks)
					one, n1 := normGateDecode(t, m, gpu, tier.NormGateFused, ids, toks)
					if n2 != 0 {
						t.Fatalf("the reference arm ran %d fused launches", n2)
					}
					if want := blocks * (1 + len(toks)); n1 != want {
						t.Fatalf("the fused arm ran %d fused launches; %d gated delta blocks over %d calls want %d",
							n1, blocks, 1+len(toks), want)
					}
					for s := range two {
						for i := range two[s] {
							if math.Float32bits(two[s][i]) != math.Float32bits(one[s][i]) {
								t.Fatalf("step %d logit %d: two launches %v, fused %v", s, i, two[s][i], one[s][i])
							}
						}
					}
					// The trial: a tuned tier decides each key at decode, and
					// whatever it keeps is the bits of the two launches on that
					// tier (the other tuners' verdicts are its own, so the
					// reference is taken there too).
					tuned, err := tier.OpenWith(append([]tier.Option{tier.WithDevices(spec),
						tier.WithTuneCache(t.TempDir()), tier.WithConfig(func(c *tier.Config) { c.NoGraph = true })},
						testTierOpts(t)...)...)
					if err != nil || tuned == nil {
						t.Fatalf("%s opened once and not twice: %v", spec, err)
					}
					tl, _ := normGateDecode(t, m, tuned, tier.NormGateTuned, ids, toks)
					st := tuned.Stats()
					tw, _ := normGateDecode(t, m, tuned, tier.NormGateTwo, ids, toks)
					tuned.Close()
					if st.NormGateTimed == 0 {
						t.Fatal("the tuned tier decided no gated-norm key")
					}
					for s := range two {
						for i := range two[s] {
							if math.Float32bits(tw[s][i]) != math.Float32bits(tl[s][i]) {
								t.Fatalf("tuned: step %d logit %d: two launches %v, tuned %v", s, i, tw[s][i], tl[s][i])
							}
						}
					}
					t.Logf("the trial timed %d keys and kept the fused launch on %d", st.NormGateTimed, st.NormGateKept)
					defer tier.BreakNormGateForTest(tier.BreakNormGateForTest(true))
					bad, nb := normGateDecode(t, m, gpu, tier.NormGateFused, ids, toks)
					if nb == 0 {
						t.Fatal("the violation reached no fused launch")
					}
					moved := 0
					for s := range two {
						for i := range two[s] {
							if bad[s][i] != two[s][i] {
								moved++
							}
						}
					}
					if moved == 0 {
						t.Fatal("the rule's output and z swapped in the fused launch, and the logits did not move")
					}
					t.Logf("%d gated delta blocks: fused in %d launches, bit-identical over %d calls; the swap moved %d logits",
						blocks, n1, len(two), moved)
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host")
	}
}
