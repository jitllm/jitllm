package model

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestGraniteOnEveryDevice holds a Granite model with every block and the head
// on a device to the host (RULE 11c): its four scales -- embedding_scale,
// residual_scale, attention.scale and logit_scale -- each one multiply that no
// shape check, buffer ledger or kernel gate sees. Two arms per device, against
// the host decoding the prompt a token at a time and then the host's own
// greedy ids: the prompt as a batched chunk (whose rows the device gathers
// from the tied head, and whose attention runs the paged prefill), then
// decode; and every token decoded a row at a time. A dense model's band is the
// order-only control -- the row-at-a-time arm at two matvec splits -- measured
// on the same device in the same run, as the paged history's gate sizes its
// own (graniteBand); a mixture's is the mixture step gate's. Then three
// sessions step as rows against each alone (stepGate).
//
// Each scale is then dropped on the device arm and must part from the host by
// ten times the band: the residual scale (tier.ScaleFaultResidual), the
// attention scale at 1/sqrt(HeadDim) (ScaleFaultAttn), the embedding scale
// the device's gather applies (ScaleFaultEmbd, where the prompt was gathered
// on the device), and the logit divisor, which the host applies after the
// device and which the arm skips by running with LogitScale 1. The greedy
// gates cannot see the last: a positive divisor moves no argmax. The step's
// own break is its residual fused into the projections unscaled
// (ScaleFaultStepResidual).
//
// JITLLM_GRANITE_MODELS names the models, comma-separated, under
// $JITLLM_MODELS; the default is one dense and one mixture Granite.
func TestGraniteOnEveryDevice(t *testing.T) {
	ran := 0
	for _, name := range graniteDevModels() {
		t.Run(strings.TrimSuffix(name[strings.LastIndex(name, "/")+1:], ".gguf"), func(t *testing.T) {
			m := openGranite(t, name)
			defer m.Close()
			prompt := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 4), true)
			const gen = 16
			seq := len(prompt) + gen + 8
			forced := hostGreedy(t, m, prompt, gen)
			host := hostRun(t, m, seq, prompt, forced)
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					rowwise := func(split int) []tier.Option {
						return []tier.Option{tier.WithSplit(split),
							tier.WithConfig(func(c *tier.Config) { c.NoBatch = true })}
					}
					// The order-only control: every token a row at a time at two
					// splits, which moves nothing but the summation order.
					rows, _, why := graniteRun(t, m, dev, seq, nil, slices.Concat(prompt, forced), rowwise(1))
					if why != "" {
						t.Skip(why)
					}
					ctlRows, _, why := graniteRun(t, m, dev, seq, nil, slices.Concat(prompt, forced), rowwise(4))
					if why != "" {
						t.Fatal(why)
					}
					// Row-at-a-time arms carry the prompt's logits too; the
					// comparison starts at the prompt's last token.
					rows, ctlRows = rows[len(prompt)-1:], ctlRows[len(prompt)-1:]
					ctl := 0.0
					for i := range rows {
						ctl = max(ctl, logitNMSE(ctlRows[i], rows[i]))
					}
					if ctl == 0 || math.IsNaN(ctl) {
						t.Fatalf("the order-only control moved nothing (%v): it cannot size the band", ctl)
					}
					band := graniteBand * ctl
					if m.Cfg.MoE() {
						// A router near a tie turns the order change into a
						// different expert, so the control is no band: on CUDA
						// granite-3.1-1b-a400m's read 3.2e-02, above the host's
						// own distance. The mixture step
						// gate's bound, which its clean arms sit under at
						// 1.2e-02..3.6e-02 (CUDA the widest).
						band = mixtureStepNMSE
					}
					chunk, cs, why := graniteRun(t, m, dev, seq, prompt, forced, nil)
					if why != "" {
						t.Fatal(why)
					}
					if cs.ResidScaled == 0 {
						t.Fatalf("no residual add took the scale (Stats.ResidScaled 0): the scaled path was not selected")
					}
					worst := 0.0
					for i := range host {
						e := max(logitNMSE(chunk[i], host[i]), logitNMSE(rows[i], host[i]))
						if math.IsNaN(e) || math.IsInf(e, 0) {
							t.Fatalf("step %d: non-finite NMSE %v", i, e)
						}
						worst = max(worst, e)
					}
					t.Logf("GRANITE %s on %s: %d prompt tokens, %d decoded: worst logit NMSE %.3e against the "+
						"host, band %.3e (order-only control %.3e), %d scaled residual adds, %d embedding gathers",
						name, dev, len(prompt), gen, worst, band, ctl, cs.ResidScaled, cs.EmbedLaunches)
					if !(worst < band) {
						t.Fatalf("a Granite on %s parts from the host: worst logit NMSE %.3e, band %.3e", dev, worst, band)
					}
					// Each scale dropped, on the chunk arm.
					violations := []struct {
						name string
						opt  tier.Option
						mut  func() func()
					}{
						{"residual scale dropped", tier.WithScaleFault(tier.ScaleFaultResidual), nil},
						{"attention at 1/sqrt(head_dim)", tier.WithScaleFault(tier.ScaleFaultAttn), nil},
						{"logit divisor skipped", nil, func() func() {
							keep := m.Cfg.LogitScale
							m.Cfg.LogitScale = 1
							return func() { m.Cfg.LogitScale = keep }
						}},
					}
					if cs.EmbedLaunches > 0 {
						violations = append(violations, struct {
							name string
							opt  tier.Option
							mut  func() func()
						}{"embedding scale dropped", tier.WithScaleFault(tier.ScaleFaultEmbd), nil})
					}
					for _, v := range violations {
						var opts []tier.Option
						if v.opt != nil {
							opts = append(opts, v.opt)
						}
						undo := func() {}
						if v.mut != nil {
							undo = v.mut()
						}
						bad, _, why := graniteRun(t, m, dev, seq, prompt, forced, opts)
						undo()
						if why != "" {
							t.Fatalf("%s: %s", v.name, why)
						}
						e := 0.0
						for i := range host {
							e = max(e, logitNMSE(bad[i], host[i]))
						}
						t.Logf("violation %q on %s: worst logit NMSE %.3e (%.0fx the band)", v.name, dev, e, e/band)
						if !(e > 10*band) {
							t.Errorf("%s: the gate cannot see it -- worst NMSE %.3e against a band of %.3e",
								v.name, e, band)
						}
					}
					// Three sessions stepping as rows against each alone, clean
					// and with the step's own break: its rows fuse the residual
					// into the projections unscaled, where each session alone
					// does not. The step's fused projections exist only for
					// quantized weights, which is why this arm is here and not
					// on the f32 fixtures.
					step := func(opts ...tier.Option) (stepResult, int) {
						g, err := openTier(t, slices.Concat([]tier.Option{tier.WithDevices(dev),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t), opts)...)
						if err != nil || g == nil {
							t.Fatalf("no %s for the step: %v", dev, err)
						}
						defer g.Close()
						r := stepGate(t, m, g, false)
						return r, g.Stats().RagResFused
					}
					// A mixture's step takes the every-architecture step gate's
					// mixture bound (stepBound): 5e-2, or eight times its own
					// order-only control where that is wider. Its break must
					// clear that band and ten times the clean step rather than
					// ten bands: the control reads up to 1.8e-02 on an
					// integrated Vulkan device, a band of 0.147, where the break reads 1.27 against a
					// clean step of 7.7e-03.
					stepBand, need := band, 10*band
					if m.Cfg.MoE() {
						stepBand = max(mixtureStepNMSE, denseStepBand*orderControl(t, m, dev))
					}
					r, _ := step()
					rb, fused := step(tier.WithScaleFault(tier.ScaleFaultStepResidual))
					t.Logf("step across sessions on %s: worst logit NMSE %.3e against each alone, band %.3e; with "+
						"its residual fused unscaled %.3e (%d fused projections)", dev, r.worst, stepBand, rb.worst, fused)
					if !(r.worst < stepBand) {
						t.Fatalf("a Granite's step across sessions on %s parts from each session alone: NMSE %.3e, "+
							"band %.3e", dev, r.worst, stepBand)
					}
					if m.Cfg.MoE() {
						need = max(stepBand, 10*r.worst)
					}
					if fused == 0 || !(rb.worst > need) {
						t.Errorf("a step's residual fused unscaled: the step gate cannot see it -- NMSE %.3e against "+
							"%.3e (band %.3e, clean step %.3e, %d fused projections)", rb.worst, need, stepBand,
							r.worst, fused)
					}
					ran++
				})
			}
		})
	}
	if ran == 0 {
		t.Skip("no device on this host ran a Granite -- this gate proved nothing")
	}
}

// graniteBand is how many times the order-only control a Granite's device arms
// may sit from the host: denseStepBand's 8, the dense gate's own. The control
// is one pair of splits, and on granite-3.3-2b it is the noisier half of the
// ratio: on this gate's prompt the device's six pairs of splits 1, 2, 4 and 8
// read 8.4e-03..2.3e-02 against each other on CUDA and
// 4.3e-03..3.1e-02 on Vulkan, and the host's four pairs sit inside them
// (9.9e-03..2.5e-02, 1.1e-02..1.8e-02). This gate's pair, splits 1 and 4, is
// the lowest of the six on Vulkan, which is the 3.2x it reads there; CUDA
// reads 2.9x and Metal 1.0x. 4x would fail on the draw alone (the host against
// split 4 is 4.2x that pair on Vulkan). The arithmetic is not the band's to
// prove: on f32 fixtures the scales read 1e-14 against the host
// (TestGraniteScalesOnEveryDevice), and the activation quantizer and the scaled
// residual add are the host's bit for bit on every device
// (TestQuantizeOnTheRoundingBoundary, TestAddScaledRoundsOnce). Every
// violation clears the band 10x and more.
const graniteBand = denseStepBand

// graniteDevModels is the models TestGraniteOnEveryDevice runs.
func graniteDevModels() []string {
	if v := os.Getenv("JITLLM_GRANITE_MODELS"); v != "" {
		return strings.Split(v, ",")
	}
	return []string{"granite/granite-3.3-2b-instruct-Q4_K_M.gguf",
		"granite/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf"}
}

// openGranite opens name, refusing a model with none of Granite's scales: the
// gate would compare a llama against itself.
func openGranite(t *testing.T, name string) *Model {
	t.Helper()
	// The container alone serves where the GGUF is not kept.
	p := firstModel(name)
	if !fileExists(p) && !fileExists(strings.TrimSuffix(p, ".gguf")+".jlm") {
		testmodels.Missing(t, "MODEL MISSING: %s nor its container (set JITLLM_MODELS) -- RULE 11, fetch it", p)
	}
	m, err := Open(jlmOf(t, p), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.ResidualScale == 1 || c.LogitScale == 1 || c.AttnScale == 1/math.Sqrt(float64(c.HeadDim)) {
		m.Close()
		t.Fatalf("%s states residual %g, logit %g, attention %g: not a Granite", name,
			c.ResidualScale, c.LogitScale, c.AttnScale)
	}
	return m
}

// graniteRun is one device arm: prompt as a chunk (nil to skip it), then the
// forced ids one at a time, every step's logits (the chunk's first) and the
// tier's counters. A non-empty reason is a placement that is not every block
// and the head, which the caller skips or fails on.
func graniteRun(t *testing.T, m *Model, dev string, seq int, prompt, forced []int32,
	opts []tier.Option) ([][]float32, tier.Stats, string) {
	t.Helper()
	g, err := openTier(t, slices.Concat([]tier.Option{tier.WithDevices(dev), tier.WithDeviceTune(tier.TuneOff)},
		testTierOpts(t), opts)...)
	if err != nil || g == nil {
		return nil, tier.Stats{}, "no " + dev + " here"
	}
	defer g.Close()
	st := m.NewState(seq)
	defer st.Close()
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.devCount() != m.Cfg.NLayer || !st.HeadOnDevice() {
		if why := g.DeclinePlan(st.planFor(0)); why != "" {
			return nil, tier.Stats{}, "THE DEVICE DECLINES THE BLOCK: " + why
		}
		return nil, tier.Stats{}, "CARD TOO SMALL: " + g.Err()
	}
	var out [][]float32
	if prompt != nil {
		lg, err := st.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	for _, id := range forced {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
	}
	if st.devCount() != m.Cfg.NLayer {
		t.Fatalf("%d of %d blocks left %s during the run: %s", m.Cfg.NLayer-st.devCount(), m.Cfg.NLayer, dev, g.Err())
	}
	return out, g.Stats(), ""
}

// TestGraniteScalesOnEveryDevice is the tight form of TestGraniteOnEveryDevice:
// Granite's four scales set on f32 fixtures of llama's block (synth-qwen2) and
// a softmax mixture (synth-mixtral), where a device and the host agree to the
// f32 reduction order rather than to two int8 quantizations, so a scale the
// device applies a little wrong reads as far outside the bound as one it
// skips. The scales are perturbed rather than waited for, as
// TestAttnScaleReachesEveryPath perturbs the attention scale: Granite's own
// published values, which no fixture's converter writes. Every block and the
// head placed; decode, a prompt chunk and three sessions stepping as rows
// (stepGate, whose rows run the per-row head), and each scale dropped on the
// device arm must move the logits.
func TestGraniteScalesOnEveryDevice(t *testing.T) {
	for _, name := range []string{"synth-qwen2", "synth-mixtral"} {
		t.Run(name, func(t *testing.T) {
			m, err := Open(hfContainer(t, name), noTune, WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.Cfg
			// granite-3.3-2b's residual_scale, logits_scaling and embedding
			// multiplier, and an attention multiplier of 1/head_dim, as
			// Granite's own configs state it.
			c.ResidualScale, c.LogitScale, c.EmbdScale = 0.22, 8, 12
			c.AttnScale = 1 / float64(c.HeadDim)
			// The block graphs are compiled at Open from the configuration:
			// rebuild them, or the device runs the graph's stale scales.
			if err := m.buildGraphs(); err != nil {
				t.Fatal(err)
			}
			ids := m.Vocab.Encode("The capital of France is a city on the river", true)
			decode := func(st *State) [][]float32 {
				var out [][]float32
				for _, id := range ids {
					lg, err := st.Forward(id)
					if err != nil {
						t.Fatal(err)
					}
					out = append(out, slices.Clone(lg))
				}
				return out
			}
			prefill := func(st *State) [][]float32 {
				lg, err := st.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				return [][]float32{slices.Clone(lg)}
			}
			onHost := func(run func(*State) [][]float32) [][]float32 {
				st := m.NewState(len(ids) + 1)
				defer st.Close()
				return run(st)
			}
			worstOf := func(want, got [][]float32) float64 {
				w := 0.0
				for p := range want {
					e := logitNMSE(got[p], want[p])
					if math.IsNaN(e) || math.IsInf(e, 0) {
						return math.Inf(1)
					}
					w = max(w, e)
				}
				return w
			}
			wantDec, wantPre := onHost(decode), onHost(prefill)
			ran := 0
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					onDevice := func(run func(*State) [][]float32, opts ...tier.Option) ([][]float32, tier.Stats) {
						g, err := openTier(t, slices.Concat([]tier.Option{tier.WithDevices(dev),
							tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t), opts)...)
						if err != nil || g == nil {
							noDevice(t, dev, err)
						}
						defer g.Close()
						st := m.NewState(len(ids) + 1)
						defer st.Close()
						if err := st.SetDevice(g); err != nil {
							t.Fatal(err)
						}
						if st.devCount() != m.Cfg.NLayer || !st.HeadOnDevice() {
							t.Fatalf("placed %d of %d blocks, head on the device %v: %v %s", st.devCount(),
								m.Cfg.NLayer, st.HeadOnDevice(), st.DeviceDeclines(), g.Err())
						}
						got := run(st)
						if st.devCount() != m.Cfg.NLayer {
							t.Fatalf("the session fell to the host: %s", g.Err())
						}
						return got, g.Stats()
					}
					worst := 0.0
					for _, arm := range []struct {
						name  string
						run   func(*State) [][]float32
						want  [][]float32
						bound float64
					}{{"decode", decode, wantDec, 1e-6}, {"prefill", prefill, wantPre, 1e-5}} {
						got, s := onDevice(arm.run)
						if s.ResidScaled == 0 {
							t.Fatalf("%s: no residual add took the scale -- the scaled path was not selected", arm.name)
						}
						w := worstOf(arm.want, got)
						t.Logf("%s, %d blocks and the head on %s: worst logit NMSE %.3e against the host "+
							"(%d scaled residual adds)", arm.name, m.Cfg.NLayer, dev, w, s.ResidScaled)
						if !(w < arm.bound) {
							t.Fatalf("%s: worst logit NMSE %.3e against the host, bound %.0e", arm.name, w, arm.bound)
						}
						worst = max(worst, w)
					}
					// Three sessions stepping as rows, against each alone.
					g, err := openTier(t, slices.Concat([]tier.Option{tier.WithDevices(dev),
						tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t))...)
					if err != nil || g == nil {
						noDevice(t, dev, err)
					}
					r := stepGate(t, m, g, false)
					g.Close()
					t.Logf("step across sessions on %s: worst logit NMSE %.3e against each alone", dev, r.worst)
					if !(r.worst < fixtureStepNMSE) {
						t.Fatalf("the step across sessions parts from each session alone: NMSE %.3e at row %d",
							r.worst, r.at)
					}
					for _, v := range []struct {
						name string
						opt  tier.Option
						mut  func() func()
					}{
						{"residual scale dropped", tier.WithScaleFault(tier.ScaleFaultResidual), nil},
						{"attention at 1/sqrt(head_dim)", tier.WithScaleFault(tier.ScaleFaultAttn), nil},
						{"logit divisor skipped", nil, func() func() {
							c.LogitScale = 1
							return func() { c.LogitScale = 8 }
						}},
					} {
						var opts []tier.Option
						if v.opt != nil {
							opts = append(opts, v.opt)
						}
						undo := func() {}
						if v.mut != nil {
							undo = v.mut()
						}
						dec, _ := onDevice(decode, opts...)
						pre, _ := onDevice(prefill, opts...)
						undo()
						bad := max(worstOf(wantDec, dec), worstOf(wantPre, pre))
						t.Logf("violation %q on %s: worst logit NMSE %.3e", v.name, dev, bad)
						if !(bad > 100*worst) || !(bad > 1e-4) {
							t.Errorf("%s: the device arm cannot see it -- worst NMSE %.3e against a clean %.3e",
								v.name, bad, worst)
						}
					}
					ran++
				})
			}
			if ran == 0 {
				t.Skip("no device on this host -- this gate proved nothing")
			}
		})
	}
}
