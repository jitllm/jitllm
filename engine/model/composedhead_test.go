package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/internal/oracle"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// headOracle is the head of m over the final residual x as the oracle computes
// it, in float64 past the projection: the output norm (a LayerNorm with
// outNormB, or an RMSNorm), the projection, the bias outB, the final softcap
// and the logit scale of c, in the model's order.
func headOracle(t *testing.T, m *Model, c *Config, x, outNormB, outB []float32) []float64 {
	t.Helper()
	w := wt(m.output)
	if w.T != quant.F32 {
		t.Fatalf("the fixture's head is %v: the oracle's matvec here is the float one", w.T)
	}
	y := make([]float32, c.NEmbd)
	if c.LayerNorm {
		oracle.LayerNorm32(y, x, m.outNorm, outNormB, c.RMSEps)
	} else {
		oracle.RMSNorm32(y, x, m.outNorm, c.RMSEps)
	}
	lg := make([]float32, w.Rows)
	if err := oracle.MatVec(lg, w.T, w.Data, y, w.Rows, w.K); err != nil {
		t.Fatal(err)
	}
	out := make([]float64, len(lg))
	for i, v := range lg {
		f := float64(v)
		if outB != nil {
			f += float64(outB[i])
		}
		if s := float64(c.FinalSoftcap); s != 0 {
			f = s * math.Tanh(f/s)
		}
		if c.LogitScale != 1 {
			f /= c.LogitScale
		}
		out[i] = f
	}
	return out
}

// headNMSE is got's distance from the oracle's want.
func headNMSE(got []float32, want []float64) float64 {
	var num, den float64
	for i, w := range want {
		g := float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			return math.Inf(1)
		}
		num += (g - w) * (g - w)
		den += w * w
	}
	return num / den
}

// headFixture is an F32 model whose head the oracle computes: synth-qwen2's is
// an RMSNorm and a projection, synth-phi2's a LayerNorm with its bias, a
// projection and its own bias, and synth-falcon's a LayerNorm over the
// embedding's rows.
type headFixture struct {
	name string
	open func(t *testing.T, opts ...Option) *Model
}

var headFixtures = []headFixture{
	{"synth-qwen2", func(t *testing.T, opts ...Option) *Model {
		m, err := Open(hfContainer(t, "synth-qwen2"), append([]Option{noTune, WithKVF16(false)}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}},
	{"synth-phi2", openHeadGGUF("synth-phi2.gguf")},
	{"synth-falcon", openHeadGGUF("synth-falcon.gguf")},
}

// openHeadGGUF opens the fixture file as a head fixture.
func openHeadGGUF(file string) func(t *testing.T, opts ...Option) *Model {
	return func(t *testing.T, opts ...Option) *Model {
		path, ok := existingModel(testmodels.Path(file))
		if !ok {
			testmodels.Missing(t, "MODEL MISSING: %s (set JITLLM_MODELS to the model directory)", testmodels.Path(file))
		}
		m, err := Open(jlmOf(t, path), append([]Option{noTune, WithKVF16(false)}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
}

// headTokens is the tokens a head fixture decodes.
func headTokens(m *Model) []int32 {
	ids := []int32{5, 11, 23, 41, 67, 89, 12, 33}
	for i := range ids {
		ids[i] %= int32(m.Cfg.NVocab)
	}
	return ids
}

// headArm is one device arm of the head gate: every block and the head on dev
// under the config cfg states and the changes mut makes to the model, held to
// the oracle over the host's final residuals xs under the config want states.
type headArm struct {
	fx         headFixture
	dev        string
	parts      tier.Parts
	xs         [][]float32
	lnb, outB  []float32
	want, cfg  func(c *Config)
	mut        func(m *Model) func()
	worst      float64
	stats      tier.Stats
	applied    []nn.HeadStage
	steps      int
	stagesWant nn.HeadStage
}

func (a *headArm) run(t *testing.T) {
	t.Helper()
	g, err := tier.OpenWith(tier.WithDevices(a.dev), tier.WithDeviceTune(tier.TuneOff),
		tier.WithConfig(func(c *tier.Config) { c.Composed, c.NoGraph = a.parts, true }))
	if err != nil || g == nil {
		noDevice(t, a.dev, err)
	}
	defer g.Close()
	m := a.fx.open(t)
	defer m.Close()
	ids := headTokens(m)
	intended := *m.Cfg
	a.want(&intended)
	wants := make([][]float64, len(ids))
	for p, x := range a.xs {
		wants[p] = headOracle(t, m, &intended, x, a.lnb, a.outB)
	}
	defer cfgMut(m, a.cfg)()
	if a.mut != nil {
		defer a.mut(m)()
	}
	a.stagesWant = m.headStages()
	st := m.NewState(len(ids) + 1)
	defer st.Close()
	st.SetDeviceFallback(false)
	if err := st.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
		t.Fatalf("wanted every block and the head on %s, got %d of %d blocks and the head %v: %s", a.dev, st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
	}
	a.worst, a.steps, a.applied = 0, len(ids), nil
	for p, id := range ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		a.worst = math.Max(a.worst, headNMSE(lg, wants[p]))
		a.applied = append(a.applied, st.head.Applied)
	}
	a.stats = g.Stats()
}

// TestComposedHeadOps holds every step of the head past its projection to the
// oracle on each device -- the LayerNorm with its bias, the output bias, the
// final softcap and the logit scale, each alone and the softcap and the scale
// in the model's order -- with every block and the head on the device, held to
// the oracle over the final residuals of the host's blocks, which the device's
// differ from by the f32 reduction order alone. The head
// runs as the composer's graph (tier.ComposedHead), and then as the tier's own
// kernels beside the host's finishing of the logits: the same logits either
// way, and nothing applied twice (nn.Head.Applied). Each op taken out of the
// composed head -- the softcap dropped, the scale inverted, the bias negated,
// the norm's bias dropped -- must part from the oracle by far more than any
// arm does.
func TestComposedHeadOps(t *testing.T) {
	const bound = 1e-9
	ran := 0
	for _, fx := range headFixtures {
		t.Run(fx.name, func(t *testing.T) {
			ref := fx.open(t)
			defer ref.Close()
			c := ref.Cfg
			ids := headTokens(ref)
			outNormB, outB := slices.Clone(ref.outNormB), slices.Clone(ref.outB)
			lnb := outNormB
			if c.LayerNorm {
				lnb = slices.Clone(ref.lnBias(outNormB))
			} else {
				lnb = nil
			}
			// The residuals the head takes: the host's blocks, which every
			// device arm runs too. The oracle reproduces the host's own head
			// from them, or neither the residual nor the oracle is right.
			host := ref.NewState(len(ids) + 1)
			var xs [][]float32
			amax := 0.0
			for p, id := range ids {
				lg, err := host.Forward(id)
				if err != nil {
					t.Fatal(err)
				}
				x := slices.Clone(host.x[:c.NEmbd])
				xs = append(xs, x)
				want := headOracle(t, ref, c, x, lnb, outB)
				if e := headNMSE(lg, want); !(e < bound) {
					t.Fatalf("pos %d: the host's own head is %.3e from the oracle: the residual or the oracle is wrong", p, e)
				}
				for _, v := range want {
					amax = math.Max(amax, math.Abs(v))
				}
			}
			host.Close()
			if amax == 0 {
				t.Fatal("the head's logits are all zero: a cap would compare nothing")
			}
			capAt := float32(amax / 2)
			arms := []struct {
				name       string
				cap, scale float64
			}{
				{name: "plain"},
				{name: "softcap", cap: float64(capAt)},
				{name: "scale", scale: 8},
				{name: "softcap then scale", cap: float64(capAt), scale: 8},
			}
			for _, dev := range stepDevices() {
				t.Run(dev, func(t *testing.T) {
					for _, arm := range arms {
						set := func(c *Config) {
							if arm.cap != 0 {
								c.FinalSoftcap = float32(arm.cap)
							}
							if arm.scale != 0 {
								c.LogitScale = arm.scale
							}
						}
						t.Run(arm.name, func(t *testing.T) {
							// The composer's head, then the tier's own with the
							// host finishing the logits.
							for _, parts := range []tier.Parts{tier.ComposedHead, 0} {
								if parts == 0 && arm.cap != 0 && outB != nil {
									// The tier's own head caps on the device and the host adds
									// the bias after it: no model has both. The composed head
									// takes the bias, the cap and the scale in the model's
									// order (finishLogits).
									continue
								}
								a := &headArm{fx: fx, dev: dev, parts: parts, xs: xs, lnb: lnb, outB: outB, want: set, cfg: set}
								a.run(t)
								n, d, _ := a.stats.ServedPart(tier.ComposedHead)
								want := a.stagesWant
								if parts == 0 {
									want &= nn.HeadCap
									if n != 0 {
										t.Fatalf("the composer served %d head steps when it was not asked", n)
									}
								} else if n != a.steps || d != 0 {
									t.Fatalf("the composed head served %d of %d steps and was declined %d: %s", n, a.steps, d, declinesOf(a.stats))
								}
								for p, got := range a.applied {
									if got != want {
										t.Fatalf("pos %d: the head applied stages %03b, want %03b (stages of the model %03b)", p, got, want, a.stagesWant)
									}
								}
								t.Logf("parts %d: worst NMSE %.3e from the oracle over %d steps, the device applied stages %03b of %03b",
									parts, a.worst, a.steps, want, a.stagesWant)
								if !(a.worst < bound) {
									t.Fatalf("parts %d: %.3e from the oracle, bound %.0e", parts, a.worst, bound)
								}
								ran++
							}
						})
					}
					// Each op taken out of the composed head parts from the oracle.
					full := func(c *Config) { c.FinalSoftcap, c.LogitScale = capAt, 8 }
					violations := []struct {
						name string
						skip bool
						cfg  func(c *Config)
						mut  func(m *Model) func()
					}{
						{name: "softcap dropped", cfg: func(c *Config) { c.FinalSoftcap, c.LogitScale = 0, 8 }},
						{name: "scale inverted", cfg: func(c *Config) { c.FinalSoftcap, c.LogitScale = capAt, 1.0/8 }},
						{name: "bias negated", skip: outB == nil, cfg: full, mut: func(m *Model) func() {
							was := m.outB
							m.outB = slices.Clone(was)
							for i := range m.outB {
								m.outB[i] = -m.outB[i]
							}
							return func() { m.outB = was }
						}},
						{name: "norm bias dropped", skip: !c.LayerNorm || outNormB == nil, cfg: full, mut: func(m *Model) func() {
							was := m.outNormB
							m.outNormB = nil
							return func() { m.outNormB = was }
						}},
					}
					for _, v := range violations {
						if v.skip {
							continue
						}
						t.Run(v.name, func(t *testing.T) {
							a := &headArm{fx: fx, dev: dev, parts: tier.ComposedHead, xs: xs, lnb: lnb, outB: outB, want: full, cfg: v.cfg, mut: v.mut}
							a.run(t)
							n, d, _ := a.stats.ServedPart(tier.ComposedHead)
							if n != a.steps || d != 0 {
								t.Fatalf("the composed head served %d of %d steps and was declined %d: %s", n, a.steps, d, declinesOf(a.stats))
							}
							t.Logf("violation %q: worst NMSE %.3e from the oracle", v.name, a.worst)
							if !(a.worst > 1e-5) {
								t.Errorf("%s: the gate cannot see it -- worst NMSE %.3e from the oracle against a bound of %.0e", v.name, a.worst, bound)
							}
						})
					}
				})
			}
		})
	}
	if ran == 0 {
		t.Fatal("no arm ran")
	}
}

// TestComposedHeadForms states the head of each fixture with a final softcap
// and a logit scale beside the ops it stores (a LayerNorm with its bias, the
// projection's bias) and has the composer form it for a GPU and a CPU device,
// each kernel lowered for its device. Every tensor the container stores for
// the head is bound: a head that forms without its bias is a model the
// composer serves wrong, and a head that is not stated at all is refused by
// name.
func TestComposedHeadForms(t *testing.T) {
	step := jit.Step{Rows: 1, Page: 64, Keys: 16, ActWindow: 32}
	for _, fx := range headFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m := fx.open(t)
			defer m.Close()
			defer cfgMut(m, func(c *Config) { c.FinalSoftcap, c.LogitScale = 30, 8 })()
			if why := m.Uncovered(-1); why != Covered {
				t.Fatalf("the head is refused by %q (%v)", why, m.UncoveredErr(-1))
			}
			f := m.Composing()
			gr := f.HeadGraph()
			bound := map[jlm.Role]bool{}
			formats := map[jlm.Role]jlm.Type{}
			for _, o := range gr.Ops {
				for _, w := range o.Weight {
					bound[w.Role] = true
					e, ok := f.Find(w.Role, jlm.DenseBlock, -1)
					if !ok {
						t.Fatalf("the head binds %v, which the container does not store", w.Role)
					}
					formats[w.Role] = e.Type
				}
			}
			for _, r := range []jlm.Role{jlm.RoleOutputNorm, jlm.RoleOutputNormBias, jlm.RoleOutputBias} {
				if _, stored := f.Find(r, jlm.DenseBlock, -1); stored && !bound[r] {
					t.Errorf("the container stores %v and the head does not bind it", r)
				}
			}
			// The norm is the model's kind, the cap's sigmoid is in the graph
			// once, and the scale is its last op.
			caps := 0
			for _, o := range gr.Ops {
				if o.Kind == graph.OpMap && o.Attr.Fn == graph.MapSigmoid {
					caps++
				}
			}
			last := gr.Ops[len(gr.Ops)-1]
			if m.Cfg.LayerNorm != (gr.Ops[0].Kind == graph.OpLayerNorm) || caps != 1 ||
				last.Kind != graph.OpMap || last.Attr.Fn != graph.MapMul || last.Attr.Const != 1.0/8 {
				t.Errorf("the head's norm is %v for LayerNorm %v, it states %d softcap(s) and ends in %v %v by %v: want one softcap and a scale of 1/8",
					gr.Ops[0].Kind, m.Cfg.LayerNorm, caps, last.Kind, last.Attr.Fn, last.Attr.Const)
			}
			for _, d := range []jit.Device{{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}, {Kind: jit.DeviceCPU, Slots: 6, Lanes: 8}} {
				_, codes, err := jit.Default().Form(jit.Request{Graph: gr, Formats: formats, Step: step, Device: d})
				if err != nil {
					t.Fatalf("%v: %v", d.Kind, err)
				}
				for _, c := range codes {
					if d.Kind == jit.DeviceGPU && c.GPU == nil || d.Kind == jit.DeviceCPU && c.CPU == nil {
						t.Errorf("%v: %s has no code for the device", d.Kind, c.Name)
					}
				}
				t.Logf("%v: the %d-op head forms into %d kernels", d.Kind, len(gr.Ops), len(codes))
			}
		})
	}
}
