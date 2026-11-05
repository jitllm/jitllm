package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The state-space hybrids (jlm.LayerSSD) held to transformers' own classes:
// scripts/ssmgold.py builds each fixture with the family's class and random
// weights -- A, D and the dt bias included -- and writes it to GGUF with
// llama.cpp's own converter. Every tensor is F32, so the bound is c6NMSE. The
// golden is the class's full-sequence forward, Mamba-2's chunked scan; every
// path here steps the recurrence, so this holds the recurrence to the scan.

const ssmGoldScript = "scripts/ssmgold.py"

func openSSM(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "ssm", ssmGoldScript, name)
}

// ssmFixtures names each fixture and what its container must carry.
func ssmBaseFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-mamba2", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "mamba2" && c.SSD() && c.SSM.Groups == 1 && c.SSM.NHeadV == 8 &&
				c.SSM.StateSize == 16 && c.SSM.Inner == 128 && !c.TiedEmbd
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && l.NoFFN && len(l.Vector(jlm.RoleSSMD)) == 8 && len(l.Vector(jlm.RoleSSMConvBias)) == 128+2*16 &&
					len(l.Vector(jlm.RoleSSMNorm)) == 128 && c.LayerKind(i).Recurrent()
			}
			return ok
		}},
	}
}

// TestSSMFamiliesMatchTransformers holds each family to transformers' own
// class on decode, prefill at every length and the batched path, and the
// container's tokenizer to the fixture's tokenizer.json.
func TestSSMFamiliesMatchTransformers(t *testing.T) {
	for _, fx := range ssmFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openSSM(t, fx.name)
			defer m.Close()
			if !fx.sel(m) {
				t.Fatalf("the container does not carry the fixture's graph: %+v", m.Cfg)
			}
			if m.Vocab == nil {
				t.Fatalf("no tokenizer: %v", m.TokErr)
			}
			for _, tk := range g.Texts {
				if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
					t.Errorf("Encode(%q)\n  ours           %v\n  tokenizer.json %v", tk.Text, got, tk.IDs)
				}
			}
			worst := c6Worst(t, m, g, func(what string, p int, lg []float32) float64 {
				t.Helper()
				nmse := llama4Cmp(g.Pos[p].Head, lg)
				if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > c6NMSE {
					t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, c6NMSE)
				}
				if am := Greedy(lg); am != g.Pos[p].Argmax {
					t.Errorf("%s pos %d: argmax %d, transformers %d", what, p, am, g.Pos[p].Argmax)
				}
				return nmse
			}, openOpts...)
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// ssdScaled replaces one per-block vector of every Mamba-2 block with f of a
// copy of it.
func ssdScaled(m *model.Model, r jlm.Role, f func(v []float32)) func() {
	return eachLayer(m, func(l *model.Layer) {
		setVec(l, r, func(v []float32) []float32 {
			w := slices.Clone(v)
			f(w)
			return w
		})
	})
}

// ssdSwapBC swaps the B and C blocks of every Mamba-2 block's convolved
// channels -- the mixed projection's rows, the convolution's filters and its
// bias -- which is what a converter that left Mamba's x | B | C order, or
// reordered it to B | C | x, would hand the engine. It needs the fixture's
// F32 weights, whose rows are byte ranges.
func ssdSwapBC(t *testing.T) func(m *model.Model) func() {
	return func(m *model.Model) func() {
		g := m.Cfg.Delta()
		n := g.QkBytes
		// A block a device took gave its host page up; the swap reads it.
		for li := range m.Layers {
			if err := m.PageIn(li, &m.Layers[li]); err != nil {
				t.Fatal(err)
			}
		}
		return eachLayer(m, func(l *model.Layer) {
			if l.Vector(jlm.RoleSSMD) == nil {
				return
			}
			w := &l.Wq
			if l.SsmIn.Rows != 0 {
				w = &l.SsmIn
			}
			if w.Packed != nil || len(w.Data) != w.Rows*w.K*4 {
				t.Fatalf("ssdSwapBC needs an F32 mixed projection, have %v", w.Typ)
			}
			rb := w.K * 4
			d := slices.Clone(w.Data)
			copy(d[:n*rb], w.Data[n*rb:2*n*rb])
			copy(d[n*rb:2*n*rb], w.Data[:n*rb])
			w.Data = d
			cw := slices.Clone(l.SsmConv1d)
			for tap := 0; tap < g.Conv; tap++ {
				p := cw[tap*g.Chans:]
				o := l.SsmConv1d[tap*g.Chans:]
				copy(p[:n], o[n:2*n])
				copy(p[n:2*n], o[:n])
			}
			l.SsmConv1d = cw
			setVec(l, jlm.RoleSSMConvBias, func(v []float32) []float32 {
				cb := slices.Clone(v)
				copy(cb[:n], v[n:2*n])
				copy(cb[n:2*n], v[:n])
				return cb
			})
		})
	}
}

// ssdViolations are each Mamba-2 feature taken out on the host. Each must
// move the logits far past c6NMSE.
func ssdViolations(t *testing.T) []c6Violation {
	return []c6Violation{
		{"A positive (A_log's sign lost)", func(m *model.Model) func() {
			return ssdScaled(m, jlm.RoleSSMA, func(v []float32) {
				for i := range v {
					v[i] = -v[i]
				}
			})
		}},
		{"no D skip", func(m *model.Model) func() {
			return ssdScaled(m, jlm.RoleSSMD, func(v []float32) { clear(v) })
		}},
		{"no convolution bias", func(m *model.Model) func() {
			return ssdScaled(m, jlm.RoleSSMConvBias, func(v []float32) { clear(v) })
		}},
		{"no dt bias", func(m *model.Model) func() {
			return ssdScaled(m, jlm.RoleSSMDtBias, func(v []float32) { clear(v) })
		}},
		{"the norm before the gate", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.SSMNormBeforeGate = true })
		}},
		{"B and C swapped", ssdSwapBC(t)},
	}
}

// ssmViolations are the Mamba-2 violations and the fixture's family's own.
func ssmViolations(t *testing.T, name string, device bool) []c6Violation {
	vs := ssdViolations(t)
	if ssmNotMamba(name) {
		vs = nil
	}
	if name == "synth-falconh1-nonorm" {
		vs = slices.DeleteFunc(vs, func(v c6Violation) bool { return v.name == "the norm before the gate" })
	}
	if f := ssmFamilyViolations(name); f != nil {
		vs = append(vs, f(device)...)
	}
	return vs
}

// TestSSDFeaturesAreLoadBearing runs each violation through decode, plus the
// two the weights cannot express: the convolution window and the state not
// carried from one token to the next.
func TestSSDFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range ssmFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openSSM(t, fx.name)
			defer m.Close()
			// refused is the error a violation's run stopped at, by name.
			var refused error
			worst := func(between func(st *Instance)) float64 {
				st := testInstance(t, m, len(g.IDs)+1, openOpts...)
				defer st.Close()
				w := 0.0
				for p, id := range g.IDs {
					lg, err := st.Forward(id)
					if err != nil {
						refused = err
						return 0
					}
					w = max(w, llama4Cmp(g.Pos[p].Head, lg))
					if between != nil {
						between(st)
					}
				}
				return w
			}
			check := func(name string, w float64) {
				t.Helper()
				if err := refused; err != nil {
					refused = nil
					t.Logf("%s: %v -- refused by name, which the gate sees", name, err)
					return
				}
				if !(w > 1e3*c6NMSE) {
					t.Errorf("%s: the gate cannot see it: worst NMSE %.3e", name, w)
				}
				t.Logf("%s: worst NMSE %.3e", name, w)
			}
			for _, v := range ssmViolations(t, fx.name, false) {
				undo := v.mut(m)
				check(v.name, worst(nil))
				undo()
			}
			check("the convolution window not carried", worst(func(st *Instance) {
				for li := range st.view.Blocks() {
					clear(st.view.Conv(li))
				}
			}))
			if m.Cfg.ShortConv() {
				return // the window is a short convolution's only state
			}
			check("the state not carried", worst(func(st *Instance) {
				for li := range st.view.Blocks() {
					clear(st.view.State(li))
				}
			}))
		})
	}
}

// TestSSMFamiliesOnEveryDevice runs each fixture with every block and the head
// on each device present against the host, then each Mamba-2 feature taken
// out of the device session alone (fixtureOnEveryDevice).
func TestSSMFamiliesOnEveryDevice(t *testing.T) {
	for _, fx := range ssmFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openSSM(t, fx.name)
			defer m.Close()
			fixtureOnEveryDevice(t, m, g, ssmViolations(t, fx.name, true), openOpts...)
		})
	}
}

// The five principles on plain Mamba-2: paging, relocation (the recurrent
// summary moving with its block) and a batch that faults its blocks in run
// it, and so does the allocation gate (decodealloc_test.go). Each family's
// file registers its own fixtures the same way.
func init() {
}

// TestSSMFamiliesPageWithTheSameAnswer is the paging principle on every
// family, the mixtures' expert pages with it: under a one-page budget every
// block is evicted and re-read, recurrent and attending alike, and decode,
// prefill and the batch must give the resident logits bit for bit.
func TestSSMFamiliesPageWithTheSameAnswer(t *testing.T) {
	for _, fx := range ssmFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			_, g, _ := openSSM(t, fx.name)
			pagesWithTheSameAnswer(t, g.IDs, jlmOf(t, testmodels.Path(fx.name+".gguf")))
		})
	}
}
