package model

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/format/jlm"
)

// The mixture families (glm4moe, qwen2moe, hunyuan, ernie4_5) held to
// transformers' own classes: scripts/moegold.py builds each fixture with the
// family's class and random weights -- the router's selection bias included,
// which transformers initialises to the identity -- and writes it to GGUF with
// llama.cpp's own converter. Every tensor is F32, so the bound is c6NMSE.

const moeGoldScript = "scripts/moegold.py"

func openMoE(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "moe", moeGoldScript, name)
}

// moeFixtures names each fixture and what its container must carry, so a
// fixture that lost a feature -- or a converter that miscounted one -- fails
// before it is compared.
var moeFixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-glm4moe", func(m *Model) bool {
		c := m.Cfg
		dense, moe := &m.layers[0], &m.layers[1]
		return c.Arch == "glm4moe" && c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm &&
			c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpert == 8 &&
			c.NExpertUsed == 2 && c.NFFNShExp == c.NFFNExp && c.NDenseLead == 1 &&
			c.NLayer == 3 && c.NMTP == 1 && !c.TiedEmbd &&
			dense.router.e == nil && dense.gate.e != nil && dense.ffnNorm != nil &&
			moe.router.e != nil && moe.expProbsB != nil && moe.shGate.e != nil && moe.shRouter == nil &&
			moe.ffnNorm != nil && moe.postAttnNorm == nil && moe.qNorm != nil &&
			moe.bq != nil && moe.bk != nil && moe.bv != nil && moe.bo == nil &&
			m.layers[3].mtp != nil
	}},
	{"synth-qwen2moe", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "qwen2moe" && c.RopeNeox && c.NRot == c.HeadDim && !c.QKNorm &&
			!c.ExpertSigmoid && c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
			c.NExpertUsed == 2 && c.NFFNShExp == 2*c.NFFNExp && c.NDenseLead == 0 && c.NMTP == 0 &&
			c.NKVHead < c.NHead && l.router.e != nil && l.expProbsB == nil && l.shGate.e != nil &&
			len(l.shRouter) == c.NEmbd && l.bq != nil && l.bk != nil && l.bv != nil && l.bo == nil &&
			l.ffnNorm != nil && !c.TiedEmbd
	}},
	{"synth-ernie45moe", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[1]
		return c.Arch == "ernie4_5-moe" && !c.RopeNeox && c.NRot == c.HeadDim && !c.QKNorm &&
			!c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
			c.NExpertUsed == 2 && c.NFFNShExp == 2*c.NFFNExp && c.NDenseLead == 1 && c.MoEStep == 2 &&
			!c.MoEAt(0) && c.MoEAt(1) && !c.MoEAt(2) && c.MoEAt(3) && c.NMTP == 0 &&
			m.layers[0].gate.e != nil && m.layers[2].gate.e != nil && m.layers[2].router.e == nil &&
			l.router.e != nil && len(l.expProbsB) == c.NExpert && l.shGate.e != nil && l.shRouter == nil &&
			l.bq == nil && l.bo == nil && c.TiedEmbd
	}},
	{"synth-ernie45", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "llama" && !c.RopeNeox && c.NRot == c.HeadDim && c.NExpert == 0 &&
			l.gate.e != nil && l.bq == nil && c.TiedEmbd
	}},
	{"synth-hunyuanmoe", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "hunyuan" && c.RopeNeox && c.NRot == c.HeadDim && c.HeadDim == 128 &&
			c.QKNorm && c.QKNormPost && !c.QKNormWide && !c.ExpertSigmoid && !c.NoExpertNorm &&
			c.ExpertScale == 1 && c.NExpert == 8 && c.NExpertUsed == 2 && c.NFFNShExp == c.NFFNExp &&
			c.NDenseLead == 0 && c.NKVHead < c.NHead && c.TiedEmbd
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.router.e != nil && l.shGate.e != nil && l.shRouter == nil &&
				l.expProbsB == nil && l.bq == nil && foldedQK(l)
		}
		return ok
	}},
	{"synth-minimaxm2", func(m *Model) bool {
		c := m.Cfg
		ok := c.Arch == "minimax-m2" && c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm && c.QKNormWide &&
			c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
			c.NExpertUsed == 2 && c.NFFNShExp == 0 && c.NDenseLead == 0 && c.NMTP == 0 &&
			c.NKVHead < c.NHead && !c.TiedEmbd
		for i := range m.layers {
			l := &m.layers[i]
			ok = ok && l.router.e != nil && len(l.expProbsB) == c.NExpert && l.shGate.e == nil &&
				len(l.qNorm) == c.NHead*c.HeadDim && len(l.kNorm) == c.NKVHead*c.HeadDim &&
				l.bq == nil && l.ffnNorm != nil
		}
		return ok
	}},
	{"synth-hunyuan", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "hunyuan" && c.RopeNeox && c.NRot == c.HeadDim && c.QKNorm && c.QKNormPost &&
			c.NExpert == 0 && c.NKVHead < c.NHead && l.gate.e != nil && l.bq == nil &&
			c.TiedEmbd && foldedQK(l)
	}},
}

// foldedQK reports a block whose after-rotary q/k norm was folded at
// conversion: k's weight is ones and q's is not (jlm.FlagQKNormPostRope).
func foldedQK(l *layer) bool {
	if l.qNorm == nil || len(l.kNorm) != len(l.qNorm) {
		return false
	}
	ones := 0
	for i := range l.kNorm {
		if l.kNorm[i] == 1 {
			ones++
		}
	}
	return ones == len(l.kNorm) && !slices.Equal(l.qNorm, l.kNorm)
}

// hunyuanNormBeforeRope runs the folded q norm before the rotary, where every
// other q/k norm runs: a weighted norm does not commute with a rotation.
func hunyuanNormBeforeRope(m *Model) func() {
	return cfgMut(m, func(c *Config) { c.QKNormPost = false })
}

// unweightedQKNorm drops the q/k norm's weights and keeps the normalisation.
func unweightedQKNorm(m *Model) func() {
	return eachLayer(m, func(l *layer) {
		if l.qNorm == nil {
			return
		}
		ones := make([]float32, len(l.qNorm))
		for i := range ones {
			ones[i] = 1
		}
		l.qNorm = ones
	})
}

// moeViolations are the features each family adds, taken out one at a time on
// the host. Each must move the logits far past c6NMSE.
var moeViolations = map[string][]c6Violation{
	"synth-glm4moe": {
		{"softmax for sigmoid", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = false }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
		{"no shared expert", noSharedExpert},
		{"no q/k norm", noQKNorm},
		{"no q/k/v biases", noQKVBias},
		{"NORM rotary for NEOX", hostNormRotary},
		{"the prediction block run as a trunk block", mtpInTrunk},
	},
	"synth-qwen2moe": {
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"the routed weights renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = false })
		}},
		{"no shared-expert gate", noSharedGate},
		{"no shared expert", noSharedExpert},
		{"no q/k/v biases", noQKVBias},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-ernie45moe": {
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no shared expert", noSharedExpert},
		{"NEOX rotary for NORM", hostNeoxRotary},
	},
	"synth-hunyuanmoe": {
		{"the q/k norm before the rotary", hunyuanNormBeforeRope},
		{"an unweighted q/k norm", unweightedQKNorm},
		{"no q/k norm", noQKNorm},
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no shared expert", noSharedExpert},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-hunyuan": {
		{"the q/k norm before the rotary", hunyuanNormBeforeRope},
		{"an unweighted q/k norm", unweightedQKNorm},
		{"no q/k norm", noQKNorm},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-ernie45": {
		{"NEOX rotary for NORM", hostNeoxRotary},
	},
	"synth-minimaxm2": {
		{"softmax for sigmoid", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = false }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"a per-head q/k norm", perHeadQK},
		{"k's norm read from q's weight", kFromQ},
		{"no q/k norm", noQKNorm},
		{"NORM rotary for NEOX", hostNormRotary},
	},
}

// moeDevViolations are the same features taken out while only the device
// session runs.
var moeDevViolations = map[string][]c6Violation{
	"synth-glm4moe": {
		// With the selection bias kept, which is the device's softmax-with-bias
		// route (ERNIE's).
		{"softmax for sigmoid", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = false }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no routed scale", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertScale = 1 }) }},
		{"no shared expert", noSharedExpert},
		{"no q/k norm", noQKNorm},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-qwen2moe": {
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"the routed weights renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = false })
		}},
		{"no shared-expert gate", noSharedGate},
		{"no shared expert", noSharedExpert},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-ernie45moe": {
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no shared expert", noSharedExpert},
		{"NEOX rotary for NORM", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = true }) }},
	},
	"synth-hunyuanmoe": {
		{"the q/k norm before the rotary", hunyuanNormBeforeRope},
		{"an unweighted q/k norm", unweightedQKNorm},
		{"no q/k norm", noQKNorm},
		{"sigmoid for softmax", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = true }) }},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"no shared expert", noSharedExpert},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-hunyuan": {
		{"the q/k norm before the rotary", hunyuanNormBeforeRope},
		{"an unweighted q/k norm", unweightedQKNorm},
		{"no q/k norm", noQKNorm},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
	"synth-ernie45": {
		{"NEOX rotary for NORM", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = true }) }},
	},
	"synth-minimaxm2": {
		{"softmax for sigmoid", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ExpertSigmoid = false }) }},
		{"no selection bias", noSelectionBias},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"a per-head q/k norm", perHeadQK},
		{"k's norm read from q's weight", kFromQ},
		{"no q/k norm", noQKNorm},
		{"NORM rotary for NEOX", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.RopeNeox = false }) }},
	},
}

// noQKNorm drops the q/k norm by the one switch both tiers read (QKNormAt).
func noQKNorm(m *Model) func() {
	undoC := cfgMut(m, func(c *Config) { c.QKNorm = false })
	undoR := dropStored(m, jlm.RoleAttnQNorm, jlm.RoleAttnKNorm)
	return func() { undoR(); undoC() }
}

// hostNormRotary pairs the host's rotary (2i, 2i+1) where the model pairs
// (i, i+NRot/2). The pairing is read twice: the table built at Open (the
// model's nn.Rope) and the rotation a State runs, which takes the segment's
// config (State.c) since the vision runner; both are flipped, or the flip
// reaches neither path and the violation measures nothing.
func hostNormRotary(m *Model) func() { return hostRotary(m, false) }

// hostNeoxRotary is hostNormRotary's opposite, for a model that pairs
// adjacent dimensions.
func hostNeoxRotary(m *Model) func() { return hostRotary(m, true) }

func hostRotary(m *Model, neox bool) func() {
	was := m.rope.Neox
	m.rope.Neox = neox
	undo := cfgMut(m, func(c *Config) { c.RopeNeox = neox })
	return func() { undo(); m.rope.Neox = was }
}

// noSelectionBias drops the router's selection-only bias from every block.
func noSelectionBias(m *Model) func() {
	return eachLayer(m, func(l *layer) { l.expProbsB = nil })
}

// noSharedExpert drops the shared expert, and its gate where it has one; the
// width goes too, since a device plan with a width and no weights is refused.
func noSharedExpert(m *Model) func() {
	undo := cfgMut(m, func(c *Config) { c.NFFNShExp = 0 })
	undoL := eachLayer(m, func(l *layer) { l.shGate, l.shUp, l.shDown, l.shRouter = tensor{}, tensor{}, tensor{}, nil })
	return func() { undoL(); undo() }
}

// noSharedGate runs the shared expert at weight one: its sigmoid gate dropped.
func noSharedGate(m *Model) func() {
	return eachLayer(m, func(l *layer) { l.shRouter = nil })
}

// mtpInTrunk runs the prediction blocks as more trunk blocks: what a converter
// that did not set them aside would hand the engine.
func mtpInTrunk(m *Model) func() {
	return cfgMut(m, func(c *Config) { c.NLayer, c.NMTP = c.NLayer+c.NMTP, 0 })
}

// TestMoEFamiliesMatchTransformers holds each mixture family to transformers'
// own class on decode, prefill at every length and the batched path, and the
// container's tokenizer to the family's tokenizer.json.
func TestMoEFamiliesMatchTransformers(t *testing.T) {
	for _, fx := range moeFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openMoE(t, fx.name)
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
			})
			t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
		})
	}
}

// TestMoEFamilyFeaturesAreLoadBearing runs each violation through decode: a
// feature the fixture cannot see is one TestMoEFamiliesMatchTransformers
// certifies nothing about.
func TestMoEFamilyFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range moeFixtures {
		vs := moeViolations[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g := openMoE(t, fx.name)
			defer m.Close()
			for _, v := range vs {
				t.Run(v.name, func(t *testing.T) {
					undo := v.mut(m)
					defer undo()
					st := m.NewState(len(g.IDs) + 1)
					defer st.Close()
					worst := 0.0
					for p, id := range g.IDs {
						lg, err := st.Forward(id)
						if err != nil {
							t.Fatal(err)
						}
						worst = max(worst, llama4Cmp(g.Pos[p].Head, lg))
					}
					if !(worst > 1e3*c6NMSE) {
						t.Errorf("the gate cannot see it: worst NMSE %.3e", worst)
					}
					t.Logf("worst NMSE %.3e", worst)
				})
			}
		})
	}
}

// TestMoEFamiliesOnEveryDevice runs each fixture with every block and the head
// on each device present against the host, then each feature taken out of the
// device session alone (fixtureOnEveryDevice).
func TestMoEFamiliesOnEveryDevice(t *testing.T) {
	for _, fx := range moeFixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openMoE(t, fx.name)
			defer m.Close()
			fixtureOnEveryDevice(t, m, g, moeDevViolations[fx.name])
		})
	}
}

// TestMoEFamiliesPageWithTheSameAnswer is the paging principle on each
// family: under a one-page budget every block and every expert page is
// evicted and re-read, and decode, prefill and the batched path must give the
// resident model's logits bit for bit. The selection check is the fault
// count: a budget that held the model would compare it with itself.
func TestMoEFamiliesPageWithTheSameAnswer(t *testing.T) {
	for _, fx := range moeFixtures {
		t.Run(fx.name, func(t *testing.T) {
			_, g := openMoE(t, fx.name)
			pagesWithTheSameAnswer(t, g.IDs, jlmOf(t, testmodels.Path(fx.name+".gguf")))
		})
	}
}

// pagesWithTheSameAnswer runs src's decode over ids, their prefill and the
// batched path in pairs, resident and under a one-page budget, and holds every
// logit bit for bit; the page-in count is the selection check.
func pagesWithTheSameAnswer(t *testing.T, ids []int32, src string) {
	t.Helper()
	run := func(opts ...Option) ([][]float32, int64) {
		m, err := Open(src, append([]Option{noTune, WithKVF16(false),
			WithJITOptions(nn.WithGEMMExact(true))}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		var out [][]float32
		st := m.NewState(len(ids) + 1)
		for _, id := range ids {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
		}
		st.Close()
		pf := m.NewState(len(ids) + 1)
		lg, err := pf.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
		pf.Close()
		b := m.NewBatch(2, len(ids)+1)
		defer b.Close()
		for p := 0; p+1 < len(ids); p += 2 {
			rows, err := b.ForwardBatch([]int32{ids[p], ids[p+1]})
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(rows[:2*m.Cfg.NVocab]))
		}
		_, in, _ := m.PageStats()
		return out, in
	}
	want, residentIn := run()
	got, pagedIn := run(WithPageBudget(1))
	if pagedIn <= residentIn {
		t.Fatalf("a one-page budget took %d page-ins against %d resident: nothing paged, "+
			"this gate proved nothing", pagedIn, residentIn)
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("row %d: the paged model's logits differ from the resident one's (NMSE %.3e)",
				i, nmseF32(want[i], got[i]))
		}
	}
	t.Logf("%d page-ins under one page against %d resident; every logit identical", pagedIn, residentIn)
}

// nmseF32 is the NMSE of got against want.
func nmseF32(want, got []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i] - want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	return num / den
}

// Hunyuan's dense fixture runs the five principles' relocation and paging gates
// (principleFixtures): its after-rotary norm caches an unweighted k on every
// tier, which a relocated history must read the same way. The mixture pages
// under TestMoEFamiliesPageWithTheSameAnswer, as the other families do --
// TestPagingDoesNotChangeTheAnswer counts frames of one page size, and a
// mixture's expert pages are another.
func init() {
	principleFixtures = append(principleFixtures, "synth-hunyuan.gguf")
}
