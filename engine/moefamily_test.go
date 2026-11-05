package engine

import (
	"math"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/convert"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/format/jlm"
)

// The mixture families (glm4moe, qwen2moe, hunyuan, ernie4_5) held to
// transformers' own classes: scripts/moegold.py builds each fixture with the
// family's class and random weights -- the router's selection bias included,
// which transformers initialises to the identity -- and writes it to GGUF with
// llama.cpp's own converter. Every tensor is F32, so the bound is c6NMSE.

const moeGoldScript = "scripts/moegold.py"

func openMoE(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "moe", moeGoldScript, name)
}

// moeFixtures names each fixture and what its container must carry, so a
// fixture that lost a feature -- or a converter that miscounted one -- fails
// before it is compared.
func moeBaseFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-glm4moe", func(m *model.Model) bool {
			c := m.Cfg
			dense, moe := &m.Layers[0], &m.Layers[1]
			return c.Arch == "glm4moe" && c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 2.5 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NFFNShExp == c.NFFNExp && c.NDenseLead == 1 &&
				c.NLayer == 3 && c.NMTP == 1 && !c.TiedEmbd &&
				dense.Router.E == nil && dense.Gate.E != nil && dense.FfnNorm != nil &&
				moe.Router.E != nil && moe.Vector(jlm.RoleExpProbsB) != nil && moe.ShGate.E != nil && moe.Vector(jlm.RoleShRouter) == nil &&
				moe.FfnNorm != nil && moe.Vector(jlm.RolePostAttnNorm) == nil && moe.QNorm != nil &&
				moe.Vector(jlm.RoleAttnQBias) != nil && moe.Vector(jlm.RoleAttnKBias) != nil && moe.Vector(jlm.RoleAttnVBias) != nil && moe.Vector(jlm.RoleAttnOutBias) == nil &&
				m.Layers[3].Mtp != nil
		}},
		{"synth-qwen2moe", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "qwen2moe" && c.RopeNeox && c.NRot == c.HeadDim && !c.QKNorm &&
				!c.ExpertSigmoid && c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NFFNShExp == 2*c.NFFNExp && c.NDenseLead == 0 && c.NMTP == 0 &&
				c.NKVHead < c.NHead && l.Router.E != nil && l.Vector(jlm.RoleExpProbsB) == nil && l.ShGate.E != nil &&
				len(l.Vector(jlm.RoleShRouter)) == c.NEmbd && l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) == nil &&
				l.FfnNorm != nil && !c.TiedEmbd
		}},
		{"synth-ernie45moe", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[1]
			return c.Arch == "ernie4_5-moe" && !c.RopeNeox && c.NRot == c.HeadDim && !c.QKNorm &&
				!c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NFFNShExp == 2*c.NFFNExp && c.NDenseLead == 1 && c.MoEStep == 2 &&
				!c.MoEAt(0) && c.MoEAt(1) && !c.MoEAt(2) && c.MoEAt(3) && c.NMTP == 0 &&
				m.Layers[0].Gate.E != nil && m.Layers[2].Gate.E != nil && m.Layers[2].Router.E == nil &&
				l.Router.E != nil && len(l.Vector(jlm.RoleExpProbsB)) == c.NExpert && l.ShGate.E != nil && l.Vector(jlm.RoleShRouter) == nil &&
				l.Vector(jlm.RoleAttnQBias) == nil && l.Vector(jlm.RoleAttnOutBias) == nil && c.TiedEmbd
		}},
		{"synth-ernie45", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "llama" && !c.RopeNeox && c.NRot == c.HeadDim && c.NExpert == 0 &&
				l.Gate.E != nil && l.Vector(jlm.RoleAttnQBias) == nil && c.TiedEmbd
		}},
		{"synth-hunyuanmoe", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "hunyuan" && c.RopeNeox && c.NRot == c.HeadDim && c.HeadDim == 128 &&
				c.QKNorm && c.QKNormPost && !c.QKNormWide && !c.ExpertSigmoid && !c.NoExpertNorm &&
				c.ExpertScale == 1 && c.NExpert == 8 && c.NExpertUsed == 2 && c.NFFNShExp == c.NFFNExp &&
				c.NDenseLead == 0 && c.NKVHead < c.NHead && c.TiedEmbd
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && l.Router.E != nil && l.ShGate.E != nil && l.Vector(jlm.RoleShRouter) == nil &&
					l.Vector(jlm.RoleExpProbsB) == nil && l.Vector(jlm.RoleAttnQBias) == nil && foldedQK(l)
			}
			return ok
		}},
		{"synth-minimaxm2", func(m *model.Model) bool {
			c := m.Cfg
			ok := c.Arch == "minimax-m2" && c.RopeNeox && 2*c.NRot == c.HeadDim && c.QKNorm && c.QKNormWide &&
				c.ExpertSigmoid && !c.NoExpertNorm && c.ExpertScale == 1 && c.NExpert == 8 &&
				c.NExpertUsed == 2 && c.NFFNShExp == 0 && c.NDenseLead == 0 && c.NMTP == 0 &&
				c.NKVHead < c.NHead && !c.TiedEmbd
			for i := range m.Layers {
				l := &m.Layers[i]
				ok = ok && l.Router.E != nil && len(l.Vector(jlm.RoleExpProbsB)) == c.NExpert && l.ShGate.E == nil &&
					len(l.QNorm) == c.NHead*c.HeadDim && len(l.KNorm) == c.NKVHead*c.HeadDim &&
					l.Vector(jlm.RoleAttnQBias) == nil && l.FfnNorm != nil
			}
			return ok
		}},
		{"synth-hunyuan", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "hunyuan" && c.RopeNeox && c.NRot == c.HeadDim && c.QKNorm && c.QKNormPost &&
				c.NExpert == 0 && c.NKVHead < c.NHead && l.Gate.E != nil && l.Vector(jlm.RoleAttnQBias) == nil &&
				c.TiedEmbd && foldedQK(l)
		}},
	}
}

// foldedQK reports a block whose after-rotary q/k norm was folded at
// conversion: k's weight is ones and q's is not (jlm.FlagQKNormPostRope).
func foldedQK(l *model.Layer) bool {
	if l.QNorm == nil || len(l.KNorm) != len(l.QNorm) {
		return false
	}
	ones := 0
	for i := range l.KNorm {
		if l.KNorm[i] == 1 {
			ones++
		}
	}
	return ones == len(l.KNorm) && !slices.Equal(l.QNorm, l.KNorm)
}

// hunyuanNormBeforeRope runs the folded q norm before the rotary, where every
// other q/k norm runs: a weighted norm does not commute with a rotation.
func hunyuanNormBeforeRope(m *model.Model) func() {
	return cfgMut(m, func(c *model.Config) { c.QKNormPost = false })
}

// unweightedQKNorm drops the q/k norm's weights and keeps the normalisation.
func unweightedQKNorm(m *model.Model) func() {
	return eachLayer(m, func(l *model.Layer) {
		if l.QNorm == nil {
			return
		}
		ones := make([]float32, len(l.QNorm))
		for i := range ones {
			ones[i] = 1
		}
		l.QNorm = ones
	})
}

// moeViolations are the features each family adds, taken out one at a time on
// the host. Each must move the logits far past c6NMSE.
func moeBaseViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-glm4moe": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"no q/k/v biases", noQKVBias},
			{"NORM rotary for NEOX", hostNormRotary},
			{"the prediction block run as a trunk block", mtpInTrunk},
		},
		"synth-qwen2moe": {
			{"sigmoid for softmax", routerSwap(true)},
			{"the routed weights renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = false })
			}},
			{"no shared-expert gate", noSharedGate},
			{"no shared expert", noSharedExpert},
			{"no q/k/v biases", noQKVBias},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-ernie45moe": {
			{"sigmoid for softmax", routerSwap(true)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no shared expert", noSharedExpert},
			{"NEOX rotary for NORM", hostNeoxRotary},
		},
		"synth-hunyuanmoe": {
			{"the q/k norm before the rotary", hunyuanNormBeforeRope},
			{"an unweighted q/k norm", unweightedQKNorm},
			{"no q/k norm", noQKNorm},
			{"sigmoid for softmax", routerSwap(true)},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
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
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"a per-head q/k norm", perHeadQK},
			{"k's norm read from q's weight", kFromQ},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", hostNormRotary},
		},
	}
}

// moeDevViolations are the same features taken out while only the device
// session runs.
func moeBaseDevViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-glm4moe": {
			// With the selection bias kept, which is the device's softmax-with-bias
			// route (ERNIE's).
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no routed scale", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 }) }},
			{"no shared expert", noSharedExpert},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-qwen2moe": {
			{"sigmoid for softmax", routerSwap(true)},
			{"the routed weights renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = false })
			}},
			{"no shared-expert gate", noSharedGate},
			{"no shared expert", noSharedExpert},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-ernie45moe": {
			{"sigmoid for softmax", routerSwap(true)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no shared expert", noSharedExpert},
			{"NEOX rotary for NORM", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = true }) }},
		},
		"synth-hunyuanmoe": {
			{"the q/k norm before the rotary", hunyuanNormBeforeRope},
			{"an unweighted q/k norm", unweightedQKNorm},
			{"no q/k norm", noQKNorm},
			{"sigmoid for softmax", routerSwap(true)},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"no shared expert", noSharedExpert},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-hunyuan": {
			{"the q/k norm before the rotary", hunyuanNormBeforeRope},
			{"an unweighted q/k norm", unweightedQKNorm},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
		"synth-ernie45": {
			{"NEOX rotary for NORM", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = true }) }},
		},
		"synth-minimaxm2": {
			{"softmax for sigmoid", routerSwap(false)},
			{"no selection bias", noSelectionBias},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"a per-head q/k norm", perHeadQK},
			{"k's norm read from q's weight", kFromQ},
			{"no q/k norm", noQKNorm},
			{"NORM rotary for NEOX", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.RopeNeox = false }) }},
		},
	}
}

// noQKNorm drops the q/k norm by the one switch both tiers read (QKNormAt).
func noQKNorm(m *model.Model) func() {
	undoC := cfgMut(m, func(c *model.Config) { c.QKNorm = false })
	undoR := dropStored(m, jlm.RoleAttnQNorm, jlm.RoleAttnKNorm)
	return func() { undoR(); undoC() }
}

// hostNormRotary pairs the host's rotary (2i, 2i+1) where the model pairs
// (i, i+NRot/2). The pairing is read twice: the table built at Open (the
// model's nn.Rope) and the rotation an instance runs, which takes the segment's
// config (Instance.c) since the vision runner; both are flipped, or the flip
// reaches neither path and the violation measures nothing.
func hostNormRotary(m *model.Model) func() { return hostRotary(m, false) }

// hostNeoxRotary is hostNormRotary's opposite, for a model that pairs
// adjacent dimensions.
func hostNeoxRotary(m *model.Model) func() { return hostRotary(m, true) }

func hostRotary(m *model.Model, neox bool) func() {
	was := m.Rope.Neox
	m.Rope.Neox = neox
	undo := cfgMut(m, func(c *model.Config) { c.RopeNeox = neox })
	return func() { undo(); m.Rope.Neox = was }
}

// noSelectionBias drops the router's selection-only bias from every block.
func noSelectionBias(m *model.Model) func() {
	return dropVecs(m, jlm.RoleExpProbsB)
}

// noSharedExpert drops the shared expert, and its gate where it has one; the
// width goes too, since a device plan with a width and no weights is refused.
func noSharedExpert(m *model.Model) func() {
	undo := cfgMut(m, func(c *model.Config) { c.NFFNShExp = 0 })
	undoL := eachLayer(m, func(l *model.Layer) { l.ShGate, l.ShUp, l.ShDown = model.Tensor{}, model.Tensor{}, model.Tensor{} })
	undoR := dropStored(m, jlm.RoleShExpGate, jlm.RoleShExpUp, jlm.RoleShExpDown)
	undoV := dropVecs(m, jlm.RoleShRouter)
	return func() { undoV(); undoR(); undoL(); undo() }
}

// noSharedGate runs the shared expert at weight one: its sigmoid gate dropped.
func noSharedGate(m *model.Model) func() {
	return dropVecs(m, jlm.RoleShRouter)
}

// routerSwap scores every expert by its logit's sigmoid, or by the softmax
// over the experts: in the model's config, which main's kernels read, and
// stated by the converter, as a container of that router states it, for the
// front end and the composer.
func routerSwap(sigmoid bool) func(*model.Model) func() {
	return func(m *model.Model) func() {
		undoC := cfgMut(m, func(c *model.Config) { c.ExpertSigmoid = sigmoid })
		c := m.Container.Config()
		was := c.RouterScore
		s := *c
		s.Flags &^= jlm.FlagExpertSigmoid
		s.Flags2 &^= jlm.Flag2ExpertSqrtSoftplus
		if sigmoid {
			s.Flags |= jlm.FlagExpertSigmoid
		}
		c.RouterScore = convert.RouterScore(&s)
		undo := restate(m, func(map[jlm.Role]graph.Shape) {})
		return func() {
			c.RouterScore = was
			undo()
			undoC()
		}
	}
}

// routerViolations are the router's features a mixture m has, each taken out
// of the model's config or its stored tensors, which main's kernels and the
// front end both read: the selection bias, the group limit, the
// renormalisation, the routed scale, the shared expert's gate, and the
// number of experts chosen one fewer. A model whose blocks the front end
// refuses runs none of them composed, so it has none, and nor does one whose
// every block is a dense lead (synth-deepseek-yarn's three): no router runs.
func routerViolations(m *model.Model) []c6Violation {
	c := m.Cfg
	routes := false
	for li := range c.NLayer {
		routes = routes || c.MoEAt(li)
	}
	if !routes || coveredBlocks(m) != c.NLayer {
		return nil
	}
	var vs []c6Violation
	stores := func(r jlm.Role) bool {
		for li := range c.NLayer {
			if m.Stores(li, r) {
				return true
			}
		}
		return false
	}
	if stores(jlm.RoleExpProbsB) {
		vs = append(vs, c6Violation{"router: the selection bias dropped", noSelectionBias})
	}
	if c.NExpertGroup > 1 && c.NExpertGroupUsed < c.NExpertGroup {
		vs = append(vs, c6Violation{"router: the group limit removed", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.NExpertGroupUsed = c.NExpertGroup })
		}})
	}
	if !c.NoExpertNorm {
		vs = append(vs, c6Violation{"router: the renormalisation removed", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
		}})
	}
	// A sparsemixer router is a top-2 the tier's own kernels run alone
	// (checkSparseMixer): with one fewer chosen it places no block, and its
	// stated weight is held by the violation below instead.
	if c.NExpertUsed > 1 && c.SparseMixer == 0 {
		vs = append(vs, c6Violation{"router: one expert fewer chosen", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.NExpertUsed-- })
		}})
	}
	if c.ExpertScale != 0 && c.ExpertScale != 1 {
		vs = append(vs, c6Violation{"router: the routed scale dropped", func(m *model.Model) func() {
			return cfgMut(m, func(c *model.Config) { c.ExpertScale = 1 })
		}})
	}
	if stores(jlm.RoleShRouter) {
		vs = append(vs, c6Violation{"router: the shared expert's gate dropped", noSharedGate})
	}
	if cc := m.Container.Config(); len(cc.RouterWeight.Terms) != 0 {
		vs = append(vs, c6Violation{"router: the stated weight replaced by the selected score", func(m *model.Model) func() {
			return statedRouter(m, cc.RouterScore, jlm.Expr{})
		}})
	}
	if stores(jlm.RoleExpScale) {
		vs = append(vs, c6Violation{"router: the per-expert scale dropped", func(m *model.Model) func() {
			return dropStored(m, jlm.RoleExpScale)
		}})
	}
	return vs
}

// mtpInTrunk runs the prediction blocks as more trunk blocks: what a converter
// that did not set them aside would hand the engine.
func mtpInTrunk(m *model.Model) func() {
	undoC := cfgMut(m, func(c *model.Config) { c.NLayer, c.NMTP = c.NLayer+c.NMTP, 0 })
	undoR := dropStored(m, jlm.RoleNextnEHProj, jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnHeadNorm,
		jlm.RoleNextnHead, jlm.RoleNextnEmbd)
	return func() { undoR(); undoC() }
}

// TestMoEFamiliesMatchTransformers holds each mixture family to transformers'
// own class on decode, prefill at every length and the batched path, and the
// container's tokenizer to the family's tokenizer.json.
func TestMoEFamiliesMatchTransformers(t *testing.T) {
	for _, fx := range moeFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openMoE(t, fx.name)
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

// TestMoEFamilyFeaturesAreLoadBearing runs each violation through decode: a
// feature the fixture cannot see is one TestMoEFamiliesMatchTransformers
// certifies nothing about.
func TestMoEFamilyFeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range moeFixtures() {
		vs := moeViolations()[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g, openOpts := openMoE(t, fx.name)
			defer m.Close()
			for _, v := range vs {
				t.Run(v.name, func(t *testing.T) {
					undo := v.mut(m)
					defer undo()
					st := testInstance(t, m, len(g.IDs)+1, openOpts...)
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
	for _, fx := range moeFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openMoE(t, fx.name)
			defer m.Close()
			fixtureOnEveryDeviceCounted(t, m, g, moeDevViolations()[fx.name], 1e-5, true, openOpts...)
		})
	}
}

// TestMoEFamiliesPageWithTheSameAnswer is the paging principle on each
// family: under a one-page budget every block and every expert page is
// evicted and re-read, and decode, prefill and the batched path must give the
// resident model's logits bit for bit. The selection check is the fault
// count: a budget that held the model would compare it with itself.
func TestMoEFamiliesPageWithTheSameAnswer(t *testing.T) {
	for _, fx := range moeFixtures() {
		t.Run(fx.name, func(t *testing.T) {
			_, g, _ := openMoE(t, fx.name)
			pagesWithTheSameAnswer(t, g.IDs, jlmOf(t, testmodels.Path(fx.name+".gguf")))
		})
	}
}

// TestHybridMixturesPageWithTheSameAnswer is the mixtures' paging gate on the
// hybrid mixtures (principleMixtures): their linear blocks' recurrent state
// stays put while the blocks' and the experts' pages are evicted and read
// back, and every logit is the resident run's.
func TestHybridMixturesPageWithTheSameAnswer(t *testing.T) {
	for _, name := range []string{"synth-qwen35moe-hybrid-mtp-q8_0.gguf", "synth-kimilinear" + q8Suffix} {
		t.Run(name, func(t *testing.T) {
			src := jlmOf(t, testmodels.Path(name))
			m, err := model.Open(src)
			if err != nil {
				t.Fatal(err)
			}
			ids := m.Vocab.Encode("The clerk counted barrels of salt on the upper floor, and the river rose", true)
			m.Close()
			pagesWithTheSameAnswer(t, ids, src)
		})
	}
}

// pagesWithTheSameAnswer runs src's decode over ids, their prefill and the
// batched path in pairs, resident and under a one-page budget, and holds every
// logit bit for bit; the page-in count is the selection check.
func pagesWithTheSameAnswer(t *testing.T, ids []int32, src string) {
	t.Helper()
	run := func(mo ...model.Option) ([][]float32, int64) {
		instOpts := []Option{WithKVF16(false)}
		m, err := model.Open(src, mo...)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		var out [][]float32
		st := testInstance(t, m, len(ids)+1, instOpts...)
		for _, id := range ids {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(lg))
		}
		st.Close()
		pf := testInstance(t, m, len(ids)+1, instOpts...)
		lg, err := pf.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, slices.Clone(lg))
		pf.Close()
		b := newSessBatch(m, 2, len(ids)+1, instOpts...)
		defer b.Close()
		for p := 0; p+1 < len(ids); p += 2 {
			rows, err := b.ForwardBatch([]int32{ids[p], ids[p+1]})
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, slices.Clone(rows[:2*m.Cfg.NVocab]))
		}
		_, in, _ := m.PageStats(m.Frames)
		return out, in
	}
	want, residentIn := run()
	got, pagedIn := run(model.WithPageBudget(1))
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
}
