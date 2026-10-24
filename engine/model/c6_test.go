package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
)

// c6Golden is scripts/c6gold.py's golden: transformers' own class over twelve
// fixed ids, and the family's real tokenizer over a few texts. The GGUF beside
// it was written by llama.cpp's convert_hf_to_gguf.py from the same weights.
type c6Golden struct {
	IDs []int32 `json:"ids"`
	Pos []struct {
		Argmax int32     `json:"argmax"`
		Max    float64   `json:"max"`
		Head   []float64 `json:"head"`
	} `json:"pos"`
	Texts []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	} `json:"texts"`
}

// c6NMSE is the bound against transformers. Every tensor is F32 and the KV
// cache f32, so reduction order is what separates the arms: measured at
// ~1e-13, and every violation below lands at 1e-5 or far above.
const c6NMSE = 1e-9

// openC6 opens a C6 fixture and its golden. A missing fixture fails rather than
// skips; the message names the script that writes it.
func openC6(t *testing.T, name string) (*Model, *c6Golden) {
	t.Helper()
	return openGolden(t, "c6", "scripts/c6gold.py", name)
}

// openGolden opens a fixture whose golden lives in testdata/golden/<dir>,
// written by script.
func openGolden(t *testing.T, dir, script, name string) (*Model, *c6Golden) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", dir, name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &c6Golden{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	path, ok := existingModel(testmodels.Path(name + ".gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(name+".gguf")+
			" (set JITLLM_MODELS to the model directory) -- run "+script+" "+name+" (RULE 11)")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

// c6Worst runs the whole golden through decode, prefill at every length and
// the batched path, and returns the worst NMSE. check reports each position.
func c6Worst(t *testing.T, m *Model, g *c6Golden, check func(what string, p int, lg []float32) float64) float64 {
	t.Helper()
	worst := 0.0
	st := m.NewState(len(g.IDs) + 1)
	for p, id := range g.IDs {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("forward", p, lg))
	}
	st.Close()
	for n := 1; n <= len(g.IDs); n++ {
		pf := m.NewState(len(g.IDs) + 1)
		lg, err := pf.Prefill(g.IDs[:n])
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("prefill", n-1, lg))
		pf.Close()
	}
	// Two sequences in lockstep through the batched graph, the second a
	// position behind.
	b := m.NewBatch(2, len(g.IDs)+1)
	defer b.Close()
	if _, err := b.PrefillSeq(0, g.IDs[:2]); err != nil {
		t.Fatal(err)
	}
	if _, err := b.PrefillSeq(1, g.IDs[:1]); err != nil {
		t.Fatal(err)
	}
	nv := m.Cfg.NVocab
	for p := 2; p < len(g.IDs); p++ {
		rows, err := b.ForwardBatch([]int32{g.IDs[p], g.IDs[p-1]})
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("batch", p, rows[:nv]))
		worst = max(worst, check("batch-lag", p-1, rows[nv:2*nv]))
	}
	return worst
}

// c6Fixtures names each fixture and what its container must carry, so a
// fixture that lost a feature fails before it is compared.
var c6Fixtures = []struct {
	name string
	sel  func(m *Model) bool
}{
	{"synth-phi2", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "phi2" && c.LayerNorm && c.Parallel && c.RopeNeox &&
			c.NRot < c.HeadDim && l.ffnNorm == nil && l.gate.e == nil &&
			l.attnNormB != nil && l.upB != nil && l.downB != nil && l.bq != nil &&
			l.bo != nil && m.outB != nil && m.outNormB != nil
	}},
	{"synth-starcoder", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "starcoder" && c.LayerNorm && !c.Parallel && c.NoPosEnc &&
			c.NKVHead == 1 && c.NHead > 1 && m.posEmbd.e != nil && l.ffnNorm != nil &&
			l.gate.e == nil && l.attnNormB != nil && l.ffnNormB != nil && l.upB != nil &&
			l.downB != nil && l.bq != nil && l.bk != nil && l.bv != nil && l.bo != nil &&
			m.outNormB != nil && c.TiedEmbd
	}},
	{"synth-stablelm", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "stablelm" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
			c.NRot*4 == c.HeadDim && l.ffnNorm != nil && l.gate.e != nil && l.bq != nil &&
			l.bo == nil && l.attnNormB != nil && m.outNormB != nil && !c.TiedEmbd
	}},
	{"synth-stablelm-par", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "stablelm" && c.LayerNorm && c.Parallel && l.ffnNorm == nil && l.gate.e != nil
	}},
	{"synth-starcoder2", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "starcoder2" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
			c.NRot == c.HeadDim && c.NKVHead < c.NHead && l.gate.e == nil && l.ffnNorm != nil &&
			l.attnNormB != nil && l.ffnNormB != nil && l.upB != nil && l.downB != nil &&
			l.bq != nil && l.bo != nil && m.outNormB != nil
	}},
	{"synth-falcon", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "falcon" && c.LayerNorm && c.Parallel && c.RopeNeox && c.NKVHead == 1 &&
			l.ffnNorm == nil && l.gate.e == nil && l.upB == nil && l.bq == nil && l.attnNormB != nil
	}},
	{"synth-falcon40", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "falcon" && c.Parallel && c.NKVHead == 2 && l.ffnNorm != nil &&
			l.ffnNormB != nil && l.attnNormB != nil && l.gate.e == nil
	}},
	{"synth-nemotron", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "nemotron" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
			2*c.NRot == c.HeadDim && c.Act == nn.ActReLU2 && l.gate.e == nil &&
			l.ffnNorm != nil && l.attnNormB != nil && l.upB != nil && l.downB != nil &&
			m.outNormB != nil && !c.TiedEmbd
	}},
	{"synth-commandr", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "command-r" && c.LayerNorm && c.Parallel && !c.RopeNeox &&
			c.LogitScale == 16 && c.TiedEmbd && l.ffnNorm == nil && l.gate.e != nil &&
			l.attnNormB == nil && m.outNormB == nil && c.NKVHead < c.NHead && c.SWAWindow == 0
	}},
	{"synth-dbrx", func(m *Model) bool {
		c, l := m.Cfg, &m.layers[0]
		return c.Arch == "dbrx" && c.LayerNorm && !c.Parallel && c.RopeNeox && c.NRot == c.HeadDim &&
			c.ClampKQV == 1.5 && c.NExpert == 4 && c.NExpertUsed == 2 && !c.NoExpertNorm &&
			c.NKVHead < c.NHead && l.router.e != nil && l.ffnNorm != nil && l.attnNormB == nil &&
			l.ffnNormB == nil && l.bq == nil && m.outNormB == nil && !c.TiedEmbd
	}},
	{"synth-cohere2", func(m *Model) bool {
		c := m.Cfg
		return c.Arch == "command-r" && c.LayerNorm && c.Parallel && c.NoPEGlobal &&
			c.SWAWindow == 4 && c.SWAPeriod == 4 && c.LogitScale == 4 &&
			c.SWA(0) && !c.SWA(3) && c.RopeAt(0) && !c.RopeAt(3)
	}},
}

// swapNorms feeds each branch of a two-norm parallel block the other's norm:
// Falcon-40B's GGUF names them the other way round from their use.
func swapNorms(m *Model) func() {
	return eachLayer(m, func(l *layer) {
		l.attnNorm, l.ffnNorm = l.ffnNorm, l.attnNorm
		l.attnNormB, l.ffnNormB = l.ffnNormB, l.attnNormB
	})
}

// onePlusAgain adds LayerNorm1p's 1 to every norm weight a second time, which
// is what an engine that applied 1p to a file whose converter already folded
// it in would compute.
func onePlusAgain(m *Model) func() {
	plus := func(w []float32) []float32 {
		o := slices.Clone(w)
		for i := range o {
			o[i]++
		}
		return o
	}
	was := m.outNorm
	m.outNorm = plus(was)
	undo := eachLayer(m, func(l *layer) { l.attnNorm, l.ffnNorm = plus(l.attnNorm), plus(l.ffnNorm) })
	return func() { undo(); m.outNorm = was }
}

// logitScale takes Cohere's logit scale away.
func logitScale(m *Model) func() { return cfgMut(m, func(c *Config) { c.LogitScale = 1 }) }

// serialFromShared turns a parallel block sharing one norm into a serial one
// with that norm on both branches.
func serialFromShared(m *Model) func() {
	undo := cfgMut(m, func(c *Config) { c.Parallel = false })
	undoL := eachLayer(m, func(l *layer) { l.ffnNorm, l.ffnNormB = l.attnNorm, l.attnNormB })
	undoR := m.restate(func(inv map[jlm.Role]graph.Shape) {
		for _, r := range [][2]jlm.Role{{jlm.RoleAttnNorm, jlm.RoleFFNNorm}, {jlm.RoleAttnNormBias, jlm.RoleFFNNormBias}} {
			if s, ok := inv[r[0]]; ok {
				inv[r[1]] = s
			}
		}
	})
	return func() { undoR(); undoL(); undo() }
}

// noNormBiases takes the pre-norms' biases away, in the model's layers and as
// the composer reads the container; rmsForLayerNorm is the config's
// mean-centring off with them.
func noNormBiases(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.attnNormB, l.ffnNormB = nil, nil })
	undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
	return func() { undoR(); undoL() }
}

func rmsForLayerNorm(m *Model) func() {
	undoC := cfgMut(m, func(c *Config) { c.LayerNorm = false })
	undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
	return func() { undoR(); undoC() }
}

// noFFNBiases takes an ungated FFN's two biases away, in the layers and as
// the composer reads the container.
func noFFNBiases(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.upB, l.downB = nil, nil })
	undoR := dropStored(m, jlm.RoleFFNUpBias, jlm.RoleFFNDownBias)
	return func() { undoR(); undoL() }
}

// oneNormForBoth makes a two-norm parallel block share its attention's norm.
func oneNormForBoth(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.ffnNorm, l.ffnNormB = nil, nil })
	undoR := dropStored(m, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	return func() { undoR(); undoL() }
}

// noPosTable takes the learned position table away.
func noPosTable(m *Model) func() {
	was := m.posEmbd
	m.posEmbd = tensor{}
	return func() { m.posEmbd = was }
}

// TestC6MatchesTransformers holds each classic-transformer graph (C6) to
// transformers' own class, on every entry point that attends: decode, prefill
// at every length and the batched path. The tokenizer half compares the
// container's tokenizer to the family's tokenizer.json.
func TestC6MatchesTransformers(t *testing.T) {
	for _, fx := range c6Fixtures {
		t.Run(fx.name, func(t *testing.T) {
			m, g := openC6(t, fx.name)
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

// c6Violation is one feature taken out of the engine's copy of the model. It
// returns the undo.
type c6Violation struct {
	name string
	mut  func(m *Model) func()
}

// eachLayer applies f to a copy of every layer and restores the originals.
func eachLayer(m *Model, f func(l *layer)) func() {
	was := slices.Clone(m.layers)
	for i := range m.layers {
		f(&m.layers[i])
	}
	return func() { copy(m.layers, was) }
}

// restate has the front end read the blocks' stored tensors through f until
// the returned undo: a violation that drops, renames or resizes tensors in the
// model's layers states the same model to the composer, which reads the
// container (frontState.restate).
func (m *Model) restate(f func(inv map[jlm.Role]graph.Shape)) func() {
	fr := &m.front
	fr.mu.Lock()
	was := fr.restate
	fr.restate = func(block int32, inv map[jlm.Role]graph.Shape) {
		if block != jlm.DenseBlock {
			f(inv)
		}
	}
	fr.restated++
	fr.mu.Unlock()
	return func() {
		fr.mu.Lock()
		fr.restate = was
		fr.restated++
		fr.mu.Unlock()
	}
}

// dropStored states the blocks without roles to the composer.
func dropStored(m *Model, roles ...jlm.Role) func() {
	return m.restate(func(inv map[jlm.Role]graph.Shape) {
		for _, r := range roles {
			delete(inv, r)
		}
	})
}

// noQKVBias, noPostAttnNorm and noPostFFNNorm are blocks without those
// tensors, in the model's layers and as the composer reads them.
func noQKVBias(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.bq, l.bk, l.bv = nil, nil, nil })
	undoR := dropStored(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias)
	return func() { undoR(); undoL() }
}

func noPostAttnNorm(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.postAttnNorm = nil })
	undoR := dropStored(m, jlm.RolePostAttnNorm)
	return func() { undoR(); undoL() }
}

func noPostFFNNorm(m *Model) func() {
	undoL := eachLayer(m, func(l *layer) { l.postFFNNorm = nil })
	undoR := dropStored(m, jlm.RolePostFFNNorm)
	return func() { undoR(); undoL() }
}

func cfgMut(m *Model, f func(c *Config)) func() {
	was := m.Cfg
	c := *was
	f(&c)
	m.Cfg = &c
	return func() { m.Cfg = was }
}

// c6Violations are the violation runs for the kit, kept as a gate: each feature
// switched off must move the logits far past c6NMSE, or the fixture cannot see
// it and TestC6MatchesTransformers certifies nothing about it.
var c6Violations = map[string][]c6Violation{
	"synth-phi2": {
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"no attn_norm bias", func(m *Model) func() { return eachLayer(m, func(l *layer) { l.attnNormB = nil }) }},
		{"no FFN biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.upB, l.downB = nil, nil })
		}},
		{"no q/k/v biases", noQKVBias},
		{"no head bias", func(m *Model) func() {
			was := m.outB
			m.outB = nil
			return func() { m.outB = was }
		}},
		{"no output norm bias", func(m *Model) func() {
			was := m.outNormB
			m.outNormB = nil
			return func() { m.outNormB = was }
		}},
		{"serial residual", func(m *Model) func() {
			undo := cfgMut(m, func(c *Config) { c.Parallel = false })
			undoL := eachLayer(m, func(l *layer) { l.ffnNorm, l.ffnNormB = l.attnNorm, l.attnNormB })
			return func() { undoL(); undo() }
		}},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-starcoder": {
		{"no position table", noPosTable},
		{"rotary instead of none", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPosEnc = false }) }},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"no norm biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.attnNormB, l.ffnNormB = nil, nil })
		}},
		{"no FFN biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.upB, l.downB = nil, nil })
		}},
		{"no k/v biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.bk, l.bv = nil, nil })
		}},
	},
	"synth-stablelm": {
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"parallel residual", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Parallel = true }) }},
		{"NORM rotary for NEOX", hostNormRotary},
		{"no q/k/v biases", noQKVBias},
	},
	"synth-stablelm-par": {
		{"serial residual", serialFromShared},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
	},
	"synth-starcoder2": {
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"parallel residual", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Parallel = true }) }},
		{"NORM rotary for NEOX", hostNormRotary},
		{"no FFN biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.upB, l.downB = nil, nil })
		}},
		{"no o bias", func(m *Model) func() { return eachLayer(m, func(l *layer) { l.bo = nil }) }},
	},
	"synth-falcon": {
		{"serial residual", serialFromShared},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"NORM rotary for NEOX", hostNormRotary},
	},
	"synth-falcon40": {
		{"the two norms swapped", swapNorms},
		{"one norm for both branches", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.ffnNorm, l.ffnNormB = nil, nil })
		}},
	},
	"synth-nemotron": {
		{"GELU for relu2", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Act = nn.ActGELU }) }},
		{"LayerNorm1p's 1 added again", onePlusAgain},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"NORM rotary for NEOX", hostNormRotary},
		{"no FFN biases", func(m *Model) func() {
			return eachLayer(m, func(l *layer) { l.upB, l.downB = nil, nil })
		}},
	},
	"synth-commandr": {
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"serial residual", serialFromShared},
		{"NEOX rotary for NORM", hostNeoxRotary},
		{"no logit scale", logitScale},
	},
	"synth-dbrx": {
		{"no clip_qkv", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ClampKQV = 0 }) }},
		{"clip_qkv at twice its value", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.ClampKQV *= 2 }) }},
		{"RMSNorm for LayerNorm", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.LayerNorm = false }) }},
		{"NORM rotary for NEOX", hostNormRotary},
		{"the routed weights not renormalised", func(m *Model) func() {
			return cfgMut(m, func(c *Config) { c.NoExpertNorm = true })
		}},
		{"parallel residual", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.Parallel = true }) }},
	},
	"synth-cohere2": {
		{"rotary on the global layer", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.NoPEGlobal = false }) }},
		{"no sliding window", func(m *Model) func() { return cfgMut(m, func(c *Config) { c.SWAWindow = 0 }) }},
		{"serial residual", serialFromShared},
		{"no logit scale", logitScale},
	},
}

func TestC6FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range c6Fixtures {
		vs := c6Violations[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g := openC6(t, fx.name)
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
