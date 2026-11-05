package engine

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
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
func openC6(t *testing.T, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	return openGolden(t, "c6", "scripts/c6gold.py", name)
}

// openGolden opens a fixture whose golden lives in testdata/golden/<dir>,
// written by script.
func openGolden(t *testing.T, dir, script, name string) (*model.Model, *c6Golden, []Option) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "testdata", "golden", dir, name+".json"))
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
	instOpts := []Option{WithKVF16(false)}
	m, err := model.Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	return m, g, instOpts
}

// c6Worst runs the whole golden through decode, prefill at every length and
// the batched path, and returns the worst NMSE. check reports each position.
func c6Worst(t *testing.T, m *model.Model, g *c6Golden, check func(what string, p int, lg []float32) float64, io ...Option) float64 {
	t.Helper()
	worst := 0.0
	st := testInstance(t, m, len(g.IDs)+1, io...)
	for p, id := range g.IDs {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("forward", p, lg))
	}
	st.Close()
	for n := 1; n <= len(g.IDs); n++ {
		pf := testInstance(t, m, len(g.IDs)+1, io...)
		lg, err := pf.Prefill(g.IDs[:n])
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("prefill", n-1, lg))
		pf.Close()
	}
	// Two sequences in lockstep through the batched graph, the second a
	// position behind.
	b := newSessBatch(m, 2, len(g.IDs)+1, io...)
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
func c6BaseFixtures() []struct {
	name string
	sel  func(m *model.Model) bool
} {
	return []struct {
		name string
		sel  func(m *model.Model) bool
	}{
		{"synth-phi2", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "phi2" && c.LayerNorm && c.Parallel && c.RopeNeox &&
				c.NRot < c.HeadDim && l.FfnNorm == nil && l.Gate.E == nil &&
				l.AttnNormB != nil && l.Vector(jlm.RoleFFNUpBias) != nil && l.Vector(jlm.RoleFFNDownBias) != nil && l.Vector(jlm.RoleAttnQBias) != nil &&
				l.Vector(jlm.RoleAttnOutBias) != nil && m.OutB != nil && m.OutNormB != nil
		}},
		{"synth-starcoder", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "starcoder" && c.LayerNorm && !c.Parallel && c.NoPosEnc &&
				c.NKVHead == 1 && c.NHead > 1 && m.PosEmbd.E != nil && l.FfnNorm != nil &&
				l.Gate.E == nil && l.AttnNormB != nil && l.FfnNormB != nil && l.Vector(jlm.RoleFFNUpBias) != nil &&
				l.Vector(jlm.RoleFFNDownBias) != nil && l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnKBias) != nil && l.Vector(jlm.RoleAttnVBias) != nil && l.Vector(jlm.RoleAttnOutBias) != nil &&
				m.OutNormB != nil && c.TiedEmbd
		}},
		{"synth-stablelm", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "stablelm" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
				c.NRot*4 == c.HeadDim && l.FfnNorm != nil && l.Gate.E != nil && l.Vector(jlm.RoleAttnQBias) != nil &&
				l.Vector(jlm.RoleAttnOutBias) == nil && l.AttnNormB != nil && m.OutNormB != nil && !c.TiedEmbd
		}},
		{"synth-stablelm-par", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "stablelm" && c.LayerNorm && c.Parallel && l.FfnNorm == nil && l.Gate.E != nil
		}},
		{"synth-starcoder2", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "starcoder2" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
				c.NRot == c.HeadDim && c.NKVHead < c.NHead && l.Gate.E == nil && l.FfnNorm != nil &&
				l.AttnNormB != nil && l.FfnNormB != nil && l.Vector(jlm.RoleFFNUpBias) != nil && l.Vector(jlm.RoleFFNDownBias) != nil &&
				l.Vector(jlm.RoleAttnQBias) != nil && l.Vector(jlm.RoleAttnOutBias) != nil && m.OutNormB != nil
		}},
		{"synth-falcon", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "falcon" && c.LayerNorm && c.Parallel && c.RopeNeox && c.NKVHead == 1 &&
				l.FfnNorm == nil && l.Gate.E == nil && l.Vector(jlm.RoleFFNUpBias) == nil && l.Vector(jlm.RoleAttnQBias) == nil && l.AttnNormB != nil
		}},
		{"synth-falcon40", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "falcon" && c.Parallel && c.NKVHead == 2 && l.FfnNorm != nil &&
				l.FfnNormB != nil && l.AttnNormB != nil && l.Gate.E == nil
		}},
		{"synth-nemotron", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "nemotron" && c.LayerNorm && !c.Parallel && c.RopeNeox &&
				2*c.NRot == c.HeadDim && c.Act == graph.ActReLU2 && l.Gate.E == nil &&
				l.FfnNorm != nil && l.AttnNormB != nil && l.Vector(jlm.RoleFFNUpBias) != nil && l.Vector(jlm.RoleFFNDownBias) != nil &&
				m.OutNormB != nil && !c.TiedEmbd
		}},
		{"synth-commandr", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "command-r" && c.LayerNorm && c.Parallel && !c.RopeNeox &&
				c.LogitScale == 16 && c.TiedEmbd && l.FfnNorm == nil && l.Gate.E != nil &&
				l.AttnNormB == nil && m.OutNormB == nil && c.NKVHead < c.NHead && c.SWAWindow == 0
		}},
		{"synth-dbrx", func(m *model.Model) bool {
			c, l := m.Cfg, &m.Layers[0]
			return c.Arch == "dbrx" && c.LayerNorm && !c.Parallel && c.RopeNeox && c.NRot == c.HeadDim &&
				c.ClampKQV == 1.5 && c.NExpert == 4 && c.NExpertUsed == 2 && !c.NoExpertNorm &&
				c.NKVHead < c.NHead && l.Router.E != nil && l.FfnNorm != nil && l.AttnNormB == nil &&
				l.FfnNormB == nil && l.Vector(jlm.RoleAttnQBias) == nil && m.OutNormB == nil && !c.TiedEmbd
		}},
		{"synth-cohere2", func(m *model.Model) bool {
			c := m.Cfg
			return c.Arch == "command-r" && c.LayerNorm && c.Parallel && c.NoPEGlobal &&
				c.SWAWindow == 4 && c.SWAPeriod == 4 && c.LogitScale == 4 &&
				c.SWA(0) && !c.SWA(3) && c.RopeAt(0) && !c.RopeAt(3)
		}},
	}
}

// swapNorms feeds each branch of a two-norm parallel block the other's norm:
// Falcon-40B's GGUF names them the other way round from their use.
func swapNorms(m *model.Model) func() {
	return eachLayer(m, func(l *model.Layer) {
		l.AttnNorm, l.FfnNorm = l.FfnNorm, l.AttnNorm
		l.AttnNormB, l.FfnNormB = l.FfnNormB, l.AttnNormB
	})
}

// onePlusAgain adds LayerNorm1p's 1 to every norm weight a second time, which
// is what an engine that applied 1p to a file whose converter already folded
// it in would compute.
func onePlusAgain(m *model.Model) func() {
	plus := func(w []float32) []float32 {
		o := slices.Clone(w)
		for i := range o {
			o[i]++
		}
		return o
	}
	was := m.OutNorm
	m.OutNorm = plus(was)
	undo := eachLayer(m, func(l *model.Layer) { l.AttnNorm, l.FfnNorm = plus(l.AttnNorm), plus(l.FfnNorm) })
	return func() { undo(); m.OutNorm = was }
}

// logitScale takes Cohere's logit scale away.
func logitScale(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.LogitScale = 1 }) }

// serialFromShared turns a parallel block sharing one norm into a serial one
// with that norm on both branches.
func serialFromShared(m *model.Model) func() {
	undo := cfgMut(m, func(c *model.Config) { c.Parallel = false })
	undoL := eachLayer(m, func(l *model.Layer) { l.FfnNorm, l.FfnNormB = l.AttnNorm, l.AttnNormB })
	undoR := restate(m, func(inv map[jlm.Role]graph.Shape) {
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
func noNormBiases(m *model.Model) func() {
	undoL := eachLayer(m, func(l *model.Layer) { l.AttnNormB, l.FfnNormB = nil, nil })
	undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
	return func() { undoR(); undoL() }
}

func rmsForLayerNorm(m *model.Model) func() {
	undoC := cfgMut(m, func(c *model.Config) { c.LayerNorm = false })
	undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
	undoD := dropStoredDense(m, jlm.RoleOutputNormBias)
	was := m.OutNormB
	m.OutNormB = nil
	return func() { m.OutNormB = was; undoD(); undoR(); undoC() }
}

// noFFNBiases takes an ungated FFN's two biases away, in the layers and as
// the composer reads the container.
func noFFNBiases(m *model.Model) func() {
	return dropVecs(m, jlm.RoleFFNUpBias, jlm.RoleFFNDownBias)
}

// oneNormForBoth makes a two-norm parallel block share its attention's norm.
func oneNormForBoth(m *model.Model) func() {
	undoL := eachLayer(m, func(l *model.Layer) { l.FfnNorm, l.FfnNormB = nil, nil })
	undoR := dropStored(m, jlm.RoleFFNNorm, jlm.RoleFFNNormBias)
	return func() { undoR(); undoL() }
}

// noPosTable takes the learned position table away.
func noPosTable(m *model.Model) func() {
	was := m.PosEmbd
	m.PosEmbd = model.Tensor{}
	undoD := dropStoredDense(m, jlm.RolePosEmbd)
	return func() { undoD(); m.PosEmbd = was }
}

// TestC6MatchesTransformers holds each classic-transformer graph (C6) to
// transformers' own class, on every entry point that attends: decode, prefill
// at every length and the batched path. The tokenizer half compares the
// container's tokenizer to the family's tokenizer.json.
func TestC6MatchesTransformers(t *testing.T) {
	for _, fx := range c6Fixtures() {
		t.Run(fx.name, func(t *testing.T) {
			m, g, openOpts := openC6(t, fx.name)
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

// c6Violation is one feature taken out of the engine's copy of the model. It
// returns the undo.
type c6Violation struct {
	name string
	mut  func(m *model.Model) func()
}

// eachLayer applies f to a copy of every layer and restores the originals.
func eachLayer(m *model.Model, f func(l *model.Layer)) func() {
	was := slices.Clone(m.Layers)
	for i := range m.Layers {
		f(&m.Layers[i])
	}
	return func() { copy(m.Layers, was) }
}

// setVec replaces block l's vector under r with what f makes of it, in a
// table of l's own, so the layers eachLayer saved keep theirs. f returning
// nil takes the vector out.
func setVec(l *model.Layer, r jlm.Role, f func(v []float32) []float32) {
	vs := make([]nn.RoleVector, 0, len(l.Vecs))
	for _, v := range l.Vecs {
		if v.Role == r {
			if v.V = f(v.V); v.V == nil {
				continue
			}
		}
		vs = append(vs, v)
	}
	l.Vecs = vs
}

// addVec gives block l a vector under r in a table of l's own: one the file
// does not store, or in place of the one it does.
func addVec(l *model.Layer, r jlm.Role, v []float32) {
	setVec(l, r, func([]float32) []float32 { return nil })
	l.Vecs = append(l.Vecs, nn.RoleVector{Role: r, V: v})
}

// dropVecs takes roles out of every block's table and out of the
// container as the composer reads it: blocks that do not store them.
func dropVecs(m *model.Model, roles ...jlm.Role) func() {
	undoL := eachLayer(m, func(l *model.Layer) {
		for _, r := range roles {
			setVec(l, r, func([]float32) []float32 { return nil })
		}
	})
	undoR := dropStored(m, roles...)
	return func() { undoR(); undoL() }
}

// restate has the front end read the blocks' stored tensors through f until
// the returned undo: a violation that drops, renames or resizes tensors in the
// model's layers states the same model to the composer, which reads the
// container (frontState.restate). It applies over the restatements already
// in place, so violations compose.
func restate(m *model.Model, f func(inv map[jlm.Role]graph.Shape)) func() {
	fr := &m.Front
	fr.Mu.Lock()
	was := fr.Restate
	fr.Restate = func(block int32, inv map[jlm.Role]graph.Shape) {
		if was != nil {
			was(block, inv)
		}
		if block != jlm.DenseBlock {
			f(inv)
		}
	}
	fr.Restated++
	fr.Mu.Unlock()
	return func() {
		fr.Mu.Lock()
		fr.Restate = was
		fr.Restated++
		fr.Mu.Unlock()
	}
}

// dropStored states the blocks without roles to the composer.
func dropStored(m *model.Model, roles ...jlm.Role) func() {
	return restate(m, func(inv map[jlm.Role]graph.Shape) {
		for _, r := range roles {
			delete(inv, r)
		}
	})
}

// dropStoredDense is dropStored for the model-wide tensors (the head's),
// over whatever the front end already reads in place of the container's.
func dropStoredDense(m *model.Model, roles ...jlm.Role) func() {
	fr := &m.Front
	fr.Mu.Lock()
	was := fr.Restate
	fr.Restate = func(block int32, inv map[jlm.Role]graph.Shape) {
		if was != nil {
			was(block, inv)
		}
		if block == jlm.DenseBlock {
			for _, r := range roles {
				delete(inv, r)
			}
		}
	}
	fr.Restated++
	fr.Mu.Unlock()
	return func() {
		fr.Mu.Lock()
		fr.Restate = was
		fr.Restated++
		fr.Mu.Unlock()
	}
}

// noQKVBias, noPostAttnNorm and noPostFFNNorm are blocks without those
// tensors, in the model's layers and as the composer reads them.
func noQKVBias(m *model.Model) func() {
	return dropVecs(m, jlm.RoleAttnQBias, jlm.RoleAttnKBias, jlm.RoleAttnVBias)
}

func noPostAttnNorm(m *model.Model) func() {
	return dropVecs(m, jlm.RolePostAttnNorm)
}

func noPostFFNNorm(m *model.Model) func() {
	return dropVecs(m, jlm.RolePostFFNNorm)
}

func cfgMut(m *model.Model, f func(c *model.Config)) func() {
	was := m.Cfg
	c := *was
	f(&c)
	m.Cfg = &c
	return func() { m.Cfg = was }
}

// c6Violations are the violation runs for the kit, kept as a gate: each feature
// switched off must move the logits far past c6NMSE, or the fixture cannot see
// it and TestC6MatchesTransformers certifies nothing about it.
func c6BaseViolations() map[string][]c6Violation {
	return map[string][]c6Violation{
		"synth-phi2": {
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no attn_norm bias", func(m *model.Model) func() {
				undoL := eachLayer(m, func(l *model.Layer) { l.AttnNormB = nil })
				undoR := dropStored(m, jlm.RoleAttnNormBias)
				return func() { undoR(); undoL() }
			}},
			{"no FFN biases", noFFNBiases},
			{"no q/k/v biases", noQKVBias},
			{"no head bias", func(m *model.Model) func() {
				was := m.OutB
				m.OutB = nil
				undoR := dropStoredDense(m, jlm.RoleOutputBias)
				return func() { undoR(); m.OutB = was }
			}},
			{"no output norm bias", func(m *model.Model) func() {
				was := m.OutNormB
				m.OutNormB = nil
				undoR := dropStoredDense(m, jlm.RoleOutputNormBias)
				return func() { undoR(); m.OutNormB = was }
			}},
			{"serial residual", func(m *model.Model) func() {
				undo := cfgMut(m, func(c *model.Config) { c.Parallel = false })
				undoL := eachLayer(m, func(l *model.Layer) { l.FfnNorm, l.FfnNormB = l.AttnNorm, l.AttnNormB })
				return func() { undoL(); undo() }
			}},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-starcoder": {
			{"no position table", noPosTable},
			{"rotary instead of none", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPosEnc = false }) }},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"no norm biases", func(m *model.Model) func() {
				undoL := eachLayer(m, func(l *model.Layer) { l.AttnNormB, l.FfnNormB = nil, nil })
				undoR := dropStored(m, jlm.RoleAttnNormBias, jlm.RoleFFNNormBias)
				return func() { undoR(); undoL() }
			}},
			{"no FFN biases", noFFNBiases},
			{"no k/v biases", func(m *model.Model) func() {
				return dropVecs(m, jlm.RoleAttnKBias, jlm.RoleAttnVBias)
			}},
		},
		"synth-stablelm": {
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"parallel residual", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.Parallel = true }) }},
			{"NORM rotary for NEOX", hostNormRotary},
			{"no q/k/v biases", noQKVBias},
		},
		"synth-stablelm-par": {
			{"serial residual", serialFromShared},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
		},
		"synth-starcoder2": {
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"parallel residual", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.Parallel = true }) }},
			{"NORM rotary for NEOX", hostNormRotary},
			{"no FFN biases", noFFNBiases},
			{"no o bias", func(m *model.Model) func() {
				return dropVecs(m, jlm.RoleAttnOutBias)
			}},
		},
		"synth-falcon": {
			{"serial residual", serialFromShared},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"NORM rotary for NEOX", hostNormRotary},
		},
		"synth-falcon40": {
			{"the two norms swapped", swapNorms},
			{"one norm for both branches", oneNormForBoth},
		},
		"synth-nemotron": {
			{"GELU for relu2", actSwap(graph.ActGELU)},
			{"LayerNorm1p's 1 added again", onePlusAgain},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"NORM rotary for NEOX", hostNormRotary},
			{"no FFN biases", noFFNBiases},
		},
		"synth-commandr": {
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"serial residual", serialFromShared},
			{"NEOX rotary for NORM", hostNeoxRotary},
			{"no logit scale", logitScale},
		},
		"synth-dbrx": {
			{"no clip_qkv", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ClampKQV = 0 }) }},
			{"clip_qkv at twice its value", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.ClampKQV *= 2 }) }},
			{"RMSNorm for LayerNorm", rmsForLayerNorm},
			{"NORM rotary for NEOX", hostNormRotary},
			{"the routed weights not renormalised", func(m *model.Model) func() {
				return cfgMut(m, func(c *model.Config) { c.NoExpertNorm = true })
			}},
			{"parallel residual", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.Parallel = true }) }},
		},
		"synth-cohere2": {
			{"rotary on the global layer", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.NoPEGlobal = false }) }},
			{"no sliding window", func(m *model.Model) func() { return cfgMut(m, func(c *model.Config) { c.SWAWindow = 0 }) }},
			{"serial residual", serialFromShared},
			{"no logit scale", logitScale},
		},
	}
}

func TestC6FeaturesAreLoadBearing(t *testing.T) {
	for _, fx := range c6Fixtures() {
		vs := c6Violations()[fx.name]
		t.Run(fx.name, func(t *testing.T) {
			if len(vs) == 0 {
				t.Fatal("no violations for this fixture")
			}
			m, g, openOpts := openC6(t, fx.name)
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
