package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// synthLlama4 is scripts/llama4gold.py's golden: transformers' own
// Llama4ForCausalLM over twelve fixed ids, and Llama 4's own tokenizer over a
// few texts.
type synthLlama4 struct {
	IDs []int32 `json:"ids"`
	BOS int32   `json:"bos"`
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

// llama4NMSE is the bound against transformers. Every tensor is F32 and the KV
// cache is f32, so what separates the two arms is reduction order: measured
// ~1e-13, with every violation below landing at 1e-5 or far above.
const llama4NMSE = 1e-9

func openSynthLlama4(t *testing.T) (*Model, *synthLlama4) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "llama4", "synth-llama4.json"))
	if err != nil {
		t.Fatal(err)
	}
	g := &synthLlama4{}
	if err := json.Unmarshal(raw, g); err != nil {
		t.Fatal(err)
	}
	path, ok := existingModel(testmodels.Path("synth-llama4.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("synth-llama4.gguf")+
			" (set JITLLM_MODELS to the model directory) -- run scripts/llama4gold.py (RULE 11)")
	}
	m, err := Open(jlmOf(t, path), noTune, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	return m, g
}

func llama4Cmp(head []float64, got []float32) float64 {
	var num, den float64
	for i, w := range head {
		d := float64(got[i]) - w
		num, den = num+d*d, den+w*w
	}
	if den == 0 {
		return math.Inf(1)
	}
	return num / den
}

// TestSynthLlama4MatchesTransformers holds the whole Llama 4 text graph to
// transformers' Llama4ForCausalLM on every entry point that attends: decode,
// prefill at EVERY length (so each position is the last row of some prefill,
// which is the only row Prefill returns), and the batched path. The fixture
// carries each feature at a value that bites inside twelve positions -- a
// chunk of 4, a temperature floor of 4, a NoPE layer, interleaved dense and
// mixture blocks, a top-1 sigmoid router weighting the expert's input, the
// weightless QK norm and llama3 rotary factors -- and the violations subtest
// removes them one at a time and demands the gate see each.
func TestSynthLlama4MatchesTransformers(t *testing.T) {
	m, g := openSynthLlama4(t)
	defer m.Close()
	c := m.Cfg
	// Selection first: every knob must be the one the script set, or the
	// reference is compared against a different model.
	if c.Arch != "llama4" || !c.SWAChunked || !c.NoPEGlobal || !c.QKL2Norm || !c.ExpertWeightIn ||
		c.SWAWindow != 4 || c.SWAPeriod != 4 || c.MoEStep != 2 || c.AttnTempFloor != 4 ||
		c.AttnTempScale == 0 || c.NFFNShExp != c.NFFNExp || m.rope.Freqs == nil {
		t.Fatalf("the container does not carry the fixture's graph: %+v", c)
	}
	if !c.MoEAt(1) || !c.MoEAt(3) || c.MoEAt(0) || c.MoEAt(2) || c.RopeAt(3) || !c.RopeAt(2) {
		t.Fatal("the per-layer rules do not select the fixture's layers")
	}
	if m.Vocab == nil {
		t.Fatalf("no tokenizer: %v", m.TokErr)
	}
	for _, tk := range g.Texts {
		if got := m.Vocab.Encode(tk.Text, false); !slices.Equal(got, tk.IDs) {
			t.Errorf("Encode(%q)\n  ours       %v\n  tokenizer.json %v", tk.Text, got, tk.IDs)
		}
	}
	if len(g.Texts) == 0 {
		t.Error("golden has no tokenizer half")
	}

	check := func(what string, p int, lg []float32) float64 {
		t.Helper()
		nmse := llama4Cmp(g.Pos[p].Head, lg)
		if math.IsNaN(nmse) || math.IsInf(nmse, 0) || nmse > llama4NMSE {
			t.Errorf("%s pos %d: NMSE %.3e against transformers (bound %.0e)", what, p, nmse, llama4NMSE)
		}
		if am := Greedy(lg); am != g.Pos[p].Argmax {
			t.Errorf("%s pos %d: argmax %d, transformers %d", what, p, am, g.Pos[p].Argmax)
		}
		return nmse
	}

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
	// position behind so the two rows sit on different sides of a chunk
	// boundary and a temperature step at every step.
	b := m.NewBatch(2, len(g.IDs)+1)
	defer b.Close()
	if _, err := b.PrefillSeq(0, g.IDs[:2]); err != nil {
		t.Fatal(err)
	}
	if _, err := b.PrefillSeq(1, g.IDs[:1]); err != nil {
		t.Fatal(err)
	}
	for p := 2; p < len(g.IDs); p++ {
		rows, err := b.ForwardBatch([]int32{g.IDs[p], g.IDs[p-1]})
		if err != nil {
			t.Fatal(err)
		}
		worst = max(worst, check("batch", p, rows[:c.NVocab]))
		worst = max(worst, check("batch-lag", p-1, rows[c.NVocab:2*c.NVocab]))
	}
	t.Logf("decode, prefill at every length and the batch: worst NMSE %.3e", worst)
}

// TestSynthLlama4FeaturesAreLoadBearing is the violation run: each Llama 4
// feature switched off must move the logits far past llama4NMSE, or the gate
// above cannot see it.
func TestSynthLlama4FeaturesAreLoadBearing(t *testing.T) {
	m, g := openSynthLlama4(t)
	defer m.Close()
	cfg := *m.Cfg
	for _, v := range []struct {
		name string
		mut  func(c *Config)
	}{
		{"sliding window instead of a chunk", func(c *Config) { c.SWAChunked = false }},
		{"every layer rotated", func(c *Config) { c.NoPEGlobal = false }},
		{"no QK L2 norm", func(c *Config) { c.QKL2Norm = false }},
		{"weight on the expert's output", func(c *Config) { c.ExpertWeightIn = false }},
		{"no attention temperature", func(c *Config) { c.AttnTempScale = 0 }},
		{"temperature offset 0", func(c *Config) { c.AttnTempOffs = 0 }},
		{"no chunk at all", func(c *Config) { c.SWAWindow = 0 }},
	} {
		t.Run(v.name, func(t *testing.T) {
			c := cfg
			v.mut(&c)
			m.Cfg = &c
			regraph(m)
			defer func() { m.Cfg = &cfg; regraph(m) }()
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
			if !(worst > 1e3*llama4NMSE) {
				t.Errorf("the gate cannot see it: worst NMSE %.3e", worst)
			}
			t.Logf("worst NMSE %.3e", worst)
		})
	}
}
