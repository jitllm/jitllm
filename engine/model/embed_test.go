package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// The embedding gates. Each model's golden (scripts/embedgold.py) carries two
// independent references for the same six texts:
//
//	llamacpp  llama.cpp b10825's llama-embedding on the CPU, from the same
//	          GGUF (or, for EmbeddingGemma, the same weights in llama.cpp's
//	          spelling -- see embedgold.py)
//	st        sentence-transformers in float32 on the model's own checkpoint:
//	          the arithmetic the weights were trained against
//
// plus llama.cpp's token ids, which the tokenizer must reproduce exactly.
//
// The models live in $JITLLM_MODELS/embed/ and NOT beside the generators, on
// purpose: every sweep over *.gguf in the model directory builds a decoder
// Instance, and an encoder has none.

// embedCosBound is the gate. A cosine is scale-free and what a vector index
// ranks by, so it is the quantity to bound; 0.9999 is the bar the work was
// set, and each model's measured value is logged beside it.
const embedCosBound = 0.9999

type embedGolden struct {
	Model    string      `json:"model"`
	GGUF     string      `json:"gguf"`
	LGGUF    string      `json:"llamacpp_gguf"`
	Texts    []string    `json:"texts"`
	IDs      [][]int32   `json:"ids"`
	Llamacpp [][]float64 `json:"llamacpp"`
	ST       [][]float64 `json:"st"`
	STIDs    [][]int32   `json:"st_ids"`
	STRepo   string      `json:"st_repo"`
	STError  string      `json:"st_error"`
}

func loadEmbedGoldens(t *testing.T) []embedGolden {
	t.Helper()
	paths, _ := filepath.Glob("../../testdata/golden/embed/*.json")
	if len(paths) == 0 {
		t.Fatal("no embedding goldens: run scripts/embedgold.py")
	}
	var out []embedGolden
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var g embedGolden
		if err := json.Unmarshal(b, &g); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		out = append(out, g)
	}
	return out
}

// openEmbedModel converts the golden's GGUF (or serves its container) and opens it.
func openEmbedModel(t *testing.T, g embedGolden, opts ...Option) *Model {
	t.Helper()
	src, ok := existingModel(testmodels.Path(filepath.Join("embed", g.GGUF)))
	if !ok {
		t.Skipf("MODEL MISSING: embed/%s (set JITLLM_MODELS to the model directory; "+
			"scripts/embedgold.py names the Ollama registry blob) -- this gate proved nothing", g.GGUF)
	}
	m, err := Open(jlmOf(t, src), opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	if !m.IsEmbedding() {
		t.Fatalf("%s: the container sets no pooling", g.Model)
	}
	return m
}

func cosine(a []float32, b []float64) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * b[i]
		aa += float64(a[i]) * float64(a[i])
		bb += b[i] * b[i]
	}
	return ab / math.Sqrt(aa*bb)
}

func l2(a []float32) float64 {
	var s float64
	for _, x := range a {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

// quantSlack is how much further from the float32 reference than llama.cpp a
// quantized file may land: with Q8_0 weights llama.cpp itself reaches only
// ~0.999 of the float32 model, so the engine owes being as faithful as the
// other engine reading the same bytes.
const quantSlack = 2e-4

// crossArchSlack is how far llama.cpp's own per-text cosine to the float32
// model moves between architectures on the same quantized file: the goldens'
// floors were taken on amd64, and llama.cpp b10825 on the M4 moves
// qwen3-embedding-0.6b's seven texts by -1.1e-04 to +2.0e-04 (text 5 by
// -8.4e-05). A host of another architecture is held to the floor less this as
// well; the measurement, not a margin, is the number.
const crossArchSlack = 2e-4

// floorSlack is quantSlack on the goldens' own architecture, and quantSlack
// plus crossArchSlack on any other.
func floorSlack() float64 {
	if runtime.GOARCH == "amd64" {
		return quantSlack
	}
	return quantSlack + crossArchSlack
}

func cos64(a, b []float64) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += a[i] * b[i]
		aa += a[i] * a[i]
		bb += b[i] * b[i]
	}
	return ab / math.Sqrt(aa*bb)
}

func worst(xs []float64) float64 {
	w := math.NaN()
	for i, x := range xs {
		if i == 0 || x < w {
			w = x
		}
	}
	return w
}

// embedAll embeds every golden text and returns its cosine against each
// reference, per text (empty where the golden has none).
//
// The ids are checked against the model's OWN tokenizer where the golden has
// it, and llama.cpp's otherwise. They differ in one known place: llama-tokenize
// does not append qwen3-embedding's EOS, which its tokenizer (and llama.cpp's
// llama-embedding) does and which is the position that model pools.
func embedAll(t *testing.T, m *Model, e *Embedder, g embedGolden, checkIDs bool) (lc, st []float64) {
	t.Helper()
	for i, text := range g.Texts {
		ids := m.EmbedIDs(text)
		want, who := g.IDs[i], "llama.cpp"
		if len(g.STIDs) > i {
			want, who = g.STIDs[i], g.STRepo
		}
		if checkIDs && !slices.Equal(ids, want) {
			t.Errorf("%s text %d: ids %v, %s %v", g.Model, i, ids, who, want)
		}
		v, err := e.Embed(ids)
		if err != nil {
			t.Fatalf("%s text %d: %v", g.Model, i, err)
		}
		for _, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s text %d: a non-finite component -- a degenerate vector is a failure, not a pass", g.Model, i)
			}
		}
		if n := l2(v); math.Abs(n-1) > 1e-5 {
			t.Errorf("%s text %d: |v| = %.7f, want 1", g.Model, i, n)
		}
		if len(g.Llamacpp) > i {
			lc = append(lc, cosine(v, g.Llamacpp[i]))
		}
		if len(g.ST) > i {
			st = append(st, cosine(v, g.ST[i]))
		}
	}
	return lc, st
}

// TestEmbeddingsMatchReference is the gate: every model's six vectors against
// both references, and its token ids against the model's own tokenizer.
func TestEmbeddingsMatchReference(t *testing.T) {
	for _, g := range loadEmbedGoldens(t) {
		t.Run(g.Model, func(t *testing.T) {
			m := openEmbedModel(t, g)
			e, err := m.NewEmbedder()
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			// The llama.cpp arm of a model llama.cpp cannot load from Ollama's
			// file was taken on a conversion WITHOUT the dense heads, so it is
			// compared before them.
			same := g.LGGUF == "" || g.LGGUF == g.GGUF
			if !same {
				e.skipDense = true
				lc, _ := embedAll(t, m, e, g, true)
				e.skipDense = false
				t.Logf("%s: worst cosine vs llama.cpp (%s, before the dense heads) %.7f", g.Model, g.LGGUF, worst(lc))
				if !(worst(lc) >= embedCosBound) {
					t.Errorf("%s: cosine vs llama.cpp %.7f < %v", g.Model, worst(lc), embedCosBound)
				}
			}
			lc, st := embedAll(t, m, e, g, true)
			// How far the FILE is from the float32 model, measured by the other
			// engine that reads it: llama.cpp against sentence-transformers.
			var floor []float64
			quantized := false
			if same && len(g.ST) == len(g.Llamacpp) {
				for i := range g.ST {
					floor = append(floor, cos64(g.Llamacpp[i], g.ST[i]))
				}
				quantized = worst(floor) < embedCosBound
			}
			if same {
				bound := embedCosBound
				if quantized {
					bound = 0.999
				}
				t.Logf("%s: worst cosine vs llama.cpp %.7f (bound %v)", g.Model, worst(lc), bound)
				if !(worst(lc) >= bound) {
					t.Errorf("%s: cosine vs llama.cpp %.7f < %v", g.Model, worst(lc), bound)
				}
			}
			if len(g.ST) == 0 {
				t.Logf("%s: no sentence-transformers arm (%s)", g.Model, g.STError)
				return
			}
			t.Logf("%s: worst cosine vs %s (float32) %.7f", g.Model, g.STRepo, worst(st))
			if !quantized {
				if !(worst(st) >= embedCosBound) {
					t.Errorf("%s: cosine vs %s %.7f < %v", g.Model, g.STRepo, worst(st), embedCosBound)
				}
				return
			}
			t.Logf("%s: the file is quantized -- llama.cpp itself reaches only %.7f of the float32 model",
				g.Model, worst(floor))
			for i := range st {
				if !(st[i] >= floor[i]-floorSlack()) {
					t.Errorf("%s text %d: cosine vs float32 %.7f, and llama.cpp on the same file reaches %.7f",
						g.Model, i, st[i], floor[i])
				}
			}
		})
	}
}

// TestEmbeddingGateDiscriminates runs the gate above against violations: each
// is a fluent-looking unit vector from the wrong model, and each must land below
// the bound on at least one text. They are applied to the loaded model, so every
// run re-checks them.
func TestEmbeddingGateDiscriminates(t *testing.T) {
	type violation struct {
		name  string
		apply func(m *Model) (undo func())
	}
	set := func(p *jlm.Pooling, v jlm.Pooling) func() {
		old := *p
		*p = v
		return func() { *p = old }
	}
	cases := map[string][]violation{
		"all-minilm-l6": {
			{"cls pooling on a mean model", func(m *Model) func() { return set(&m.Cfg.Pooling, jlm.PoolCLS) }},
		},
		"mxbai-embed-large": {
			{"mean pooling on a CLS model", func(m *Model) func() { return set(&m.Cfg.Pooling, jlm.PoolMean) }},
		},
		"embeddinggemma-300m": {
			{"causal attention on a bidirectional model", func(m *Model) func() {
				m.Cfg.NonCausal = false
				return func() { m.Cfg.NonCausal = true }
			}},
			{"last-token pooling on a mean model", func(m *Model) func() { return set(&m.Cfg.Pooling, jlm.PoolLast) }},
			// Only the golden's long text is past the window, which is why it
			// is in the golden. Widening the window moves only the mask.
			{"a sliding window four times too wide", func(m *Model) func() {
				old := m.Cfg.SWAWindow
				m.Cfg.SWAWindow = 4 * old
				return func() { m.Cfg.SWAWindow = old }
			}},
			{"no dense heads", func(m *Model) func() {
				d1, d2 := m.dense1, m.dense2
				m.dense1, m.dense2 = tensor{}, tensor{}
				undo := dropStoredDense(m, jlm.RoleEmbdDense1, jlm.RoleEmbdDense2)
				return func() { undo(); m.dense1, m.dense2 = d1, d2 }
			}},
		},
		"qwen3-embedding-0.6b": {
			{"mean pooling on a last-token model", func(m *Model) func() { return set(&m.Cfg.Pooling, jlm.PoolMean) }},
			{"bidirectional attention on a causal model", func(m *Model) func() {
				m.Cfg.NonCausal = true
				return func() { m.Cfg.NonCausal = false }
			}},
			{"no EOS appended", func(m *Model) func() {
				old := m.Vocab.AddEOS
				m.Vocab.AddEOS = false
				return func() { m.Vocab.AddEOS = old }
			}},
		},
	}
	ran := 0
	for _, g := range loadEmbedGoldens(t) {
		vs := cases[g.Model]
		if len(vs) == 0 || len(g.ST) == 0 && len(g.Llamacpp) == 0 {
			continue
		}
		t.Run(g.Model, func(t *testing.T) {
			m := openEmbedModel(t, g)
			for _, v := range vs {
				undo := v.apply(m)
				e, err := m.NewEmbedder()
				if err != nil {
					undo()
					t.Fatal(err)
				}
				// The reference that includes every stage: sentence-transformers
				// where it exists, llama.cpp otherwise.
				lcOK := g.LGGUF == "" || g.LGGUF == g.GGUF
				lcs, sts := embedAll(t, m, e, g, false)
				e.Close()
				undo()
				lc, st := worst(lcs), worst(sts)
				ref, which := st, "st"
				if len(g.ST) == 0 {
					if !lcOK {
						t.Fatalf("%s: no reference covers the whole pipeline", g.Model)
					}
					ref, which = lc, "llama.cpp"
				}
				t.Logf("%s / %s: worst cosine vs %s %.5f", g.Model, v.name, which, ref)
				if ref >= embedCosBound {
					t.Errorf("%s: the gate PASSED %q (cosine %.7f) -- it does not discriminate", g.Model, v.name, ref)
				}
				ran++
			}
		})
	}
	if ran == 0 {
		testmodels.Missing(t, "%s", "no violation ran: the discriminating gate proved nothing")
	}
}
