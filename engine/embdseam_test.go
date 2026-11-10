package engine

import (
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// eachModel runs fn over every language model present. A model that will not
// load is skipped, not failed.
func eachModel(t *testing.T, fn func(t *testing.T, m *model.Model, io []Option), opts ...Option) {
	paths := languageModels(t)
	if p := os.Getenv("JITLLM_MODEL"); p != "" {
		paths = []string{testmodels.Resolve(p)}
	}
	if len(paths) == 0 {
		testmodels.Missing(t, "%s", "MODEL MISSING: no models in "+testmodels.Dir()+" (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
	}
	sort.Strings(paths)
	for _, p := range paths {
		// Small models only: the seam (embed(), the span walk, the chunk loop)
		// is architecture-independent, and the big files push the package past
		// its timeout. Pin one with JITLLM_MODEL when a big one matters.
		if fi, err := os.Stat(p); err == nil && fi.Size() > 2<<30 && os.Getenv("JITLLM_MODEL") == "" {
			continue
		}
		t.Run(filepath.Base(p), func(t *testing.T) {
			instOpts := opts
			m, err := model.Open(jlmOf(t, p))
			if err != nil {
				t.Skipf("cannot load: %v", err)
			}
			defer m.Close()
			fn(t, m, instOpts)
		})
	}
}

// promptIDs is n distinct in-range token ids.
func promptIDs(m *model.Model, n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(1 + i%(min(64, m.Cfg.NVocab-1)))
	}
	return ids
}

// embdRows returns the embeddings the token path would look up for ids, scaled
// as the caller must scale them. The embedding seam is gated by identity, not
// tolerance: ForwardEmbd/PrefillMixed reach the same block stack with the same
// values and the fill is a copy, so any difference is a defect (a scale applied
// twice, a wrong stride, a position not advanced). It needs no VLM.
func embdRows(t *testing.T, m *model.Model, ids []int32) []float32 {
	t.Helper()
	c := m.Cfg
	e := make([]float32, len(ids)*c.NEmbd)
	embd := &refDense(t, m).Embd
	for i, id := range ids {
		row := e[i*c.NEmbd : (i+1)*c.NEmbd]
		// The shipping lookup, not nn.Row: m.embd is in the device layout, and
		// reading it row-major produces numbers, not an error.
		if err := embedRowRef(embd, row, int(id)); err != nil {
			t.Fatal(err)
		}
		// The caller applies EmbdScale, the seam does not: a spliced image
		// feature must not be scaled again.
		if c.EmbdScale != 1 {
			for j := range row {
				row[j] *= float32(c.EmbdScale)
			}
		}
	}
	return e
}

// seamTokens makes ids[lo:hi] -- the rows a gate sends through the embedding
// seam -- token 0 on a model whose rows read their token past the embedding:
// per-layer embeddings (Gemma 4's E models) and a router that routes by token
// (DeepSeek V4's hash-routed blocks). An embedding row has no token, and the
// seam takes token 0's (ple.go, as llama.cpp does for an image's rows; a
// token-routed gather reads row 0 for a row naming none): the token path
// equals the seam exactly for that token and no other, so that is the token
// the comparison runs.
func seamTokens(m *model.Model, ids []int32, lo, hi int) []int32 {
	if m.Cfg.PLEDim == 0 && !routesByToken(m) {
		return ids
	}
	ids = append([]int32(nil), ids...)
	clear(ids[lo:hi])
	return ids
}

// routesByToken reports whether a block of m routes its rows by their token
// (jlm.RoleHashExperts).
func routesByToken(m *model.Model) bool {
	for li := range m.Cfg.NLayer {
		if m.Stores(li, jlm.RoleHashExperts) {
			return true
		}
	}
	return false
}

// TestForwardEmbdMatchesForward runs with tuning off for
// TestPrefillMixedMatchesPrefill's reason: on arm64 a timed shape picks its
// matvec over its first calls, and two states would compare two kernels.
func TestForwardEmbdMatchesForward(t *testing.T) {
	eachModel(t, func(t *testing.T, m *model.Model, io []Option) {
		ids := promptIDs(m, 6)
		ids = seamTokens(m, ids, 0, len(ids))

		a := testInstance(t, m, 64, io...)
		defer a.Close()
		var want []float32
		var err error
		for _, id := range ids {
			if want, err = a.Forward(id); err != nil {
				t.Fatal(err)
			}
		}

		b := testInstance(t, m, 64, io...)
		defer b.Close()
		e := embdRows(t, m, ids)
		row := func(i int) []float32 { return e[i*m.Cfg.NEmbd : (i+1)*m.Cfg.NEmbd] }
		step := func(i int) ([]float32, error) { return b.ForwardEmbd(row(i)) }
		what := "ForwardEmbd"
		var got []float32
		for i := range ids {
			if got, err = step(i); err != nil {
				t.Fatal(err)
			}
		}
		diffBits(t, what, want, got)
	})
}

// TestPrefillMixedMatchesPrefill runs with tuning off: on arm64 a timed shape
// picks its matvec over its first calls, so two states decoding the same row
// may run different kernels and part by rounding.
func TestPrefillMixedMatchesPrefill(t *testing.T) { prefillMixedMatchesPrefill(t) }

// prefillMixedMatchesPrefill is TestPrefillMixedMatchesPrefill with opts on every model it opens.
func prefillMixedMatchesPrefill(t *testing.T, opts ...Option) {
	eachModel(t, func(t *testing.T, m *model.Model, io []Option) {
		ids := promptIDs(m, 24)
		// The same prompt as three spans: tokens, embeddings, tokens. The split
		// is not on a chunk boundary, so a straddling span is exercised.
		cut1, cut2 := 7, 19
		ids = seamTokens(m, ids, cut1, cut2)

		a := testInstance(t, m, 64, io...)
		defer a.Close()
		want, err := a.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}

		e := embdRows(t, m, ids[cut1:cut2])
		mid := model.Span{Embd: e}
		b := testInstance(t, m, 64, io...)
		defer b.Close()
		got, err := b.PrefillMixed(
			model.Span{Tokens: ids[:cut1]},
			mid,
			model.Span{Tokens: ids[cut2:]},
		)
		if err != nil {
			t.Fatal(err)
		}
		diffBits(t, "PrefillMixed", want, got)
		if a.pos != b.pos {
			t.Errorf("position: token path %d, mixed path %d", a.pos, b.pos)
		}
		// And the rows after it: decode resumes where the mixed prompt left
		// the rotary, which for rows that advance like text is the cache
		// position (an image grid on an M-RoPE model moves it; mrope.go).
		next := ids[len(ids)-1]
		want, err = a.Forward(next)
		if err != nil {
			t.Fatal(err)
		}
		if got, err = b.Forward(next); err != nil {
			t.Fatal(err)
		}
		diffBits(t, "Forward after PrefillMixed", want, got)
	}, opts...)
}

func diffBits(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: %d logits, want %d", what, len(got), len(want))
	}
	bad := 0
	worst := 0.0
	for i := range want {
		if want[i] != got[i] {
			bad++
			if d := want[i] - got[i]; float64(d) > worst || -float64(d) > worst {
				worst = float64(d)
				if worst < 0 {
					worst = -worst
				}
			}
		}
	}
	if bad != 0 {
		t.Errorf("%s: %d of %d logits differ, worst |delta| %.3e -- the fill is a copy, so this must be exact",
			what, bad, len(want), worst)
	}
}
