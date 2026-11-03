package model

import (
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// eachModel runs fn over every language model present. A model that will not
// load is skipped, not failed.
func eachModel(t *testing.T, fn func(t *testing.T, m *Model), opts ...Option) {
	paths := languageModels(t)
	if p := os.Getenv("JITLLM_MODEL"); p != "" {
		paths = []string{testmodels.Resolve(p)}
	}
	if len(paths) == 0 {
		t.Skip("MODEL MISSING: no models in " + testmodels.Dir() + " (set JITLLM_MODELS to the model directory) -- this gate proved nothing")
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
			m, err := Open(jlmOf(t, p), opts...)
			if err != nil {
				t.Skipf("cannot load: %v", err)
			}
			defer m.Close()
			fn(t, m)
		})
	}
}

// promptIDs is n distinct in-range token ids.
func promptIDs(m *Model, n int) []int32 {
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
func embdRows(t *testing.T, m *Model, ids []int32) []float32 {
	t.Helper()
	c := m.Cfg
	e := make([]float32, len(ids)*c.NEmbd)
	for i, id := range ids {
		row := e[i*c.NEmbd : (i+1)*c.NEmbd]
		// The shipping lookup, not nn.Row: m.embd is in the device layout, and
		// reading it row-major produces numbers, not an error.
		if err := m.embedRow(row, int(id)); err != nil {
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
// seam -- token 0 on a model with per-layer embeddings (Gemma 4's E models).
// Such a row's per-layer input reads its token, which an embedding row does
// not have, and the seam takes token 0's (ple.go, as llama.cpp does for an
// image's rows): the token path equals the seam exactly for that token and no
// other, so that is the token the comparison runs.
func seamTokens(m *Model, ids []int32, lo, hi int) []int32 {
	if m.Cfg.PLEDim == 0 {
		return ids
	}
	ids = append([]int32(nil), ids...)
	clear(ids[lo:hi])
	return ids
}

// TestForwardEmbdMatchesForward runs with tuning off for
// TestPrefillMixedMatchesPrefill's reason: on arm64 a timed shape picks its
// matvec over its first calls, and two states would compare two kernels.
func TestForwardEmbdMatchesForward(t *testing.T) {
	eachModel(t, func(t *testing.T, m *Model) {
		ids := promptIDs(m, 6)
		ids = seamTokens(m, ids, 0, len(ids))

		a := m.NewInstance(64)
		defer a.Close()
		var want []float32
		var err error
		for _, id := range ids {
			if want, err = a.Forward(id); err != nil {
				t.Fatal(err)
			}
		}

		b := m.NewInstance(64)
		defer b.Close()
		e := embdRows(t, m, ids)
		row := func(i int) []float32 { return e[i*m.Cfg.NEmbd : (i+1)*m.Cfg.NEmbd] }
		step := func(i int) ([]float32, error) { return b.ForwardEmbd(row(i)) }
		what := "ForwardEmbd"
		// A block that routes by token id cannot run a row with none, as its
		// references cannot (refusesIDless); a one-row span naming its id is
		// the same decode.
		if routesByID(m) {
			refusesIDless(t, m, func(s *Instance) error {
				_, err := s.ForwardEmbd(row(0))
				return err
			})
			step = func(i int) ([]float32, error) { return b.PrefillMixed(Span{Embd: row(i), Tokens: ids[i : i+1]}) }
			what = "one-row embedding spans naming their ids"
		}
		var got []float32
		for i := range ids {
			if got, err = step(i); err != nil {
				t.Fatal(err)
			}
		}
		diffBits(t, what, want, got)
	}, noTune())
}

// routesByID reports whether some block of m picks its experts by the row's
// token id (DeepSeek V4's hash-routed blocks).
func routesByID(m *Model) bool { return m.Cfg.NHashLayers > 0 }

// refusesIDless holds a model that routes by token id to its references: a
// supplied embedding with no id is refused by name, not routed somewhere.
// transformers' DeepseekV4HashRouter indexes tid2eid by input_ids, which
// inputs_embeds alone do not give it, and llama.cpp's deepseek4 builder
// gathers the table at the batch's tokens. The run is an instance of its own, so a
// refusal half way through a block leaves nothing behind for the gate.
func refusesIDless(t *testing.T, m *Model, run func(*Instance) error) {
	t.Helper()
	s := m.NewInstance(64)
	defer s.Close()
	if err := run(s); err == nil || !strings.Contains(err.Error(), "routes by token id") {
		t.Fatalf("an embedding row with no token id on a model that routes by it: %v, want a refusal naming it", err)
	}
}

// TestPrefillMixedMatchesPrefill runs with tuning off: on arm64 a timed shape
// picks its matvec over its first calls, so two states decoding the same row
// may run different kernels and part by rounding.
func TestPrefillMixedMatchesPrefill(t *testing.T) {
	eachModel(t, func(t *testing.T, m *Model) {
		ids := promptIDs(m, 24)
		// The same prompt as three spans: tokens, embeddings, tokens. The split
		// is not on a chunk boundary, so a straddling span is exercised.
		cut1, cut2 := 7, 19
		ids = seamTokens(m, ids, cut1, cut2)

		a := m.NewInstance(64)
		defer a.Close()
		want, err := a.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}

		e := embdRows(t, m, ids[cut1:cut2])
		mid := Span{Embd: e}
		// A model that routes by token id refuses the embedding rows without
		// their ids, and runs them naming them; a wrong name routes them
		// elsewhere, which the logits must show.
		if routesByID(m) {
			refusesIDless(t, m, func(s *Instance) error {
				_, err := s.PrefillMixed(Span{Tokens: ids[:cut1]}, mid, Span{Tokens: ids[cut2:]})
				return err
			})
			wrong := make([]int32, cut2-cut1)
			for i := range wrong {
				wrong[i] = ids[cut1+i] + 1
			}
			w := m.NewInstance(64)
			bad, err := w.PrefillMixed(Span{Tokens: ids[:cut1]}, Span{Embd: e, Tokens: wrong}, Span{Tokens: ids[cut2:]})
			if err != nil {
				t.Fatal(err)
			}
			if slices.Equal(bad, want) {
				t.Error("the embedding rows named by the wrong ids gave the right logits: the ids reach no router")
			}
			w.Close()
			mid.Tokens = ids[cut1:cut2]
		}
		b := m.NewInstance(64)
		defer b.Close()
		got, err := b.PrefillMixed(
			Span{Tokens: ids[:cut1]},
			mid,
			Span{Tokens: ids[cut2:]},
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
	}, noTune())
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
