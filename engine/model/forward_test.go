package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

type runCase struct {
	Prompt string `json:"prompt"`
	N      int    `json:"n"`
	Out    string `json:"out"`
}

var models = map[string]string{
	"stories260K": "../../testdata/models/stories260K.gguf",
	"stories15M":  testmodels.Path("stories15M-q4_0.gguf"),
	"tinyllama":   testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"),
	// A real mixture of experts, so the tests that iterate this map cover the
	// MoE graph on the generated kernels rather than only on the F32 fixture.
	"qwen3moe": testmodels.Path("Qwen3-MOE-4x0.6B-Q4_K_M.gguf"),
	// qwen2vl is qwen2's block with M-RoPE; with no image rows the (t, h, w)
	// position triple collapses to the token index, so its text path must be
	// exactly qwen2's.
	"qwen2vl": testmodels.Path("Qwen2-VL-2B-Instruct-Q4_K_M.gguf"),
}

// tieMargin is how close the top two logits must be for a disagreement with
// llama.cpp to be acceptable.
//
// jitllm and llama.cpp do not compute the same function (activation
// quantization and reduction order differ), so they diverge wherever two tokens
// are nearly tied. The gate is therefore: agree until a near-tie, and if you
// disagree, the margin must be tiny. A disagreement with a wide margin fails.
const tieMargin = 0.5

// greedyLoadOpts is extra load options for the gate below, for a caller that
// cannot reach its Open (ssemKVF16 pins the reference's binary16 KV cache on the
// SSE tier). A test variable, so no other model in the process inherits it.
var greedyLoadOpts []Option

func TestGreedyMatchesLlamaCpp(t *testing.T) {
	for name, path := range models {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "golden", "run", name+".json"))
			if err != nil {
				t.Skipf("no goldens (run scripts/rungold.py): %v", err)
			}
			var cases []runCase
			if err := json.Unmarshal(raw, &cases); err != nil {
				t.Fatal(err)
			}
			mp, ok := existingModel(path)
			if !ok {
				t.Skipf("model not present: %s (nor its container) (set JITLLM_MODELS to the model directory)", path)
			}
			m, err := Open(jlmOf(t, mp), greedyLoadOpts...)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()

			// A tie at token 0 compares nothing: fall through to the next
			// prompt, and fail only if no prompt of the model compared a single
			// token (the pack-width tuner can flip an opening tie either way).
			compared := 0
			for _, c := range cases {
				ids := m.Vocab.Encode(c.Prompt, true)
				st := m.NewState(len(ids) + c.N + 1)
				defer st.Close() // a State owns a pinned pool; see container_test.go
				var logits []float32
				for _, id := range ids {
					if logits, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				var out []int32
				matched := 0
				for i := 0; i < c.N; i++ {
					next := Greedy(logits)
					if next == m.Vocab.EOS {
						break
					}
					out = append(out, next)
					got := m.Vocab.Decode(out)
					// Agreement is a prefix relationship, which sidesteps the
					// fact that re-tokenizing text is not always its inverse.
					if !strings.HasPrefix(c.Out, got) && !strings.HasPrefix(got, c.Out) {
						top := TopK(logits, 2)
						margin := top[0].Logit - top[1].Logit
						if margin > tieMargin {
							t.Errorf("%s %q: diverged at token %d with margin %.4f (%q %.4f vs %q %.4f)\n got  %q\n want %q",
								name, c.Prompt, i, margin,
								m.Vocab.Text(top[0].ID), top[0].Logit,
								m.Vocab.Text(top[1].ID), top[1].Logit, got, c.Out)
						} else {
							t.Logf("%s %q: %d/%d tokens exact, then a %.4f-margin tie (%q vs %q) — acceptable",
								name, c.Prompt, matched, c.N,
								margin, m.Vocab.Text(top[0].ID), m.Vocab.Text(top[1].ID))
						}
						break
					}
					matched = i + 1
					if logits, err = st.Forward(next); err != nil {
						t.Fatal(err)
					}
				}
				compared += matched
			}
			if compared == 0 {
				t.Errorf("%s: no prompt compared a single token with llama.cpp -- this gate proved nothing", name)
			}
		})
	}
}

// TestKVCacheConsistency: feeding a prompt token by token must give the same
// logits as continuing an existing session, and a Reset must be a true reset.
// A stale KV entry produces fluent, subtly wrong text and nothing else notices.
func TestKVCacheConsistency(t *testing.T) {
	m, err := Open(jlmOf(t, models["stories260K"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	ids := m.Vocab.Encode("Once upon a time", true)

	run := func() []float32 {
		st := m.NewState(len(ids) + 1)
		defer st.Close()
		var lg []float32
		for _, id := range ids {
			if lg, err = st.Forward(id); err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), lg...)
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("two identical runs differ at logit %d: %v vs %v", i, a[i], b[i])
		}
	}

	// Closed: a State left open keeps its JIT's view of the P-cores' crew for
	// the life of the process, and a later gate's decode is then never that
	// crew's only user, so no run of it fuses (TestKVPagesRelocateWithTheLayer
	// opened none, on every model, whenever this ran first).
	st := m.NewState(len(ids) + 1)
	defer st.Close()
	for _, id := range ids {
		if _, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	st.Reset()
	var c []float32
	for _, id := range ids {
		if c, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	for i := range a {
		if math.Abs(float64(a[i]-c[i])) > 1e-12 {
			t.Fatalf("after Reset, logit %d is %v, want %v", i, c[i], a[i])
		}
	}
}

func TestStateRejects(t *testing.T) {
	m, err := Open(jlmOf(t, models["stories260K"]))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	st := m.NewState(2)
	defer st.Close()
	if _, err := st.Forward(int32(m.Cfg.NVocab)); err == nil {
		t.Error("Forward accepted an out-of-range token")
	}
	if _, err := st.Forward(-1); err == nil {
		t.Error("Forward accepted a negative token")
	}
	for i := 0; i < 2; i++ {
		if _, err := st.Forward(1); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := st.Forward(1); err == nil {
		t.Error("Forward accepted a token past the end of the KV cache")
	}
}
