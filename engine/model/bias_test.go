package model

import (
	"math"
	"os"
	"testing"
)

// TestAttentionBiasesReachTheModel injects each attention bias (a bias is just a
// vector) and requires it to move the logits, so it needs no biased model. Each
// is tested on its own because they enter at different points: attn_output.bias
// was once added to a buffer the next RMSNorm overwrote, and a shared "some bias
// moved the logits" check would pass with three of four dead.
// TestGreedyMatchesLlamaCpp covers a real biased model (Qwen2).
func TestAttentionBiasesReachTheModel(t *testing.T) {
	for name, path := range models {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				t.Skipf("model not present: %s", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if m.Vocab == nil {
				t.Skipf("no tokenizer: %v", m.TokErr)
			}
			ids := m.Vocab.Encode("The capital of France is", true)
			base := logitsOf(t, m, ids)

			c := m.Cfg
			for _, b := range []struct {
				name string
				set  func(l *layer, v []float32)
			}{
				{"attn_q.bias", func(l *layer, v []float32) { l.bq = v }},
				{"attn_k.bias", func(l *layer, v []float32) { l.bk = v }},
				{"attn_v.bias", func(l *layer, v []float32) { l.bv = v }},
				{"attn_output.bias", func(l *layer, v []float32) { l.bo = v }},
			} {
				t.Run(b.name, func(t *testing.T) {
					// Width by role: q is nHead*headDim, k and v are the KV
					// width, and the output projection is back at n_embd.
					w := c.NEmbd
					switch b.name {
					case "attn_q.bias":
						w = c.NHead * c.HeadDim
					case "attn_k.bias", "attn_v.bias":
						w = c.KVDim()
					}
					v := make([]float32, w)
					for i := range v {
						v[i] = 0.05 // small, but far outside f32 rounding
					}
					for li := range m.layers {
						b.set(&m.layers[li], v)
					}
					defer func() {
						for li := range m.layers {
							b.set(&m.layers[li], nil)
						}
					}()

					got := logitsOf(t, m, ids)
					var d float64
					for i := range base {
						if x := math.Abs(float64(base[i] - got[i])); x > d {
							d = x
						}
					}
					// The assertion is that it moved: a bias added to a dead
					// buffer is indistinguishable from a correct no-op otherwise.
					if d == 0 {
						t.Errorf("%s changed no logit: it is being computed and discarded", b.name)
					}
					t.Logf("%s moves the logits by %.4f", b.name, d)
				})
			}
		})
	}
}

// TestBiasedBatchMatchesForward is the second half: ForwardBatch must apply the
// same biases as Forward, which the plain batch gate cannot see on unbiased
// models.
func TestBiasedBatchMatchesForward(t *testing.T) {
	for name, path := range models {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				t.Skipf("model not present: %s", path)
			}
			m, err := Open(jlmOf(t, path), noTune)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			c := m.Cfg
			fill := func(n int, v float64) []float32 {
				x := make([]float32, n)
				for i := range x {
					x[i] = float32(v)
				}
				return x
			}
			// Different values per role, so a bias applied to the wrong vector
			// is a disagreement rather than a coincidence.
			for li := range m.layers {
				l := &m.layers[li]
				l.bq = fill(c.NHead*c.HeadDim, 0.05)
				l.bk = fill(c.KVDim(), -0.03)
				l.bv = fill(c.KVDim(), 0.07)
				l.bo = fill(c.NEmbd, -0.02)
			}

			const steps = 8
			ids := make([]int32, steps)
			for j := range ids {
				ids[j] = int32((7 + j*13) % c.NVocab)
			}
			ref := m.NewState(steps + 1)
			defer ref.Close()
			var want []float32
			for _, id := range ids {
				if want, err = ref.Forward(id); err != nil {
					t.Fatal(err)
				}
			}
			b := m.NewBatch(1, steps+1)
			defer b.Close()
			var got []float32
			for _, id := range ids {
				if _, err := b.ForwardBatch([]int32{id}); err != nil {
					t.Fatal(err)
				}
				got = b.BatchLogits(0)
			}
			var sse, sy2 float64
			for i := range want {
				d := float64(want[i] - got[i])
				sse += d * d
				sy2 += float64(want[i]) * float64(want[i])
			}
			nmse := sse / sy2
			// A batch of one runs the GEMM path where Forward runs MatVec, so
			// they reassociate; a dropped bias is orders above this bar.
			if nmse > 1e-3 || math.IsNaN(nmse) {
				t.Errorf("biased batch-of-one vs Forward: NMSE %.3e -- a bias is "+
					"reaching one path and not the other", nmse)
			}
			t.Logf("biased batch-of-one vs Forward: NMSE %.3e", nmse)

			// And prefill, the third copy. With the GEMM pinned out this is bit
			// equality, not a bar.
			t.Run("prefill", func(t *testing.T) {
				pm, err := Open(jlmOf(t, path), noTune, noGEMM)
				if err != nil {
					t.Fatal(err)
				}
				defer pm.Close()
				for li := range pm.layers {
					l := &pm.layers[li]
					l.bq = fill(c.NHead*c.HeadDim, 0.05)
					l.bk = fill(c.KVDim(), -0.03)
					l.bv = fill(c.KVDim(), 0.07)
					l.bo = fill(c.NEmbd, -0.02)
				}
				ref := pm.NewState(steps + 1)
				defer ref.Close()
				var w []float32
				for _, id := range ids {
					if w, err = ref.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				pre := pm.NewState(steps + 1)
				defer pre.Close()
				g, err := pre.Prefill(ids)
				if err != nil {
					t.Fatal(err)
				}
				var d float64
				for i := range w {
					if x := math.Abs(float64(w[i] - g[i])); x > d {
						d = x
					}
				}
				if d != 0 {
					t.Errorf("biased prefill vs Forward: max|dlogit| %.4e with the "+
						"GEMM disabled -- these must be bit-identical", d)
				}
				t.Logf("biased prefill vs Forward: bit-identical")
			})
		})
	}
}

// logitsOf prefills ids on a fresh state and returns the last-token logits.
func logitsOf(t *testing.T, m *Model, ids []int32) []float32 {
	t.Helper()
	st := m.NewState(len(ids) + 8)
	defer st.Close()
	var out []float32
	var err error
	for _, id := range ids {
		if out, err = st.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	cp := make([]float32, len(out))
	copy(cp, out)
	return cp
}

// TestPrefillAfterForwardContinues covers the multi-turn shape: a prompt, some
// generated tokens, then another prompt appended to the same session. Forward
// must advance bpos, which Prefill reads, or the second prompt restarts at the
// last Prefill's end and overwrites the generated tokens' cache entries.
func TestPrefillAfterForwardContinues(t *testing.T) {
	for name, path := range models {
		if name == "tinyllama" && os.Getenv("JITLLM_SLOW") == "" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(path); err != nil {
				t.Skipf("model not present: %s", path)
			}
			// noGEMM so the comparison is bit equality.
			m, err := Open(jlmOf(t, path), noTune, noGEMM)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := []int32{7, 19, 33, 51, 64}
			for i := range ids {
				ids[i] %= int32(m.Cfg.NVocab)
			}

			// All five through Forward, one at a time.
			ref := m.NewState(len(ids) + 4)
			defer ref.Close()
			var want []float32
			for _, id := range ids {
				if want, err = ref.Forward(id); err != nil {
					t.Fatal(err)
				}
			}
			// The same five: two decoded, then three PREFILLED on top.
			mix := m.NewState(len(ids) + 4)
			defer mix.Close()
			for _, id := range ids[:2] {
				if _, err = mix.Forward(id); err != nil {
					t.Fatal(err)
				}
			}
			got, err := mix.Prefill(ids[2:])
			if err != nil {
				t.Fatal(err)
			}
			if mix.Pos() != len(ids) {
				t.Errorf("position is %d after 2 Forward + 3 Prefill, want %d",
					mix.Pos(), len(ids))
			}
			var d float64
			for i := range want {
				if x := math.Abs(float64(want[i] - got[i])); x > d {
					d = x
				}
			}
			if d != 0 {
				t.Errorf("Prefill after Forward: max|dlogit| %.4e -- the second "+
					"call started at the wrong position", d)
			}
			t.Logf("Prefill after Forward: bit-identical, pos %d", mix.Pos())
		})
	}
}
