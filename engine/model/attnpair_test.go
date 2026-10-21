package model

import (
	"math"
	"os"
	"testing"
)

// TestPairedAttentionIsBitIdentical holds decode with the paired attention
// kernels to decode without them, logit for logit.
//
// The paired kernels share a load and change no arithmetic, and jit_old/cpu's gate
// holds them bit-identical to two single-head calls, so decode must match too;
// anything else means the pairing is wrong (a pair across a kv-head boundary, a
// dropped odd tail, a task boundary splitting a pair). All four settings run,
// since scores and accumulate can each be wrong alone.
func TestPairedAttentionIsBitIdentical(t *testing.T) {
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
			ids := []int32{7, 19, 33, 51, 64, 12, 88, 41}
			for i := range ids {
				ids[i] %= int32(m.Cfg.NVocab)
			}
			gqa := m.Cfg.GQA()
			run := func(pair, chunk int) []float32 {
				st := m.NewState(len(ids) + 8)
				defer st.Close()
				st.SetAttnPair(pair)
				st.SetAttnChunk(chunk)
				var out []float32
				for _, id := range ids {
					if out, err = st.Forward(id); err != nil {
						t.Fatal(err)
					}
				}
				// The path must have run: with odd gqa (stories15M's is 1) no
				// heads pair and every assertion holds vacuously.
				if pair > 0 && gqa%2 == 0 && st.AttnPaired() == 0 {
					t.Fatalf("gqa=%d is even but no head pair was served: the "+
						"paired path did not run and this proves nothing", gqa)
				}
				return append([]float32(nil), out...)
			}
			if gqa%2 != 0 {
				t.Skipf("gqa=%d is odd, so heads never pair on this model", gqa)
			}
			// The baseline runs the same schedule, so the pool-chunk change is
			// not folded in.
			base := run(-1, 2)
			for _, c := range []struct {
				pair, chunk int
				what        string
			}{
				{1, 2, "scores paired"},
				{2, 2, "accumulate paired"},
				{3, 2, "both paired"},
				{3, 4, "both paired, 4 heads per task"},
				{3, 8, "both paired, 8 heads per task"},
			} {
				got := run(c.pair, c.chunk)
				// Bit patterns, not |a-b|: a difference cannot see a NaN or a
				// signed zero, which is what an uninitialised accumulator or a
				// skipped store produces.
				for i := range base {
					if math.Float32bits(base[i]) != math.Float32bits(got[i]) {
						t.Errorf("%s: logit %d is %v (bits %#x), unpaired gives %v (bits %#x)"+
							" -- the paired kernels are bit-identical to two single calls,"+
							" so decode must be too",
							c.what, i, got[i], math.Float32bits(got[i]),
							base[i], math.Float32bits(base[i]))
						break
					}
				}
			}
		})
	}
}
