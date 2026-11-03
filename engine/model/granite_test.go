package model

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGraniteMatchesLlamaCppValues holds Granite's four scaling constants to
// llama.cpp's own intermediates, because the greedy gate cannot see two of
// them: logit_scale DIVIDES every logit by one positive number, which moves no
// argmax, and a wrong residual scale is a fluent model on a short prompt.
//
// The oracle is llama-eval-callback on the same GGUF, same prompt:
//
//	llama-eval-callback -m <file> -p "The capital" -n 1 -ngl 0 -c 64
//
// l_out-0 is block 0's output at both positions (embedding_scale and
// residual_scale are in it; the dump prints each row's first and last three),
// and result_output is the last position's logits (attention.scale and
// logit_scale are in them). Both engines quantize the activations, so the
// bounds are loose; every violation lands far outside them.
func TestGraniteMatchesLlamaCppValues(t *testing.T) {
	cases := []struct {
		file      string
		ids       []int32
		lOut0     [2][6]float32 // each position's first and last three
		logitsSum float64
		logits    [6]float32
	}{
		{"granite-3.3-2b-instruct-Q4_K_M.gguf", []int32{1318, 18926},
			[2][6]float32{{-0.0112, 0.2801, 0.1442, 0.0053, 0.2461, -0.1647},
				{-0.2819, 0.2363, -0.1898, 0.0730, -0.0562, -0.0268}},
			-36229.632812, [6]float32{2.9563, 0.6259, 1.9656, -2.3235, -1.5073, -1.6497}},
		// granitemoe: the residual scale reaches the mixture through the
		// router's routed scale, not through an add of its own.
		{"granite-3.1-1b-a400m-instruct-Q4_K_M.gguf", []int32{1318, 18926},
			[2][6]float32{{-0.2197, 0.1535, -0.1296, -0.1672, -0.0629, 0.1385},
				{0.1279, 0.2047, -0.1128, -0.0206, 0.0541, -0.0182}},
			379907.8125, [6]float32{19.3710, 2.7529, 2.7197, 5.6961, 5.8532, 8.0791}},
		// granite-4.0-micro: head_count_kv is a per-layer array here, and a
		// shared-expert width sits beside expert_count 0.
		{"granite-4.0-micro-Q4_K_M.gguf", []int32{791, 6864},
			[2][6]float32{{0.4195, 0.4019, -0.1699, -0.1860, -0.1602, 0.0357},
				{0.3087, 0.2906, -0.4681, -0.5217, 0.0767, -0.1074}},
			205086.59375, [6]float32{14.0201, 13.8513, 6.7002, -4.1401, -4.1264, -4.1352}},
	}
	ends := func(v []float32) [6]float32 {
		n := len(v)
		return [6]float32{v[0], v[1], v[2], v[n-3], v[n-2], v[n-1]}
	}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			// The Granite files sit in their own subdirectory of the model
			// directory, as qwen35/ and embed/ do, or at its top level.
			path, ok := existingModel(testmodels.Path(filepath.Join("granite", tc.file)))
			if !ok {
				path, ok = existingModel(testmodels.Path(tc.file))
			}
			if !ok {
				testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path(tc.file)+" (set JITLLM_MODELS) -- RULE 11, fetch it")
			}
			// noTune(): on arm64 the packed matvec times its kernels in place for
			// a shape's first calls, so Forward and ForwardBatch would otherwise
			// run different kernels (an integer and a float super-block fold)
			// for the same token.
			m, err := Open(jlmOf(t, path), noTune(), WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := m.Vocab.Encode("The capital", true)
			if len(ids) != len(tc.ids) || ids[0] != tc.ids[0] || ids[1] != tc.ids[1] {
				t.Fatalf("ids %v, llama.cpp's %v", ids, tc.ids)
			}
			var lOut0 [][6]float32
			m.Trace(func(l int, name string, v []float32) {
				if l == 0 && name == "ffn_resid" {
					lOut0 = append(lOut0, ends(v))
				}
			})
			st := m.NewInstance(8)
			defer st.Close()
			var logits []float32
			for _, id := range ids {
				if logits, err = st.Forward(id); err != nil {
					t.Fatal(err)
				}
			}
			sum := 0.0
			for _, x := range logits {
				sum += float64(x)
			}
			got := ends(logits)
			t.Logf("l_out-0 %v (llama.cpp %v), logits sum %.3f (llama.cpp %.3f), ends %v (llama.cpp %v)",
				lOut0, tc.lOut0, sum, tc.logitsSum, got, tc.logits)
			if len(lOut0) != 2 {
				t.Fatalf("traced block 0 at %d positions, want 2", len(lOut0))
			}
			for p := range lOut0 {
				for i := range lOut0[p] {
					if d := math.Abs(float64(lOut0[p][i] - tc.lOut0[p][i])); !(d < 0.02) {
						t.Errorf("block 0 output, position %d value %d: %.4f against llama.cpp's %.4f",
							p, i, lOut0[p][i], tc.lOut0[p][i])
					}
				}
			}
			if rel := math.Abs(sum-tc.logitsSum) / math.Abs(tc.logitsSum); !(rel < 0.05) {
				t.Errorf("logits sum to %.3f against llama.cpp's %.3f (%.1f%%)", sum, tc.logitsSum, 100*rel)
			}
			for i := range got {
				if d := math.Abs(float64(got[i] - tc.logits[i])); !(d < 0.25) {
					t.Errorf("logit %d is %.4f against llama.cpp's %.4f", i, got[i], tc.logits[i])
				}
			}

			// And the batched path, which adds its own residuals and makes its
			// own logits; nothing else that opens a Granite reaches ForwardBatch.
			m.Trace(nil)
			b := newSessBatch(m, 1, 8)
			defer b.Close()
			for _, id := range ids {
				if _, err := b.ForwardBatch([]int32{id}); err != nil {
					t.Fatal(err)
				}
			}
			var sse, sy2 float64
			for i, v := range b.BatchLogits(0) {
				d := float64(v - logits[i])
				sse, sy2 = sse+d*d, sy2+float64(logits[i])*float64(logits[i])
			}
			if nmse := sse / sy2; !(nmse < 1e-3) {
				t.Errorf("ForwardBatch's logits against Forward's: NMSE %.3e", nmse)
			}
		})
	}
}
