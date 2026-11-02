package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGPTOSSMatchesLlamaCppIntermediates holds gpt-oss-20b's graph to
// llama.cpp's running graph on one token, node by node. The numbers are
// llama-eval-callback's (b10825, CPU) for the prompt "The", token 976, on
// gpt-oss-20b-Q4_K_M.gguf:
//
//	llama-eval-callback -m gpt-oss-20b-Q4_K_M.gguf -p "The" -n 1 -ngl 0 -c 64
//
// A sum is weak alone and strong in sequence: each node is the next one's
// input, so a missing sink, bias, norm position, YaRN magnitude, gating or
// activation moves every sum after it. The bounds are the engines'
// quantization noise (llama.cpp quantizes a k-quant matvec's input per 256,
// this engine per 32).
//
// Position 0 is an identity rotation, so the roped K is the unroped K times
// YaRN's magnitude, 1.3466 -- which is what makes the K row a YaRN check even
// at one token. And at one position the sink is the whole difference between
// a softmax of 1 and 1/(1+exp(sink-score)).
func TestGPTOSSMatchesLlamaCppIntermediates(t *testing.T) {
	path, ok := existingModel(testmodels.Path("gpt-oss-20b-Q4_K_M.gguf"))
	if !ok {
		testmodels.Missing(t, "%s", "MODEL MISSING: "+testmodels.Path("gpt-oss-20b-Q4_K_M.gguf")+" (set JITLLM_MODELS to the model directory) -- RULE 11, fetch it; this gate proves nothing without it")
	}
	m, err := Open(jlmOf(t, path))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	sums := map[string]float64{}
	var topk []float32
	m.Trace(func(l int, name string, v []float32) {
		if name == "moe_topk" && l == 0 {
			topk = append([]float32(nil), v...)
		}
		var s float64
		for _, x := range v {
			s += float64(x)
		}
		sums[fmt.Sprintf("%s-%d", name, l)] = s
	})
	st := m.NewInstance(8)
	defer st.Close()
	logits, err := st.Forward(976)
	if err != nil {
		t.Fatal(err)
	}
	sums["moe_out-0"] = sums["ffn_resid-0"] - sums["attn_resid-0"]
	for _, c := range []struct {
		ours, theirs string
		want, tol    float64 // tol is relative
	}{
		{"attn_norm-0", "attn_norm-0", -27.703680, 1e-5},
		{"k-0", "Kcur-0 (roped)", -4.469628, 2e-3},
		{"attn-0", "kqv_out-0", -38.742336, 5e-3},
		{"attn_resid-0", "ffn_inp-0", -96.851616, 5e-3},
		{"moe_out-0", "ffn_moe_out-0", 87.151596, 3e-2},
	} {
		got, ok := sums[c.ours]
		if !ok {
			t.Fatalf("no trace point %s", c.ours)
		}
		rel := math.Abs(got-c.want) / math.Abs(c.want)
		if !(rel <= c.tol) {
			t.Errorf("%s (llama.cpp %s): %.6f against %.6f, relative %.2e over %.0e",
				c.ours, c.theirs, got, c.want, rel, c.tol)
			continue
		}
		t.Logf("%-13s %-16s %14.6f  llama.cpp %14.6f  rel %.2e", c.ours, c.theirs, got, c.want, rel)
	}
	// The router's choice, in score order, is exact or it is a different FFN.
	if want := []float32{13, 21, 17, 2}; fmt.Sprint(topk) != fmt.Sprint(want) {
		t.Errorf("layer 0 selected experts %v, llama.cpp %v", topk, want)
	}
	for i, want := range []float32{-0.7553, 3.0155, -0.6948} {
		if d := math.Abs(float64(logits[i] - want)); !(d < 0.1) {
			t.Errorf("logit %d: %.4f against llama.cpp's %.4f", i, logits[i], want)
		}
	}
	t.Logf("layer 0 experts %v; logits[0:3] %v (llama.cpp -0.7553 3.0155 -0.6948)", topk, logits[:3])
}
