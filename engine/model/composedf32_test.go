package model

import (
	"math"
	"os"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestComposedPromptMatchesLlamaCppF32 holds the composed prompt -- its
// attention on the matrix unit over binary16 operands -- to llama.cpp's CPU
// arithmetic with a float32 KV cache and no flash attention
// (JITLLM_LLAMACPP_CPU), teacher-forced over a 2048-token document: at every
// 64th position of its second half, each prefix prefilled whole by the
// composer on cuda:0, the log-softmax over llama.cpp's top-16-nat window is
// held to a median and worst NMSE and a mean KL(reference||arm). Each bound
// is the arm's own reading plus the spread llama.cpp's CUDA arms (flash
// attention on and off) and main's CUDA tier take against the same
// reference: what a reduction order alone moves. The violation, every
// position's logits read against the next position's, must fall outside a
// bound.
func TestComposedPromptMatchesLlamaCppF32(t *testing.T) {
	dir := os.Getenv("JITLLM_LLAMACPP_CPU")
	if dir == "" {
		t.Fatal("JITLLM_LLAMACPP_CPU is unset: it names a llama.cpp build holding llama-perplexity, the reference")
	}
	text, err := os.ReadFile("testdata/prompt-28k.txt")
	if err != nil {
		t.Fatal(err)
	}
	const nctx, stride = 2048, 64
	for _, c := range []struct {
		file              string
		median, worst, kl float64
	}{
		{"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 3.89e-5 + 0.54e-5, 5.63e-5 + 3.85e-5, 0.00278 + 0.00021},
		{"Qwen3-1.7B-Q4_K_M.gguf", 1.55e-4 + 0.32e-4, 1.80e-3 + 0.32e-3, 0.00751 + 0.00235},
	} {
		t.Run(c.file, func(t *testing.T) {
			gguf := testmodels.Path(c.file)
			if !fileExists(gguf) {
				t.Fatalf("MODEL MISSING: %s (set JITLLM_MODELS)", gguf)
			}
			toks, ref := lcppLogProbs(t, dir, gguf, string(text), nctx, 0, "-ctk", "f32", "-ctv", "f32", "-fa", "off")
			var at []int
			for i := 0; i < len(ref)-1; i += stride {
				at = append(at, i)
			}
			m, err := Open(jlmOf(t, gguf), noTune())
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			ids := slices.Clone(toks)
			if m.Vocab.AddBOS && m.Vocab.BOS >= 0 {
				ids[0] = int32(m.Vocab.BOS)
			}
			got := map[int][]float32{}
			for _, i := range at {
				l, st, _ := composedDecode(t, m, "cuda:0", ids[:nctx/2+i+1], nil, gpu.ComposedBlock|gpu.ComposedPrompt, false, false)
				if n, d := st.ServedPart(gpu.ComposedPrompt); n == 0 || d != 0 {
					t.Fatalf("position %d: the composer served %d block steps of the prompt and declined %d", nctx/2+i, n, d)
				}
				got[i] = logSoftmax(l[0])
			}
			score := func(off int) (float64, float64, float64) {
				var ns []float64
				var kl float64
				for _, i := range at {
					n, k := lpCompare(got[i], ref[i+off])
					ns = append(ns, n)
					kl += k
				}
				slices.Sort(ns)
				return ns[len(ns)/2], ns[len(ns)-1], kl / float64(len(at))
			}
			med, worst, kl := score(0)
			t.Logf("%d positions: NMSE median %.3g worst %.3g, mean KL %.3g (bounds %.3g, %.3g, %.3g)", len(at), med, worst, kl, c.median, c.worst, c.kl)
			if !(med <= c.median && worst <= c.worst && kl <= c.kl) {
				t.Errorf("against llama.cpp's float32 CPU arithmetic the composed prompt reads NMSE median %.3g worst %.3g and mean KL %.3g, past %.3g, %.3g and %.3g",
					med, worst, kl, c.median, c.worst, c.kl)
			}
			bm, bw, bk := score(1)
			t.Logf("one position off: NMSE median %.3g worst %.3g, mean KL %.3g", bm, bw, bk)
			if bm <= c.median && bw <= c.worst && bk <= c.kl {
				t.Errorf("one position off reads NMSE median %.3g worst %.3g and mean KL %.3g, inside every bound: the gate cannot see it", bm, bw, bk)
			}
		})
	}
}

// lpCompare is the NMSE of a's log-probabilities against ref's over ref's
// top-16-nat window, and KL(ref||a) over it.
func lpCompare(a, ref []float32) (float64, float64) {
	top := slices.Max(ref)
	var num, den, kl float64
	for i, r := range ref {
		if math.IsInf(float64(r), -1) || math.IsInf(float64(a[i]), -1) || r < top-15.5 {
			continue
		}
		d := float64(a[i] - r)
		num += d * d
		den += float64(r) * float64(r)
		kl -= math.Exp(float64(r)) * d
	}
	return num / den, kl
}
