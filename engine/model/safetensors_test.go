package model

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// The end-to-end gate on the safetensors input: the container it writes must
// load and decode like the same model's GGUF. convert/ checks the readers
// tensor by tensor; this covers everything after load. The arms are bf16 and
// Q8_0 of the same weights, so they must agree until a near-tie rather than
// for a fixed token count.

// jlmOfSafetensors converts a HuggingFace directory to a container beside the
// weights and returns that path (convertDir, which jlmOf takes for a
// directory too). It resolves symlinks first so the container lands next to
// the real files, not next to a link in models/.
func jlmOfSafetensors(t testing.TB, dir string) string {
	t.Helper()
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir, err)
	}
	dst, err := convertDir(real, false)
	if err != nil {
		t.Fatalf("convert %s: %v", dir, err)
	}
	return dst
}

// greedy runs one container and returns the ids it generates plus, per step,
// the gap between the winning logit and the runner-up.
func greedy(t testing.TB, path, prompt string, n int) (ids []int32, margins []float32, text string) {
	t.Helper()
	m, err := Open(path, noTune)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if !m.Container {
		t.Fatal("loaded, but not as a container -- this gate proved nothing")
	}
	if m.Vocab == nil {
		t.Fatalf("no tokenizer: %v", m.TokErr)
	}
	in := m.Vocab.Encode(prompt, true)
	st := m.NewInstance(len(in) + n + 1)
	// An instance owns a worker pool; leaking one leaves DecodeCores() threads
	// spinning for the rest of the test binary.
	defer st.Close()
	logits, err := st.Prefill(in)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		best, bv, second := int32(0), float32(-1e30), float32(-1e30)
		for j, v := range logits {
			if v > bv {
				best, bv, second = int32(j), v, bv
			} else if v > second {
				second = v
			}
		}
		ids = append(ids, best)
		margins = append(margins, bv-second)
		if logits, err = st.Forward(best); err != nil {
			t.Fatal(err)
		}
	}
	return ids, margins, m.Vocab.Decode(ids)
}

// TestSafetensorsContainerDecodesLikeItsGGUF is the gate.
func TestSafetensorsContainerDecodesLikeItsGGUF(t *testing.T) {
	hf := testmodels.Path("SmolLM2-360M-Instruct")
	gg := testmodels.Path("SmolLM2-360M-Instruct-Q8_0.gguf")
	if _, err := os.Stat(gg); err != nil {
		t.Skipf("MODEL MISSING: %s (%v) (set JITLLM_MODELS to the model directory) -- this gate proved nothing", gg, err)
	}
	const prompt, n = "The capital of France is", 16

	stPath := jlmOfSafetensors(t, hf)
	stIDs, stM, stText := greedy(t, stPath, prompt, n)
	ggIDs, ggM, ggText := greedy(t, jlmOf(t, gg), prompt, n)

	t.Logf("safetensors %v\n            %q", stIDs, stText)
	t.Logf("gguf        %v\n            %q", ggIDs, ggText)

	// A degenerate answer is a failure: miswired weights still produce ids,
	// but not variety.
	distinct := map[int32]bool{}
	for _, id := range stIDs {
		distinct[id] = true
	}
	if len(distinct) < 4 {
		t.Fatalf("%d generated tokens hold %d distinct ids: %v", n, len(distinct), stIDs)
	}

	for i := range stIDs {
		if stIDs[i] == ggIDs[i] {
			continue
		}
		// Parting at a near-tie is allowed; parting at a confident token
		// (margin over 0.5, the teacher-forced bar) is a reader bug.
		if stM[i] > 0.5 || ggM[i] > 0.5 {
			t.Fatalf("token %d: safetensors %d (margin %.3f) against gguf %d (margin %.3f) -- "+
				"a confident disagreement, not a near-tie\n  safetensors %q\n  gguf        %q",
				i, stIDs[i], stM[i], ggIDs[i], ggM[i], stText, ggText)
		}
		t.Logf("token %d parts at a near-tie: margins %.3f and %.3f", i, stM[i], ggM[i])
		return
	}
	t.Logf("%d token ids identical across both input formats", n)
}
