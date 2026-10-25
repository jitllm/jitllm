package model

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestGGUFCannotBeInferred: the engine reads one weight format, the container;
// gguf.Open survives only for conversion. The gate is the refusal, because a
// call site that skips on an open error would otherwise go green silently.
func TestGGUFCannotBeInferred(t *testing.T) {
	src := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	m, err := Open(src)
	if err == nil {
		m.Close()
		t.Fatal("a GGUF opened for inference -- the engine reads one format and this is not it")
	}
	var nc *NotConvertedError
	if !errors.As(err, &nc) {
		t.Fatalf("refused with %v, want a *NotConvertedError naming the convert command", err)
	}
	// The error's whole job is to hand back the next line to type.
	if got := nc.Error(); !contains(got, "jitllm convert") || !contains(got, Ext) {
		t.Fatalf("the refusal does not say how to fix it:\n%s", got)
	}
	t.Logf("refused, and said how:\n%s", nc)
}

// TestContainerRunsOnTheCPU: the container the refusal points at must actually
// run. Same model, converted, decoding real tokens through the packed kernels.
func TestContainerRunsOnTheCPU(t *testing.T) {
	path := jlmOf(t, testmodels.Path("tinyllama-1.1b-q3_K_M.gguf"))
	m, err := Open(path)
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
	ids := m.Vocab.Encode("The capital of France is", true)
	st := m.NewState(len(ids) + 8 + 1)
	// Closed: a State owns a worker pool, and a leaked one keeps threads
	// spinning on the same cores for the rest of the test binary.
	defer st.Close()
	logits, err := st.Prefill(ids)
	if err != nil {
		t.Fatal(err)
	}
	out := append([]int32(nil), ids...)
	for i := 0; i < 8; i++ {
		best, bv := int32(0), float32(-1e30)
		for j, v := range logits {
			if v > bv {
				best, bv = int32(j), v
			}
		}
		out = append(out, best)
		if logits, err = st.Forward(best); err != nil {
			t.Fatal(err)
		}
	}
	// A degenerate answer is a failure: a kernel reading the wrong words still
	// produces ids, but not variety (a collapse to token 0, or repetition).
	distinct := map[int32]bool{}
	for _, id := range out[len(ids):] {
		distinct[id] = true
	}
	if len(distinct) < 3 {
		t.Fatalf("8 generated tokens hold %d distinct ids: %v", len(distinct), out)
	}
	t.Logf("ids %v", out)
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// TestAnUpperCaseExtensionIsStillAContainer: a container is a container
// whatever case its extension is written in (the desktop catalog matches .JLM
// case-insensitively).
func TestAnUpperCaseExtensionIsStillAContainer(t *testing.T) {
	src := testmodels.Path("stories260K.jlm")
	b, err := os.ReadFile(src)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- RULE 11, fetch it; this gate proves nothing without it", err)
	}
	up := filepath.Join(t.TempDir(), "stories260K.JLM")
	if err := os.WriteFile(up, b, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open(up)
	if err != nil {
		t.Fatalf("an upper-case .JLM was refused: %v", err)
	}
	m.Close()
}
