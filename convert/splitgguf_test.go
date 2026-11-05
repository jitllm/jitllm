package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestSplitGGUFConvertsLikeTheWholeFile converts a model split by llama.cpp's
// llama-gguf-split and the same model whole, and demands byte-identical
// containers. The parts are made once with
//
//	llama-gguf-split --split --split-max-tensors 60 tinyllama-1.1b-q3_K_M.gguf split-test/tl
//
// in JITLLM_MODELS. Every tensor served from the wrong part reads in-bounds
// bytes of another tensor, so the check is the bytes, not an error.
func TestSplitGGUFConvertsLikeTheWholeFile(t *testing.T) {
	whole := testmodels.Path("tinyllama-1.1b-q3_K_M.gguf")
	part := testmodels.Path(filepath.Join("split-test", "tl-00001-of-00004.gguf"))
	for _, p := range []string{whole, part} {
		if _, err := os.Stat(p); err != nil {
			testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
		}
	}
	dir, err := os.MkdirTemp(filepath.Dir(part), "splitgate")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	var out [2][]byte
	for i, src := range []string{whole, part} {
		dst := filepath.Join(dir, filepath.Base(src)+".jlm")
		if _, err := FromGGUF(src, dst, jlm.Fingerprint{}); err != nil {
			t.Fatal(err)
		}
		if out[i], err = os.ReadFile(dst); err != nil {
			t.Fatal(err)
		}
	}
	if !bytes.Equal(out[0], out[1]) {
		for i := range out[0] {
			if i >= len(out[1]) || out[0][i] != out[1][i] {
				t.Fatalf("containers differ at byte %d (%d and %d bytes)", i, len(out[0]), len(out[1]))
			}
		}
		t.Fatalf("containers differ in length: %d and %d", len(out[0]), len(out[1]))
	}
	t.Logf("split and whole convert to the same %d bytes", len(out[0]))
}
