package convert

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/convert/safetensors"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestShardsReadInPlaceWriteTheSameContainer converts one model twice: from its
// directory, and from shards handed over as byte-range readers with no file
// behind them -- what a streamed conversion is. The two containers must be the
// same bytes, with and without Q8.
func TestShardsReadInPlaceWriteTheSameContainer(t *testing.T) {
	dir := testmodels.Path("synth-kimilinear")
	_, shards, err := hfShards(dir)
	if err != nil {
		testmodels.Missing(t, "MODEL MISSING: %s: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", dir, err)
	}
	for _, q8 := range []bool{false, true} {
		var opts []Option
		if q8 {
			opts = append(opts, WithQ8())
		}
		tmp := t.TempDir()
		a, b := filepath.Join(tmp, "a.jlm"), filepath.Join(tmp, "b.jlm")
		if _, err := FromSafetensors(dir, a, jlm.Fingerprint{Host: "t"}, opts...); err != nil {
			t.Fatal(err)
		}
		var open []*safetensors.File
		for _, p := range shards {
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			f, err := safetensors.OpenAt(p, bytes.NewReader(raw), int64(len(raw)))
			if err != nil {
				t.Fatal(err)
			}
			open = append(open, f)
		}
		if _, err := FromShards(os.DirFS(dir), dir, open, b, jlm.Fingerprint{Host: "t"}, opts...); err != nil {
			t.Fatal(err)
		}
		ab, _ := os.ReadFile(a)
		bb, _ := os.ReadFile(b)
		if len(ab) == 0 || !bytes.Equal(ab, bb) {
			t.Fatalf("q8=%v: %d bytes from the directory, %d from the readers, and they differ",
				q8, len(ab), len(bb))
		}
	}
}
