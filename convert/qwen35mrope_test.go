package convert

import (
	"maps"
	"os"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/convert/gguf"
	"github.com/jitllm/jitllm/convert/meta"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestQwen35SectionsAreReadAsEveryMRopes: Qwen3.5's rope sections are held to
// the rule every interleaved M-RoPE is (ropeSectionsFit): a fourth section,
// and a split that leaves a pair to the axis ggml and transformers turn
// differently, are refused; the file's own sections convert.
func TestQwen35SectionsAreReadAsEveryMRopes(t *testing.T) {
	path := testmodels.Path("synth-qwen35vl.gguf")
	if _, err := os.Stat(path); err != nil {
		testmodels.Missing(t, "MODEL MISSING: %v (see docs/testing.md) -- this gate proved nothing", err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	key := f.Arch() + ".rope.dimension_sections"
	v, ok := f.KV[key]
	if !ok {
		t.Fatalf("%s states no %s", path, key)
	}
	secs, err := v.Int32s()
	if err != nil || len(secs) < 3 || secs[0] < 2 || secs[2] < 1 {
		t.Fatalf("sections %v: %v", secs, err)
	}
	with := func(s []int32) error {
		kv := maps.Clone(f.KV)
		kv[key] = meta.MakeInt32s(s)
		_, err := configOf(meta.New(f.Path, kv, f.Tensors))
		return err
	}
	if err := with(secs[:3]); err != nil {
		t.Fatalf("the file's sections %v: %v", secs[:3], err)
	}
	for _, r := range []struct {
		name string
		secs []int32
		err  string
	}{
		{"a fourth section", []int32{secs[0], secs[1], secs[2] - 1, 1}, "fourth section"},
		{"a pair left to the fourth axis", []int32{secs[0] - 1, secs[1], secs[2] + 1}, "fourth axis"},
	} {
		if err := with(r.secs); err == nil || !strings.Contains(err.Error(), r.err) {
			t.Errorf("%s %v: %v, want an error saying %q", r.name, r.secs, err, r.err)
		}
	}
}
