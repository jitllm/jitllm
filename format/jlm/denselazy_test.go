package jlm_test

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestDenseOnDemandReadsWhatIsAsked holds frames that read the dense region
// entry by entry to the region read whole: every dense entry's planes are
// the same bytes, and frames asked for none hold no dense memory.
func TestDenseOnDemandReadsWhatIsAsked(t *testing.T) {
	src, _ := expertSource(t)
	dst := filepath.Join(t.TempDir(), "m.jlm")
	if _, err := jlm.Write(dst, src, jlm.Fingerprint{Host: "t"}); err != nil {
		t.Fatal(err)
	}
	eager := openFrames(t, dst)
	c := eager.File()
	lazy := framesOf(t, c, jlm.DenseOnDemand())
	if got := lazy.MappedBytes(); got != 0 {
		t.Fatalf("frames that read no dense entry hold %d bytes of it (the region read whole: %d)", got, eager.MappedBytes())
	}
	n := 0
	for i := range c.Entries() {
		e := &c.Entries()[i]
		if !c.InDense(e) {
			continue
		}
		a, b, d := eager.Span(e)
		x, y, z := lazy.Span(e)
		if !bytes.Equal(a, x) || !bytes.Equal(b, y) || !bytes.Equal(d, z) {
			t.Fatalf("%s: read on demand, its planes differ from the dense region's", e.Name)
		}
		n++
	}
	if n == 0 {
		t.Fatal("the container has no dense entry: the gate proved nothing")
	}
}
