package gpu

import (
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestEveryContainerTypeIsADeviceFormat: every type a container stores
// has a device format (quantOf) that names the container type back
// (jlmType), so the card refuses no weight or table for its format. The
// violation, a container type quantOf does not map, is seen.
func TestEveryContainerTypeIsADeviceFormat(t *testing.T) {
	n := 0
	for c := range jlm.Type(255) {
		src, ok := jlm.SourceType(c)
		if !ok {
			continue
		}
		n++
		q, ok := quantOf(src)
		if !ok {
			t.Errorf("container type %v (%v) has no device format", c, src)
			continue
		}
		back, ok := jlmType(q)
		if !ok {
			t.Errorf("device format %v of %v names no container type", q, c)
			continue
		}
		if s, _ := jlm.SourceType(back); s != src {
			t.Errorf("container type %v reads as %v, its device format %v as %v", c, src, q, s)
		}
	}
	if n < 10 {
		t.Fatalf("%d container types: the gate read nothing", n)
	}
	t.Logf("%d container types, each a device format", n)
}
