package composer

import (
	"testing"

	"github.com/jitllm/jitllm/format/quant"
)

// TestActWindowFollowsTheScales holds the model-wide activation window to
// its rule: the k-quant super-block only where every matrix is a k-quant,
// the 32-wide block wherever one matrix keeps 32-wide scales, and the
// narrow window for a model of float matrices alone.
func TestActWindowFollowsTheScales(t *testing.T) {
	for _, c := range []struct {
		name  string
		types []quant.Type
		want  int
	}{
		{"k-quants", []quant.Type{quant.Q4_K, quant.Q6_K}, 256},
		{"a 32-wide block among k-quants", []quant.Type{quant.Q4_K, quant.Q8_0}, 32},
		{"floats alone", []quant.Type{quant.F16, quant.F32}, 32},
		{"k-quants beside floats", []quant.Type{quant.F32, quant.Q5_K}, 256},
		{"nothing", nil, 32},
	} {
		if got := ActWindow(c.types); got != c.want {
			t.Errorf("%s: window %d, want %d", c.name, got, c.want)
		}
	}
}
