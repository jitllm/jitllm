package convert

import (
	"errors"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
)

// TestAValueOfAnotherWidthIsRefused: outside an MLA model the history holds
// a key and a value of one head width, so a file whose value heads are
// another width is refused at conversion with ErrNotImplemented; an MLA
// model's narrower value, and a value stated at the key's width, convert.
func TestAValueOfAnotherWidthIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name   string
		v, lat uint32
		refuse bool
	}{
		{"unstated", 0, 0, false},
		{"the key's width", 128, 0, false},
		{"narrower", 64, 0, true},
		{"wider", 256, 0, true},
		{"an MLA model's narrower value", 64, 16, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := valueWidthFits(&Config{Config: &jlm.Config{HeadDim: 128, HeadDimV: tc.v, KVLoraRank: tc.lat}})
			switch {
			case tc.refuse && !errors.Is(err, ErrNotImplemented):
				t.Fatalf("refused with %v, want ErrNotImplemented", err)
			case !tc.refuse && err != nil:
				t.Fatal(err)
			}
		})
	}
}
