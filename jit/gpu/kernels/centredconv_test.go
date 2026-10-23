package kernels

import (
	"testing"

	"github.com/jitllm/jitllm/jit/ir"
)

// TestConv3CentredValidates: the kernel builds and passes ir.Validate (its
// input and output are separate params), and refuses an empty shape. backend
// runs it against the host's loop (TestConv3CentredMatchesTheLoop).
func TestConv3CentredValidates(t *testing.T) {
	for _, tc := range []struct{ chans, rows int }{{8, 1}, {8, 6}, {1024, 64}, {3, 7}} {
		k, err := Conv3Centred(tc.chans, tc.rows)
		if err != nil {
			t.Fatal(err)
		}
		if err := k.Validate(); err != nil {
			t.Fatalf("Conv3Centred(%d, %d): %v", tc.chans, tc.rows, err)
		}
	}
	if _, err := Conv3Centred(0, 4); err == nil {
		t.Error("an empty shape built")
	}
}

// TestConformerKernelsValidate: the FastConformer block's kernels build and
// pass ir.Validate (no buffer read and written), and refuse empty shapes.
// backend runs them against their loops (TestConformerKernelsMatchTheLoops).
func TestConformerKernelsValidate(t *testing.T) {
	for _, k := range []struct {
		name string
		ok   func() error
		bad  func() error
	}{
		{"relposscores", func() error { k, err := RelPosScores(376, 8, 64); return validated(k, err) },
			func() error { _, err := RelPosScores(0, 8, 64); return err }},
		{"attnaccrows", func() error { k, err := AttnAccRows(376, 8, 64); return validated(k, err) },
			func() error { _, err := AttnAccRows(4, 0, 64); return err }},
		{"dwconv1drows", func() error { k, err := DWConv1dRows(9, 512, 376); return validated(k, err) },
			func() error { _, err := DWConv1dRows(4, 512, 376); return err }},
		{"affinerows", func() error { k, err := AffineRows(512, 376); return validated(k, err) },
			func() error { _, err := AffineRows(0, 376); return err }},
	} {
		if err := k.ok(); err != nil {
			t.Errorf("%s: %v", k.name, err)
		}
		if k.bad() == nil {
			t.Errorf("%s: an empty or even shape built", k.name)
		}
	}
}

func validated(k *ir.Kernel, err error) error {
	if err != nil {
		return err
	}
	return k.Validate()
}
