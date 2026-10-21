package paritytest

import (
	"fmt"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestQuantizeIsMains holds the composer's int8 operand to main's Quantize,
// and of rows padded to whole blocks to QuantizePadded.
func TestQuantizeIsMains(t *testing.T) {
	for _, c := range []struct{ rows, k, win int }{{1, 64, 32}, {4, 128, 32}, {3, 256, 256}, {2, 144, 32}, {5, 72, 32}, {1, 200, 32}} {
		kp := (c.k + 31) / 32 * 32
		win := min(c.win, kp)
		name := fmt.Sprintf("r%d_k%d_w%d", c.rows, c.k, c.win)
		t.Run(name, func(t *testing.T) {
			var mk *ir.Kernel
			var err error
			if kp == c.k {
				mk, err = mainkernels.Quantize(c.rows*c.k, win)
			} else {
				mk, err = mainkernels.QuantizePadded(c.rows, c.k, kp, win)
			}
			if err != nil {
				t.Fatal(err)
			}
			g := &compose.Row{Width: c.rows * kp / 4, Rows: 1}
			x, a, ax := g.Buf(compose.BufRow), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut)
			if kp == c.k {
				g.Quantize8(x, a, ax, win)
			} else {
				g.Quantize8Padded(x, a, ax, win, c.k, kp)
			}
			same(t, name, []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
		})
	}
}
