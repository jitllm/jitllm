package paritytest

import (
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestXIELUIsMains holds the composer's xIELU to main's XIELU.
func TestXIELUIsMains(t *testing.T) {
	mk, err := mainkernels.XIELU(256)
	if err != nil {
		t.Fatal(err)
	}
	g := &compose.Row{Width: 256, Rows: 1}
	x, p, out := g.Buf(compose.BufRow), g.Buf(compose.BufScalar), g.Buf(compose.BufOut)
	g.Store(out, g.XIELU(g.Load(x), p))
	same(t, "xielu", []*ir.Kernel{mk}, planned(t, g, compose.Facts{}))
}
