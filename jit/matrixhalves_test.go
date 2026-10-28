package jit_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/jit"
)

// TestProjectionsOfOneRowShareTheirHalves forms three projections of one
// input over a step of rows the matrix unit runs, and holds the row's
// binary16 words to one kernel the three read.
func TestProjectionsOfOneRowShareTheirHalves(t *testing.T) {
	const K, R = 1024, 128
	b := graph.New()
	in := b.In("x", K)
	var outs []graph.ValueID
	for _, role := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV} {
		outs = append(outs, b.MatVec(in, role, 256))
	}
	b.Out(outs...)
	jt, _ := jlm.TypeOf(quant.Q4_K)
	dev := jit.Device{Kind: jit.DeviceGPU, Slots: 163840, Lanes: 32, Units: 80, Shared: 98304, ExactDiv: true, ExactArith: true, Matrix: [3]int{16, 16, 16}}
	_, codes, err := jit.Default().Form(jit.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt, jlm.RoleAttnK: jt, jlm.RoleAttnV: jt},
		Step: jit.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	halves, readers := 0, 0
	for _, c := range codes {
		if strings.HasSuffix(c.Name, ".halves") {
			halves++
		}
		for _, bd := range c.Binds {
			if strings.HasSuffix(bd.Operand.Name, ".halves") && !bd.Writes {
				readers++
			}
		}
	}
	if halves != 1 || readers != 3 {
		t.Fatalf("%d kernels write the row's binary16 words and %d read them: want one the three projections read", halves, readers)
	}
}
