package composer_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestExpertGroupingIsShared forms a mixture's two projections a selection
// routes over a step of rows the matrix unit runs, and holds the grouping
// of the selection's pairs by expert to one: one kernel writes its counts,
// and both projections' kernels read them.
func TestExpertGroupingIsShared(t *testing.T) {
	const E, P, K, rows, R = 8, 2, 256, 96, 128
	b := graph.New()
	x := b.In("x", P*K)
	sel := b.In("sel", P)
	b.G.Values[sel].Elem = graph.ElemU32
	var outs []graph.ValueID
	for _, role := range []jlm.Role{jlm.RoleExpGateBank, jlm.RoleExpUpBank} {
		y := b.Emit(graph.OpExpertMatVec, []graph.ValueID{x, sel}, []graph.Result{{Name: role.String(), Width: P * rows}},
			[]graph.Binding{{Role: role, K: K, Rows: rows, Sheets: E}}, graph.Attrs{}, nil, nil)
		outs = append(outs, y[0])
	}
	b.Out(outs...)
	jt, _ := jlm.TypeOf(quant.Q4_K)
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 163840, Lanes: 32, Units: 80, Shared: 98304, ExactDiv: true, ExactArith: true, Matrix: [3]int{16, 16, 16}}
	_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleExpGateBank: jt, jlm.RoleExpUpBank: jt},
		Step: composer.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	writes, reads := 0, 0
	for _, c := range codes {
		for _, bd := range c.Binds {
			if !strings.HasSuffix(bd.Operand.Name, ".count") {
				continue
			}
			if bd.Writes {
				writes++
			} else {
				reads++
			}
		}
	}
	if writes != 1 || reads < 2 {
		t.Fatalf("the selection's counts are written by %d kernels and read by %d: want one grouping both projections read", writes, reads)
	}
}
