package composer_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/format/quant"
	"github.com/jitllm/jitllm/graph"
)

// TestProjectionsOfOneRowShareTheirHalves forms three projections of one
// input over a step of rows the matrix unit runs, and holds the row's
// binary16 words to one kernel the three read, each projection where
// Siblings runs it, alone or beside another.
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
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 163840, Lanes: 32, Units: 80, Shared: 98304, ExactDiv: true, Matrix: [3]int{16, 16, 16}}
	_, codes, err := composer.Default().Form(composer.Request{Graph: b.G, Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt, jlm.RoleAttnK: jt, jlm.RoleAttnV: jt},
		Step: composer.Step{Rows: R, Page: 64, Keys: 1, ActWindow: 32}, Device: dev})
	if err != nil {
		t.Fatal(err)
	}
	halves, readers := 0, 0
	for _, c := range codes {
		if strings.HasSuffix(c.Name, ".halves") {
			halves++
		}
		reads := slices.ContainsFunc(c.Binds, func(bd composer.Binding) bool {
			return strings.HasSuffix(bd.Operand.Name, ".halves") && !bd.Writes
		})
		for _, r := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV} {
			if reads && bindsRole(c, r) {
				readers++
			}
		}
	}
	if halves != 1 || readers != 3 {
		t.Fatalf("%d kernels write the row's binary16 words and %d read them: want one the three projections read", halves, readers)
	}
}
