package composer_test

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestMatrixPrepLaunches forms a normed input's three projections and a
// wide one on a V100's stated units at a decode step's rows and a prompt
// chunk's, and holds what each row factor costs before its projections to
// one kernel: its scale and, where the stage does not convert it, its
// binary16 words, written together.
func TestMatrixPrepLaunches(t *testing.T) {
	dev := composer.Device{Kind: composer.DeviceGPU, Slots: 80 * 2048, Lanes: 32, Units: 80, Shared: 96 << 10, ExactDiv: true, Matrix: [3]int{16, 16, 16}}
	jt := jlm.TypeQ4S
	for _, rows := range []int{16, 64, 512} {
		for _, convert := range []int{0, -1, 1} {
			const K = 4096
			b := graph.New()
			x := b.In("x", K)
			h := b.Norm(x, jlm.RoleAttnNorm, 1e-5)
			var outs []graph.ValueID
			for _, role := range []jlm.Role{jlm.RoleAttnQ, jlm.RoleAttnK, jlm.RoleAttnV} {
				outs = append(outs, b.MatVec(h, role, 1024))
			}
			outs = append(outs, b.MatVec(x, jlm.RoleFFNUp, 14336))
			b.Out(outs...)
			d := dev
			if convert != 0 {
				d.Tile = gpuMatrixTiles()[0]
				d.Tile.Convert = convert
			}
			_, codes, err := composer.Default().Form(composer.Request{Graph: b.G,
				Formats: map[jlm.Role]jlm.Type{jlm.RoleAttnQ: jt, jlm.RoleAttnK: jt, jlm.RoleAttnV: jt, jlm.RoleFFNUp: jt, jlm.RoleAttnNorm: jlm.TypeF32},
				Step:    composer.Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}, Device: d})
			if err != nil {
				t.Fatal(err)
			}
			prep, halves := 0, 0
			for _, c := range codes {
				writes := map[string]bool{}
				for _, bd := range c.Binds {
					for _, suf := range []string{".halves", ".rowscale"} {
						if bd.Writes && strings.HasSuffix(bd.Operand.Name, suf) {
							writes[suf] = true
						}
					}
				}
				if len(writes) > 0 {
					prep++
				}
				if writes[".halves"] {
					halves++
					if !writes[".rowscale"] {
						t.Errorf("rows %d convert %d: %s writes a row factor's binary16 words apart from its scale", rows, convert, c.Name)
					}
				}
			}
			t.Logf("rows %d convert %d: %d prep launches for two row factors, %d writing binary16 words", rows, convert, prep, halves)
			if prep != 2 {
				for _, c := range codes {
					t.Logf("%s, %d workgroups of %d, passes %v", c.Name, c.Groups, c.GPU.Group[0], c.Passes)
				}
				t.Fatalf("rows %d convert %d: %d prep launches for two row factors, not one each", rows, convert, prep)
			}
			if convert != 0 && (halves == 0) != (convert > 0) {
				t.Fatalf("rows %d convert %d: %d kernels write binary16 words", rows, convert, halves)
			}
		}
	}
}
