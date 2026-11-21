package composer

import (
	"maps"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/composer/ir"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPUPad widens rows of ragged widths with zeros to the next
// whole block (and a whole block to the next) on the CPU, then reads them by
// a matrix stored that wide whose padded columns are not zero, as the tower
// block reads a row narrower than its matrices: the product is the matrix's
// first columns' alone, held to the oracle. A pad whose tail repeats the
// row's last element is the violation.
func TestComposedCPUPad(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	dev := cpuDevice(hostArm())
	const E = 5
	for _, w := range []int{1, 7, 20, 31, 32, 33, 144} {
		p := (w + 31) / 32 * 32
		if p == w {
			p += 32
		}
		for _, rows := range []int{1, 5} {
			b := graph.New()
			x := b.In("x", w)
			b.Out(b.MatVec(b.Pad(x, p), jlm.RoleVAttnQ, E))
			gr := b.G
			in := RandRow(rng, rows*w, 1)
			wt := RandRow(rng, E*p, 1/math.Sqrt(float64(w)))
			want := make([]float32, rows*E)
			for r := range rows {
				for e := range E {
					s := 0.0
					for c := range w {
						s += float64(wt[e*p+c]) * float64(in[r*w+c])
					}
					want[r*E+e] = float32(s)
				}
			}
			roles := map[jlm.Role][]float32{jlm.RoleVAttnQ: wt}
			formats := map[jlm.Role]jlm.Type{jlm.RoleVAttnQ: jlm.TypeF32}
			step := Step{Rows: rows, Page: 64, Keys: 1, ActWindow: 32}
			run := func(r *Registry) ([]float32, error) {
				_, codes, err := r.Form(Request{Graph: gr, Formats: formats, Step: step, Device: dev})
				if err != nil {
					return nil, err
				}
				return gridRowsRun(hostArm(), codes, gr, step.Rows, in, roles, [3][]uint32{})
			}
			got, err := run(Default().WithoutInt8(DeviceCPU))
			if err != nil {
				t.Fatalf("%d wide to %d at %d rows: %v", w, p, rows, err)
			}
			e := rowsNMSE(got, want)
			if math.IsNaN(e) || !(e < 1e-10) {
				t.Fatalf("%d wide to %d at %d rows: NMSE %.3e against the oracle", w, p, rows, e)
			}
			bad, err := run(padRepeats(Default().WithoutInt8(DeviceCPU)))
			if err != nil {
				t.Fatalf("violation, %d wide to %d: %v", w, p, err)
			}
			if be := rowsNMSE(bad, want); !(be > 1e-6) && !math.IsNaN(be) {
				t.Fatalf("VIOLATION: %d wide to %d with its tail the row's last element reads the same (NMSE %.3e)", w, p, be)
			}
		}
	}
}

// padRepeats is r with a pad whose tail repeats the row's last element.
func padRepeats(r *Registry) *Registry {
	v := *r
	v.cache = newFormCache()
	v.Decompositions = maps.Clone(r.Decompositions)
	v.Decompositions[graph.OpPad] = func(o graph.Op, c *Context) ([]Prim, error) {
		in, sh := c.rows(o.In[0]), c.rows(o.Out[0])
		var x exprs
		r, e := x.coord(0), x.coord(1)
		x.at(0, in, r, x.sel(ir.U32, x.op(OpLess, ir.Pred, e, x.u(uint32(in[1]))), e, x.u(uint32(in[1]-1))))
		return []Prim{{Kind: PrimMap, Shape: sh, In: []Operand{c.value(o.In[0])}, Out: c.value(o.Out[0]), Expr: x.e}}, nil
	}
	return &v
}
