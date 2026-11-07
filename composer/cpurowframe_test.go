package composer

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// TestComposedCPURowFrame frames a merged grid's rows on the CPU as
// HunyuanVL's merger does: a newline row after each of the grid's rows
// (graph.OpRowBreak), then a begin row before them and an end row after
// (graph.OpWrapRows) -- the grid's width the request's patch grid over the
// merge and its height the request's, each output row the input row or
// stored row its place names, and no row past the framed ones written. A
// grid stated a window wider is the row break's violation, a row of
// windows more the row wrap's.
func TestComposedCPURowFrame(t *testing.T) {
	r := Default().WithoutInt8(DeviceCPU)
	dev := Device{Kind: DeviceCPU, Slots: 1, Lanes: 8, ISA: uint64(cpu.Probe().ISA()), ExactDiv: true}
	f32 := map[jlm.Role]jlm.Type{jlm.RoleVImgNewline: jlm.TypeF32, jlm.RoleVImgBegin: jlm.TypeF32, jlm.RoleVImgEnd: jlm.TypeF32}
	for _, c := range []struct{ gw, gh, S, E int }{{6, 4, 2, 8}, {10, 6, 2, 5}, {9, 6, 3, 16}} {
		W, H := c.gw/c.S, c.gh/c.S
		M, L := W*H, H*(W+1)
		vec := graph.Shape{K: c.E}
		inv := map[jlm.Role]graph.Shape{jlm.RoleVImgNewline: vec, jlm.RoleVImgBegin: vec, jlm.RoleVImgEnd: vec}
		row := func(v float32) []float32 {
			out := make([]float32, c.E)
			for i := range out {
				out[i] = v - float32(i)
			}
			return out
		}
		nl, begin, end := row(-1), row(-1000), row(-2000)
		vecs := map[jlm.Role][]float32{jlm.RoleVImgNewline: nl, jlm.RoleVImgBegin: begin, jlm.RoleVImgEnd: end}
		// form composes op over an input of in rows into out rows; run runs
		// it at a request grid gw x gh.
		form := func(op graph.OpKind, ws []graph.Binding, a graph.Attrs, in, out int) func(x []float32, gw, gh int) []float32 {
			b := graph.NewBlock(-1)
			x := b.In("rows", c.E)
			b.Out(b.Emit(op, []graph.ValueID{x}, []graph.Result{{Name: "framed", Width: c.E, Rows: out}}, ws, a, nil, nil)[0])
			if err := graph.Verify(b.G, graph.Facts{Inventory: inv}); err != nil {
				t.Fatal(err)
			}
			_, codes, err := r.Form(Request{Graph: b.G, Formats: f32, Step: Step{Rows: in, Page: 64, Keys: 1, ActWindow: 32}, Device: dev, Apart: true})
			if err != nil {
				t.Fatalf("%+v %s: %v", c, op, err)
			}
			return func(x []float32, gw, gh int) []float32 {
				req := make([]uint32, RequestRows+out)
				req[RequestGrid], req[RequestGridH] = uint32(gw), uint32(gh)
				pos := make([]uint32, gw*gh+1)
				pos[0] = uint32(gw * gh)
				got, err := runCPUBound(hostArm(), codes, x, out*c.E, [3][]uint32{}, vecs, func(o Operand) []uint32 {
					switch o.Name {
					case "request":
						return req
					case "rope.pos":
						return pos
					}
					return nil
				})
				if err != nil {
					t.Fatalf("%+v %s: %v", c, op, err)
				}
				return got
			}
		}
		check := func(name string, got []float32, want func(r int) []float32, n int) {
			for r := range n {
				for i, v := range want(r) {
					if got[r*c.E+i] != v {
						t.Fatalf("%+v %s: row %d element %d is %v, want %v", c, name, r, i, got[r*c.E+i], v)
					}
				}
			}
		}
		same := func(a, b []float32, n int) bool {
			for i := range n {
				if a[i] != b[i] {
					return false
				}
			}
			return true
		}

		in := make([]float32, M*c.E)
		for i := range in {
			in[i] = float32(i) + 0.25
		}
		brk := form(graph.OpRowBreak, []graph.Binding{{Role: jlm.RoleVImgNewline, K: c.E}}, graph.Attrs{Span: c.S}, M, 2*M)
		lines := brk(in, c.gw, c.gh)
		check("row break", lines, func(r int) []float32 {
			if k := r % (W + 1); k < W {
				src := (r/(W+1))*W + k
				return in[src*c.E : (src+1)*c.E]
			}
			return nl
		}, L)
		if same(brk(in, c.gw+c.S, c.gh), lines, L*c.E) {
			t.Fatalf("%+v: a row break over a grid a window wider writes the same rows", c)
		}

		wrap := form(graph.OpWrapRows, []graph.Binding{{Role: jlm.RoleVImgBegin, K: c.E}, {Role: jlm.RoleVImgEnd, K: c.E}},
			graph.Attrs{Span: c.S, Width: 1}, 2*M, 2*M+2)
		framed := wrap(lines, c.gw, c.gh)
		check("row wrap", framed, func(r int) []float32 {
			switch r {
			case 0:
				return begin
			case L + 1:
				return end
			}
			return lines[(r-1)*c.E : r*c.E]
		}, L+2)
		if same(wrap(lines, c.gw, c.gh+c.S), framed, (L+2)*c.E) {
			t.Fatalf("%+v: a row wrap over a row of windows more writes the same rows", c)
		}
		// No row past the framed ones: a grid a row of windows shorter
		// leaves the last rows as they were.
		short := wrap(lines, c.gw, c.gh-c.S)
		for j := (L - (W + 1) + 2) * c.E; j < len(short); j++ {
			if !math.IsNaN(float64(short[j])) {
				t.Fatalf("%+v: a row wrap a row of windows shorter wrote element %d of %d (%v)", c, j, len(short), short[j])
			}
		}
		t.Logf("%dx%d patches by %d: %d rows framed, a newline after each of %d", c.gw, c.gh, c.S, L+2, H)
	}
}
