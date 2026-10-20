package tier

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/graph"
	"github.com/jitllm/jitllm/jit/gpu/backend"
)

// primCase is one op over values of the given widths, run at rows rows: its
// inputs, its bound vector (or nil), and its definition in Go per row.
type primCase struct {
	name string
	op   devOp
	ins  []int     // input widths
	idx  []bool    // which inputs are index rows
	outs []int     // output widths
	vec  []float32 // the op's bound vector, or nil
	ref  func(row int, in [][]float32) [][]float32
}

// TestPrimitivesMatchTheirDefinitions runs each generic primitive's generated
// kernel on every device at three rows -- broadcasts, a bound vector, a
// constant, the fused multiply-add, tile, concat with a repeated input, a
// head-wise slice, a weightless head norm and a sigmoid gate times a row --
// and holds it to the primitive's definition (graph.OpMap and its kin) to a
// few ulps. A row's stride is its own value's width: a wrong stride parts.
func TestPrimitivesMatchTheirDefinitions(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows = 3
	fill := func(n int, s float64) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32(math.Sin(float64(i)*s+0.3) * 1.7)
		}
		return f
	}
	m := func(fn graph.MapFn, c float64) devOp {
		return devOp{Kind: dkMap, A: graph.Attrs{Fn: fn, Const: c}}
	}
	cases := []primCase{
		{name: "add, the second broadcast a group of 4", op: m(graph.MapAdd, 0), ins: []int{12, 3}, outs: []int{12},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 12)
				for i := range o {
					o[i] = in[0][i] + in[1][i/4]
				}
				return [][]float32{o}
			}},
		{name: "mul by a bound vector", op: func() devOp { o := m(graph.MapMul, 0); o.Role = 1; return o }(), ins: []int{10}, outs: []int{10},
			vec: fill(10, 0.7),
			ref: func(_ int, in [][]float32) [][]float32 {
				v := fill(10, 0.7)
				o := make([]float32, 10)
				for i := range o {
					o[i] = in[0][i] * v[i]
				}
				return [][]float32{o}
			}},
		{name: "mul by a constant", op: m(graph.MapMul, 0.25), ins: []int{7}, outs: []int{7},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 7)
				for i := range o {
					o[i] = in[0][i] * 0.25
				}
				return [][]float32{o}
			}},
		{name: "fma, the second broadcast a group of 5", op: m(graph.MapFma, 0), ins: []int{15, 3, 15}, outs: []int{15},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 15)
				for i := range o {
					o[i] = float32(math.FMA(float64(in[0][i]), float64(in[1][i/5]), float64(in[2][i])))
				}
				return [][]float32{o}
			}},
		{name: "tile a row of 4 to 12", op: devOp{Kind: dkTile, A: graph.Attrs{Span: 4}}, ins: []int{4}, outs: []int{12},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 12)
				for i := range o {
					o[i] = in[0][i%4]
				}
				return [][]float32{o}
			}},
		{name: "concat with a repeated input", op: devOp{Kind: dkConcat}, ins: []int{3, 5, 3}, outs: []int{11},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := append(append(append([]float32(nil), in[0]...), in[1]...), in[2]...)
				return [][]float32{o}
			}},
		{name: "slice each head's 2 from offset 1 of 4", op: devOp{Kind: dkSlice, A: graph.Attrs{HeadDim: 4, Offset: 1, Width: 2}}, ins: []int{12}, outs: []int{6},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 6)
				for h := 0; h < 3; h++ {
					o[2*h], o[2*h+1] = in[0][4*h+1], in[0][4*h+2]
				}
				return [][]float32{o}
			}},
		{name: "weightless head norm, heads of 8", op: devOp{Kind: dkHeadNorm, HD: 8, A: graph.Attrs{Eps: 1e-6}}, ins: []int{24}, outs: []int{24},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 24)
				for h := 0; h < 3; h++ {
					var ss float32
					for _, x := range in[0][8*h : 8*h+8] {
						ss += x * x
					}
					r := 1 / float32(math.Sqrt(float64(ss/8+1e-6)))
					for i := 0; i < 8; i++ {
						o[8*h+i] = in[0][8*h+i] * r
					}
				}
				return [][]float32{o}
			}},
		{name: "sigmoid gate times a row", op: devOp{Kind: dkActMul, Act: graph.ActSigmoid}, ins: []int{9, 9}, outs: []int{9},
			ref: func(_ int, in [][]float32) [][]float32 {
				o := make([]float32, 9)
				for i := range o {
					o[i] = float32(1/(1+math.Exp(-float64(in[0][i])))) * in[1][i]
				}
				return [][]float32{o}
			}},
	}
	// The concat case reads its first input twice.
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := &devTier{devShared: &devShared{dev: d}, devSess: &devSess{}, lane: &lane{}, Stats: &Stats{}}
			bs := &blockScratch{}
			for _, c := range cases {
				prog := &devProgram{}
				o := c.op
				var inVals []devVal
				for j, w := range c.ins {
					e := elemF32
					if j < len(c.idx) && c.idx[j] {
						e = elemU32
					}
					if c.name == "concat with a repeated input" && j == 2 {
						inVals = append(inVals, inVals[0])
						continue
					}
					inVals = append(inVals, prog.val("in", devType{Elem: e, Layout: layoutRow, Width: w}, slotAuto))
				}
				o.In = inVals
				for _, w := range c.outs {
					o.Out = append(o.Out, prog.val("out", devType{Elem: elemF32, Layout: layoutRow, Width: w}, slotAuto))
				}
				var vecBuf backend.Buf
				if c.vec != nil {
					b, err := d.Alloc(4 * len(c.vec))
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					if err := b.Write(f32le(c.vec)); err != nil {
						t.Fatal(err)
					}
					o.V, vecBuf = b, b
				}
				ls := g.genOp(&o, prog, bs, rows)
				if len(ls) == 0 || ls[0].k == nil {
					t.Errorf("%s: no generated kernel (%v; %s)", c.name, o.Why, g.LastErr)
					continue
				}
				// Inputs, row by row.
				inData := make([][]float32, len(c.ins))
				bufs := map[devVal]backend.Buf{}
				for j, w := range c.ins {
					if c.name == "concat with a repeated input" && j == 2 {
						inData[j] = inData[0]
						continue
					}
					inData[j] = fill(rows*w, 0.37+0.11*float64(j))
					b, err := d.Alloc(4 * len(inData[j]))
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					if err := b.Write(f32le(inData[j])); err != nil {
						t.Fatal(err)
					}
					bufs[inVals[j]] = b
				}
				for k, w := range c.outs {
					b, err := d.Alloc(4 * rows * w)
					if err != nil {
						t.Fatal(err)
					}
					defer b.Free()
					bufs[o.Out[k]] = b
				}
				for _, l := range ls {
					args := make([]backend.Buf, l.n)
					for j := 0; j < int(l.n); j++ {
						switch a := l.args[j]; {
						case a >= gaIn0 && a <= gaIn3:
							args[j] = bufs[o.In[a-gaIn0]]
						case a >= gaOut0 && a <= gaOut2:
							args[j] = bufs[o.Out[a-gaOut0]]
						case a == gaV:
							args[j] = vecBuf
						default:
							t.Fatalf("%s: argument kind %d", c.name, a)
						}
					}
					if err := l.k.Launch(l.groups, l.width, args...); err != nil {
						t.Fatalf("%s: %v", c.name, err)
					}
				}
				for k, w := range c.outs {
					raw := make([]byte, 4*rows*w)
					if err := bufs[o.Out[k]].Read(raw); err != nil {
						t.Fatal(err)
					}
					for r := 0; r < rows; r++ {
						row := make([][]float32, len(c.ins))
						for j, iw := range c.ins {
							row[j] = inData[j][r*iw : (r+1)*iw]
						}
						want := c.ref(r, row)[k]
						for i := 0; i < w; i++ {
							got := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*(r*w+i):]))
							if d := math.Abs(float64(got - want[i])); d > 4e-6*math.Max(1, math.Abs(float64(want[i]))) {
								t.Errorf("%s: row %d element %d is %g, the definition's %g", c.name, r, i, got, want[i])
								break
							}
						}
					}
				}
			}
		})
	}
}

// f32le is v as little-endian bytes.
func f32le(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return b
}
