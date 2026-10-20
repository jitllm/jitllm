package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestComposedRotaryMatchesRoPERows holds the rotary op emitter to RoPERows
// bit for bit: NORM and NEOX pairs, a whole head turned and half of one,
// over rows each turned by its own row of the table.
func TestComposedRotaryMatchesRoPERows(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const heads, hd, rows = 4, 64, 3
	const width = heads * hd
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			for _, c := range []struct {
				nrot        int
				neox, split bool
			}{{64, false, false}, {64, true, false}, {32, false, false}, {32, true, false}, {64, true, true}, {32, true, true}} {
				src := make([]float32, rows*width)
				for i := range src {
					src[i] = float32(math.Sin(float64(i) * 0.13))
				}
				tabW := c.nrot
				if c.split {
					tabW = 2 * c.nrot // XD-RoPE: table A then table B a row
				}
				cs := make([]float32, rows*tabW)
				for i := 0; i < rows*tabW; i += 2 {
					th := float64(i) * 0.07
					cs[i], cs[i+1] = float32(math.Cos(th)), float32(math.Sin(th))
				}
				off := make([]byte, rows*4)
				for r := 0; r < rows; r++ {
					binary.LittleEndian.PutUint32(off[r*4:], uint32(r*width))
				}
				tk, err := mainkernels.RoPERows(heads, hd, c.nrot, c.neox, rows)
				if c.split {
					tk, err = mainkernels.RoPERowsSplit(heads, hd, c.nrot, rows, 0)
				}
				if err != nil {
					t.Fatal(err)
				}
				gr := &compose.Row{Width: width, Rows: rows}
				in, tab, out := gr.Buf(compose.BufRow), gr.Buf(compose.BufVec), gr.Buf(compose.BufOut)
				gr.Store(out, gr.Rotary(in, tab, hd, c.nrot, c.neox, c.split))
				ck, err := gr.Emit(compose.Schedule{Group: 256})
				if err != nil {
					t.Fatal(err)
				}
				tc, err := d.Compile(tk)
				if err != nil {
					t.Fatal(err)
				}
				cc, err := d.Compile(ck)
				if err != nil {
					t.Fatal(err)
				}
				// The template writes the turned dimensions over a copy of the
				// row; the composed group writes the whole row.
				want := g.up(f32bytes(src))
				pairs := heads * c.nrot / 2 * rows
				if err := tc.Launch((pairs+127)/128, 128, g.up(f32bytes(src)), g.up(f32bytes(cs)), g.up(off), want); err != nil {
					t.Fatal(err)
				}
				got := g.up(make([]byte, rows*width*4))
				if err := cc.Launch(rows, 256, g.up(f32bytes(src)), g.up(f32bytes(cs)), got); err != nil {
					t.Fatal(err)
				}
				w, gb := make([]byte, rows*width*4), make([]byte, rows*width*4)
				if err := want.Read(w); err != nil {
					t.Fatal(err)
				}
				if err := got.Read(gb); err != nil {
					t.Fatal(err)
				}
				// Bit for bit where the backend contracts as the template does
				// (PTX); within a few ulps where a driver fuses a multiply and
				// an add at its own whim (SPIR-V): kernels may differ per
				// device inside the gated margin.
				worst := 0.0
				for i := 0; i < rows*width; i++ {
					a := math.Float32frombits(binary.LittleEndian.Uint32(w[i*4:]))
					b := math.Float32frombits(binary.LittleEndian.Uint32(gb[i*4:]))
					// The inputs are at most 1 in magnitude: a contraction
					// moves a result by an ulp of 1, about 1.2e-7.
					worst = math.Max(worst, math.Abs(float64(a-b)))
				}
				t.Logf("nrot %d neox %v split %v: worst difference %.3g", c.nrot, c.neox, c.split, worst)
				if d.API() == "ptx" && !bytes.Equal(w, gb) || worst > 4.8e-7 {
					t.Errorf("nrot %d neox %v split %v: the composed rotary parts from the template (worst %.3g)",
						c.nrot, c.neox, c.split, worst)
				}
				tc.Close()
				cc.Close()
			}
		})
	}
}

// TestComposedGatesMatchTheTemplates holds the two sigmoid-gated row ops to
// their templates bit for bit: a gated output, sigmoid(gate) * row
// (SigmoidMul), and a gated add, acc + sigmoid(the row's logit) * branch
// (SharedExpertAdd). A unary sigmoid that computed another activation passed
// every other gate, which covered no op that reads one.
func TestComposedGatesMatchTheTemplates(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const k, rows = 256, 3
	fill := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.Sin(float64(i)*s) * (1 + float64(i%5)))
		}
		return v
	}
	acc, br, logit := fill(rows*k, 0.21), fill(rows*k, 0.13), fill(rows, 1.7)
	gateOut := func(act kernels.ActKind) *compose.Row {
		g := &compose.Row{Width: k, Rows: rows}
		gt, row := g.Load(g.Buf(compose.BufRow)), g.Load(g.Buf(compose.BufRow))
		g.Store(g.Buf(compose.BufOut), g.Bin(compose.OpMul, g.Act(act, gt), row))
		return g
	}
	gatedAdd := func(act kernels.ActKind) *compose.Row {
		g := &compose.Row{Width: k, Rows: rows}
		a, b := g.Load(g.Buf(compose.BufRow)), g.Load(g.Buf(compose.BufRow))
		lg := g.Load(g.Buf(compose.BufRowScalar))
		g.Store(g.Buf(compose.BufOut), g.Bin(compose.OpAdd, a, g.Bin(compose.OpMul, g.Act(act, lg), b)))
		return g
	}
	emit := func(r *compose.Row) *ir.Kernel {
		ker, err := r.Emit(compose.Schedule{Group: 128})
		if err != nil {
			t.Fatal(err)
		}
		return ker
	}
	sigMul, err := mainkernels.SigmoidMul(rows * k)
	if err != nil {
		t.Fatal(err)
	}
	shAdd, err := mainkernels.SharedExpertAdd(k, rows)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			run := func(ker *ir.Kernel, groups int, ins ...[]float32) []byte {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				var bufs []backend.Buf
				for _, v := range ins {
					bufs = append(bufs, g.up(f32bytes(v)))
				}
				out := g.up(make([]byte, rows*k*4))
				if err := c.Launch(groups, 128, append(bufs, out)...); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, rows*k*4)
				if err := out.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			want := run(sigMul, (rows*k+127)/128, br, acc)
			if got := run(emit(gateOut(kernels.ActSigmoid)), rows, br, acc); !bytes.Equal(got, want) {
				t.Errorf("sigmoid(gate) * row: the composed kernel parts from SigmoidMul")
			}
			if bad := run(emit(gateOut(kernels.ActSiLU)), rows, br, acc); bytes.Equal(bad, want) {
				t.Errorf("a SiLU gate matched SigmoidMul: the gate cannot see the activation")
			}
			// SharedExpertAdd's order: acc, branch, logit, out.
			want = run(shAdd, (rows*k+127)/128, acc, br, logit)
			if got := run(emit(gatedAdd(kernels.ActSigmoid)), rows, acc, br, logit); !bytes.Equal(got, want) {
				t.Errorf("acc + sigmoid(logit) * branch: the composed kernel parts from SharedExpertAdd")
			}
			if bad := run(emit(gatedAdd(kernels.ActSiLU)), rows, acc, br, logit); bytes.Equal(bad, want) {
				t.Errorf("a SiLU gate matched SharedExpertAdd: the gate cannot see the activation")
			}
		})
	}
}

// TestComposedSplitQGMatchesSplitHeadGate holds the composed query/gate split
// (two indexed loads and two stores a column) to SplitHeadGate bit for bit; a
// gate read from the wrong half is its violation.
func TestComposedSplitQGMatchesSplitHeadGate(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const heads, hd, rows = 4, 32, 3
	const w = heads * hd
	src := make([]float32, rows*2*w)
	for i := range src {
		src[i] = float32(math.Sin(float64(i) * 0.19))
	}
	build := func(gateOff int) *ir.Kernel {
		gr := &compose.Row{Width: w, Rows: rows}
		in, q, gt := gr.Buf(compose.BufRow), gr.Buf(compose.BufOut), gr.Buf(compose.BufOut)
		col := gr.Column()
		head, j := gr.UBin(compose.OpDiv, col, gr.UConst(hd)), gr.UBin(compose.OpRem, col, gr.UConst(hd))
		at := gr.UBin(compose.OpAdd, gr.UBin(compose.OpAdd, gr.UBin(compose.OpMul, gr.RowIdx(), gr.UConst(2*w)),
			gr.UBin(compose.OpMul, head, gr.UConst(2*hd))), j)
		gr.Store(q, gr.LoadAt(in, at))
		gr.Store(gt, gr.LoadAt(in, gr.UBin(compose.OpAdd, at, gr.UConst(gateOff))))
		k, err := gr.Emit(compose.Schedule{Group: 128})
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			run := func(ker *ir.Kernel, groups int) []byte {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				q, gt := g.up(make([]byte, rows*w*4)), g.up(make([]byte, rows*w*4))
				if err := c.Launch(groups, 128, g.up(f32bytes(src)), q, gt); err != nil {
					t.Fatal(err)
				}
				a, b := make([]byte, rows*w*4), make([]byte, rows*w*4)
				if err := q.Read(a); err != nil {
					t.Fatal(err)
				}
				if err := gt.Read(b); err != nil {
					t.Fatal(err)
				}
				return append(a, b...)
			}
			tk, err := mainkernels.SplitHeadGate(heads, hd, rows)
			if err != nil {
				t.Fatal(err)
			}
			want := run(tk, (rows*w+127)/128)
			if got := run(build(hd), rows); !bytes.Equal(got, want) {
				t.Errorf("the composed split parts from SplitHeadGate")
			}
			if got := run(build(0), rows); bytes.Equal(got, want) {
				t.Errorf("a gate read from the query's half matched")
			}
		})
	}
}

// TestComposedCombineMatchesExpertCombine holds the composed weighted sum of
// a step's selected experts (each slot's fused multiply-add in slot order
// from zero) to ExpertCombine bit for bit; the weights read in reverse order
// are its violation.
func TestComposedCombineMatchesExpertCombine(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const n, slots = 192, 6
	e := make([]float32, slots*n)
	for i := range e {
		e[i] = float32(math.Sin(float64(i)*0.31)) * 2
	}
	w := []float32{0.31, 0.22, 0.17, 0.12, 0.11, 0.07}
	build := func(rev bool) *ir.Kernel {
		gr := &compose.Row{Width: n, Rows: 1}
		eb, wb, out := gr.Buf(compose.BufRow), gr.Buf(compose.BufRow), gr.Buf(compose.BufOut)
		acc := gr.Const(0)
		for j := 0; j < slots; j++ {
			jw := j
			if rev {
				jw = slots - 1 - j
			}
			x := gr.LoadAt(eb, gr.UBin(compose.OpAdd, gr.UConst(j*n), gr.Column()))
			acc = gr.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: gr.LoadAt(wb, gr.UConst(jw)), B: x, C: acc})
		}
		gr.Store(out, acc)
		k, err := gr.Emit(compose.Schedule{Group: 128})
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			run := func(ker *ir.Kernel, groups int) []byte {
				c, err := d.Compile(ker)
				if err != nil {
					t.Fatal(err)
				}
				defer c.Close()
				out := g.up(make([]byte, n*4))
				if err := c.Launch(groups, 128, g.up(f32bytes(e)), g.up(f32bytes(w)), out); err != nil {
					t.Fatal(err)
				}
				raw := make([]byte, n*4)
				if err := out.Read(raw); err != nil {
					t.Fatal(err)
				}
				return raw
			}
			tk, err := mainkernels.ExpertCombine(n, slots)
			if err != nil {
				t.Fatal(err)
			}
			want := run(tk, (n+127)/128)
			if got := run(build(false), 1); !bytes.Equal(got, want) {
				t.Errorf("the composed combine parts from ExpertCombine")
			}
			if got := run(build(true), 1); bytes.Equal(got, want) {
				t.Errorf("weights in reverse order matched")
			}
		})
	}
}
