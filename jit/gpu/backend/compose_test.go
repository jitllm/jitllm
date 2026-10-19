package backend_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestComposedRowsMatchTheTemplates holds kernels the row former composes
// from op fragments to the templates they replace, bit for bit: an add, its
// RMSNorm and the int8 operand of the result (RMSNormQuantRows), and a gated
// activation (ActMul). A composed group that parts from its template in one
// bit fails.
func TestComposedRowsMatchTheTemplates(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const k, rows, win = 512, 3, 32
	const eps = float32(1e-6)
	fill := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(math.Sin(float64(i)*s) * (1 + float64(i%7)))
		}
		return v
	}
	x, y, w := fill(rows*k, 0.37), fill(rows*k, 0.11), fill(k, 0.05)
	// add -> RMSNorm -> quantize, as op fragments.
	norm := &compose.Row{Width: k, Rows: rows,
		Bufs: []compose.Buf{{Kind: compose.BufRow}, {Kind: compose.BufRow}, {Kind: compose.BufVec},
			{Kind: compose.BufOut}, {Kind: compose.BufOut}, {Kind: compose.BufQ8}, {Kind: compose.BufQ8X}},
		Nodes: []compose.Node{
			{Kind: compose.NLoad, Buf: 0}, {Kind: compose.NLoad, Buf: 1},
			{Kind: compose.NBinary, Op: compose.OpAdd, A: 0, B: 1},
			{Kind: compose.NBinary, Op: compose.OpMul, A: 2, B: 2},
			{Kind: compose.NReduce, Red: compose.RedSum, A: 3},
			{Kind: compose.NConst, F: 1 / float32(k)},
			{Kind: compose.NBinary, Op: compose.OpMul, A: 4, B: 5},
			{Kind: compose.NConst, F: eps},
			{Kind: compose.NBinary, Op: compose.OpAdd, A: 6, B: 7},
			{Kind: compose.NUnary, Op: compose.OpRsqrt, A: 8},
			{Kind: compose.NLoad, Buf: 2},
			{Kind: compose.NBinary, Op: compose.OpMul, A: 2, B: 9},
			{Kind: compose.NBinary, Op: compose.OpMul, A: 11, B: 10},
		},
		Stores: []compose.Store{{Buf: 3, Val: 2}, {Buf: 4, Val: 12}},
		Q8:     &compose.Q8{Val: 12, A: 5, X: 6, Window: win, Pad: k},
	}
	composed, err := norm.Emit(compose.Schedule{Group: kernels.RMSNormGroup})
	if err != nil {
		t.Fatal(err)
	}
	// The violation: the add a subtract. It must part from the template.
	bad := *norm
	bad.Nodes = append([]compose.Node(nil), norm.Nodes...)
	bad.Nodes[2].Op = compose.OpSub
	violated, err := bad.Emit(compose.Schedule{Group: kernels.RMSNormGroup})
	if err != nil {
		t.Fatal(err)
	}
	template, err := kernels.RMSNormQuantRows(k, rows, eps, false, true, win)
	if err != nil {
		t.Fatal(err)
	}
	// act(gate) * up, as fragments.
	const n = rows * k
	gated := &compose.Row{Width: n, Rows: 1,
		Bufs: []compose.Buf{{Kind: compose.BufRow}, {Kind: compose.BufRow}, {Kind: compose.BufOut}},
		Nodes: []compose.Node{{Kind: compose.NLoad, Buf: 0}, {Kind: compose.NLoad, Buf: 1},
			{Kind: compose.NBinary, Op: compose.OpActMul, Act: kernels.ActSiLU, A: 0, B: 1}},
		Stores: []compose.Store{{Buf: 2, Val: 2}},
	}
	composedAct, err := gated.Emit(compose.Schedule{Group: 128})
	if err != nil {
		t.Fatal(err)
	}
	templateAct, err := kernels.ActMul(n, kernels.ActSiLU)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			run := func(kern backend.Kernel, groups, width int, outs []int, ins ...[]float32) [][]byte {
				var bufs []backend.Buf
				for _, v := range ins {
					bufs = append(bufs, g.up(f32bytes(v)))
				}
				var got []backend.Buf
				for _, nb := range outs {
					b := g.up(make([]byte, nb))
					got = append(got, b)
					bufs = append(bufs, b)
				}
				if err := kern.Launch(groups, width, bufs...); err != nil {
					t.Fatal(err)
				}
				var res [][]byte
				for i, b := range got {
					raw := make([]byte, outs[i])
					if err := b.Read(raw); err != nil {
						t.Fatal(err)
					}
					res = append(res, raw)
				}
				return res
			}
			ck, err := d.Compile(composed)
			if err != nil {
				t.Fatalf("composed: %v", err)
			}
			defer ck.Close()
			tk, err := d.Compile(template)
			if err != nil {
				t.Fatal(err)
			}
			defer tk.Close()
			outs := []int{rows * k * 4, rows * k * 4, rows * k, rows * k / 32 * 3 * 4}
			// The template's order: x, y, w, sum, out, a, ax.
			want := run(tk, rows, kernels.RMSNormGroup, outs, x, y, w)
			got := run(ck, rows, kernels.RMSNormGroup, outs, x, y, w)
			for i, name := range []string{"sum", "normed", "int8", "scales"} {
				if !bytes.Equal(got[i], want[i]) {
					t.Errorf("add, RMSNorm, quantize: the composed %s parts from the template's", name)
				}
			}
			vk, err := d.Compile(violated)
			if err != nil {
				t.Fatal(err)
			}
			defer vk.Close()
			if v := run(vk, rows, kernels.RMSNormGroup, outs, x, y, w); bytes.Equal(v[1], want[1]) {
				t.Errorf("a subtract in the add's place matched the template: the gate cannot see a wrong op")
			}
			cka, err := d.Compile(composedAct)
			if err != nil {
				t.Fatalf("composed act: %v", err)
			}
			defer cka.Close()
			tka, err := d.Compile(templateAct)
			if err != nil {
				t.Fatal(err)
			}
			defer tka.Close()
			wantA := run(tka, (n+127)/128, 128, []int{n * 4}, x, y)
			gotA := run(cka, 1, 128, []int{n * 4}, x, y)
			if !bytes.Equal(gotA[0], wantA[0]) {
				t.Errorf("silu(gate) * up: the composed kernel parts from ActMul")
			}
		})
	}
}

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
				nrot int
				neox bool
			}{{64, false}, {64, true}, {32, false}, {32, true}} {
				src := make([]float32, rows*width)
				for i := range src {
					src[i] = float32(math.Sin(float64(i) * 0.13))
				}
				cs := make([]float32, rows*c.nrot)
				for i := 0; i < rows*c.nrot; i += 2 {
					th := float64(i) * 0.07
					cs[i], cs[i+1] = float32(math.Cos(th)), float32(math.Sin(th))
				}
				off := make([]byte, rows*4)
				for r := 0; r < rows; r++ {
					binary.LittleEndian.PutUint32(off[r*4:], uint32(r*width))
				}
				tk, err := kernels.RoPERows(heads, hd, c.nrot, c.neox, rows)
				if err != nil {
					t.Fatal(err)
				}
				gr := &compose.Row{Width: width, Rows: rows}
				in, tab, out := gr.Buf(compose.BufRow), gr.Buf(compose.BufVec), gr.Buf(compose.BufOut)
				gr.Store(out, gr.Rotary(in, tab, hd, c.nrot, c.neox))
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
				t.Logf("nrot %d neox %v: worst difference %.3g", c.nrot, c.neox, worst)
				if d.API() == "ptx" && !bytes.Equal(w, gb) || worst > 4.8e-7 {
					t.Errorf("nrot %d neox %v: the composed rotary parts from RoPERows (worst %.3g)", c.nrot, c.neox, worst)
				}
				tc.Close()
				cc.Close()
			}
		})
	}
}

// TestMatVecEpilogueMatchesTheMenu holds a matvec whose output passes through
// a composed epilogue (compose.Epi) to the menu form it replaces, bit for
// bit: a residual added (the menu's Bias slot carrying the residual) and
// silu(gate)*out (the menu's Gate), over Q8_0 and binary16 weights, unsplit and
// split in the group. A residual subtracted in the epilogue must part.
func TestMatVecEpilogueMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const rows, k = 96, 512
	rng := rand.New(rand.NewSource(7))
	acts := randActs(k, rng)
	resid := randActs(rows, rng)
	gate := randActs(rows, rng)
	epi := func(op compose.Op, gated bool) *compose.Epi {
		g := &compose.Row{Width: rows, Rows: 1}
		in := g.Add(compose.Node{Kind: compose.NInput})
		other := g.Load(g.Buf(compose.BufRow))
		if gated {
			return &compose.Epi{G: g, Out: g.Bin(compose.OpMul, g.Act(kernels.ActSiLU, other), in)}
		}
		return &compose.Epi{G: g, Out: g.Bin(op, in, other)}
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			g := newGPU(t, d)
			defer g.free()
			for _, q := range []kernels.Quant{kernels.Q8_0, kernels.Float16} {
				qs, dw, sc, err := kernels.PackWeights(q, randWeights(q, rows, k, rng), rows, k)
				if err != nil {
					t.Fatal(err)
				}
				bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(sc))
				bX := g.up(f32bytes(acts))
				// The int8 operand, quantized on the device as a step does.
				bA, bAX := g.up(make([]byte, k)), g.up(make([]byte, k/32*3*4))
				qk, err := kernels.Quantize(k, 32)
				if err != nil {
					t.Fatal(err)
				}
				qc, err := d.Compile(qk)
				if err != nil {
					t.Fatal(err)
				}
				if err := qc.Launch((kernels.QuantizeThreads(k/32)+127)/128, 128, bX, bA, bAX); err != nil {
					t.Fatal(err)
				}
				qc.Close()
				dArg := bD
				if kernels.IsFloat(q) {
					dArg = bX // a float matvec reads the float row in the scale slot
				}
				for _, grp := range []int{1, 4} {
					s := kernels.MatVecShape{T: q, K: k, Rows: rows, Split: grp, GroupSplit: grp > 1}
					width := 128
					if grp > 1 {
						width = kernels.GroupSplitWidth(grp)
					}
					launch := func(ker *ir.Kernel, extra ...backend.Buf) []byte {
						c, err := d.Compile(ker)
						if err != nil {
							t.Fatalf("%v split %d: compile: %v", q, grp, err)
						}
						defer c.Close()
						out := g.up(make([]byte, rows*4))
						args := append([]backend.Buf{bQS, dArg, bSC, bA, bAX, out}, extra...)
						if err := c.Launch((rows*grp+width-1)/width, width, args...); err != nil {
							t.Fatal(err)
						}
						raw := make([]byte, rows*4)
						if err := out.Read(raw); err != nil {
							t.Fatal(err)
						}
						return raw
					}
					build := func(s kernels.MatVecShape, e kernels.Epilogue) *ir.Kernel {
						ker, err := kernels.MatVecEpi(s, e)
						if err != nil {
							t.Fatalf("%v split %d: %v", q, grp, err)
						}
						return ker
					}
					sb, sg := s, s
					sb.Bias = true
					sg.Gate, sg.GateAct = true, kernels.ActSiLU
					menuRes := build(sb, nil)
					menuGate := build(sg, nil)
					if want, got := launch(menuRes, g.up(f32bytes(resid))), launch(build(s, epi(compose.OpAdd, false)), g.up(f32bytes(resid))); !bytes.Equal(got, want) {
						t.Errorf("%v split %d: the composed residual parts from the menu's", q, grp)
					}
					if want, got := launch(menuGate, g.up(f32bytes(gate))), launch(build(s, epi(0, true)), g.up(f32bytes(gate))); !bytes.Equal(got, want) {
						t.Errorf("%v split %d: the composed gate parts from the menu's", q, grp)
					}
					if bad := launch(build(s, epi(compose.OpSub, false)), g.up(f32bytes(resid))); bytes.Equal(bad, launch(menuRes, g.up(f32bytes(resid)))) {
						t.Errorf("%v split %d: a subtracted residual matched: the gate cannot see the epilogue", q, grp)
					}
				}
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
	sigMul, err := kernels.SigmoidMul(rows * k)
	if err != nil {
		t.Fatal(err)
	}
	shAdd, err := kernels.SharedExpertAdd(k, rows)
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
