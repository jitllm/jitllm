package backend_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// Stream ops as composed row groups, each held to the menu kernel it replaces
// bit for bit on every device that compiles it (the tier's generators,
// jit/gpu/tier/gen_streams_a.go, build these same groups from an op's values
// and attributes), each with a violation that must part from the menu.

// streamsFill is n deterministic floats of varied magnitude.
func streamsFill(n int, s float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*s) * (1 + float64(i%5)))
	}
	return v
}

// streamsRun compiles ker on d, launches it over groups groups of its own
// width on the float inputs, and reads outN floats back.
func streamsRun(t *testing.T, d backend.Device, g *gpu, ker *ir.Kernel, groups, outN int, ins ...[]float32) []byte {
	t.Helper()
	c, err := d.Compile(ker)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	var bufs []backend.Buf
	for _, v := range ins {
		bufs = append(bufs, g.up(f32bytes(v)))
	}
	out := g.up(make([]byte, outN*4))
	if err := c.Launch(groups, ker.Group[0], append(bufs, out)...); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, outN*4)
	if err := out.Read(raw); err != nil {
		t.Fatal(err)
	}
	return raw
}

// streamsEmit forms row group r at the group size the tier's schedule gives
// it: the narrowest power of two over its width, at least 32, at most max.
func streamsEmit(t *testing.T, r *compose.Row, max int) *ir.Kernel {
	t.Helper()
	g := 32
	for g < r.Width && g < max {
		g *= 2
	}
	k, err := r.Emit(compose.Schedule{Group: g})
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// streamsFlat is a flat menu kernel's launch over n threads.
func streamsFlat(n int) int { return (n + 127) / 128 }

func fma(r *compose.Row, a, b, c int) int {
	return r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: a, B: b, C: c})
}

// TestComposedAltUpMatchesTheMenu holds AltUp's router (a serial dot product
// per element, then tanh), its prediction, its correction, the scaled active
// row and the finish to AltUpRoute, AltUpPredict, AltUpCorrect, MulRows and
// AltUpFinish.
func TestComposedAltUpMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const s, d, rows = 4, 96, 3
	plane := rows * d
	h, w := streamsFill(rows*d, 0.17), streamsFill(s*d, 0.07)
	x, m := streamsFill(s*plane, 0.11), streamsFill(rows*s, 0.9)
	pt, ct := streamsFill(s*s*s, 0.3), streamsFill(s*s, 0.7)
	o, sc, p := streamsFill(rows*d, 0.23), streamsFill(d, 0.41), streamsFill(rows*d, 0.19)
	for i := range m {
		m[i] /= 5 // the router's tanh range
	}
	route := func(act bool) *ir.Kernel {
		r := &compose.Row{Width: s, Rows: rows}
		hb, wb := r.Buf(compose.BufRow), r.Buf(compose.BufVec)
		c := r.Loop(d)
		ctr := r.Iter(c)
		hat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), r.UConst(d)), ctr)
		wat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.Column(), r.UConst(d)), ctr)
		dot := r.Fold(c, r.Const(0))
		r.SetFold(dot, r.Add(compose.Node{Kind: compose.NBinary, Op: compose.OpFma, A: r.LoadAt(hb, hat), B: r.LoadAt(wb, wat), C: dot}))
		xx := r.Bin(compose.OpMul, dot, r.Const(1/float32(d)))
		if act {
			// The violation: a sigmoid where the tanh is.
			r.Store(r.Buf(compose.BufOut), r.Act(kernels.ActSigmoid, xx))
			return streamsEmit(t, r, 256)
		}
		e := r.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpExp, A: r.Bin(compose.OpMul, r.Const(2), xx)})
		one := r.Const(1)
		r.Store(r.Buf(compose.BufOut), r.Bin(compose.OpSub, one, r.Bin(compose.OpDiv, r.Const(2), r.Bin(compose.OpAdd, e, one))))
		return streamsEmit(t, r, 256)
	}
	stream := func(width int) (*compose.Row, int, int) {
		r := &compose.Row{Width: width, Rows: s * rows}
		return r, r.UBin(compose.OpDiv, r.RowIdx(), r.UConst(rows)), r.UBin(compose.OpRem, r.RowIdx(), r.UConst(rows))
	}
	predict := func(transposed bool) *ir.Kernel {
		r, k, row := stream(d)
		xb, mb, tb := r.Buf(compose.BufRow), r.Buf(compose.BufRow), r.Buf(compose.BufVec)
		mrow, kb := r.UBin(compose.OpMul, row, r.UConst(s)), r.UBin(compose.OpMul, k, r.UConst(s))
		acc := r.Load(xb)
		for j := 0; j < s; j++ {
			c := r.Const(0)
			for i := 0; i < s; i++ {
				at := r.UBin(compose.OpAdd, kb, r.UConst(i*s*s+j))
				if transposed {
					at = r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(1)), r.UConst(i*s*s+j*s))
				}
				c = fma(r, r.LoadAt(mb, r.UBin(compose.OpAdd, mrow, r.UConst(i))), r.LoadAt(tb, at), c)
			}
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpAdd, r.UConst(j*rows), row), r.UConst(d)), r.Column())
			acc = fma(r, c, r.LoadAt(xb, at), acc)
		}
		r.Store(r.Buf(compose.BufOut), acc)
		return streamsEmit(t, r, 256)
	}
	correct := func(rowStride bool) *ir.Kernel {
		r, k, row := stream(d)
		pb, ob, mb, tb := r.Buf(compose.BufRow), r.Buf(compose.BufRow), r.Buf(compose.BufRow), r.Buf(compose.BufVec)
		c := r.Const(0)
		for i := 0; i < s; i++ {
			at := r.UBin(compose.OpAdd, k, r.UConst(i*s))
			if rowStride {
				// The violation: T read by stream, not by coefficient.
				at = r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(s)), r.UConst(i))
			}
			mi := r.LoadAt(mb, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(s)), r.UConst(i)))
			c = fma(r, mi, r.LoadAt(tb, at), c)
		}
		c = r.Bin(compose.OpAdd, c, r.Const(1))
		rem := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(d)), r.Column())
		r.Store(r.Buf(compose.BufOut), fma(r, c, r.Bin(compose.OpSub, r.LoadAt(ob, rem), r.LoadAt(pb, rem)), r.Load(pb)))
		return streamsEmit(t, r, 256)
	}
	scale := func(add bool) *ir.Kernel {
		r := &compose.Row{Width: d, Rows: rows}
		op := compose.OpMul
		if add {
			op = compose.OpAdd
		}
		x, w := r.Load(r.Buf(compose.BufRow)), r.Load(r.Buf(compose.BufVec))
		r.Store(r.Buf(compose.BufOut), r.Bin(op, x, w))
		return streamsEmit(t, r, 256)
	}
	finish := func(first bool) *ir.Kernel {
		r, _, row := stream(d)
		src, pb := r.Buf(compose.BufRow), r.Buf(compose.BufRow)
		pat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(d)), r.Column())
		rest := r.Less(r.UConst(rows-1), r.RowIdx())
		if first {
			// The violation: the row added to the first stream instead.
			rest = r.Less(r.RowIdx(), r.UConst(rows))
		}
		r.Store(r.Buf(compose.BufOut), r.Bin(compose.OpAdd, r.Load(src), r.Select(rest, r.LoadAt(pb, pat), r.Const(0))))
		return streamsEmit(t, r, 256)
	}
	menu := func(k *ir.Kernel, err error) *ir.Kernel {
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	for _, dev := range devs {
		defer dev.Close()
		t.Run(dev.API(), func(t *testing.T) {
			g := newGPU(t, dev)
			defer g.free()
			check := func(name string, want []byte, good, bad *ir.Kernel, groups, outN int, ins ...[]float32) {
				t.Helper()
				if got := streamsRun(t, dev, g, good, groups, outN, ins...); !bytes.Equal(got, want) {
					t.Errorf("%s: the composed kernel parts from the menu's", name)
				}
				if got := streamsRun(t, dev, g, bad, groups, outN, ins...); bytes.Equal(got, want) {
					t.Errorf("%s: a deliberately wrong composed kernel matched the menu's", name)
				}
			}
			check("route", streamsRun(t, dev, g, menu(kernels.AltUpRoute(rows, d, s)), streamsFlat(rows*s), rows*s, h, w),
				route(false), route(true), rows, rows*s, h, w)
			check("predict", streamsRun(t, dev, g, menu(kernels.AltUpPredict(rows, d, s)), streamsFlat(s*plane), s*plane, x, m, pt),
				predict(false), predict(true), s*rows, s*plane, x, m, pt)
			check("correct", streamsRun(t, dev, g, menu(kernels.AltUpCorrect(rows, d, s)), streamsFlat(s*plane), s*plane, x, o, m, ct),
				correct(false), correct(true), s*rows, s*plane, x, o, m, ct)
			check("scale", streamsRun(t, dev, g, menu(kernels.MulRows(rows, d)), streamsFlat(rows*d), rows*d, o, sc),
				scale(false), scale(true), rows, rows*d, o, sc)
			check("finish", streamsRun(t, dev, g, menu(kernels.AltUpFinish(rows, d, s)), streamsFlat(s*plane), s*plane, x, p),
				finish(false), finish(true), s*rows, s*plane, x, p)
		})
	}
}

// TestComposedHyperConnectionMatchesTheMenu holds a hyper-connection's grouped
// stream norm (the residual's stream-major rows and a projection's row-major
// ones, at a width inside one group and one past it), its gated mean, its
// placement weights (gated, plain and halved) and the placement itself to
// Q4EGroupNorm, Q4EGateMix, Q4EInject and Q4EPlace.
func TestComposedHyperConnectionMatchesTheMenu(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const h, rows = 4, 3
	const eps = float32(1e-6)
	norm := func(t *testing.T, d int, byRow, byColumn bool) *ir.Kernel {
		r := &compose.Row{Width: d, Rows: rows * h}
		in, w := r.Buf(compose.BufRow), r.Buf(compose.BufVec)
		k := r.UBin(compose.OpRem, r.RowIdx(), r.UConst(h))
		var x int
		if byRow {
			x = r.Load(in)
		} else {
			rr := r.UBin(compose.OpDiv, r.RowIdx(), r.UConst(h))
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(rows)), rr), r.UConst(d)), r.Column())
			x = r.LoadAt(in, at)
		}
		wat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(d)), r.Column())
		if byColumn {
			wat = r.Column() // the violation: one row of the weight for every stream
		}
		r.Store(r.Buf(compose.BufOut), r.Bin(compose.OpMul, r.Bin(compose.OpMul, x, r.RMSScale(x, d, eps)), r.LoadAt(w, wat)))
		return streamsEmit(t, r, kernels.RMSNormGroup)
	}
	const dm = 96
	mix := func(plain bool) *ir.Kernel {
		r := &compose.Row{Width: dm, Rows: rows}
		xn, gate := r.Buf(compose.BufRow), r.Buf(compose.BufRow)
		acc := r.Const(0)
		for k := 0; k < h; k++ {
			at := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, r.RowIdx(), r.UConst(h*dm)), r.UBin(compose.OpAdd, r.Column(), r.UConst(k*dm)))
			x := r.LoadAt(xn, at)
			if !plain {
				x = r.Bin(compose.OpMul, r.Act(kernels.ActSigmoid, r.LoadAt(gate, at)), x)
			}
			acc = r.Bin(compose.OpAdd, acc, x)
		}
		r.Store(r.Buf(compose.BufOut), r.Bin(compose.OpMul, acc, r.Const(1/float32(h))))
		return streamsEmit(t, r, 256)
	}
	inject := func(scale float32, logits bool) *ir.Kernel {
		r := &compose.Row{Width: h, Rows: rows}
		w := r.Const(scale)
		if logits {
			x := r.Bin(compose.OpMul, r.Load(r.Buf(compose.BufRow)), r.Const(1/float32(h)))
			w = r.Act(kernels.ActSigmoid, x)
			if scale != 1 {
				w = r.Bin(compose.OpMul, r.Const(scale), w)
			}
		}
		r.Store(r.Buf(compose.BufOut), w)
		return streamsEmit(t, r, 256)
	}
	place := func(byStream bool) *ir.Kernel {
		r := &compose.Row{Width: dm, Rows: h * rows}
		x, y, w := r.Buf(compose.BufRow), r.Buf(compose.BufRow), r.Buf(compose.BufRow)
		k, row := r.UBin(compose.OpDiv, r.RowIdx(), r.UConst(rows)), r.UBin(compose.OpRem, r.RowIdx(), r.UConst(rows))
		wat := r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(h)), k)
		if byStream {
			// The violation: the weights indexed stream-major.
			wat = r.UBin(compose.OpAdd, r.UBin(compose.OpMul, k, r.UConst(rows)), row)
		}
		yt := r.LoadAt(y, r.UBin(compose.OpAdd, r.UBin(compose.OpMul, row, r.UConst(dm)), r.Column()))
		r.Store(r.Buf(compose.BufOut), fma(r, r.LoadAt(w, wat), yt, r.Load(x)))
		return streamsEmit(t, r, 256)
	}
	menu := func(k *ir.Kernel, err error) *ir.Kernel {
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	for _, dev := range devs {
		defer dev.Close()
		t.Run(dev.API(), func(t *testing.T) {
			g := newGPU(t, dev)
			defer g.free()
			same := func(name string, want []byte, k *ir.Kernel, groups, outN int, ins ...[]float32) {
				t.Helper()
				if got := streamsRun(t, dev, g, k, groups, outN, ins...); !bytes.Equal(got, want) {
					t.Errorf("%s: the composed kernel parts from the menu's", name)
				}
			}
			differs := func(name string, want []byte, k *ir.Kernel, groups, outN int, ins ...[]float32) {
				t.Helper()
				if got := streamsRun(t, dev, g, k, groups, outN, ins...); bytes.Equal(got, want) {
					t.Errorf("%s: a deliberately wrong composed kernel matched the menu's", name)
				}
			}
			for _, d := range []int{dm, 1100} {
				x, wt := streamsFill(rows*h*d, 0.013), streamsFill(h*d, 0.021)
				for _, sm := range []bool{true, false} {
					want := streamsRun(t, dev, g, menu(kernels.Q4EGroupNorm(d, h, rows, eps, sm)), rows*h, rows*h*d, x, wt)
					same("norm", want, norm(t, d, !sm, false), rows*h, rows*h*d, x, wt)
					differs("norm by column", want, norm(t, d, !sm, true), rows*h, rows*h*d, x, wt)
					differs("norm in the other layout", want, norm(t, d, sm, false), rows*h, rows*h*d, x, wt)
				}
			}
			xn, gate := streamsFill(rows*h*dm, 0.05), streamsFill(rows*h*dm, 0.09)
			gated := streamsRun(t, dev, g, menu(kernels.Q4EGateMix(dm, h, rows, false)), streamsFlat(rows*dm), rows*dm, xn, gate)
			plain := streamsRun(t, dev, g, menu(kernels.Q4EGateMix(dm, h, rows, true)), streamsFlat(rows*dm), rows*dm, xn, gate)
			same("gated mean", gated, mix(false), rows, rows*dm, xn, gate)
			same("plain mean", plain, mix(true), rows, rows*dm, xn, gate)
			differs("a gated mean against the plain mean", plain, mix(false), rows, rows*dm, xn, gate)
			inj := streamsFill(rows*h, 0.6)
			var weights [3][]byte
			for fault, c := range []struct {
				name  string
				scale float32
				mk    bool
			}{{"gated weights", 2, true}, {"plain weights", 1, false}, {"half weights", 1, true}} {
				weights[fault] = streamsRun(t, dev, g, menu(kernels.Q4EInject(rows*h, h, fault)), streamsFlat(rows*h), rows*h, inj)
				ins := [][]float32{inj}
				if !c.mk {
					ins = nil // no logits: the weight is the scale itself
				}
				same(c.name, weights[fault], inject(c.scale, c.mk), rows, rows*h, ins...)
			}
			for a := range weights {
				for b := range weights {
					if a != b && bytes.Equal(weights[a], weights[b]) {
						t.Errorf("placement weights %d and %d are the same bits: the gate cannot tell them apart", a, b)
					}
				}
			}
			xs, y, wp := streamsFill(h*rows*dm, 0.015), streamsFill(rows*dm, 0.033), streamsFill(rows*h, 0.4)
			want := streamsRun(t, dev, g, menu(kernels.Q4EPlace(dm, h, rows)), streamsFlat(h*rows*dm), h*rows*dm, xs, y, wp)
			same("place", want, place(false), h*rows, h*rows*dm, xs, y, wp)
			differs("place by stream", want, place(true), h*rows, h*rows*dm, xs, y, wp)
		})
	}
}
