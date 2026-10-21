package backend_test

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestComposedNormQuantizeMatchesMain holds a row group the composer forms
// from nodes -- an RMSNorm by a weight, and its int8 operand as a stage of
// the same workgroup (Quantize8Stage) -- to main's RMSNormQuantRows, bit for
// bit: the normed rows, the codes, the scales and the sums. Several rows, a
// window of 32 and of the whole row, the tree and the butterfly reduction.
func TestComposedNormQuantizeMatchesMain(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const eps = float32(1e-6)
	rng := rand.New(rand.NewSource(5))
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct {
			k, rows, window int
			warp            bool
		}{{256, 3, 32, false}, {1024, 2, 32, false}, {256, 2, 256, false}, {512, 3, 32, true}} {
			x := make([]float32, c.k*c.rows)
			for i := range x {
				x[i] = float32(rng.NormFloat64() * 3)
			}
			w := make([]float32, c.k)
			for i := range w {
				w[i] = float32(0.5 + rng.Float64())
			}
			up := func(b []byte) backend.Buf {
				buf, err := d.Alloc(max(len(b), 4))
				if err != nil {
					t.Fatal(err)
				}
				if err := buf.Write(b); err != nil {
					t.Fatal(err)
				}
				return buf
			}
			nAX := 4 * (c.rows*c.k/32 + c.rows*c.k/16)
			read := func(b backend.Buf, n int) []byte {
				out := make([]byte, n)
				if err := b.Read(out); err != nil {
					t.Fatal(err)
				}
				return out
			}
			in, wb := up(f32le(x)), up(f32le(w))
			run := func(k *compileArgs) (o, a, ax []byte) {
				ob, ab, axb := up(make([]byte, 4*c.k*c.rows)), up(make([]byte, c.k*c.rows)), up(make([]byte, nAX))
				defer ob.Free()
				defer ab.Free()
				defer axb.Free()
				kern, err := d.Compile(k.k)
				if err != nil {
					t.Fatal(err)
				}
				defer kern.Close()
				if err := kern.Launch(c.rows, k.group, k.args(in, wb, ob, ab, axb)...); err != nil {
					t.Fatal(err)
				}
				return read(ob, 4*c.k*c.rows), read(ab, c.k*c.rows), read(axb, nAX)
			}
			var mk *compileArgs
			if c.warp {
				k, err := mainkernels.RMSNormQuantRowsWarp(c.k, c.rows, eps, false, false, c.window)
				if err != nil {
					t.Fatal(err)
				}
				mk = &compileArgs{k: k, group: mainkernels.RMSNormGroup, args: mainNormArgs}
			} else {
				k, err := mainkernels.RMSNormQuantRows(c.k, c.rows, eps, false, false, c.window)
				if err != nil {
					t.Fatal(err)
				}
				mk = &compileArgs{k: k, group: mainkernels.RMSNormGroup, args: mainNormArgs}
			}
			mo, ma, max_ := run(mk)
			// The composer's: x, w, out, a, ax.
			g := &compose.Row{Width: c.k, Rows: c.rows}
			bx, bw, bo, ba, bax := g.Buf(compose.BufRow), g.Buf(compose.BufVec), g.Buf(compose.BufOut), g.Buf(compose.BufOutU32), g.Buf(compose.BufOut)
			xv := g.Load(bx)
			ss := g.Reduce(compose.RedSum, g.Bin(compose.OpMul, xv, xv))
			mean := g.Bin(compose.OpAdd, g.Bin(compose.OpMul, ss, g.Const(1/float32(c.k))), g.Const(eps))
			scale := g.Bin(compose.OpDiv, g.Const(1), g.Add(compose.Node{Kind: compose.NUnary, Op: compose.OpSqrt, A: mean}))
			y := g.Bin(compose.OpMul, g.Bin(compose.OpMul, xv, scale), g.Load(bw))
			g.Store(bo, y)
			g.Quantize8Stage(y, ba, bax, c.window)
			s := compose.Schedule{Group: mainkernels.RMSNormGroup, Warp: c.warp}
			ck, err := g.Emit(s)
			if err != nil {
				t.Fatal(err)
			}
			co, ca, cax := run(&compileArgs{k: ck, group: s.Group, args: func(in, wb, o, a, ax backend.Buf) []backend.Buf {
				return []backend.Buf{in, wb, o, a, ax}
			}})
			in.Free()
			wb.Free()
			for _, p := range []struct {
				what       string
				main, comp []byte
			}{{"normed rows", mo, co}, {"codes", ma, ca}, {"scales and sums", max_, cax}} {
				if !bytes.Equal(p.main, p.comp) {
					t.Errorf("%s %+v: the %s part from main's", d.API(), c, p.what)
				}
			}
		}
	}
}

// compileArgs is a kernel, its group and how its parameters are bound.
type compileArgs struct {
	k     *ir.Kernel
	group int
	args  func(in, w, out, a, ax backend.Buf) []backend.Buf
}

// mainNormArgs is main's RMSNormQuantRows binding: pX, pW, pOut, pA, pAX.
func mainNormArgs(in, w, out, a, ax backend.Buf) []backend.Buf {
	return []backend.Buf{in, w, out, a, ax}
}
