package backend_test

import (
	"bytes"
	"encoding/binary"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestComposedMMAMatchesMain holds the Q8_0 contraction the composer forms
// from nodes (Row.MatVec) under the warp-tile schedule -- its integer dot
// on the int8 matrix instruction, the fragments the dot's operands at each
// lane's coordinates -- to main's MatVecMMA, bit for bit, on every device
// whose target has the instruction: tiles of 1 to 4 instructions each way,
// a bias, two and eight sub-blocks.
func TestComposedMMAMatchesMain(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(11))
	ran := 0
	for _, d := range devs {
		defer d.Close()
		tg := compose.Target{API: compose.APIOf(d.API()), ISA: backend.ArchOf(d)}
		if !tg.IntMMA() {
			continue
		}
		for _, c := range []struct{ rows, k, toks, mt, nt int }{
			{64, 64, 32, 4, 4}, {128, 256, 64, 4, 4}, {256, 256, 32, 2, 1}, {32, 64, 8, 1, 1}, {64, 256, 16, 1, 2}} {
			const blockB = 34
			nb := c.k / 32
			src := make([]byte, c.rows*nb*blockB)
			for r := 0; r < c.rows; r++ {
				for b := 0; b < nb; b++ {
					blk := src[(r*nb+b)*blockB:]
					binary.LittleEndian.PutUint16(blk, f16bits(float32(0.01+0.02*rng.Float64())))
					for l := 0; l < 32; l++ {
						blk[2+l] = byte(int8(rng.Intn(255) - 127))
					}
				}
			}
			qs, dw, _, err := kernels.PackWeights(kernels.Q8_0, src, c.rows, c.k)
			if err != nil {
				t.Fatal(err)
			}
			codes := make([]uint32, c.toks*c.k/4)
			for i := range codes {
				codes[i] = rng.Uint32() &^ 0x80808080 // no byte is -128
			}
			ax := make([]float32, c.toks*nb+c.toks*c.k/16)
			for i := range ax[:c.toks*nb] {
				ax[i] = float32(0.003 + 0.01*rng.Float64())
			}
			bias := make([]float32, c.rows)
			for r := range bias {
				bias[r] = float32(rng.Float64() - 0.5)
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
			bq, bd, ba, bx, bb, bsc := up(wordsLE(qs)), up(wordsLE(dw)), up(wordsLE(codes)), up(f32le(ax)), up(f32le(bias)), up(make([]byte, 4))
			run := func(k *ir.Kernel, threads int, args ...backend.Buf) []byte {
				out := up(make([]byte, 4*c.rows*c.toks))
				defer out.Free()
				kern, err := d.Compile(k)
				if err != nil {
					t.Fatalf("%s: %v", d.API(), err)
				}
				defer kern.Close()
				if err := kern.Launch((threads+127)/128, 128, append(args, out)...); err != nil {
					t.Fatal(err)
				}
				got := make([]byte, 4*c.rows*c.toks)
				if err := out.Read(got); err != nil {
					t.Fatal(err)
				}
				return got
			}
			mk, err := mainkernels.MatVecMMA(mainkernels.MatVecShape{T: mainkernels.Q8_0, K: c.k, Rows: c.rows, NTok: c.toks,
				MT: c.mt, NT: c.nt, Bias: true})
			if err != nil {
				t.Fatal(err)
			}
			threads := c.rows / (16 * c.mt) * (c.toks / (8 * c.nt)) * 32
			// main's parameters: pQS, pD, pSC, pA, pAX, pOut, pBias.
			mainOut := func() []byte {
				out := up(make([]byte, 4*c.rows*c.toks))
				defer out.Free()
				kern, err := d.Compile(mk)
				if err != nil {
					t.Fatal(err)
				}
				defer kern.Close()
				if err := kern.Launch((threads+127)/128, 128, bq, bd, bsc, ba, bx, out, bb); err != nil {
					t.Fatal(err)
				}
				got := make([]byte, 4*c.rows*c.toks)
				if err := out.Read(got); err != nil {
					t.Fatal(err)
				}
				return got
			}()
			g := &compose.Row{Width: c.rows, Rows: c.toks}
			gq, gd, ga, gx, gb, gout := g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufRow),
				g.Buf(compose.BufVec), g.Buf(compose.BufOut)
			v, err := g.MatVec(compose.MatVecWeight{T: kernels.Q8_0, K: c.k, Rows: c.rows, QS: gq, D: gd, SC: -1},
				compose.MatVecOperand{A: ga, AX: gx, Toks: c.toks})
			if err != nil {
				t.Fatal(err)
			}
			g.Store(gout, g.Bin(compose.OpAdd, v, g.Load(gb)))
			ck, err := g.Emit(compose.Schedule{Group: 128, Flat: true, MMA: true, MT: c.mt, NT: c.nt, Target: tg})
			if err != nil {
				t.Fatal(err)
			}
			mmas := 0
			for _, o := range ck.Ops {
				if o.Kind == ir.OpMMA {
					mmas++
				}
			}
			if mmas != c.mt*c.nt {
				t.Fatalf("%+v: %d matrix instructions, want %d", c, mmas, c.mt*c.nt)
			}
			got := run(ck, threads, bq, bd, ba, bx, bb)
			if !bytes.Equal(got, mainOut) {
				n := 0
				for i := 0; i < len(got); i += 4 {
					if !bytes.Equal(got[i:i+4], mainOut[i:i+4]) {
						n++
					}
				}
				t.Errorf("%s %+v: %d of %d outputs part from main's", d.API(), c, n, c.rows*c.toks)
			}
			ran++
			for _, b := range []backend.Buf{bq, bd, ba, bx, bb, bsc} {
				b.Free()
			}
		}
	}
	if ran == 0 {
		t.Skip("no device with the int8 matrix instruction")
	}
}
