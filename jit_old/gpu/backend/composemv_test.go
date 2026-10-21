package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/compose"
	"github.com/jitllm/jitllm/jit_old/gpu/kernels"
)

// TestComposedMatVecMatchesItsDefinition holds the matvec the composer
// forms from primitive nodes (Row.MatVec under the flat schedule) to its
// definition, out[t][e] = sum over k of w[e][k] a[t][k], with w the GGUF
// blocks' d*q and a the int8 operand's scale times its codes: random Q8_0
// weights packed by the vocabulary (kernels.PackWeights), at one token and
// at several, an odd row count included (rows share a scale word in pairs).
// The sums are integers scaled once a block, so the device's f32 order
// parts from the reference only by rounding.
func TestComposedMatVecMatchesItsDefinition(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	rng := rand.New(rand.NewSource(7))
	for _, d := range devs {
		defer d.Close()
		for _, c := range []struct{ rows, k, toks, tile, cols, split int }{{64, 64, 1, 1, 1, 1}, {33, 256, 3, 1, 1, 1},
			{33, 256, 3, 3, 1, 1}, {128, 512, 5, 1, 1, 1}, {128, 512, 4, 4, 1, 1}, {128, 512, 4, 2, 2, 1},
			{64, 256, 1, 1, 4, 1}, {64, 256, 8, 4, 2, 1}, {64, 256, 1, 1, 1, 4}, {64, 512, 4, 2, 2, 2}} {
			const blockB = 34 // Q8_0: an f16 scale and 32 signed codes
			nb := c.k / 32
			src := make([]byte, c.rows*nb*blockB)
			wv := make([]float64, c.rows*c.k)
			for r := 0; r < c.rows; r++ {
				for b := 0; b < nb; b++ {
					blk := src[(r*nb+b)*blockB:]
					h := f16bits(float32(0.01 + 0.02*rng.Float64()))
					binary.LittleEndian.PutUint16(blk, h)
					for l := 0; l < 32; l++ {
						q := int8(rng.Intn(255) - 127)
						blk[2+l] = byte(q)
						wv[r*c.k+b*32+l] = f16value(h) * float64(q)
					}
				}
			}
			qs, dw, _, err := kernels.PackWeights(kernels.Q8_0, src, c.rows, c.k)
			if err != nil {
				t.Fatal(err)
			}
			// The operand: codes four to a word, a scale a 32-block, then
			// the sums per 16 (zero here: Q8_0 has no bias to correct).
			codes := make([]uint32, c.toks*c.k/4)
			ax := make([]float32, c.toks*nb+c.toks*c.k/16)
			av := make([]float64, c.toks*c.k)
			for tk := 0; tk < c.toks; tk++ {
				for b := 0; b < nb; b++ {
					sc := float32(0.003 + 0.01*rng.Float64())
					ax[tk*nb+b] = sc
					for l := 0; l < 32; l++ {
						q := int8(rng.Intn(255) - 127)
						i := tk*c.k + b*32 + l
						codes[i/4] |= uint32(uint8(q)) << (8 * (i % 4))
						av[i] = float64(sc) * float64(q)
					}
				}
			}
			bias := make([]float32, c.rows)
			for r := range bias {
				bias[r] = float32(rng.Float64() - 0.5)
			}
			g := &compose.Row{Width: c.rows, Rows: c.toks}
			bq, bd, ba, bx := g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufRow)
			bb, bo := g.Buf(compose.BufVec), g.Buf(compose.BufOut)
			v, err := g.MatVec(compose.MatVecWeight{T: kernels.Q8_0, K: c.k, Rows: c.rows, QS: bq, D: bd, SC: -1},
				compose.MatVecOperand{A: ba, AX: bx, Toks: c.toks})
			if err != nil {
				t.Fatal(err)
			}
			g.Store(bo, g.Bin(compose.OpAdd, v, g.Load(bb)))
			const group = 128
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
			out := up(make([]byte, 4*c.rows*c.toks))
			named := []backend.Buf{up(wordsLE(qs)), up(wordsLE(dw)), up(wordsLE(codes)), up(f32le(ax)), up(f32le(bias)), out}
			bufs := append([]backend.Buf(nil), named...)
			run := func(r *compose.Row, s compose.Schedule, threads int, args []backend.Buf) {
				k, err := r.Emit(s)
				if err != nil {
					t.Fatal(err)
				}
				kern, err := d.Compile(k)
				if err != nil {
					t.Fatalf("%s: %v", d.API(), err)
				}
				defer kern.Close()
				if err := kern.Launch((threads+group-1)/group, group, args...); err != nil {
					t.Fatal(err)
				}
			}
			sched := compose.Schedule{Group: group, Flat: true, Tile: c.tile, TileCols: c.cols}
			threads := c.rows / c.cols * c.toks / c.tile
			if c.split <= 1 {
				run(g, sched, threads, named)
			} else {
				// The split rewrite: the segments' planes (the bias in the
				// first's), then their sum.
				first, second, b1, b2, err := g.SplitFold(c.split)
				if err != nil {
					t.Fatal(err)
				}
				planes := up(make([]byte, 4*c.rows*c.toks*c.split))
				bufs = append(bufs, planes)
				pick := func(ix []int) []backend.Buf {
					var out []backend.Buf
					for _, i := range ix {
						if i < 0 {
							out = append(out, planes)
						} else {
							out = append(out, named[i])
						}
					}
					return out
				}
				sched.Split = c.split
				run(first, sched, threads*c.split, pick(b1))
				run(second, compose.Schedule{Group: group, Flat: true}, c.rows*c.toks, pick(b2))
			}
			got := make([]byte, 4*c.rows*c.toks)
			if err := out.Read(got); err != nil {
				t.Fatal(err)
			}
			for tk := 0; tk < c.toks; tk++ {
				for r := 0; r < c.rows; r++ {
					want := float64(bias[r])
					for j := 0; j < c.k; j++ {
						want += wv[r*c.k+j] * av[tk*c.k+j]
					}
					g := math.Float32frombits(binary.LittleEndian.Uint32(got[4*(tk*c.rows+r):]))
					if math.Abs(float64(g)-want) > 1e-4*math.Max(1, math.Abs(want)) {
						t.Fatalf("%s %+v: out[%d][%d] = %g, the definition's %g", d.API(), c, tk, r, g, want)
					}
				}
			}
			for _, b := range bufs {
				b.Free()
			}
		}
	}
}

func wordsLE(v []uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}
