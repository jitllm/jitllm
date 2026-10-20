package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/compose"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
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
			g := &compose.Row{Width: c.rows, Rows: c.toks}
			bq, bd, ba, bx, bo := g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufU32), g.Buf(compose.BufRow), g.Buf(compose.BufOut)
			v, err := g.MatVec(compose.MatVecWeight{T: kernels.Q8_0, K: c.k, Rows: c.rows, QS: bq, D: bd, SC: -1},
				compose.MatVecOperand{A: ba, AX: bx, Toks: c.toks})
			if err != nil {
				t.Fatal(err)
			}
			g.Store(bo, v)
			const group = 128
			k, err := g.Emit(compose.Schedule{Group: group, Flat: true, Tile: c.tile, TileCols: c.cols, Split: c.split})
			if err != nil {
				t.Fatal(err)
			}
			kern, err := d.Compile(k)
			if err != nil {
				t.Fatalf("%s: %v", d.API(), err)
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
			out := up(make([]byte, 4*c.rows*c.toks*c.split))
			bufs := []backend.Buf{up(wordsLE(qs)), up(wordsLE(dw)), up(wordsLE(codes)), up(f32le(ax)), out}
			if err := kern.Launch((c.rows/c.cols*c.toks/c.tile*c.split+group-1)/group, group, bufs...); err != nil {
				t.Fatal(err)
			}
			if c.split > 1 {
				// The segments' planes summed by a group of their own.
				red := &compose.Row{Width: c.rows, Rows: c.toks}
				in, o := red.Buf(compose.BufRow), red.Buf(compose.BufOut)
				red.Store(o, red.SumPlanes(in, c.split))
				rk, err := red.Emit(compose.Schedule{Group: group, Flat: true})
				if err != nil {
					t.Fatal(err)
				}
				rkern, err := d.Compile(rk)
				if err != nil {
					t.Fatal(err)
				}
				sum := up(make([]byte, 4*c.rows*c.toks))
				if err := rkern.Launch((c.rows*c.toks+group-1)/group, group, out, sum); err != nil {
					t.Fatal(err)
				}
				rkern.Close()
				bufs = append(bufs, sum)
				out = sum
			}
			got := make([]byte, 4*c.rows*c.toks)
			if err := out.Read(got); err != nil {
				t.Fatal(err)
			}
			for tk := 0; tk < c.toks; tk++ {
				for r := 0; r < c.rows; r++ {
					var want float64
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
			kern.Close()
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
