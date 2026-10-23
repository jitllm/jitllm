package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestRaggedAttentionReadsEachRowsOwnSequence: AttnScoresRagged and
// AttnAccRagged against a scalar reference, for a batch whose rows are
// different sequences at scattered bases of one cache -- counts from 1 to 128,
// GQA 4, K transposed and row-major. Positions past a row's count must score
// -Inf, and a row must never read another row's keys or values.
func TestRaggedAttentionReadsEachRowsOwnSequence(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	const nh, hd, gqa, maxSeq, capPos = 8, 64, 4, 256, 800
	kvDim := nh / gqa * hd
	bases := []uint32{0, 100, 250, 400, 600}
	counts := []uint32{1, 37, 128, 3, 90}
	rows := len(bases)
	rng := rand.New(rand.NewSource(5))
	rnd := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	q := rnd(rows * nh * hd)
	kRow := rnd((capPos + 1) * kvDim) // [pos][kvDim]
	v := rnd((capPos + 1) * kvDim)
	scale := float32(1 / math.Sqrt(hd))
	pn := []uint32{0}
	for _, c := range counts {
		pn[0] = max(pn[0], c)
		pn = append(pn, c)
	}
	for _, d := range devs {
		defer d.Close()
		for _, kst := range []int{0, capPos + 1} {
			name := fmt.Sprintf("%s/kstride%d", d.API(), kst)
			kbuf := kRow
			if kst > 0 { // transposed: [kvDim][cap+1]
				kbuf = make([]float32, kvDim*kst)
				for p := 0; p <= capPos; p++ {
					for e := 0; e < kvDim; e++ {
						kbuf[e*kst+p] = kRow[p*kvDim+e]
					}
				}
			}
			sk, err := kernels.AttnScoresRagged(nh, hd, kvDim, gqa, maxSeq, scale, rows, 2, kst, 0)
			if err != nil {
				t.Fatal(err)
			}
			ak, err := kernels.AttnAccRagged(nh, hd, kvDim, gqa, maxSeq, rows)
			if err != nil {
				t.Fatal(err)
			}
			for _, k := range []interface{ Validate() error }{sk, ak} {
				if err := k.Validate(); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
			}
			ks, err := d.Compile(sk)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			ka, err := d.Compile(ak)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			g := newGPU(t, d)
			u32 := func(x []uint32) []byte {
				b := make([]byte, 4*len(x))
				for i, v := range x {
					binary.LittleEndian.PutUint32(b[4*i:], v)
				}
				return b
			}
			nanBuf := func(n int) backend.Buf {
				p := make([]float32, n)
				for i := range p {
					p[i] = float32(math.NaN())
				}
				return g.up(f32bytes(p))
			}
			bq, bk, bv := g.up(f32bytes(q)), g.up(f32bytes(kbuf)), g.up(f32bytes(v))
			bn, bb := g.up(u32(pn)), g.up(u32(bases))
			bs := nanBuf(rows * nh * maxSeq)
			kt := 2
			groups := (int(pn[0]) + kt - 1) / kt
			if err := ks.Launch((groups*nh*rows+127)/128, 128, bq, bk, bn, bb, bs); err != nil {
				t.Fatal(err)
			}
			got := make([]byte, rows*nh*maxSeq*4)
			if err := bs.Read(got); err != nil {
				t.Fatal(err)
			}
			probs := make([]float32, rows*nh*maxSeq)
			for r := 0; r < rows; r++ {
				for h := 0; h < nh; h++ {
					kvh := h / gqa * hd
					for tt := 0; tt < int(pn[0]); tt++ {
						at := (r*nh+h)*maxSeq + tt
						g := math.Float32frombits(binary.LittleEndian.Uint32(got[4*at:]))
						if tt >= int(counts[r]) {
							if !math.IsInf(float64(g), -1) {
								t.Fatalf("%s: row %d head %d pos %d past count %d scored %v, want -Inf", name, r, h, tt, counts[r], g)
							}
							continue
						}
						var want float64
						for i := 0; i < hd; i++ {
							want += float64(q[(r*nh+h)*hd+i]) * float64(kRow[(int(bases[r])+tt)*kvDim+kvh+i])
						}
						want *= float64(scale)
						if math.Abs(float64(g)-want) > 1e-4*(1+math.Abs(want)) {
							t.Fatalf("%s: row %d head %d pos %d score %v, want %v", name, r, h, tt, g, want)
						}
						probs[at] = float32(rng.Float64())
					}
				}
			}
			ba, bo := g.up(f32bytes(probs)), nanBuf(rows*nh*hd)
			if err := ka.Launch((rows*nh*hd+127)/128, 128, ba, bv, bn, bb, bo); err != nil {
				t.Fatal(err)
			}
			out := make([]byte, rows*nh*hd*4)
			if err := bo.Read(out); err != nil {
				t.Fatal(err)
			}
			for r := 0; r < rows; r++ {
				for h := 0; h < nh; h++ {
					for i := 0; i < hd; i++ {
						var want float64
						for tt := 0; tt < int(counts[r]); tt++ {
							want += float64(probs[(r*nh+h)*maxSeq+tt]) * float64(v[(int(bases[r])+tt)*kvDim+h/gqa*hd+i])
						}
						at := (r*nh+h)*hd + i
						g := float64(math.Float32frombits(binary.LittleEndian.Uint32(out[4*at:])))
						if math.IsNaN(g) || math.Abs(g-want) > 1e-4*(1+math.Abs(want)) {
							t.Fatalf("%s: acc row %d head %d dim %d = %v, want %v", name, r, h, i, g, want)
						}
					}
				}
			}
			ks.Close()
			ka.Close()
			g.free()
		}
	}
}
