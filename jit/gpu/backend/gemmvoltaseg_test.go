package backend_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGemmVoltaSegmentsIsTheLaunches: a GemmVoltaSegments launch -- q, k and v
// as one grid over one binary16 operand -- writes exactly what each segment's
// own GemmVolta launch writes, bit for bit: every segment runs GemmVolta's
// body for its shape, unsplit. Mixed formats that share an operand layout
// (Q4_K and Q8_0's 32-element sub-blocks), mixed row counts, and the sm_70
// and m16n8 tile families the device lowers. Outputs are poisoned with NaN
// and carry a guard word. The segments' weights handed over the wrong way
// round must disagree: the violation.
func TestGemmVoltaSegmentsIsTheLaunches(t *testing.T) {
	gpuLock(t)
	ran := 0
	for _, d := range backend.Open() {
		defer d.Close()
		if d.API() != "ptx" {
			continue
		}
		for _, tl := range []mainkernels.VoltaTile{
			{MT: 4, NT: 4, WM: 1, WN: 4, KB: 1},
			{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 16},
			{MT: 4, NT: 4, WM: 2, WN: 2, KB: 1, F16K: 8},
		} {
			segs := []mainkernels.MatVecShape{
				{T: mainkernels.Q4_K, K: 1024, Rows: 512, NTok: 128},
				{T: mainkernels.Q4_K, K: 1024, Rows: 512, NTok: 128},
				{T: mainkernels.Q8_0, K: 1024, Rows: 256, NTok: 128, Bias: true},
			}
			if gemmSegCase(t, d, segs, tl) {
				ran++
			}
		}
	}
	if ran == 0 {
		t.Skip("no CUDA device lowers a GemmVolta tile here")
	}
}

func gemmSegCase(t *testing.T, d backend.Device, segs []mainkernels.MatVecShape, tl mainkernels.VoltaTile) bool {
	name := fmt.Sprintf("%s %+v", d.Name(), tl)
	tiles := make([]mainkernels.VoltaTile, len(segs))
	for i := range tiles {
		tiles[i] = tl
	}
	kk, err := mainkernels.GemmVoltaSegments(segs, tiles)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	seg, err := d.Compile(kk)
	if err != nil {
		t.Logf("%s: not lowered here: %v", name, err)
		return false
	}
	defer seg.Close()
	blocks, nth, err := mainkernels.GemmVoltaSegmentsGrid(segs, tiles)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	g := newGPU(t, d)
	defer g.free()
	k, ntok := segs[0].K, segs[0].NTok
	// The operand: finite binary16 values, two to a word.
	bw := make([]byte, ntok*k*2)
	for i := 0; i < len(bw); i += 2 {
		h := uint16(rng.Intn(2))<<15 | uint16(12+rng.Intn(6))<<10 | uint16(rng.Intn(1024))
		binary.LittleEndian.PutUint16(bw[i:], h)
	}
	bB := g.up(bw)
	const guard = 0x5EED5EED
	poisoned := func(n int) backend.Buf {
		p := make([]float32, n+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[n*4:], guard)
		return g.up(b)
	}
	type w struct{ qs, d, sc, bias backend.Buf }
	ws := make([]w, len(segs))
	refs, gots, swapped := make([]backend.Buf, len(segs)), make([]backend.Buf, len(segs)), make([]backend.Buf, len(segs))
	for i, s := range segs {
		qs, dw, scw, err := mainkernels.PackWeights(s.T, rawWeights(s.T, s.Rows, k, rng), s.Rows, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(scw) == 0 {
			scw = []uint32{0}
		}
		ws[i] = w{qs: g.up(u32bytes(qs)), d: g.up(u32bytes(dw)), sc: g.up(u32bytes(scw))}
		if s.Bias {
			bv := make([]float32, s.Rows)
			for j := range bv {
				bv[j] = float32(rng.NormFloat64())
			}
			ws[i].bias = g.up(f32bytes(bv))
		}
		n := s.Rows * ntok
		refs[i], gots[i], swapped[i] = poisoned(n), poisoned(n), poisoned(n)
		one, err := mainkernels.GemmVolta(s, tl)
		if err != nil {
			t.Fatalf("%s: segment %d alone: %v", name, i, err)
		}
		k1, err := d.Compile(one)
		if err != nil {
			t.Fatalf("%s: segment %d alone: %v", name, i, err)
		}
		args := []backend.Buf{ws[i].qs, ws[i].d, ws[i].sc, bB, refs[i]}
		if s.Bias {
			args = append(args, ws[i].bias)
		}
		if err := k1.Launch(mainkernels.GemmVoltaGroups(s, tl), tl.Threads(), args...); err != nil {
			t.Fatalf("%s: segment %d alone: %v", name, i, err)
		}
		k1.Close()
	}
	launch := func(out []backend.Buf, swap bool) {
		args := []backend.Buf{bB}
		for i, s := range segs {
			x := ws[i]
			if swap && i < 2 {
				// The first two segments are one format and shape; each reads
				// the other's weights.
				x.qs, x.d, x.sc = ws[1-i].qs, ws[1-i].d, ws[1-i].sc
			}
			args = append(args, x.qs, x.d, x.sc, out[i])
			if s.Bias {
				args = append(args, x.bias)
			}
		}
		total := 0
		for _, n := range blocks {
			total += n
		}
		if err := seg.Launch(total, nth, args...); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	launch(gots, false)
	launch(swapped, true)
	same := 0
	for i, s := range segs {
		n := s.Rows * ntok
		rb, gb, sb := make([]byte, (n+1)*4), make([]byte, (n+1)*4), make([]byte, (n+1)*4)
		if refs[i].Read(rb) != nil || gots[i].Read(gb) != nil || swapped[i].Read(sb) != nil {
			t.Fatalf("%s: read back failed", name)
		}
		for j := 0; j < n; j++ {
			if v := math.Float32frombits(binary.LittleEndian.Uint32(rb[4*j:])); math.IsNaN(float64(v)) {
				t.Fatalf("%s: segment %d's reference word %d is NaN -- a degenerate oracle", name, i, j)
			}
		}
		if !bytes.Equal(rb, gb) {
			bad := 0
			for j := 0; j <= n; j++ {
				if binary.LittleEndian.Uint32(rb[4*j:]) != binary.LittleEndian.Uint32(gb[4*j:]) {
					bad++
				}
			}
			t.Fatalf("%s: segment %d: %d of %d words differ from its own launch", name, i, bad, n+1)
		}
		if i == 0 && bytes.Equal(rb[:4*n], sb[:4*n]) {
			same++
		}
	}
	if same > 0 {
		t.Fatalf("%s: the weights handed over swapped wrote the same rows; the gate cannot see a swap", name)
	}
	t.Logf("%s: %d segments in %v workgroups, bit-identical to their own launches", name, len(segs), blocks)
	return true
}
