package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestMatVecGate2IsTheTwoLaunches: a Gate2 matvec -- gate's row and up's row
// summed in one launch over two weight streams -- writes exactly what the
// gate's own matvec followed by up's Gate matvec write, bit for bit, at Split
// 1 and at the in-group splits, for SiLU and GELU. The output is poisoned
// with NaN and carries a guard word. The same launch with the two weights
// handed over the other way round must disagree: the violation the gate has
// to see.
func TestMatVecGate2IsTheTwoLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		for _, q := range []mainkernels.Quant{mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.Q6_K} {
			for _, sh := range [][2]int{{1024, 4096}, {256, 2048}, {96, 768}, {8192, 2048}} {
				for _, sp := range []int{1, 2, 4, 8, 16, 32, 64} {
					for _, a := range []mainkernels.ActKind{mainkernels.ActSiLU, mainkernels.ActGELU} {
						if sp > 1 && !mainkernels.GroupSplitOK(sh[0], sp) || (sh[1]/q.Elems()*q.Elems()/32)%sp != 0 {
							continue
						}
						name := fmt.Sprintf("%s/%v/%dx%d/s%d/%v", d.API(), q, sh[0], sh[1], sp, a)
						gate2Case(t, d, name, q, sh[0], sh[1], sp, a)
						ran++
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Fatal("no case ran")
	}
}

// gate2Weights is a random q matrix of rows x k, raw GGUF blocks.
func gate2Weights(q mainkernels.Quant, rows, k int, rng *rand.Rand) []byte {
	nsuper := k / q.Elems()
	raw := make([]byte, rows*nsuper*q.BlockBytes())
	switch q {
	case mainkernels.Q4_K:
		exact := make([][]float64, rows)
		for r := range exact {
			exact[r] = make([]float64, k)
		}
		buildQ4K(raw, exact, rows, nsuper, rng)
	case mainkernels.Q8_0:
		for i := 0; i < len(raw); i += 34 {
			binary.LittleEndian.PutUint16(raw[i:], uint16(0x2000|rng.Intn(0x0C00)))
			for j := 2; j < 34; j++ {
				raw[i+j] = byte(rng.Intn(256))
			}
		}
	default:
		// Q6_K: 128 low bytes, 64 high, 16 signed scales, an f16 d.
		bb := q.BlockBytes()
		for i := 0; i < len(raw); i += bb {
			for j := 0; j < bb-2; j++ {
				raw[i+j] = byte(rng.Intn(256))
			}
			for j := 192; j < 208; j++ {
				raw[i+j] = byte(rng.Intn(64) - 32)
			}
			binary.LittleEndian.PutUint16(raw[i+bb-2:], uint16(0x1800|rng.Intn(0x0400)))
		}
	}
	return raw
}

func gate2Case(t *testing.T, d backend.Device, name string, q mainkernels.Quant, rows, k, split int, a mainkernels.ActKind) {
	rng := rand.New(rand.NewSource(int64(rows*7 + k + split)))
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	pack := func(raw []byte) (qs, dw, scw []uint32) {
		qs, dw, scw, err := mainkernels.PackWeights(q, raw, rows, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(scw) == 0 {
			scw = []uint32{0}
		}
		return qs, dw, scw
	}
	gq, gd, gsc := pack(gate2Weights(q, rows, k, rng))
	uq, ud, usc := pack(gate2Weights(q, rows, k, rng))
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	grp := split > 1
	mk := func(s mainkernels.MatVecShape) backend.Kernel {
		t.Helper()
		s.T, s.K, s.Rows, s.Split, s.GroupSplit, s.GateAct = q, k, rows, split, grp, a
		kk, err := mainkernels.MatVec(s)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := kk.Validate(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		c, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("%s: compile: %v", name, err)
		}
		return c
	}
	plain, gated, both := mk(mainkernels.MatVecShape{}), mk(mainkernels.MatVecShape{Gate: true}), mk(mainkernels.MatVecShape{Gate2: true})
	defer plain.Close()
	defer gated.Close()
	defer both.Close()

	g := newGPU(t, d)
	defer g.free()
	const guard = 0x5EED5EED
	poisoned := func() backend.Buf {
		p := make([]float32, rows+1)
		for i := range p {
			p[i] = float32(math.NaN())
		}
		b := f32bytes(p)
		binary.LittleEndian.PutUint32(b[rows*4:], guard)
		return g.up(b)
	}
	bGQ, bGD, bGS := g.up(u32bytes(gq)), g.up(u32bytes(gd)), g.up(u32bytes(gsc))
	bUQ, bUD, bUS := g.up(u32bytes(uq)), g.up(u32bytes(ud)), g.up(u32bytes(usc))
	bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	gOut, ref, got, swapped := poisoned(), poisoned(), poisoned(), poisoned()
	w := 128
	if grp {
		w = mainkernels.GroupSplitWidth(split)
	}
	groups := (rows*split + w - 1) / w
	if err := plain.Launch(groups, w, bGQ, bGD, bGS, bA, bAX, gOut); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := gated.Launch(groups, w, bUQ, bUD, bUS, bA, bAX, ref, gOut); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := both.Launch(groups, w, bUQ, bUD, bUS, bA, bAX, got, bGQ, bGD, bGS); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := both.Launch(groups, w, bGQ, bGD, bGS, bA, bAX, swapped, bUQ, bUD, bUS); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rb, gb, sb := make([]byte, (rows+1)*4), make([]byte, (rows+1)*4), make([]byte, (rows+1)*4)
	if ref.Read(rb) != nil || got.Read(gb) != nil || swapped.Read(sb) != nil {
		t.Fatalf("%s: read back failed", name)
	}
	// Bit identity is the bar where the lowering pins every rounding (PTX's
	// .rn arithmetic). SPIR-V lowers Fma as an OpFMul and OpFAdd without
	// NoContraction and MSL compiles with fast math, so there the driver may
	// contract the two kernels' sums differently; the tier offers Gate2 only
	// on PTX (mainkernels.Gate2Exact), and elsewhere this holds the rows to the
	// f32 band.
	exact := mainkernels.Gate2Exact(d.API())
	bad, same := 0, 0
	var num, den float64
	for i := 0; i <= rows; i++ {
		w, v := binary.LittleEndian.Uint32(rb[4*i:]), binary.LittleEndian.Uint32(gb[4*i:])
		if i < rows && math.IsNaN(float64(math.Float32frombits(w))) {
			t.Fatalf("%s: the reference row %d is NaN -- a degenerate oracle", name, i)
		}
		if !exact && i < rows {
			fw, fv := float64(math.Float32frombits(w)), float64(math.Float32frombits(v))
			num, den = num+(fw-fv)*(fw-fv), den+fw*fw
			if binary.LittleEndian.Uint32(sb[4*i:]) == w {
				same++
			}
			continue
		}
		if w != v {
			if bad++; bad <= 3 {
				t.Errorf("%s: word %d = %#x, want %#x", name, i, v, w)
			}
		}
		if i < rows && binary.LittleEndian.Uint32(sb[4*i:]) == w {
			same++
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d of %d words differ", name, bad, rows+1)
	}
	if !exact && (math.IsNaN(num) || num > 1e-10*den) {
		t.Fatalf("%s: NMSE %.3g against the two launches", name, num/den)
	}
	if same == rows {
		t.Fatalf("%s: the weights handed over swapped wrote the same rows; the gate cannot see a swap", name)
	}
}
