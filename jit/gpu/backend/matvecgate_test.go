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

// TestMatVecGateIsActMulAfterIt: a Gate matvec writes exactly what the plain
// matvec followed by mainkernels.ActMul writes -- bit for bit, since it is the same
// multiply on the same two floats -- at Split 1 and at the in-group splits the
// tuner picks, for SiLU and GELU. The output is poisoned with NaN and carries a
// guard word.
func TestMatVecGateIsActMulAfterIt(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		for _, q := range []mainkernels.Quant{mainkernels.Q8_0, mainkernels.Q4_K} {
			for _, sh := range [][2]int{{1024, 4096}, {256, 2048}} {
				for _, sp := range []int{1, 4, 32} {
					for _, a := range []mainkernels.ActKind{mainkernels.ActSiLU, mainkernels.ActGELU} {
						name := fmt.Sprintf("%s/%v/%dx%d/s%d/%v", d.API(), q, sh[0], sh[1], sp, a)
						gateCase(t, d, name, q, sh[0], sh[1], sp, a)
					}
				}
			}
		}
	}
}

func gateCase(t *testing.T, d backend.Device, name string, q mainkernels.Quant, rows, k, split int, a mainkernels.ActKind) {
	rng := rand.New(rand.NewSource(int64(rows + k + split)))
	nsuper := k / q.Elems()
	raw := make([]byte, rows*nsuper*q.BlockBytes())
	if q == mainkernels.Q4_K {
		exact := make([][]float64, rows)
		for r := range exact {
			exact[r] = make([]float64, k)
		}
		buildQ4K(raw, exact, rows, nsuper, rng)
	} else {
		for i := 0; i < len(raw); i += 34 {
			binary.LittleEndian.PutUint16(raw[i:], uint16(0x2000|rng.Intn(0x0C00)))
			for j := 2; j < 34; j++ {
				raw[i+j] = byte(rng.Intn(256))
			}
		}
	}
	x := make([]float32, k)
	gate := make([]float32, rows)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	for i := range gate {
		gate[i] = float32(rng.NormFloat64() * 3)
	}
	qs, dw, scw, err := mainkernels.PackWeights(q, raw, rows, k)
	if err != nil {
		t.Fatal(err)
	}
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	grp := split > 1
	mk := func(gated bool) backend.Kernel {
		t.Helper()
		kk, err := mainkernels.MatVec(mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
			GroupSplit: grp, Gate: gated, GateAct: a})
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
	plain, gatedK := mk(false), mk(true)
	defer plain.Close()
	defer gatedK.Close()
	am, err := mainkernels.ActMul(rows, a)
	if err != nil {
		t.Fatal(err)
	}
	actK, err := d.Compile(am)
	if err != nil {
		t.Fatal(err)
	}
	defer actK.Close()

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
	bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	bA, bAX := g.up(u32bytes(av)), g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	bG := g.up(f32bytes(gate))
	up, ref, got := poisoned(), poisoned(), poisoned()
	w := 128
	if grp {
		w = mainkernels.GroupSplitWidth(split)
	}
	groups := (rows*split + w - 1) / w
	if err := plain.Launch(groups, w, bQS, bD, bSC, bA, bAX, up); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := actK.Launch((rows+127)/128, 128, bG, up, ref); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := gatedK.Launch(groups, w, bQS, bD, bSC, bA, bAX, got, bG); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rb, gb := make([]byte, (rows+1)*4), make([]byte, (rows+1)*4)
	if ref.Read(rb) != nil || got.Read(gb) != nil {
		t.Fatalf("%s: read back failed", name)
	}
	bad := 0
	for i := 0; i <= rows; i++ {
		w, v := binary.LittleEndian.Uint32(rb[4*i:]), binary.LittleEndian.Uint32(gb[4*i:])
		if i < rows && math.IsNaN(float64(math.Float32frombits(w))) {
			t.Fatalf("%s: the reference row %d is NaN -- a degenerate oracle", name, i)
		}
		if w != v {
			if bad++; bad <= 3 {
				t.Errorf("%s: word %d = %#x, want %#x", name, i, v, w)
			}
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d of %d words differ", name, bad, rows+1)
	}
}
