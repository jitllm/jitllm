package backend_test

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestGate2InIsTheTwoLaunches: a Gate2 matvec with the norm's operand
// prologue (mainkernels.ActIn{Norm}) -- the FFN norm, the quantize, the gate's
// row and up's row in one launch -- writes, bit for bit on PTX, what
// RMSNormQuantRows followed by the Gate2 matvec writes (and Gate2 is the
// gate's launch then up's gated launch, TestMatVecGate2IsTheTwoLaunches).
// Split 1 and the in-group splits, SiLU and GELU, the warp and tree norms,
// both windows, the output NaN-poisoned with a guard word. The violation: the
// two weights handed over the other way round must move the rows.
func TestGate2InIsTheTwoLaunches(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	ran := 0
	for _, d := range devs {
		defer d.Close()
		if !mainkernels.Gate2Exact(d.API()) {
			continue // the tier offers Gate2 only where it is exact
		}
		for _, q := range []kernels.Quant{kernels.Q8_0, kernels.Q4_K, kernels.Q6_K} {
			for _, sh := range [][2]int{{1024, 4096}, {512, 2048}, {128, 768}} {
				for _, sp := range []int{1, 4, 16} {
					for _, win := range []int{32, 256} {
						for _, warp := range []bool{false, true} {
							a := kernels.ActSiLU
							if win == 256 {
								a = kernels.ActGELU
							}
							if sp > 1 && !mainkernels.GroupSplitOK(sh[0], sp) || (sh[1]/32)%sp != 0 || sh[1]%win != 0 {
								continue
							}
							name := fmt.Sprintf("%s/%v/%dx%d/s%d/w%d/warp%v/%v", d.API(), q, sh[0], sh[1], sp, win, warp, a)
							if gate2InCase(t, d, name, q, sh[0], sh[1], sp, win, warp, a) {
								ran++
							}
						}
					}
				}
			}
		}
	}
	if ran == 0 {
		t.Skip("no backend here makes Gate2 exact")
	}
	t.Logf("%d cases bit-identical", ran)
}

func gate2InCase(t *testing.T, d backend.Device, name string, q kernels.Quant, rows, k, split, win int, warp bool, a kernels.ActKind) bool {
	rng := rand.New(rand.NewSource(int64(rows*13 + k + split + win)))
	x, nw := make([]float32, k), make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * float32(1+i%5)
		nw[i] = float32(rng.NormFloat64())
	}
	pack := func(raw []byte) (qs, dw, scw []uint32) {
		qs, dw, scw, err := kernels.PackWeights(q, raw, rows, k)
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
	g := newGPU(t, d)
	defer g.free()
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
	in := mainkernels.ActIn{On: true, Norm: true, Eps: 1e-5, Warp: warp, Window: win}
	norm, err := mainkernels.RMSNormQuantRows(k, 1, in.Eps, false, false, win)
	if warp {
		norm, err = mainkernels.RMSNormQuantRowsWarp(k, 1, in.Eps, false, false, win)
	}
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	normK, err := d.Compile(norm)
	if err != nil {
		t.Fatalf("%s: compile the norm: %v", name, err)
	}
	defer normK.Close()
	grp := split > 1
	s := mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split, GroupSplit: grp, Gate2: true, GateAct: a}
	mk := func(s mainkernels.MatVecShape) backend.Kernel {
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
	two := mk(s)
	defer two.Close()
	s.In = in
	one := mk(s)
	defer one.Close()

	bX, bNW := g.up(f32bytes(x)), g.up(f32bytes(nw))
	bGQ, bGD, bGS := g.up(u32bytes(gq)), g.up(u32bytes(gd)), g.up(u32bytes(gsc))
	bUQ, bUD, bUS := g.up(u32bytes(uq)), g.up(u32bytes(ud)), g.up(u32bytes(usc))
	bA, bAX := poisoned(k/4), poisoned(k/32+k/16)
	ref, got, swapped := poisoned(rows), poisoned(rows), poisoned(rows)
	if err := normK.Launch(1, mainkernels.RMSNormGroup, bX, bNW, poisoned(k), bA, bAX); err != nil {
		t.Fatalf("%s: norm: %v", name, err)
	}
	w := 128
	if grp {
		w = mainkernels.GroupSplitWidth(split)
	}
	groups := (rows*split + w - 1) / w
	if err := two.Launch(groups, w, bUQ, bUD, bUS, bA, bAX, ref, bGQ, bGD, bGS); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := one.Launch(groups, w, bUQ, bUD, bUS, bX, bNW, got, bGQ, bGD, bGS); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if err := one.Launch(groups, w, bGQ, bGD, bGS, bX, bNW, swapped, bUQ, bUD, bUS); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rb, gb, sb := make([]byte, (rows+1)*4), make([]byte, (rows+1)*4), make([]byte, (rows+1)*4)
	if ref.Read(rb) != nil || got.Read(gb) != nil || swapped.Read(sb) != nil {
		t.Fatalf("%s: read back failed", name)
	}
	bad, same := 0, 0
	for i := 0; i <= rows; i++ {
		rw, gw := binary.LittleEndian.Uint32(rb[4*i:]), binary.LittleEndian.Uint32(gb[4*i:])
		if i < rows && math.IsNaN(float64(math.Float32frombits(rw))) {
			t.Fatalf("%s: the reference row %d is NaN -- a degenerate oracle", name, i)
		}
		if rw != gw {
			if bad++; bad <= 3 {
				t.Errorf("%s: word %d = %#x, want %#x", name, i, gw, rw)
			}
		}
		if i < rows && binary.LittleEndian.Uint32(sb[4*i:]) == rw {
			same++
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d of %d words differ", name, bad, rows+1)
	}
	if same == rows {
		t.Fatalf("%s: the weights handed over swapped wrote the same rows; the gate cannot see a swap", name)
	}
	return true
}
