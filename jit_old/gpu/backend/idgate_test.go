package backend_test

import (
	"bytes"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/backend"
	"github.com/jitllm/jitllm/jit_old/gpu/ir"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestIndexedGateAndBiasMatchTheSeparateKernels gates the mixture's fused
// epilogues -- an expert matvec adding its expert's own bias, and the up
// projection writing act(gate)*up itself -- against the separate launches they
// replace: the plain indexed matvec, mainkernels.IndexedBiasAdd and mainkernels.ActMul.
//
// The bar is bit equality: the fused epilogue does the same float operations in
// the same order, so anything else is an index (the bias read at the slot
// instead of the expert id, or the gate read at slot 0's row). The cases carry a permuted
// selection with a repeated expert, both SiLU and gpt-oss's swiglu-oai (whose
// clamps bite: the values are scaled past 7), the in-group and the one-split
// forms, and MXFP4.
func TestIndexedGateAndBiasMatchTheSeparateKernels(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.MXFP4} {
				for _, act := range []mainkernels.ActKind{mainkernels.ActSiLU, mainkernels.ActSwiGLUOAI} {
					for _, c := range []struct {
						rows, k, experts, split int
						grp                     bool
						sel                     []uint32
					}{
						{64, 256, 8, 1, false, []uint32{5, 2, 5, 0}},
						{96, 512, 4, 4, true, []uint32{3, 1}},
						{64, 2880, 32, 10, true, []uint32{31, 7, 0, 12}},
					} {
						name := q.String() + "/" + act.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) +
							"/split" + itoa(c.split)
						t.Run(name, func(t *testing.T) {
							idGateCase(t, d, q, act, c.rows, c.k, c.experts, c.split, c.grp, c.sel)
						})
					}
				}
			}
		})
	}
}

func idGateCase(t *testing.T, d backend.Device, q mainkernels.Quant, act mainkernels.ActKind,
	rows, k, experts, split int, grp bool, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	slots := len(sel)
	rng := rand.New(rand.NewSource(int64(rows*13+k+split) + int64(q)*7 + int64(act)))
	g := newGPU(t, d)
	defer g.free()
	pack := func() (backend.Buf, backend.Buf, backend.Buf) {
		raw := randWeights(q, experts*rows, k, rng)
		if q == mainkernels.MXFP4 {
			for i := 0; i < len(raw); i += 17 {
				raw[i] = byte(120 + rng.Intn(14))
			}
		}
		per := len(raw) / experts
		var qs, dw, scw []uint32
		for e := 0; e < experts; e++ {
			q1, d1, s1, err := mainkernels.PackWeights(q, raw[e*per:(e+1)*per], rows, k)
			if err != nil {
				t.Fatal(err)
			}
			qs, dw, scw = append(qs, q1...), append(dw, d1...), append(scw, s1...)
		}
		if len(scw) == 0 {
			scw = []uint32{0}
		}
		return g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	}
	gQS, gD, gSC := pack()
	uQS, uD, uSC := pack()
	av, as, asum, err := mainkernels.PackActivations(randActs(k, rng))
	if err != nil {
		t.Fatal(err)
	}
	bA := g.up(u32bytes(av))
	bAX := g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	bSel := g.up(u32bytes(sel))
	biasOf := func() backend.Buf {
		v := make([]float32, experts*rows)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 4)
		}
		return g.up(f32bytes(v))
	}
	gBias, uBias := biasOf(), biasOf()
	n := rows * slots
	w := 128
	if grp {
		w = mainkernels.GroupSplitWidth(split)
	}
	launch := func(kk *ir.Kernel, err error, groups, width int, args ...backend.Buf) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		kern, err := d.Compile(kk)
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer kern.Close()
		if err := kern.Launch(groups, width, args...); err != nil {
			t.Fatal(err)
		}
	}
	read := func(b backend.Buf) []byte {
		p := make([]byte, n*4)
		if err := b.Read(p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	shape := mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split, GroupSplit: grp,
		Experts: experts, Slots: slots}
	groups := (n*split + w - 1) / w

	// The separate launches: plain matvec, IndexedBiasAdd, ActMul.
	gRaw, uRaw := g.up(make([]byte, n*4)), g.up(make([]byte, n*4))
	kk, err := mainkernels.MatVec(shape)
	launch(kk, err, groups, w, gQS, gD, gSC, bA, bAX, gRaw, bSel)
	launch(kk, err, groups, w, uQS, uD, uSC, bA, bAX, uRaw, bSel)
	gWant, uWant := g.up(make([]byte, n*4)), g.up(make([]byte, n*4))
	kb, err := mainkernels.IndexedBiasAdd(slots, rows)
	launch(kb, err, (n+127)/128, 128, gRaw, bSel, gBias, gWant)
	launch(kb, err, (n+127)/128, 128, uRaw, bSel, uBias, uWant)
	want := g.up(make([]byte, n*4))
	ka, err := mainkernels.ActMul(n, act)
	launch(ka, err, (n+127)/128, 128, gWant, uWant, want)

	// The fused epilogues: the gate with its bias, then up with its bias
	// and the gate.
	poison := bytes.Repeat([]byte{0xFF, 0xFF, 0xC0, 0x7F}, n) // NaN
	gGot, got := g.up(poison), g.up(poison)
	bs := shape
	bs.Bias = true
	kgb, err := mainkernels.MatVec(bs)
	launch(kgb, err, groups, w, gQS, gD, gSC, bA, bAX, gGot, gBias, bSel)
	if a, b := read(gWant), read(gGot); !bytes.Equal(a, b) {
		t.Fatalf("the gate with its fused expert bias differs from matvec + IndexedBiasAdd")
	}
	us := bs
	us.Gate, us.GateAct = true, act
	kug, err := mainkernels.MatVec(us)
	launch(kug, err, groups, w, uQS, uD, uSC, bA, bAX, got, uBias, gGot, bSel)
	a, b := read(want), read(got)
	if !bytes.Equal(a, b) {
		t.Fatalf("up with its fused bias and gate differs from matvec + IndexedBiasAdd + ActMul")
	}
	// A degenerate oracle is a failure, not a pass (RULE 10).
	live := 0
	for i := 0; i < n; i++ {
		v := math.Float32frombits(uint32(a[4*i]) | uint32(a[4*i+1])<<8 | uint32(a[4*i+2])<<16 | uint32(a[4*i+3])<<24)
		if v != 0 && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			live++
		}
	}
	// SiLU of a large negative gate is 0, so a quarter live is the floor.
	if live < n/5 {
		t.Fatalf("only %d of %d outputs are finite and non-zero", live, n)
	}
}
