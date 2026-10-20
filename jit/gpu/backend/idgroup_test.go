package backend_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestIndexedGroupSplitMatchesTheReduce gates the indexed in-group split: an
// expert matvec whose Split segments meet in threadgroup memory rather than in
// a partial plane summed by a Reduce launch.
//
// The oracle is the partial-writing form of the same kernel and the bar is
// equality: both compute the same partials in the same order, so any
// difference is addressing (the slot decoded from the group index, or the
// store's slot stride). The partial-writing form is itself gated by
// TestIndexedMatVecMatchesDense.
//
// The splits include ones that are not powers of two, because gpt-oss's k of
// 2880 is 90 activation blocks and 2 is the only power of two dividing it.
func TestIndexedGroupSplitMatchesTheReduce(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.Q6_K,
				mainkernels.MXFP4} {
				for _, c := range []struct {
					rows, k, experts, split int
					slotAct                 bool
					sel                     []uint32
				}{
					{64, 256, 4, 2, false, []uint32{3, 1, 0, 2}},
					{64, 256, 8, 4, true, []uint32{5, 2}},
					{128, 512, 4, 4, false, []uint32{1, 1}},
					// gpt-oss's shape class: k = 90 blocks, splits 5, 9 and 18.
					{96, 2880, 8, 9, false, []uint32{7, 0, 3, 5}},
					{96, 2880, 8, 18, true, []uint32{2, 6, 1, 4}},
					{64, 2880, 4, 5, true, []uint32{3, 1}},
					// One slot: the top-1 router shape, where the slot is 0.
					{64, 512, 4, 8, false, []uint32{2}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) + "/e" + itoa(c.experts) +
						"s" + itoa(len(c.sel)) + "/split" + itoa(c.split)
					if c.slotAct {
						name += "/slotact"
					}
					t.Run(name, func(t *testing.T) {
						idGroupCase(t, d, q, c.rows, c.k, c.experts, c.split, c.slotAct, c.sel)
					})
				}
			}
		})
	}
}

func idGroupCase(t *testing.T, d backend.Device, q mainkernels.Quant, rows, k, experts, split int,
	slotAct bool, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	slots := len(sel)
	rng := rand.New(rand.NewSource(int64(rows*31+k*7+split) + int64(q)))
	raw := randWeights(q, experts*rows, k, rng)
	if q == mainkernels.MXFP4 {
		// The E8M0 byte leads each 17-byte block; a random one reaches 2^128.
		for i := 0; i < len(raw); i += 17 {
			raw[i] = byte(120 + rng.Intn(14))
		}
	}
	perExpert := len(raw) / experts
	var qs, dw, scw []uint32
	for e := 0; e < experts; e++ {
		q1, d1, s1, err := mainkernels.PackWeights(q, raw[e*perExpert:(e+1)*perExpert], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		qs, dw, scw = append(qs, q1...), append(dw, d1...), append(scw, s1...)
	}
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	nact := 1
	if slotAct {
		nact = slots
	}
	var x []float32
	for j := 0; j < nact; j++ {
		x = append(x, randActs(k, rng)...)
	}
	// One Quantize over the whole vector's layout: all scales, then all sums.
	var av []uint32
	var as, asum []float32
	for j := 0; j < nact; j++ {
		a1, s1, u1, err := mainkernels.PackActivations(x[j*k : (j+1)*k])
		if err != nil {
			t.Fatal(err)
		}
		av, as, asum = append(av, a1...), append(as, s1...), append(asum, u1...)
	}

	g := newGPU(t, d)
	defer g.free()
	bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	bA := g.up(u32bytes(av))
	bAX := g.up(f32bytes(append(append([]float32{}, as...), asum...)))
	bSel := g.up(u32bytes(sel))

	shape := mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
		Experts: experts, Slots: slots, SlotAct: slotAct}
	want := g.run(t, shape, rows, slots, split, bQS, bD, bSC, bA, bAX, bSel)

	shape.GroupSplit = true
	kk, err := mainkernels.MatVec(shape)
	if err != nil {
		t.Fatalf("the in-group indexed shape is refused: %v", err)
	}
	kern, err := d.Compile(kk)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kern.Close()
	n := rows * slots
	out := g.up(make([]byte, n*4))
	w := mainkernels.GroupSplitWidth(split)
	if err := kern.Launch(n*split/w, w, bQS, bD, bSC, bA, bAX, out, bSel); err != nil {
		t.Fatalf("launch: %v", err)
	}
	rb := make([]byte, n*4)
	if err := out.Read(rb); err != nil {
		t.Fatal(err)
	}
	finite := 0
	for i := 0; i < n; i++ {
		v := math.Float32frombits(binary.LittleEndian.Uint32(rb[i*4:]))
		if v != want[i] {
			t.Fatalf("slot %d (expert %d) row %d: in-group %v, reduce %v",
				i/rows, sel[i/rows], i%rows, v, want[i])
		}
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) && v != 0 {
			finite++
		}
	}
	// A degenerate oracle is a failure, not a pass (RULE 10).
	if finite < n/2 {
		t.Fatalf("only %d of %d outputs are finite and non-zero: the comparison proves nothing", finite, n)
	}
}
