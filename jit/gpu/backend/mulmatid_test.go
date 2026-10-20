package backend_test

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// TestIndexedMatVecMatchesDense is the correctness gate on MUL_MAT_ID.
//
// It compares against the dense kernel on each expert's own sheet, so it is
// exact rather than NMSE: the only thing that can differ is address arithmetic.
// A wrong expert offset reads real weights of the wrong rows and produces
// plausible numbers, which only a differential against a known-good kernel
// sees.
func TestIndexedMatVecMatchesDense(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.Q6_K,
				mainkernels.Float32, mainkernels.BFloat16} {
				for _, c := range []struct {
					rows, k, experts, split int
					sel                     []uint32
				}{
					// A permutation, so a kernel that ignores pSel and always
					// reads expert 0 fails on three slots out of four.
					{64, 256, 4, 1, []uint32{3, 1, 0, 2}},
					// Fewer slots than experts: the top-k case, and the one
					// where an output stride of Experts*Rows rather than
					// Slots*Rows writes past the buffer.
					{64, 256, 8, 1, []uint32{5, 2}},
					{128, 512, 4, 1, []uint32{2}},
					// The same expert twice: two slots must get equal answers,
					// which catches an output index that uses the expert id.
					{64, 256, 4, 1, []uint32{1, 1}},
					// Split, where splitSeg and slot share one decoded value if
					// the two meanings were ever collapsed.
					{128, 512, 4, 4, []uint32{3, 0, 2}},
					{256, 2048, 8, 8, []uint32{7, 4, 1}},
					// qwen3moe's real expert shape.
					{768, 2048, 8, 4, []uint32{6, 0}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) +
						"/e" + itoa(c.experts) + "s" + itoa(len(c.sel)) + "/split" + itoa(c.split)
					t.Run(name, func(t *testing.T) {
						indexedCase(t, d, q, c.rows, c.k, c.experts, c.split, c.sel)
					})
				}
			}
		})
	}
}

func indexedCase(t *testing.T, d backend.Device, q mainkernels.Quant, rows, k, experts, split int, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	bank := experts * rows
	rng := rand.New(rand.NewSource(int64(bank*7919 + k + int(q))))

	// One sheet per expert, which is what the model writes and the kernel
	// reads. Packing the bank as one tall sheet would interleave it by
	// sub-block and word, scattering expert e's rows, while the kernel's three
	// bases (wrow/wrowD/wrowSC) address one expert's sheet.
	raw := randWeights(q, bank, k, rng)
	x := randActs(k, rng)

	perExpert := len(raw) / experts
	var qs, dw, scw []uint32
	for e := 0; e < experts; e++ {
		q1, d1, s1, err := mainkernels.PackWeights(q, raw[e*perExpert:(e+1)*perExpert], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		qs = append(qs, q1...)
		dw = append(dw, d1...)
		scw = append(scw, s1...)
	}
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}

	g := newGPU(t, d)
	defer g.free()
	if len(scw) == 0 {
		scw = []uint32{0}
	}
	// dOf is the scale-plane slot: a float format's is the float activation.
	dOf := func(d []uint32) backend.Buf {
		if mainkernels.IsFloat(q) {
			return g.up(f32bytes(x))
		}
		return g.up(u32bytes(d))
	}
	bQS := g.up(u32bytes(qs))
	bD := dOf(dw)
	bSC := g.up(u32bytes(scw))
	bA := g.up(u32bytes(av))
	bAX := g.up(f32bytes(append(append([]float32{}, as...), asum...)))

	// The oracle is the dense kernel on one expert's sheet, not on the bank.
	// A dense run over experts*rows of a per-sheet buffer would read across
	// sheet boundaries, which is the very layout confusion under test.
	want := make([][]float32, experts)
	for e := 0; e < experts; e++ {
		q1, d1, s1, err := mainkernels.PackWeights(q, raw[e*perExpert:(e+1)*perExpert], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(s1) == 0 {
			s1 = []uint32{0}
		}
		eQS, eD, eSC := g.up(u32bytes(q1)), dOf(d1), g.up(u32bytes(s1))
		want[e] = g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split},
			rows, 1, split, eQS, eD, eSC, bA, bAX, nil)
	}

	// The kernel under test: the same buffer, one expert's sheet per slot.
	bSel := g.up(u32bytes(sel))
	got := g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
		Experts: experts, Slots: len(sel)}, rows, len(sel), split,
		bQS, bD, bSC, bA, bAX, bSel)

	for j, e := range sel {
		for r := 0; r < rows; r++ {
			w := want[int(e)][r]
			v := got[j*rows+r]
			if v != w {
				t.Fatalf("slot %d (expert %d) row %d: got %v want %v (that expert's own sheet)",
					j, e, r, v, w)
			}
		}
	}

	// The oracle must be able to discriminate: two distinct experts must
	// disagree, or the comparison above passes whatever the index did.
	if experts > 1 {
		same := true
		for r := 0; r < rows && same; r++ {
			if want[0][r] != want[1][r] {
				same = false
			}
		}
		if same {
			t.Fatal("experts 0 and 1 produce identical rows: the test cannot detect a wrong index")
		}
	}
}

// TestIndexedMatVecPerSlotActivations covers MatVecShape.SlotAct, which is what
// an MoE down-projection needs: slot j consumes the FFN activations that expert
// e_j produced, so one launch reads Slots different activation vectors.
//
// An off-by-one-slot read gives finite, plausible numbers. The oracle is again
// the dense kernel, once per slot with that slot's own activations, and the bar
// is equality.
func TestIndexedMatVecPerSlotActivations(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q8_0, mainkernels.Q4_K, mainkernels.Q6_K,
				mainkernels.Float32, mainkernels.BFloat16} {
				for _, c := range []struct {
					rows, k, experts, split int
					sel                     []uint32
				}{
					{64, 256, 4, 1, []uint32{3, 1, 0, 2}},
					{64, 256, 8, 1, []uint32{5, 2}},
					{128, 512, 4, 4, []uint32{3, 0, 2}},
					// qwen3moe's down shape: NEmbd rows from NFFNExp inputs.
					{2048, 768, 8, 4, []uint32{6, 0}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) +
						"/e" + itoa(c.experts) + "s" + itoa(len(c.sel)) + "/split" + itoa(c.split)
					t.Run(name, func(t *testing.T) {
						slotActCase(t, d, q, c.rows, c.k, c.experts, c.split, c.sel)
					})
				}
			}
		})
	}
}

func slotActCase(t *testing.T, d backend.Device, q mainkernels.Quant, rows, k, experts, split int, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	slots := len(sel)
	bank := experts * rows
	rng := rand.New(rand.NewSource(int64(bank*104729 + k + int(q))))

	// One sheet per expert -- the layout the model writes; see indexedCase.
	raw := randWeights(q, bank, k, rng)
	perExpert := len(raw) / experts
	sheet := func(e int) (qs, d, sc []uint32) {
		qs, d, sc, err := mainkernels.PackWeights(q, raw[e*perExpert:(e+1)*perExpert], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(sc) == 0 {
			sc = []uint32{0}
		}
		return qs, d, sc
	}
	var qs, dw, scw []uint32
	for e := 0; e < experts; e++ {
		q1, d1, s1 := sheet(e)
		qs, dw, scw = append(qs, q1...), append(dw, d1...), append(scw, s1...)
	}

	g := newGPU(t, d)
	defer g.free()
	bQS := g.up(u32bytes(qs))
	bSC := g.up(u32bytes(scw))
	bSel := g.up(u32bytes(sel))

	// pAX is flat: all the scales, then all the sums, as one mainkernels.Quantize
	// over the whole Slots*K vector produces on the device.
	var allA []uint32
	var allScale, allSum []float32
	perSlot := make([][]float32, slots)
	perSlotA := make([][]uint32, slots)
	perSlotX := make([][]float32, slots)
	var allX []float32
	for j := 0; j < slots; j++ {
		x := randActs(k, rng)
		perSlotX[j], allX = x, append(allX, x...)
		av, as, asum, err := mainkernels.PackActivations(x)
		if err != nil {
			t.Fatal(err)
		}
		perSlotA[j] = av
		perSlot[j] = append(append([]float32{}, as...), asum...) // the dense oracle's own layout
		allA = append(allA, av...)
		allScale = append(allScale, as...)
		allSum = append(allSum, asum...)
	}
	allAX := append(allScale, allSum...)
	// The scale-plane slot: a float format's carries the float activations.
	bD := g.up(u32bytes(dw))
	if mainkernels.IsFloat(q) {
		bD = g.up(f32bytes(allX))
	}
	got := g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
		Experts: experts, Slots: slots, SlotAct: true}, rows, slots, split,
		bQS, bD, bSC, g.up(u32bytes(allA)), g.up(f32bytes(allAX)), bSel)

	// Oracle: the dense kernel on that expert's own sheet, once per slot, with
	// that slot's own activations. Running it over the whole bank would read
	// across sheet boundaries -- see indexedCase.
	for j, e := range sel {
		q1, d1, s1 := sheet(int(e))
		eD := g.up(u32bytes(d1))
		if mainkernels.IsFloat(q) {
			eD = g.up(f32bytes(perSlotX[j]))
		}
		want := g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split}, rows, 1, split,
			g.up(u32bytes(q1)), eD, g.up(u32bytes(s1)),
			g.up(u32bytes(perSlotA[j])), g.up(f32bytes(perSlot[j])), nil)
		for r := 0; r < rows; r++ {
			if v, w := got[j*rows+r], want[r]; v != w {
				t.Fatalf("slot %d (expert %d) row %d: got %v want %v", j, e, r, v, w)
			}
		}
	}
}

// TestIndexedMatVecMatchesACompactBank is the kernel half of "stream the routed
// experts instead of holding the whole bank".
//
// The unchanged kernel must serve a bank holding only the k experts a token
// routed, addressed 0..k-1, and match the full bank addressed by real ids. It
// can because a bank is independent sheets back to back (jlm.sheetsOf), so the
// selected experts concatenated are a bank of k experts.
//
// The bar is bit equality: same arithmetic on the same bytes in the same order.
func TestIndexedMatVecMatchesACompactBank(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []mainkernels.Quant{mainkernels.Q4_0, mainkernels.Q4_K, mainkernels.Q6_K} {
				for _, c := range []struct {
					rows, k, experts, split int
					sel                     []uint32
				}{
					// Out of order, so a compact bank built in selection order
					// is NOT the same as a prefix of the full one.
					{64, 256, 8, 1, []uint32{5, 2, 7}},
					{128, 512, 16, 4, []uint32{11, 0, 15, 3}},
					// The same expert twice: two slots share one sheet in the
					// full bank and must get two sheets in the compact one, or
					// the remap is a lie about what is resident.
					{64, 256, 8, 1, []uint32{4, 4}},
					// Qwen3-Next's own shape: 10 of many, wide k.
					{512, 2048, 32, 4, []uint32{31, 17, 2, 28, 9, 14, 0, 23, 5, 30}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) +
						"/e" + itoa(c.experts) + "->" + itoa(len(c.sel)) + "/split" + itoa(c.split)
					t.Run(name, func(t *testing.T) {
						compactCase(t, d, q, c.rows, c.k, c.experts, c.split, c.sel)
					})
				}
			}
		})
	}
}

func compactCase(t *testing.T, d backend.Device, q mainkernels.Quant, rows, k, experts, split int, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	rng := rand.New(rand.NewSource(int64(experts*rows*7919 + k + int(q))))
	raw := randWeights(q, experts*rows, k, rng)
	x := randActs(k, rng)
	perExpert := len(raw) / experts

	pack := func(ids []uint32) (qs, dw, scw []uint32) {
		for _, e := range ids {
			q1, d1, s1, err := mainkernels.PackWeights(q, raw[int(e)*perExpert:(int(e)+1)*perExpert], rows, k)
			if err != nil {
				t.Fatal(err)
			}
			qs = append(qs, q1...)
			dw = append(dw, d1...)
			scw = append(scw, s1...)
		}
		if len(scw) == 0 {
			scw = []uint32{0}
		}
		return
	}
	all := make([]uint32, experts)
	for i := range all {
		all[i] = uint32(i)
	}
	av, as, asum, err := mainkernels.PackActivations(x)
	if err != nil {
		t.Fatal(err)
	}

	g := newGPU(t, d)
	defer g.free()
	bA := g.up(u32bytes(av))
	bAX := g.up(f32bytes(append(append([]float32{}, as...), asum...)))

	// A: the whole bank, addressed by real expert ids -- what ships.
	fq, fd, fs := pack(all)
	full := g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
		Experts: experts, Slots: len(sel)}, rows, len(sel), split,
		g.up(u32bytes(fq)), g.up(u32bytes(fd)), g.up(u32bytes(fs)), bA, bAX,
		g.up(u32bytes(sel)))

	// B: only the routed experts, in selection order, addressed 0..k-1 -- what
	// a streamed placement would upload.
	cq, cd, cs := pack(sel)
	slots := make([]uint32, len(sel))
	for i := range slots {
		slots[i] = uint32(i)
	}
	compact := g.run(t, mainkernels.MatVecShape{T: q, K: k, Rows: rows, Split: split,
		Experts: len(sel), Slots: len(sel)}, rows, len(sel), split,
		g.up(u32bytes(cq)), g.up(u32bytes(cd)), g.up(u32bytes(cs)), bA, bAX,
		g.up(u32bytes(slots)))

	for i := range full {
		if full[i] != compact[i] {
			t.Fatalf("slot %d row %d: compact bank gives %v, the full bank %v -- the "+
				"id-to-slot remap is not equivalent", i/rows, i%rows, compact[i], full[i])
		}
	}
	// The comparison must be able to fail: two distinct selected experts must
	// disagree.
	if len(sel) > 1 && sel[0] != sel[1] {
		same := true
		for r := 0; r < rows && same; r++ {
			if full[r] != full[rows+r] {
				same = false
			}
		}
		if same {
			t.Fatal("two different selected experts produce identical rows: this " +
				"cannot detect a wrong remap")
		}
	}
	t.Logf("%d of %d experts uploaded, addressed 0..%d: bit-identical over %d slot(s)",
		len(sel), experts, len(sel)-1, len(sel))
}
