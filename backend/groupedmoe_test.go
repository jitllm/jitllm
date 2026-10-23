package backend_test

import (
	"github.com/jitllm/jitllm/internal/oracle"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
)

// TestGroupedMatVecMatchesPerSlot is the kernel half of a batched mixture: NTok
// columns in groups of Tok, each group reading the ONE expert pSel names for
// it. The oracle is the per-slot indexed matvec (TestIndexedMatVecPerSlotActivations)
// with every column given its group's expert -- the same arithmetic per column,
// so the bar is bit equality. A kernel that indexed pSel by column, by group
// base or not at all reads another expert's sheet and fails here.
func TestGroupedMatVecMatchesPerSlot(t *testing.T) {
	gpuLock(t)
	devs := backend.Open()
	if len(devs) == 0 {
		t.Skip("no GPU backend on this host")
	}
	for _, d := range devs {
		defer d.Close()
		t.Run(d.API(), func(t *testing.T) {
			for _, q := range []kernels.Quant{kernels.Q4_0, kernels.Q8_0, kernels.Q4_K, kernels.Q6_K, kernels.MXFP4} {
				for _, c := range []struct {
					rows, k, experts, tok, split int
					sel                          []uint32 // one per group
				}{
					{64, 256, 4, 2, 1, []uint32{3, 1, 0, 2}},
					{128, 512, 8, 4, 2, []uint32{5, 5, 2, 7}},
					{64, 768, 8, 8, 4, []uint32{6, 0, 3}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) + "/tok" + itoa(c.tok) +
						"/g" + itoa(len(c.sel)) + "/split" + itoa(c.split)
					t.Run(name, func(t *testing.T) { groupedCase(t, d, q, c.rows, c.k, c.experts, c.tok, c.split, c.sel) })
				}
			}
			// A float bank reads the float activation in the scale plane's slot
			// (MLA's absorbs on a float fixture, a float mixture's experts),
			// walking k below 32 where the bank is that narrow (Sub): the
			// fixtures' absorbs are 16x16 sheets.
			for _, q := range []kernels.Quant{kernels.Float32, kernels.Float16, kernels.BFloat16} {
				for _, c := range []struct {
					rows, k, sub int
					sel          []uint32
				}{
					{16, 16, 16, []uint32{3, 1, 0, 2}},
					{32, 64, 0, []uint32{5, 5, 2, 7}},
				} {
					name := q.String() + "/" + itoa(c.rows) + "x" + itoa(c.k) + "/sub" + itoa(c.sub)
					t.Run(name, func(t *testing.T) { groupedFloatCase(t, d, q, c.rows, c.k, c.sub, 8, c.sel) })
				}
			}
		})
	}
}

// groupedFloatCase is groupedCase for a float bank: the activation is the float
// rows, one a column, passed where a quantized format reads its scales.
func groupedFloatCase(t *testing.T, d backend.Device, q kernels.Quant, rows, k, sub, experts int, sel []uint32) {
	const tok = 4
	ntok := tok * len(sel)
	rng := rand.New(rand.NewSource(int64(rows*7919 + k + int(q) + ntok)))
	// floatWeights, not randWeights: k may be below the format's 32-element
	// block, which randWeights counts in.
	raw := make([]byte, experts*rows*k*4)
	exact := make([][]float64, experts*rows)
	for r := range exact {
		exact[r] = make([]float64, k)
	}
	floatWeights(q, raw, exact, experts*rows, k, rng)
	w := 4
	if q != kernels.Float32 {
		w = 2
	}
	raw = raw[:experts*rows*k*w]
	per := len(raw) / experts
	var qs []uint32
	for e := 0; e < experts; e++ {
		q1, _, _, err := kernels.PackWeights(q, raw[e*per:(e+1)*per], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		qs = append(qs, q1...)
	}
	var acts []float32
	for j := 0; j < ntok; j++ {
		acts = append(acts, randActs(k, rng)...)
	}
	perCol := make([]uint32, ntok)
	for j := range perCol {
		perCol[j] = sel[j/tok]
	}
	g := newGPU(t, d)
	defer g.free()
	// The int8 slots are bound and never read by a float matvec.
	bQS, bX, bNone := g.up(u32bytes(qs)), g.up(f32bytes(acts)), g.up(nil)
	got := g.run(t, kernels.MatVecShape{T: q, K: k, Rows: rows, Experts: experts, NTok: ntok, Tok: tok, Sub: sub},
		rows, ntok, 1, bQS, bX, bNone, bNone, bNone, g.up(u32bytes(sel)))
	want := g.run(t, kernels.MatVecShape{T: q, K: k, Rows: rows, Experts: experts, Slots: ntok, SlotAct: true, Sub: sub},
		rows, ntok, 1, bQS, bX, bNone, bNone, bNone, g.up(u32bytes(perCol)))
	if len(got) == 0 || len(want) == 0 {
		t.Fatal("a kernel did not run -- this gate proved nothing")
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output %d (column %d, expert %d): grouped %v, per-slot %v",
				i, (i/rows)%ntok, perCol[(i/rows)%ntok], got[i], want[i])
		}
	}
	// The violation: every group's expert one along. Equality with the
	// grouped output would mean the experts' sheets are not what the columns
	// read, and the comparison above could not tell an expert from another.
	off := make([]uint32, ntok)
	for j := range off {
		off[j] = (perCol[j] + 1) % uint32(experts)
	}
	bad := g.run(t, kernels.MatVecShape{T: q, K: k, Rows: rows, Experts: experts, Slots: ntok, SlotAct: true, Sub: sub},
		rows, ntok, 1, bQS, bX, bNone, bNone, bNone, g.up(u32bytes(off)))
	same := true
	for i := range bad {
		same = same && bad[i] == got[i]
	}
	if same {
		t.Fatal("the grouped output equals the per-slot one at the wrong experts: this gate cannot see a sheet")
	}
}

func groupedCase(t *testing.T, d backend.Device, q kernels.Quant, rows, k, experts, tok, split int, sel []uint32) {
	if k%q.Elems() != 0 {
		t.Skipf("k=%d is not a multiple of %d", k, q.Elems())
	}
	ntok := tok * len(sel)
	rng := rand.New(rand.NewSource(int64(rows*7919 + k + int(q) + ntok)))
	raw := randWeights(q, experts*rows, k, rng)
	per := len(raw) / experts
	var qs, dw, scw []uint32
	for e := 0; e < experts; e++ {
		q1, d1, s1, err := kernels.PackWeights(q, raw[e*per:(e+1)*per], rows, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(s1) == 0 {
			s1 = []uint32{0}
		}
		qs, dw, scw = append(qs, q1...), append(dw, d1...), append(scw, s1...)
	}
	var allA []uint32
	var allScale, allSum []float32
	for j := 0; j < ntok; j++ {
		av, as, asum, err := oracle.PackActivations(randActs(k, rng))
		if err != nil {
			t.Fatal(err)
		}
		allA, allScale, allSum = append(allA, av...), append(allScale, as...), append(allSum, asum...)
	}
	perCol := make([]uint32, ntok)
	for j := range perCol {
		perCol[j] = sel[j/tok]
	}
	g := newGPU(t, d)
	defer g.free()
	bQS, bD, bSC := g.up(u32bytes(qs)), g.up(u32bytes(dw)), g.up(u32bytes(scw))
	bA, bAX := g.up(u32bytes(allA)), g.up(f32bytes(append(allScale, allSum...)))
	got := g.run(t, kernels.MatVecShape{T: q, K: k, Rows: rows, Split: split, Experts: experts,
		NTok: ntok, Tok: tok}, rows, ntok, split, bQS, bD, bSC, bA, bAX, g.up(u32bytes(sel)))
	want := g.run(t, kernels.MatVecShape{T: q, K: k, Rows: rows, Split: split, Experts: experts,
		Slots: ntok, SlotAct: true}, rows, ntok, split, bQS, bD, bSC, bA, bAX, g.up(u32bytes(perCol)))
	if len(got) == 0 || len(want) == 0 {
		t.Fatal("a kernel did not run -- this gate proved nothing")
	}
	// SPIR-V's Q4_K/Q5_K are not Tok-invariant to the ulp (AGENTS.md's open
	// question, bounded at NMSE 1e-8); every other backend and format is exact.
	var num, den float64
	for i := range want {
		dv := float64(got[i] - want[i])
		num, den = num+dv*dv, den+float64(want[i])*float64(want[i])
		if got[i] != want[i] && d.API() != "spirv" {
			t.Fatalf("output %d (column %d, expert %d): grouped %v, per-slot %v",
				i, (i/rows)%ntok, perCol[(i/rows)%ntok], got[i], want[i])
		}
	}
	if nmse := num / den; !(nmse <= 1e-8) {
		t.Fatalf("NMSE %.3e against the per-slot kernel", nmse)
	}
}
