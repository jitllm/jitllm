package model

import (
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestOpenHonoursAPageBudget asserts the budget reaches the reader before the
// first page does, so a model larger than memory can load. It checks the
// budget actually binds before comparing tokens, since a budget that did
// nothing would give two identical fully-resident models.
func TestOpenHonoursAPageBudget(t *testing.T) {
	path, ok := models["stories260K"]
	if !ok {
		t.Skip("no small model in the table")
	}
	src := jlmOf(t, path)

	full, err := Open(src)
	if err != nil {
		t.Fatal(err)
	}
	defer full.Close()

	// One byte of budget is one frame, which is the tightest configuration the
	// reader has: every block but the resident one faults on every visit.
	tight, err := Open(src, WithPageBudget(1))
	if err != nil {
		t.Fatalf("Open under a one-page budget: %v", err)
	}
	defer tight.Close()

	nb := int(tight.container.H.NBlocks)
	if nb < 2 {
		t.Skipf("model has %d blocks; a budget cannot bind", nb)
	}
	// A zero budget is unlimited, not empty. The configuration under test is
	// 0 < budget < every page, which makes a page-in evict.
	var allPages uint64
	for i := 0; i < tight.container.NPages(); i++ {
		allPages += tight.container.PageBytes(i)
	}
	if got := tight.container.Budget(); got == 0 || got >= allPages {
		t.Fatalf("budget is %d bytes against %d of pages: it did not bind (0 means "+
			"unlimited), so this test would compare two fully-resident models "+
			"and prove nothing", got, allPages)
	}
	if got := full.container.Budget(); got != 0 && got < allPages {
		t.Fatalf("the unbudgeted open has a %d-byte budget against %d of pages: both "+
			"arms are paged and the comparison is not paired", got, allPages)
	}
	// build() reads only the always-resident dense region, so opening faults
	// nothing and placement can happen before any block is read.
	if in, _ := tight.container.Faults(); in != 0 {
		t.Logf("%d page-ins during open (expected 0 now that norms are dense)", in)
	}

	// And the paged model answers identically: residency is a memory decision,
	// never an arithmetic one.
	ids := []int32{1, 7, 3, 11}
	a, b := full.NewState(16), tight.NewState(16)
	defer a.Close()
	defer b.Close()
	for _, id := range ids {
		la, err := a.Forward(id % int32(full.Cfg.NVocab))
		if err != nil {
			t.Fatal(err)
		}
		lb, err := b.Forward(id % int32(tight.Cfg.NVocab))
		if err != nil {
			t.Fatal(err)
		}
		if Greedy(la) != Greedy(lb) {
			t.Fatalf("paged model picked %d, resident picked %d", Greedy(lb), Greedy(la))
		}
		for i := range la {
			if la[i] != lb[i] {
				t.Fatalf("logit %d differs: %v vs %v", i, la[i], lb[i])
			}
		}
	}
}

// TestForwardBatchFaultsItsBlocksIn runs the batched forward under a budget
// that cannot hold the model (one frame), and requires it to equal decode.
// A block that is not faulted in reads another block's bytes from a reused
// frame, so the failure is a wrong (often non-finite) answer rather than an
// unbound-tensor error; it is invisible on a model that fits.
//
// Equal, not close: the GEMM's float epilogue (GEMMExact) sums every batched
// matmul in the matvec's order, and keeps the matvec on the whole of k where a
// wide pool would split it (tinyllama's head on a 14-worker pool read
// NMSE 5.87e-14 before it did), so paging is the only variable left. The
// shipped GEMM on a pre-VNNI amd64 or an arm64 host sums a k-quant's
// super-block in integers (cpu.StationaryIntAcc), and int8 re-quantization
// carries that ~1e-12 per matmul to NMSE 2.85e-03 on qwen3moe -- past this
// gate's old 1e-3 bound. TestBatchMatchesForward's "shipped" arm
// is where that form is judged.
func TestForwardBatchFaultsItsBlocksIn(t *testing.T) {
	for name, path := range models {
		if name == "stories260K" {
			continue // its blocks are smaller than a frame; nothing evicts
		}
		t.Run(name, func(t *testing.T) { forwardBatchFaultsItsBlocksIn(t, path) })
	}
	// The principle fixtures (paging_test.go), one per architecture beyond
	// the map's.
	for _, name := range append(principleFixtures, principleMixtures...) {
		t.Run(name, func(t *testing.T) { forwardBatchFaultsItsBlocksIn(t, testmodels.Path(name)) })
	}
}

// forwardBatchFaultsItsBlocksIn is TestForwardBatchFaultsItsBlocksIn on one
// model.
func forwardBatchFaultsItsBlocksIn(t *testing.T, path string) {
	p, ok := existingModel(path)
	if !ok {
		t.Skipf("model not present: %s", path)
	}
	src := jlmOf(t, p)
	tight, err := Open(src, noTune, WithKVF16(false), WithPageBudget(1),
		WithJITOptions(nn.WithGEMMExact(true)))
	if err != nil {
		t.Fatalf("Open under a one-page budget: %v", err)
	}
	defer tight.Close()

	// Assert the budget binds; a zero budget is unlimited.
	nb := int(tight.container.H.NBlocks)
	var allPages uint64
	for i := 0; i < tight.container.NPages(); i++ {
		allPages += tight.container.PageBytes(i)
	}
	if got := tight.container.Budget(); nb < 2 || got == 0 || got >= allPages {
		t.Fatalf("%d block(s), budget %d against %d of pages: nothing is "+
			"ever evicted, so this gate proved nothing", nb, got, allPages)
	}

	const nseq, steps = 2, 4
	streams := make([][]int32, nseq)
	for i := range streams {
		streams[i] = make([]int32, steps)
		for j := range streams[i] {
			streams[i][j] = int32((7 + i*101 + j*13) % tight.Cfg.NVocab)
		}
	}
	// Decode on the same tightly budgeted model, so the batched path
	// is the only variable.
	want := make([][]float32, nseq)
	for i := range streams {
		st := tight.NewState(steps + 1)
		for _, id := range streams[i] {
			lg, err := st.Forward(id)
			if err != nil {
				st.Close()
				t.Fatalf("Forward under a one-page budget: %v", err)
			}
			want[i] = append([]float32(nil), lg...)
		}
		st.Close()
	}

	b := tight.NewBatch(nseq, steps+1)
	defer b.Close()
	for j := 0; j < steps; j++ {
		row := make([]int32, nseq)
		for i := range streams {
			row[i] = streams[i][j]
		}
		if _, err := b.ForwardBatch(row); err != nil {
			t.Fatalf("ForwardBatch under a one-page budget: %v", err)
		}
	}
	var sse, sy2 float64
	for i := 0; i < nseq; i++ {
		lg := b.BatchLogits(i)
		if at := firstNonFinite(lg); at >= 0 {
			t.Fatalf("sequence %d is non-finite at logit %d -- the batched "+
				"path read a page it did not fault in", i, at)
		}
		for x, v := range lg {
			d := float64(v - want[i][x])
			sse += d * d
			sy2 += float64(want[i][x]) * float64(want[i][x])
		}
	}
	nmse := 0.0
	if sy2 > 0 {
		nmse = sse / sy2
	}
	t.Logf("%d block(s), budget %d of %d page bytes, batch vs decode NMSE %.3e",
		nb, tight.container.Budget(), allPages, nmse)
	if nmse != 0 {
		t.Fatalf("ForwardBatch and Forward disagree under paging: NMSE %.3e", nmse)
	}

}
