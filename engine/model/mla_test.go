package model

import (
	"math"
	"os"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// TestMLARunsUnderAPageBudget runs Multi-head Latent Attention under the pager,
// where every block is re-bound on each visit. The small fixtures otherwise
// stay resident, so a missing rebind of the MLA matrices, or of the per-head
// views l.kbHead/l.vbHead (slices of wkb/wvb), would go unseen: a stale view
// reads another block's weights of the right shape.
func TestMLARunsUnderAPageBudget(t *testing.T) {
	for _, name := range []string{"synth-deepseek", "synth-deepseek-lite"} {
		t.Run(name, func(t *testing.T) {
			src := hfContainer(t, name)

			full, err := Open(src, noTune(), WithKVF16(false))
			if err != nil {
				t.Fatal(err)
			}
			defer full.Close()
			if !full.Cfg.MLA() {
				t.Fatalf("%s is not an MLA model -- this gate proved nothing", name)
			}

			// One byte is one frame, the tightest the reader has: every block
			// is re-bound at every layer of every token.
			tight, err := Open(src, noTune(), WithKVF16(false), WithPageBudget(1))
			if err != nil {
				t.Fatalf("Open under a one-page budget: %v", err)
			}
			defer tight.Close()

			// Assert the budget actually evicts. A zero budget is unlimited,
			// not empty.
			nb := int(tight.container.H.NBlocks)
			var allPages uint64
			for i := 0; i < tight.container.NPages(); i++ {
				allPages += tight.container.PageBytes(i)
			}
			if got := tight.container.Budget(); nb < 2 || got == 0 || got >= allPages {
				t.Fatalf("%d block(s), budget %d bytes against %d of pages: nothing "+
					"is ever evicted, so this gate proved nothing", nb, got, allPages)
			}
			t.Logf("%d block(s), budget %d bytes against %d of pages",
				nb, tight.container.Budget(), allPages)

			ids := []int32{1, 2, 3, 4, 5, 6, 7, 8}
			want := teacherForce(t, full, ids)
			got := teacherForce(t, tight, ids)

			// Bit equality: paging changes which bytes are resident, not the
			// arithmetic.
			for p := range want {
				for i := range want[p] {
					if want[p][i] != got[p][i] {
						t.Fatalf("pos %d logit %d: resident %v, paged %v -- "+
							"a weight moved when its page did",
							p, i, want[p][i], got[p][i])
					}
				}
			}
		})
	}
}

// teacherForce decodes ids one at a time and returns every position's logits.
func teacherForce(t *testing.T, m *Model, ids []int32) [][]float32 {
	t.Helper()
	st := m.NewInstance(len(ids) + 1)
	defer st.Close()
	out := make([][]float32, 0, len(ids))
	for _, id := range ids {
		lg, err := st.Forward(id)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, append([]float32(nil), lg...))
	}
	return out
}

// mlaFixtures is the list both batched MLA gates walk. JITLLM_MLA_MODEL points
// them at a real DeepSeek instead (no fixture carries rope_scaling), e.g.
//
//	./scripts/cap 24G -- taskset -c 0,2,4,6,8,10 go test ./engine/model/ \
//	  -run 'TestMLA(Prefill|Batch)MatchesDecode' -count=1 -v \
//	  -args -test.timeout=30m
//	  # with JITLLM_MLA_MODEL=DeepSeek-V2-Lite.Q4_K_M
func mlaFixtures() []string {
	if p := os.Getenv("JITLLM_MLA_MODEL"); p != "" {
		return []string{p}
	}
	return []string{"synth-deepseek", "synth-deepseek-lite", "synth-deepseek-dense"}
}

// mlaOpen resolves a fixture name or an override path.
func mlaOpen(t *testing.T, name string, opts ...Option) *Model {
	t.Helper()
	src := name
	if os.Getenv("JITLLM_MLA_MODEL") == "" {
		src = hfContainer(t, name)
	} else if p, ok := existingModel(testmodels.Resolve(name)); ok {
		src = jlmOf(t, p)
	} else {
		t.Fatalf("JITLLM_MLA_MODEL=%s: not on this box (RULE 11: fetch it)", name)
	}
	m, err := Open(src, append([]Option{noTune(), WithKVF16(false)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// mlaBound is effectively zero on an F32 fixture, where both paths run the same
// kernel in the same order, and the reassociation band on a quantised model,
// where the batched side runs the tiled packed GEMM.
func mlaBound(m *Model) float64 {
	if os.Getenv("JITLLM_MLA_MODEL") == "" {
		return 1e-10
	}
	return 1e-3
}

// TestMLAPrefillMatchesDecode holds MLA's prefill (batched projections, per-row
// rotary, absorb and cache write) to its decode, the path verified against
// transformers. It catches what a single-token gate cannot: a cache row at the
// wrong position or a rotary half from another token.
func TestMLAPrefillMatchesDecode(t *testing.T) {
	for _, name := range mlaFixtures() {
		t.Run(name, func(t *testing.T) {
			m := mlaOpen(t, name)
			defer m.Close()
			if !m.Cfg.MLA() {
				t.Fatalf("%s is not an MLA model -- this gate proved nothing", name)
			}

			// Long enough that rows attend to earlier rows of their OWN chunk,
			// which is the case a one-token prompt cannot reach.
			ids := []int32{5, 11, 23, 41, 67, 89, 101, 127, 151, 173}
			want := teacherForce(t, m, ids)

			ps := m.NewInstance(len(ids) + 1)
			defer ps.Close()
			got, err := ps.Prefill(ids)
			if err != nil {
				t.Fatalf("Prefill: %v", err)
			}
			if ps.Pos() != len(ids) {
				t.Fatalf("prefill left the cache at %d for %d token(s)", ps.Pos(), len(ids))
			}

			last := want[len(want)-1]
			var num, den float64
			for i := range last {
				d := float64(got[i] - last[i])
				num += d * d
				den += float64(last[i]) * float64(last[i])
			}
			nmse := num / den
			t.Logf("%d token(s), prefill vs decode NMSE %.3e", len(ids), nmse)
			if !(nmse < mlaBound(m)) {
				t.Fatalf("prefill and decode disagree: NMSE %.3e", nmse)
			}
		})
	}
}

// TestMLABatchMatchesDecode is TestMLAPrefillMatchesDecode's twin for
// ForwardBatch. A batch's rows are different sequences, each with its own cache
// slot, so it reaches what prefill cannot: a query read from row 0's buffer, a
// latent written into another sequence's slot. It fails against every row
// using slot 0. Its composed arm runs both sides on the CPU device composed,
// every block step served.
func TestMLABatchMatchesDecode(t *testing.T) {
	for _, name := range mlaFixtures() {
		t.Run(name+"/composed-cpu", func(t *testing.T) { mlaBatchMatchesDecode(t, name, true) })
	}
}

func mlaBatchMatchesDecode(t *testing.T, name string, composed bool) {
	t.Helper()
	m := mlaOpen(t, name)
	defer m.Close()
	if !m.Cfg.MLA() {
		t.Fatalf("%s is not an MLA model -- this gate proved nothing", name)
	}

	// Three sequences that share no token, so a row reading another
	// row's cache cannot look right by coincidence.
	const nseq, steps = 3, 6
	streams := make([][]int32, nseq)
	for i := range streams {
		streams[i] = make([]int32, steps)
		for j := range streams[i] {
			streams[i][j] = int32((7 + i*101 + j*13) % m.Cfg.NVocab)
		}
	}

	want := make([][]float32, nseq)
	for i := range streams {
		s := m.NewInstance(steps + 1)
		for _, id := range streams[i] {
			lg, err := s.Forward(id)
			if err != nil {
				s.Close()
				t.Fatal(err)
			}
			want[i] = append([]float32(nil), lg...)
		}
		cpuServed(t, s, "a sequence's decode", steps*m.Cfg.NLayer, composed)
		s.Close()
	}

	b := newSessBatch(m, nseq, steps+1)
	defer b.Close()
	var sse, sy2 float64
	for j := 0; j < steps; j++ {
		row := make([]int32, nseq)
		for i := range streams {
			row[i] = streams[i][j]
		}
		if _, err := b.ForwardBatch(row); err != nil {
			t.Fatalf("step %d: %v", j, err)
		}
	}
	cpuServed(t, b.Instance, "the batch", steps*m.Cfg.NLayer, composed)
	// Check each arm for non-finite values before subtracting, so a
	// failure names the arm rather than reporting NMSE NaN.
	for i := 0; i < nseq; i++ {
		if at := firstNonFinite(want[i]); at >= 0 {
			t.Fatalf("the HOST reference for sequence %d is non-finite at logit %d "+
				"-- Forward, not ForwardBatch, is what failed here", i, at)
		}
		if at := firstNonFinite(b.BatchLogits(i)); at >= 0 {
			t.Fatalf("ForwardBatch sequence %d is non-finite at logit %d", i, at)
		}
	}
	for i := 0; i < nseq; i++ {
		lg := b.BatchLogits(i)
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
	t.Logf("%d sequence(s) x %d step(s), batch vs decode NMSE %.3e", nseq, steps, nmse)
	if !(nmse < mlaBound(m)) {
		t.Fatalf("ForwardBatch and Forward disagree: NMSE %.3e", nmse)
	}
}

// TestAttnScaleReachesEveryPath asserts that the softmax multiplier is read
// from the CONFIG on all three entry points, by perturbing the config and
// demanding all three move together. Without a YaRN log multiplier AttnScale
// equals 1/sqrt(HeadDim), so a path deriving that itself would agree on every
// fixture; perturbing the field by DeepSeek-V2-Lite's correction exposes it
// without that model.
func TestAttnScaleReachesEveryPath(t *testing.T) {
	// An F32 MLA fixture: exact between the three paths when they agree (so
	// the bound can be 1e-10 rather than a reassociation band), and the
	// architecture the divergence is actually about.
	m, err := Open(hfContainer(t, "synth-deepseek-dense"), noTune(), WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	// The size DeepSeek-V2-Lite's own correction is: mscale = 0.1*0.707*ln(40)
	// + 1 = 1.2608, and the score carries it squared.
	const mscale2 = 1.5896261651208736
	if m.Cfg.AttnScale == 0 {
		t.Fatal("AttnScale is zero -- this gate would compare two zeroed softmaxes")
	}
	m.Cfg.AttnScale *= mscale2

	ids := []int32{5, 11, 23, 41, 67, 89}
	want := teacherForce(t, m, ids)
	last := want[len(want)-1]

	nmseAgainstLast := func(got []float32) float64 {
		var num, den float64
		for i := range last {
			d := float64(got[i] - last[i])
			num += d * d
			den += float64(last[i]) * float64(last[i])
		}
		return num / den
	}

	ps := m.NewInstance(len(ids) + 1)
	defer ps.Close()
	got, err := ps.Prefill(ids)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if n := nmseAgainstLast(got); !(n < 1e-10) {
		t.Errorf("Prefill does not read Config.AttnScale: NMSE %.3e against Forward", n)
	} else {
		t.Logf("Prefill  vs Forward NMSE %.3e", n)
	}

	// ForwardBatch, one row, so its logits are comparable to the same chain.
	b := newSessBatch(m, 1, len(ids)+1)
	defer b.Close()
	for _, id := range ids {
		if _, err := b.ForwardBatch([]int32{id}); err != nil {
			t.Fatal(err)
		}
	}
	if n := nmseAgainstLast(b.BatchLogits(0)); !(n < 1e-10) {
		t.Errorf("ForwardBatch does not read Config.AttnScale: NMSE %.3e against Forward", n)
	} else {
		t.Logf("ForwardBatch vs Forward NMSE %.3e", n)
	}
}

// firstNonFinite is the index of the first NaN or Inf, or -1.
func firstNonFinite(v []float32) int {
	for i, x := range v {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return i
		}
	}
	return -1
}
