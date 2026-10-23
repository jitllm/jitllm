package model

import (
	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/jit"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testmodels"
)

// principleFixtures are the architectures whose gates of the five principles
// (AGENTS.md) run a synthetic fixture beside tinyllama: a file adding an
// architecture appends its fixture here, so the relocation and paging gates
// run every graph rather than one.
var principleFixtures []string

// principleModels is tinyllama and every principle fixture.
func principleModels() []string {
	return append([]string{"tinyllama-1.1b-q3_K_M.gguf"}, principleFixtures...)
}

// principleMixtures are mixture fixtures that run the relocation, batch-fault
// and allocation gates. They are not principleFixtures because
// TestPagingDoesNotChangeTheAnswer counts frames of one page size and a
// mixture's expert pages are another: their paging gate is
// TestMoEFamiliesPageWithTheSameAnswer.
var principleMixtures []string

// relocModels is principleModels and every principle mixture.
func relocModels() []string { return append(principleModels(), principleMixtures...) }

// TestPagingDoesNotChangeTheAnswer compares token ids across page budgets, from
// fully resident down to one frame. After an eviction a frame holds another
// block, so a stale slice reads the wrong weights and produces fluent wrong
// text rather than a crash.
func TestPagingDoesNotChangeTheAnswer(t *testing.T) {
	for _, name := range principleModels() {
		t.Run(name, func(t *testing.T) { pagingDoesNotChangeTheAnswer(t, jlmOf(t, testmodels.Path(name))) })
		// And on a q8_0 KV cache: a block paged out and back is the same
		// block whatever its history is stored as.
		t.Run(name+"/q8_0", func(t *testing.T) {
			pagingDoesNotChangeTheAnswer(t, jlmOf(t, testmodels.Path(name)), WithKVType(KVQ8_0))
		})
	}
}

// TestPlanPagesWithTheSameAnswer is TestPagingDoesNotChangeTheAnswer on the
// composer's plan, on every device: the same budgets, the same ids.
func TestPlanPagesWithTheSameAnswer(t *testing.T) {
	devs := backend.Open()
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	if len(devs) == 0 {
		t.Fatal("no device: the plan's paging gate proved nothing")
	}
	gpu := jit.Device{Kind: jit.DeviceGPU, Slots: 80 * 2048, Lanes: 32}
	for _, name := range principleModels() {
		for _, dev := range devs {
			t.Run(name+"/"+dev.API(), func(t *testing.T) {
				pagingDoesNotChangeTheAnswer(t, jlmOf(t, testmodels.Path(name)), WithPlanDevice(dev, gpu))
			})
		}
	}
}

// pagingDoesNotChangeTheAnswer is TestPagingDoesNotChangeTheAnswer on one model.
// A frame count at or past the model's block count holds it whole and pages
// nothing, so a fixture of three or four blocks runs the counts below it.
func pagingDoesNotChangeTheAnswer(t *testing.T, path string, opts ...Option) {

	// Tuning off: each arm opens its own model, and a tuner that times its
	// kernel choice per process (an arm64 host picks a matvec per shape) can give the
	// arms different reduction orders, which flips tinyllama's 0.082 tie at
	// token 7 -- a difference in tuning, read as one in paging.
	ids := func(frames int) ([]int32, int, int64) {
		devIns, devBlocks := 0, 0
		m, err := Open(path, append([]Option{noTune}, opts...)...)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if frames > 0 {
			m.SetPageBudget(uint64(frames) * m.PageSize())
		}
		in := m.Vocab.Encode("The capital of France is", true)
		st := m.NewState(len(in) + 8 + 1)
		defer st.Close()
		// On the composer's plan the device holds the same count of blocks
		// as the host's frames: its budget is what no page-out gives back
		// and frames of its widest block.
		res := m.PlanResidency()
		if frames > 0 && res != nil {
			m.SetPlanBudget(res.Permanent() + uint64(frames)*res.Widest())
		}
		ins0 := 0
		if res != nil {
			ins0 = res.PageIns
		}
		logits, err := st.Prefill(in)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]int32, 0, 8)
		for i := 0; i < 8; i++ {
			best, bv := int32(0), float32(-1e30)
			for j, v := range logits {
				if v > bv {
					best, bv = int32(j), v
				}
			}
			out = append(out, best)
			if logits, err = st.Forward(best); err != nil {
				t.Fatal(err)
			}
		}
		got, faults, _ := m.PageStats()
		if frames > 0 && res != nil && res.Resident() >= res.Blocks() {
			// Blocks narrower than the widest fit more of them in the
			// frames' bytes, as the host's check above allows.
			t.Logf("%2d frames' bytes hold all %d of the plan's device blocks", frames, res.Blocks())
		} else if frames > 0 && res != nil {
			devIns, devBlocks = res.PageIns-ins0, res.Blocks()
			if devIns <= devBlocks {
				t.Fatalf("%d frames: the plan's device paged in %d blocks over %d tokens of a %d-block model -- it did not page", frames, devIns, 8+len(in), devBlocks)
			}
			t.Logf("%2d frames: the plan's device paged in %d blocks", frames, devIns)
		}
		return out, got, faults
	}

	m0, err := Open(path, append([]Option{noTune}, opts...)...)
	if err != nil {
		if strings.Contains(err.Error(), "q8_0 KV cache is not implemented") {
			t.Skipf("refused by name: %v", err)
		}
		t.Fatal(err)
	}
	if m0.opt.kvTypeSet {
		s := m0.NewState(8)
		got := s.KVType()
		s.Close()
		if got != m0.opt.kvType {
			t.Fatalf("asked for a %v KV cache and got %v", m0.opt.kvType, got)
		}
	}
	nb := int(m0.container.H.NBlocks)
	mixture := m0.Cfg.NExpert > 0
	m0.Close()
	want, _, residentFaults := ids(0)
	for _, frames := range []int{8, 4, 2, 1} {
		if frames >= nb {
			continue
		}
		got, n, faults := ids(frames)
		// Check the budget actually paged, or agreement proves nothing. The
		// budget is frames pages of the largest block, so blocks smaller than
		// that (Gemma 4's single-width FFNs beside its double ones, Nemotron-H's
		// mixer-only blocks) and a mixture's expert pages (container v26) fit
		// more pages in it -- and at a large count, every block.
		if !mixture && n >= nb && frames > 1 {
			t.Logf("%2d frames' bytes hold all %d blocks", frames, nb)
			continue
		}
		if n < frames || (!mixture && n >= nb) {
			t.Fatalf("asked for %d frames, got %d -- this gate proved nothing", frames, n)
		}
		if faults <= residentFaults {
			t.Fatalf("%d frames took %d faults against %d fully resident -- nothing paged",
				frames, faults, residentFaults)
		}
		if len(got) != len(want) {
			t.Fatalf("%d frames: %d ids, want %d", frames, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%d frames: id %d is %d, resident says %d\n  resident %v\n  paged    %v",
					frames, i, got[i], want[i], want, got)
			}
		}
		t.Logf("%2d frames: %d page-in(s), ids identical", frames, faults)
	}
}
