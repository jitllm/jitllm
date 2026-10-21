package model

import (
	"strings"
	"testing"

	"github.com/jitllm/jitllm/jit_old/gpu/tier"
)

// TestASpeculatingRunVerifiesAtTheReservedStepWidth: a verification is a
// ragged step of a few rows. With tier.Config.StepRows at SpecRows the
// device reserved that step's scratch at placement, so on a card the
// placement filled every verification runs at the reserved width. The
// violation is the same budget without the reservation: the narrow scratch is
// refused after placement, and the rows run padded to the prompt width.
func TestASpeculatingRunVerifiesAtTheReservedStepWidth(t *testing.T) {
	m := fitModel(t)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) { specStepFits(t, m, spec) })
	}
}

func specStepFits(t *testing.T, m *Model, spec string) {
	prompt := m.Vocab.Encode(strings.Repeat("The clerk counted barrels of salt on the upper floor. ", 40), true)
	cfg := func(reserve bool) func(*tier.Config) {
		return func(c *tier.Config) {
			if reserve {
				c.StepRows = SpecRows()
			}
		}
	}
	// run returns the budget spent before the first verification and the
	// stats after twelve rounds.
	run := func(limit uint64, reserve bool) (uint64, tier.Stats) {
		g := fitTier(t, spec, limit, cfg(reserve))
		defer g.Close()
		st := m.NewState(len(prompt) + 64)
		defer st.Close()
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		sp, err := st.Speculate(WithSpecDrafter(SpecDraftLookup), WithSpecDraft(3))
		if err != nil {
			t.Fatal(err)
		}
		defer sp.Close()
		if _, err := sp.Start(prompt, nil); err != nil {
			t.Fatal(err)
		}
		placed := g.Stats().BudgetUsed
		for range 12 {
			if _, err := sp.Next(nil); err != nil {
				t.Fatal(err)
			}
		}
		if st.GPULayers() != m.Cfg.NLayer {
			t.Skipf("CARD TOO SMALL: %d of %d blocks placed -- this gate proved nothing here (%s)",
				st.GPULayers(), m.Cfg.NLayer, g.Err())
		}
		return placed, g.Stats()
	}
	// The budget is what the reserved placement spent: the card is full at
	// the end of it, and nothing is left for a scratch asked for later.
	limit, _ := run(0, true)
	_, sr := run(limit, true)
	t.Logf("reserved: step %d rows, %d ragged steps, %d rows of padding", sr.ReservedStep, sr.RowsSteps, sr.RowsPadded)
	if sr.ReservedStep == 0 || sr.RowsSteps == 0 {
		t.Fatalf("no step reserved (%d) or no verification ran on the device (%d steps): %s",
			sr.ReservedStep, sr.RowsSteps, sr.LastErr)
	}
	if sr.RowsPadded > sr.RowsSteps*(sr.ReservedStep-1) {
		t.Fatalf("the verifications ran %d rows of padding over %d steps, wider than the reserved %d-row step",
			sr.RowsPadded, sr.RowsSteps, sr.ReservedStep)
	}
	// The violation's budget is what its own placement spent, before any
	// step asked for a scratch: the narrow width is asked of a full card.
	ulimit, _ := run(0, false)
	_, su := run(ulimit, false)
	t.Logf("unreserved: %d ragged steps, %d rows of padding: %s", su.RowsSteps, su.RowsPadded, su.LastErr)
	if su.RowsSteps == 0 {
		t.Fatalf("without the reservation no verification ran on the device: %s", su.LastErr)
	}
	if su.RowsPadded <= su.RowsSteps*(sr.ReservedStep-1) {
		// A device whose step grain is narrower than the matrix
		// instruction's (Vulkan's 8 rows) asks for a scratch small enough
		// that the pool's free pages give it room: it never pads, so there is
		// nothing for the reservation to prevent.
		t.Skipf("without the reservation the verifications ran %d rows of padding over %d steps: the narrow "+
			"scratch fitted this device, so the violation proved nothing here", su.RowsPadded, su.RowsSteps)
	}
}
