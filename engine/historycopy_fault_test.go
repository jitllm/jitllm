//go:build jitllmfault

package engine

import (
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// TestAHistoryReadAcrossCardsDiscriminates runs TestAHistoryReadAcrossCards'
// split placement with the copies taking none of the rows their sources
// write once they were made (gpu.Faults.SetPagedFault "copystale"): the
// readers attend over a copy made before the prompt. The logits must leave
// the band the gate holds them to, or the gate cannot see a copy that is not
// appended.
func TestAHistoryReadAcrossCardsDiscriminates(t *testing.T) {
	for _, f := range []struct {
		name string
		open func(*testing.T, string) (*model.Model, *c6Golden, []Option)
	}{{"synth-gemma4-e", openGemma4}, {"synth-gemma3n", openGemma3n}} {
		t.Run(f.name, func(t *testing.T) {
			m, _, io := f.open(t, f.name)
			defer m.Close()
			pairs := historyCopyPairs(t)
			if len(pairs) == 0 {
				noDevice(t, "two cards", nil)
			}
			for _, p := range pairs {
				t.Run(p[0]+","+p[1], func(t *testing.T) {
					var fs gpu.Faults
					defer fs.SetPagedFault("")
					got, band := historyReadAcrossCards(t, m, io, p[0], p[1], func() { fs.SetPagedFault("copystale") }, gpu.WithFaults(&fs))
					out := 0
					for i := range got {
						if got[i] > max(3*band[i], 1e-6) {
							out++
						}
					}
					t.Logf("NMSE from the CPU's with the copies not appended: %.2e; one card: %.2e", got, band)
					if out == 0 {
						t.Fatal("copies left as they were made stood inside the band: the gate cannot see a copy that is not appended")
					}
				})
			}
		})
	}
}
