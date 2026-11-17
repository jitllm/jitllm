package engine

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestATemperatureOffsetIsAWholeNumber: the attention temperature 1 + s ln(1
// + floor((pos + o) / f)) is stated over the position op's whole-number
// offset o, the only kind a model states (Llama 4's is 1). A fractional one
// is refused at conversion, never truncated to the whole number below it,
// and offset 2 moves the Llama 4 fixture's logits from offset 1's (the
// temperature steps a position sooner) on the composed CPU and with every
// block on a card.
func TestATemperatureOffsetIsAWholeNumber(t *testing.T) {
	src := testmodels.Path("synth-llama4.gguf")
	if !sourcePresent(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", src)
	}
	err := convertStated(src, filepath.Join(t.TempDir(), "half"+model.Ext), func(s *jlm.Source) { s.Config.AttnTempOffset = 1.5 })
	if err == nil || !strings.Contains(err.Error(), "not an integer") {
		t.Fatalf("a temperature offset of 1.5 converts (%v): it must be refused, not truncated", err)
	}
	m, g, opts := openSynthLlama4(t)
	defer m.Close()
	gc, _ := openGPU(t)
	if gc != nil {
		defer gc.Close()
	}
	type arm struct {
		name string
		run  func() *Instance
	}
	arms := []arm{{"composed CPU", func() *Instance { return testInstance(t, m, len(g.IDs)+1, opts...) }}}
	if gc != nil {
		arms = append(arms, arm{"card", func() *Instance { return cardInstance(t, m, gc) }})
	}
	for _, a := range arms {
		t.Run(a.name, func(t *testing.T) {
			decode := func(offs float64) [][]float32 {
				defer cfgMut(m, func(c *model.Config) { c.AttnTempOffs = offs })()
				st := a.run()
				defer st.Close()
				out := make([][]float32, 0, len(g.IDs))
				for _, id := range g.IDs {
					lg, err := st.Forward(id)
					if err != nil {
						t.Fatalf("offset %v: %v", offs, err)
					}
					out = append(out, slices.Clone(lg))
				}
				return out
			}
			whole, further := decode(1), decode(2)
			moved := false
			for p := range whole {
				moved = moved || !slices.Equal(whole[p], further[p])
			}
			if !moved {
				t.Fatal("offset 2 reads offset 1's logits at every position: the gate cannot see the temperature")
			}
			t.Logf("%d positions: offset 2 moves offset 1's logits", len(whole))
		})
	}
}
