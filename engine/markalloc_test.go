package engine

import (
	"path/filepath"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestMarkerDoesNotAllocate is the allocation principle on the marker
// readouts (a marker head's scorer, a pointer): a warm question's engine
// half -- its prompt run to its markers' scores and the readout's answer --
// makes no heap allocation, on the CPU and with every block on each card, as
// TestDecisionDoesNotAllocate holds a label readout's.
func TestMarkerDoesNotAllocate(t *testing.T) {
	for _, mc := range markerCases() {
		t.Run(filepath.Base(mc.src), func(t *testing.T) {
			src, ok := existingModel(testmodels.Path(mc.src))
			if !ok {
				testmodels.Missing(t, "MODEL MISSING: %s (scripts/decisiongold.sh names its source)", testmodels.Path(mc.src))
			}
			m, err := model.Open(jlmOf(t, src))
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			t.Run("cpu", func(t *testing.T) { markerAllocs(t, m, nil) })
			for _, spec := range stepDevices() {
				t.Run(spec, func(t *testing.T) {
					g, err := openGPU(t, gpu.WithDevices(spec), gpu.WithDeviceTune(gpu.TuneOff))
					if err != nil || g == nil {
						noDevice(t, spec, err)
					}
					defer g.Close()
					markerAllocs(t, m, g)
				})
			}
		})
	}
}

func markerAllocs(t *testing.T, m *model.Model, g *gpu.GPU) {
	in, err := NewInstance(m, 1024, WithKVF16(false))
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	if g != nil {
		if err := in.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if in.GPULayers() != m.Cfg.NLayer {
			t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), m.Cfg.NLayer, g.Err())
		}
	}
	d, err := NewDecider(in)
	if err != nil {
		t.Fatal(err)
	}
	state, qs := decisionRequest(t, "support")
	type prompt struct {
		ids   []int32
		marks []int
		qtype int
	}
	var prompts []prompt
	for i := range qs {
		ids, marks, err := markedPrompt(d.mp, d.encodePlain, state, &qs[i], in.MaxSeq(), 0, 0)
		if err != nil {
			t.Fatal(err)
		}
		prompts = append(prompts, prompt{ids, marks, int(qs[i].Type)})
	}
	one := func() {
		for _, p := range prompts {
			n := len(p.marks)
			if err := in.MarkScores(p.ids, p.qtype, p.marks, d.scores[:n]); err != nil {
				t.Fatal(err)
			}
			for j := range n {
				d.keep[j], d.vals[j*len(d.vals)/len(d.keep)] = 1, d.scores[j]
			}
			if !in.decideScores(d.vals, d.keep, d.params, d.prior, d.probs, d.stats) {
				t.Fatal("the readout declined")
			}
		}
	}
	one()
	one()
	if n := testing.AllocsPerRun(2, one); n != 0 {
		t.Errorf("%v allocations over %d warm questions", n, len(prompts))
	}
}
