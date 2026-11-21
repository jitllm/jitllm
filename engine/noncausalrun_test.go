package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/model"
)

// TestNonCausalPromptIsOneRun prefills TestCentredRunMatchesTheCPU's
// non-causal decoder (centredModel) through Instance.Prefill, its 1150
// tokens marked nothing: a non-causal model's prompt is one bidirectional
// run whatever chunks a device takes. On the CPU it is the marked run's
// logits to the bit, in one instance; with every block on each card it is
// held to the CPU's run within three times the band a run one submission
// holds reads there.
func TestNonCausalPromptIsOneRun(t *testing.T) {
	m := centredModel(t)
	defer m.Close()
	a := newCentredArm(t, m)
	ids := a.long[0].Tokens
	plain := func(g *gpu.GPU) (lg, run []float32) {
		in := testInstance(t, m, len(ids)+1, WithKVF16(false), withCPUFloat())
		defer in.Close()
		if g != nil {
			if err := in.SetDeviceLayers(g, m.Cfg.NLayer); err != nil {
				t.Fatal(err)
			}
			if in.GPULayers() != m.Cfg.NLayer {
				t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), m.Cfg.NLayer, g.Err())
			}
		}
		got, err := in.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		lg = append([]float32(nil), got...)
		in.Reset()
		if got, err = in.PrefillMixed(model.Span{Tokens: ids, Bidir: true}); err != nil {
			t.Fatal(err)
		}
		return lg, append([]float32(nil), got...)
	}
	lg, run := plain(nil)
	for i := range lg {
		if math.Float32bits(lg[i]) != math.Float32bits(run[i]) {
			nm, _ := nmseOf(lg, run)
			t.Fatalf("the CPU's unmarked prompt is not its marked run: logit %d reads %v against %v (NMSE %.3e)", i, lg[i], run[i], nm)
		}
	}
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			defer g.Close()
			band, _ := nmseOf(centredLogits(t, m, g, m.Cfg.NLayer, a.short), a.cpuShort)
			lg, _ := plain(g)
			nm, worst := nmseOf(lg, a.cpuLong)
			t.Logf("%s: an unmarked prompt of %d rows against the CPU's run: logit NMSE %.3e (max|d| %.4f); band %.3e",
				spec, len(ids), nm, worst, band)
			if math.IsNaN(band) || math.IsNaN(nm) || nm > centredBound*max(band, 1e-6) {
				t.Errorf("an unmarked prompt of %d rows on %s reads NMSE %.3e against the CPU's run, past %gx the band %.3e",
					len(ids), spec, nm, centredBound, band)
			}
		})
	}
}
