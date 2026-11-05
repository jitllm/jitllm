package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
)

// TestACardServesAModelAgainAfterARelease places a model on a card, gives
// it back (the instance and the model closed), and places it again, three
// times: every placement prefills the logits the first did. A card keeps one
// state for every session view of it, and a session body bound to the first
// view that ever called ran every later call on that view, its scratch and
// its session's per-layer inputs -- right while the first was placed, wrong
// once a release had let it go. synth-gemma4-e computes its per-layer inputs
// in the input rows on the card and reads them back (Stats.EmbedReadBacks,
// counted); the input rows' body bound to the first view is the violation,
// and fails it.
func TestACardServesAModelAgainAfterARelease(t *testing.T) {
	for _, name := range []string{"synth-gemma4-e.gguf"} {
		t.Run(name, func(t *testing.T) {
			path := jlmOf(t, testmodels.Path(name))
			d, err := openGPU(t, gpu.WithDeviceTune(gpu.TuneOff))
			if err != nil || d == nil {
				noDevice(t, "device", err)
			}
			defer d.Close()
			var first []float32
			for run := range 3 {
				m, err := model.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				in := m.Vocab.Encode("The capital of France is", true)
				st := testInstance(t, m, len(in)+1)
				st.SetDevice(d)
				before := d.Stats().EmbedReadBacks
				l, err := st.Prefill(in)
				if err != nil {
					t.Fatal(err)
				}
				placed, reads := st.GPULayers(), d.Stats().EmbedReadBacks-before
				l = append([]float32(nil), l...)
				st.Close()
				m.Close()
				if placed == 0 || reads == 0 {
					t.Fatalf("run %d: %d blocks on the card, %d input rows read back on it: this gate proved nothing", run, placed, reads)
				}
				if run == 0 {
					first = l
					continue
				}
				worst := 0.0
				for i := range l {
					worst = math.Max(worst, math.Abs(float64(l[i]-first[i])))
				}
				if worst > 1e-4 || argmaxOf(l) != argmaxOf(first) {
					t.Fatalf("placement %d after %d releases: argmax %d against the first's %d, logits apart by up to %.3g",
						run+1, run, argmaxOf(l), argmaxOf(first), worst)
				}
				t.Logf("placement %d after %d releases: %d blocks, the first's logits (apart by up to %.3g)", run+1, run, placed, worst)
			}
		})
	}
}
