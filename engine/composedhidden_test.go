package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/nn"
)

// composedHiddenRows is the rows a speculation step of the hidden gates
// runs after the prompt: a verification's handful.
const composedHiddenRows = 6

// TestComposedHeadWritesItsHidden runs speculation steps, which read the
// trunk's normed rows back with the logits (nn.Head.Hidden), with every
// block on each device and the composer serving the blocks and the head: a
// verification's every row as ragged rows, one row as decode, and a
// prompt chunk's folded last row. The head the composer serves writes its
// normed rows where the readback reads them (Stats.HiddenHeads and
// HiddenRows count it), and each row is the tier's own head's at its
// position: nearer it, by ten times, than to its neighbour's.
func TestComposedHeadWritesItsHidden(t *testing.T) {
	m := composedRowsModel(t)
	defer m.Close()
	for _, dev := range stepDevices() {
		t.Run(dev, func(t *testing.T) {
			want, _ := composedHidden(t, m, nil, dev, 0, composedHiddenRows, false)
			for _, a := range []struct {
				name  string
				parts nn.Parts
				k     int
				fold  bool
				at    []int
			}{
				{"rows", composedParts, composedHiddenRows, false, []int{0, 1, 2, 3, 4, 5}},
				{"decode", composedParts, 1, false, []int{0}},
				{"fold", composedParts | nn.ComposedPrompt, composedHiddenRows, true, []int{composedHiddenRows - 1}},
			} {
				t.Run(a.name, func(t *testing.T) {
					got, st := composedHidden(t, m, nil, dev, a.parts, a.k, a.fold)
					if st.HiddenHeads == 0 || st.HiddenRows < len(a.at) {
						t.Fatalf("the composer served %d heads whose %d normed rows were read back, where the step read %d",
							st.HiddenHeads, st.HiddenRows, len(a.at))
					}
					if hs, hd := st.ServedPart(nn.ComposedHead); hs == 0 || hd != 0 {
						t.Fatalf("the composer served the head %d times and declined it %d (%s)", hs, hd, declinesOf(st))
					}
					worst, off := composedHiddenGap(t, m, got, want, a.at)
					if !(off > max(1e-9, 10*worst)) {
						t.Fatalf("the served head's normed rows sit %.3e from the tier's own, and %.3e from a neighbour's: "+
							"they are not the rows it normed", worst, off)
					}
					t.Logf("%d head(s) served with %d normed rows read back: worst NMSE %.3e from the tier's own, %.3e from a neighbour's",
						st.HiddenHeads, st.HiddenRows, worst, off)
				})
			}
		})
	}
}

// composedHidden prefills a prompt with every block and the head on device
// dev, the composer asked for parts, and runs one speculation step of k rows
// after it: every row wanted, or with fold a prompt chunk wanting its last.
// It returns the normed rows read back and the tier's stats.
func composedHidden(t *testing.T, m *model.Model, io []Option, dev string, parts nn.Parts, k int, fold bool, extra ...gpu.Option) ([]float32, gpu.Stats) {
	t.Helper()
	g, err := openGPU(t, append([]gpu.Option{gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.PromptRows = 32 })}, extra...)...)
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	c := m.Cfg
	prompt := m.Vocab.Encode("The capital of France is", true)
	rows := m.Vocab.Encode(" Paris, and the capital of Spain is Madrid", false)[:k]
	s := testInstance(t, m, len(prompt)+len(rows)+4, io...)
	defer s.Close()
	s.SetDeviceFallback(false)
	if err := s.SetDevice(g); err != nil {
		t.Fatal(err)
	}
	if s.GPULayers() != c.NLayer || !s.HeadOnDevice() {
		t.Fatalf("placed %d of %d blocks, head %v: %s", s.GPULayers(), c.NLayer, s.HeadOnDevice(), g.Err())
	}
	if _, err := s.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	out := rowsOut{logits: make([]float32, k*c.NVocab), hidden: make([]float32, k*c.NEmbd), prompt: fold}
	if fold {
		out.from = k - 1
	}
	func() {
		defer m.EnterPager()()
		s.kv.note(s.seqPos(0), rows...)
		if err := s.stepRows(tokenRows(rows), &out); err != nil {
			t.Fatal(err)
		}
	}()
	return out.hidden[:(k-out.from)*c.NEmbd], g.Stats()
}

// composedHiddenGap is the worst NMSE of got's rows from want's at the
// positions at, and the least from a neighbouring position's.
func composedHiddenGap(t *testing.T, m *model.Model, got, want []float32, at []int) (worst, off float64) {
	t.Helper()
	d := m.Cfg.NEmbd
	row := func(v []float32, i int) []float32 { return v[i*d : (i+1)*d] }
	off = math.Inf(1)
	for j, i := range at {
		e := logitNMSE(row(got, j), row(want, i))
		if math.IsNaN(e) || math.IsInf(e, 0) {
			t.Fatalf("row %d: the normed row is not finite against the tier's (%v)", i, e)
		}
		worst = max(worst, e)
		for _, n := range []int{i - 1, i + 1} {
			if n >= 0 && n < len(want)/d {
				off = min(off, logitNMSE(row(got, j), row(want, n)))
			}
		}
	}
	return worst, off
}
