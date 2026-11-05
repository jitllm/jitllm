//go:build jitllmfault

package engine

import (
	"math"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/model"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/engine/nn"
)

// TestComposedRowsDiscriminate runs the composed rows' gates under each
// violation of the rows the composer serves and demands each fires (RULE
// 10): a ragged row stating half its position ("pos") and every row reading the
// first row's pages ("alias") must part a row from the host past a tie, as
// must a per-row head that projects the first wanted row alone ("headrow"),
// whose other rows' logits are another step's; and a folded head reading the
// chunk's first row ("foldrow") must part from the tail call
// TestPrefillFoldsTheHead holds it to.
func TestComposedRowsDiscriminate(t *testing.T) {
	m := composedRowsModel(t)
	defer m.Close()
	ids := make([][]int32, len(composedRowsPrompts()))
	for i, p := range composedRowsPrompts() {
		ids[i] = m.Vocab.Encode(p, true)
	}
	const gen = 12
	want := make([][]int32, len(ids))
	for i := range ids {
		want[i] = hostGreedy(t, m, ids[i], gen)
	}
	for _, dev := range stepDevices() {
		for _, f := range []struct {
			name string
			arm  func(*gpu.Faults, string)
		}{{"pos", (*gpu.Faults).SetPagedFault}, {"alias", (*gpu.Faults).SetPagedFault}, {"headrow", (*gpu.Faults).SetRowsFault}} {
			t.Run(dev+"/"+f.name, func(t *testing.T) {
				var faults gpu.Faults
				f.arm(&faults, f.name)
				got, _, stats := batchGreedyStats(t, m, ids, gen, 256, []gpu.Option{gpu.WithDevices(dev), gpu.WithFaults(&faults)})
				composedRowsServed(t, m, stats)
				i, j, gap := composedRowsPart(t, m, ids, want, got)
				if i < 0 {
					t.Fatalf("under %q every row matched the host to a tie: the gate cannot see it", f.name)
				}
				t.Logf("%q: row %d parts from the host at token %d, %.3f apart -- caught", f.name, i, j, gap)
			})
		}
		t.Run(dev+"/foldrow", func(t *testing.T) {
			composedFoldParts(t, m, dev)
		})
		t.Run(dev+"/nohidden", func(t *testing.T) {
			want, _ := composedHidden(t, m, nil, dev, 0, composedHiddenRows, false)
			var faults gpu.Faults
			faults.SetRowsFault("nohidden")
			got, st := composedHidden(t, m, nil, dev, composedParts, composedHiddenRows, false, gpu.WithFaults(&faults))
			if st.HiddenHeads == 0 {
				t.Fatalf("no served head's normed rows were read back: the arm did not select the served head")
			}
			at := []int{0, 1, 2, 3, 4, 5}
			d := m.Cfg.NEmbd
			worst, off := 0.0, math.Inf(1)
			for j, i := range at {
				e := logitNMSE(got[j*d:(j+1)*d], want[i*d:(i+1)*d])
				if math.IsNaN(e) || math.IsInf(e, 0) {
					worst = math.Inf(1)
					break
				}
				worst = max(worst, e)
				if i > 0 {
					off = min(off, logitNMSE(got[j*d:(j+1)*d], want[(i-1)*d:i*d]))
				}
			}
			if off > max(1e-9, 10*worst) {
				t.Fatalf("under \"nohidden\" the normed rows sit %.3e from the tier's and %.3e from a neighbour's: the gate cannot see it",
					worst, off)
			}
			t.Logf("\"nohidden\": the normed rows read back sit %.3e from the tier's -- caught", worst)
		})
	}
}

// composedFoldParts prefills a prompt that ends partway into a chunk with
// every block on device dev and the composer serving the prompt and the
// head, folded into the last chunk and as the tail call, under "foldrow",
// and demands the logits part.
func composedFoldParts(t *testing.T, m *model.Model, dev string, io ...Option) {
	// Chunks of at most 32 rows: the device reserves and warms the widths up
	// to 32, each a form of the composer's.
	var faults gpu.Faults
	g, err := openGPU(t, gpu.WithDevices(dev), gpu.WithDeviceTune(gpu.TuneOff),
		gpu.WithConfig(func(c *gpu.Config) { c.PromptRows = 32 }), gpu.WithFaults(&faults))
	if err != nil || g == nil {
		noDevice(t, dev, err)
	}
	defer g.Close()
	text := strings.Repeat("The river rose every spring until the bridges were islands, "+
		"and the clerk counted barrels of salt on the upper floor. ", 8)
	ids := m.Vocab.Encode(text, true)[:40]
	run := func(fold bool) ([]float32, int) {
		st := testInstance(t, m, len(ids)+4, io...)
		defer st.Close()
		st.SetDeviceFallback(false)
		if err := st.SetDevice(g); err != nil {
			t.Fatal(err)
		}
		if st.GPULayers() != m.Cfg.NLayer || !st.HeadOnDevice() {
			t.Fatalf("placed %d of %d blocks, head %v: %s", st.GPULayers(), m.Cfg.NLayer, st.HeadOnDevice(), g.Err())
		}
		st.noHeadFold = !fold
		before := g.Stats().HeadFolds
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		return append([]float32(nil), lg...), g.Stats().HeadFolds - before
	}
	want, _ := run(false)
	faults.SetRowsFault("foldrow")
	got, folds := run(true)
	faults.SetRowsFault("")
	if folds == 0 {
		t.Fatalf("no head folded: the arm did not select the fold")
	}
	if hs, hd := g.Stats().ServedPart(nn.ComposedHead); hs == 0 || hd != 0 {
		t.Fatalf("the composer served the head %d times and declined it %d (%s)", hs, hd, declinesOf(g.Stats()))
	}
	nmse := logitNMSE(got, want)
	if math.IsNaN(nmse) || !(nmse > 1e-10) {
		t.Fatalf("under \"foldrow\" the folded head sits %.3e from the tail call: the fold gate cannot see it", nmse)
	}
	t.Logf("\"foldrow\": the folded head parts from the tail call by NMSE %.3e -- caught", nmse)
}
