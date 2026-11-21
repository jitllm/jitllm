package engine

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/gpu"
	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// windowSpan is the rows of each window of the windowed run: a window ends
// inside a card's second piece, and so does a convolution's tap across it.
const windowSpan = 300

// segmentWindows is each of n rows' window, the span rows of its own
// segment: windows a tower's square hands as runs, here one a row.
func segmentWindows(n, span int) []nn.KeyRun {
	wins := make([]nn.KeyRun, n)
	for r := range wins {
		lo := r / span * span
		wins[r] = nn.KeyRun{Lo: lo, Hi: min(lo+span, n)}
	}
	return wins
}

// windowedRows is every row's residual after the blocks, layers of them on g
// where g is not nil, of a run over spans whose rows read wins
// (nn.RowWindowDevice), as a marker readout's run hands them
// (Instance.MarkScoresAfter): one bidirectional run, handed its rows'
// windows, every row kept. With no windows it is the prompt's prefill, its
// run handed as a key run (nn.KeyRunDevice).
func windowedRows(t *testing.T, m *model.Model, g *gpu.GPU, layers int, spans []model.Span, wins []nn.KeyRun) []float32 {
	t.Helper()
	return runRows(t, m, g, layers, spans, wins, wins != nil)
}

// runRows is windowedRows with the run named: readout runs it as a marker
// readout's run hands it, with wins or none (nil), every block of a
// non-causal decoder told the prompt is non-causal and no key run; else it
// is the prompt's prefill.
func runRows(t *testing.T, m *model.Model, g *gpu.GPU, layers int, spans []model.Span, wins []nn.KeyRun, readout bool) []float32 {
	t.Helper()
	n := model.SpanPositions(spans, m.Cfg.NEmbd)
	in := testInstance(t, m, n+1, WithKVF16(false), withCPUFloat())
	defer in.Close()
	if g != nil {
		if err := in.SetDeviceLayers(g, layers); err != nil {
			t.Fatal(err)
		}
		if in.GPULayers() != layers {
			t.Fatalf("the card took %d of %d blocks: %s", in.GPULayers(), layers, g.Err())
		}
	}
	out := make([]float32, n*m.Cfg.NEmbd)
	in.Reset()
	if !readout {
		in.rowTap = func(base int, rows []float32) { copy(out[base*m.Cfg.NEmbd:], rows) }
	} else {
		in.mark = embedMode{bidir: true, whole: true, total: n, out: out, rows: true, rowWins: wins}
		in.emb = &in.mark
	}
	_, err := in.PrefillMixed(spans...)
	in.emb, in.rowTap = nil, nil
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("row %d's residual holds %v", i/m.Cfg.NEmbd, v)
		}
	}
	return out
}

// TestCentredWindowsMatchTheCPU runs a run whose blocks read their rows'
// windows (graph.Attrs.RowWindows) and convolve their rows centred: LFM2 350M
// restated non-causal with a media prefix (jlm.Flag2MediaPrefix), its
// attention and its short convolutions windowed, each row handed its
// segment of 300 rows (segmentWindows). Its blocks are causal blocks, the
// model reading out through no marker head, so a card runs the 1150-row run
// in pieces (gpu.GPU.phased), and a piece past the first holds the end of
// one window and the start of the next. The card's rows past position 0, and
// those past its first submission, every block there and half, at the
// reserved width as placed and narrowed to 64 rows, are held to the CPU's
// within three times the band a prompt's run one submission holds, with no
// windows, reads on the same card. The windows are load-bearing: the CPU's
// rows with them stand from its rows without them past the bound many times
// over.
func TestCentredWindowsMatchTheCPU(t *testing.T) {
	m := centredModelAs(t, jlm.Flag2MediaPrefix)
	defer m.Close()
	long, short := centredPrompt(t, m)
	n, ns := len(long[0].Tokens), len(short[0].Tokens)
	w, past := m.Cfg.NEmbd, nn.MaxDevicePrefillChunk*m.Cfg.NEmbd
	wins := segmentWindows(n, windowSpan)
	cpuLong := windowedRows(t, m, nil, 0, long, wins)
	cpuShort := windowedRows(t, m, nil, 0, short, nil)
	cpuOpen := windowedRows(t, m, nil, 0, long, segmentWindows(n, n))
	open, _ := nmseOf(cpuOpen[w:], cpuLong[w:])
	openPast, _ := nmseOf(cpuOpen[past:], cpuLong[past:])
	t.Logf("the CPU's rows with the windows against its rows without them: NMSE %.3e past row 0, %.3e past a submission", open, openPast)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			band, _ := nmseOf(windowedRows(t, m, g, m.Cfg.NLayer, short, nil)[w:], cpuShort[w:])
			g.Close()
			t.Logf("%s: a prompt's run of %d rows, one submission, no windows, against the CPU's: NMSE %.3e", spec, ns, band)
			bound := centredBound * max(band, 1e-6)
			if math.IsNaN(band) {
				t.Fatalf("a run one submission holds reads NMSE %v on %s", band, spec)
			}
			if !(min(open, openPast) > 10*bound) {
				t.Fatalf("the CPU's rows stand %.3e (%.3e past a submission) from its rows with no windows, inside 10x the bound %.3e: the windows hold nothing to see",
					open, openPast, bound)
			}
			for _, rows := range []int{0, 64} {
				g := openCentredCard(t, spec, rows)
				for _, layers := range []int{m.Cfg.NLayer, m.Cfg.NLayer / 2} {
					halos, phases := haloPieces(g), runPhases(g)
					got := windowedRows(t, m, g, layers, long, wins)
					halos, phases = haloPieces(g)-halos, runPhases(g)-phases
					nm, worst := nmseOf(got[w:], cpuLong[w:])
					later, _ := nmseOf(got[past:], cpuLong[past:])
					t.Logf("%s, reserved %d, %d of %d blocks: against the CPU's, NMSE %.3e past row 0 (max|d| %.4f), %.3e past a submission; bound %.3e; %d phase calls, %d pieces with a halo",
						spec, rows, layers, m.Cfg.NLayer, nm, worst, later, bound, phases, halos)
					if halos == 0 || phases == 0 {
						t.Fatalf("%d phase calls and %d pieces with a halo: the run did not go in pieces", phases, halos)
					}
					if math.IsNaN(nm) || nm > bound || math.IsNaN(later) || later > bound {
						t.Errorf("a windowed run of %d rows on %s, reserved %d, %d blocks, reads NMSE %.3e (%.3e past a submission) against the CPU's, past %gx the band %.3e",
							n, spec, rows, layers, nm, later, centredBound, band)
					}
				}
				g.Close()
			}
		})
	}
}
