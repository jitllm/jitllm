package engine

import (
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/model"
	"github.com/jitllm/jitllm/nn"
)

// mixedWindowsModel is centredModelAs(jlm.Flag2MediaPrefix) with blocks 2
// and 3 of every four restated to read no windows: their attention and
// their centred convolutions read their rows' run, the other blocks' their
// rows' windows (graph.Attrs.RowWindows). Of LFM2 350M's blocks, attention
// and convolution blocks both fall on each side, three attentions a side.
func mixedWindowsModel(t *testing.T) *model.Model {
	t.Helper()
	src := testmodels.Path("lfm2/LFM2-350M-Q4_K_M.gguf")
	if !sourcePresent(src) {
		testmodels.Missing(t, "MODEL MISSING: %s (see docs/testing.md) -- this gate proved nothing", src)
	}
	path := filepath.Join(t.TempDir(), "mixed"+model.Ext)
	if err := convertStated(src, path, func(s *jlm.Source) {
		s.Config.Flags |= jlm.FlagNonCausal
		s.Config.Flags2 |= jlm.Flag2MediaPrefix
	}); err != nil {
		t.Fatal(err)
	}
	m, err := model.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	restated := map[int]string{}
	for li := range m.Cfg.NLayer {
		if li%4 < 2 {
			continue
		}
		g := cloned(m.Composing().BlockGraph(li))
		for i := range g.Ops {
			g.Ops[i].Attr.RowWindows = false
		}
		b, err := graph.Encode(g)
		if err != nil {
			m.Close()
			t.Fatal(err)
		}
		restated[li] = string(b)
	}
	m.Close()
	if _, err := jlm.Restate(path, func(f *jlm.File, c *jlm.Config) (bool, error) {
		c.Blocks.Graphs, c.Blocks.Of = slices.Clone(c.Blocks.Graphs), slices.Clone(c.Blocks.Of)
		for li, b := range restated {
			c.Blocks.Graphs = append(c.Blocks.Graphs, b)
			c.Blocks.Of[li] = uint16(len(c.Blocks.Graphs) - 1)
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}
	if m, err = model.Open(path); err != nil {
		t.Fatal(err)
	}
	kinds := map[[2]bool]int{}
	for li, g := range m.FrontGraphs().Blocks {
		for _, o := range g.Ops {
			if o.Kind == graph.OpAttend || o.Kind == graph.OpRowConv {
				kinds[[2]bool{o.Kind == graph.OpAttend, statesWindows(g)}]++
				if statesWindows(g) != (li%4 < 2) {
					m.Close()
					t.Fatalf("block %d reads windows %v: the restatement did not take", li, statesWindows(g))
				}
			}
		}
	}
	for _, k := range [][2]bool{{true, true}, {true, false}, {false, true}, {false, false}} {
		if kinds[k] == 0 {
			m.Close()
			t.Fatalf("no block attends %v and reads windows %v (%v): the gate would mix nothing", k[0], k[1], kinds)
		}
	}
	return m
}

// statesWindows says an op of g reads its rows' windows
// (graph.Attrs.RowWindows).
func statesWindows(g *graph.Graph) bool {
	return slices.ContainsFunc(g.Ops, func(o graph.Op) bool { return o.Attr.RowWindows })
}

// mixedSpan is the rows of each window of the mixed model's runs: narrow, so
// a block reading its window parts from one reading its run past what a
// card's band hides.
const mixedSpan = 64

// TestMixedWindowsMatchTheCPU runs a model some of whose blocks read their
// rows' windows and the rest not (mixedWindowsModel), each row
// handed its segment of 64 rows (segmentWindows) as a marker readout hands
// them: a block reads the windows only where its graph states them
// (nn.RowWindowDevice), on the CPU device and on a card alike. The card runs
// the 1150-row run in pieces (gpu.GPU.phased), a block at a time, and the
// 400-row run in one submission, where one call holds blocks of both
// kinds; every block there and half, at the
// reserved width as placed and narrowed to 64 rows. The card's rows past
// position 0 are held to the CPU's within three times the band a prompt's
// run one submission holds reads on the same card. The mix is load-bearing:
// the CPU's rows stand past the bound both from those of the model whose
// every block reads the windows and from its own with no windows.
func TestMixedWindowsMatchTheCPU(t *testing.T) {
	m := mixedWindowsModel(t)
	defer m.Close()
	long, short := centredPrompt(t, m)
	n, ns := len(long[0].Tokens), len(short[0].Tokens)
	w := m.Cfg.NEmbd
	wins, shortWins := segmentWindows(n, mixedSpan), segmentWindows(ns, mixedSpan)
	cpuLong := windowedRows(t, m, nil, 0, long, wins)
	cpuShort := windowedRows(t, m, nil, 0, short, shortWins)
	cpuPrompt := windowedRows(t, m, nil, 0, short, nil)
	every := centredModelAs(t, jlm.Flag2MediaPrefix)
	standEvery, _ := nmseOf(windowedRows(t, every, nil, 0, long, wins)[w:], cpuLong[w:])
	every.Close()
	standNone, _ := nmseOf(windowedRows(t, m, nil, 0, long, nil)[w:], cpuLong[w:])
	stand := min(standEvery, standNone)
	t.Logf("the CPU's rows against those of the model whose every block reads the windows: NMSE %.3e past row 0; against its own with no windows: %.3e",
		standEvery, standNone)
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			band, _ := nmseOf(windowedRows(t, m, g, m.Cfg.NLayer, short, nil)[w:], cpuPrompt[w:])
			g.Close()
			bound := centredBound * max(band, 1e-6)
			t.Logf("%s: a prompt's run of %d rows, one submission, against the CPU's: NMSE %.3e", spec, ns, band)
			if math.IsNaN(band) {
				t.Fatalf("a run one submission holds reads NMSE %v on %s", band, spec)
			}
			if !(stand > bound) {
				t.Fatalf("the mixed model's rows stand %.3e from those of a model whose every block reads the windows or none, inside the bound %.3e: the mix holds nothing to see",
					stand, bound)
			}
			for _, rows := range []int{0, 64} {
				g := openCentredCard(t, spec, rows)
				for _, layers := range []int{m.Cfg.NLayer, m.Cfg.NLayer / 2} {
					for _, arm := range []struct {
						spans []model.Span
						wins  []nn.KeyRun
						cpu   []float32
					}{{long, wins, cpuLong}, {short, shortWins, cpuShort}} {
						phases := runPhases(g)
						got := windowedRows(t, m, g, layers, arm.spans, arm.wins)
						phases = runPhases(g) - phases
						nm, worst := nmseOf(got[w:], arm.cpu[w:])
						rn := len(arm.spans[0].Tokens)
						t.Logf("%s, reserved %d, %d of %d blocks, %d rows: against the CPU's, NMSE %.3e (max|d| %.4f); bound %.3e; %d phase calls",
							spec, rows, layers, m.Cfg.NLayer, rn, nm, worst, bound, phases)
						if rn == n && phases == 0 {
							t.Fatalf("the %d-row run made no phase call: it did not go in pieces", n)
						}
						if math.IsNaN(nm) || nm > bound {
							t.Errorf("a mixed run of %d rows on %s, reserved %d, %d blocks, reads NMSE %.3e against the CPU's, past %gx the band %.3e",
								rn, spec, rows, layers, nm, centredBound, band)
						}
					}
				}
				g.Close()
			}
		})
	}
}

// TestUnwindowedRunMatchesTheCPU reads out a non-causal decoder no block of
// which states windows (centredModel) over one bidirectional run of 1150
// rows, as a marker readout hands it to its devices (runRows): with no
// windows, where every row reads the whole run, and with windows of 300
// rows, which no block reads. On the CPU both are the prompt's own run: the
// same rows as the prompt's prefill, the windows changing nothing. On each
// card -- every block there and half, at the reserved width as placed and
// narrowed to 64 rows -- both are held to the CPU's within three times the
// band a prompt's run one submission holds reads on the same card.
func TestUnwindowedRunMatchesTheCPU(t *testing.T) {
	m := centredModel(t)
	defer m.Close()
	long, short := centredPrompt(t, m)
	n, ns := len(long[0].Tokens), len(short[0].Tokens)
	w := m.Cfg.NEmbd
	wins := segmentWindows(n, windowSpan)
	prompt := windowedRows(t, m, nil, 0, long, nil)
	none := runRows(t, m, nil, 0, long, nil, true)
	windowed := runRows(t, m, nil, 0, long, wins, true)
	cpuPrompt := windowedRows(t, m, nil, 0, short, nil)
	asPrompt, _ := nmseOf(none[w:], prompt[w:])
	unread, _ := nmseOf(windowed[w:], none[w:])
	t.Logf("the CPU's readout against its prompt's run: NMSE %.3e; with windows no block reads against none: NMSE %.3e", asPrompt, unread)
	if !(asPrompt <= 1e-12) {
		t.Errorf("the CPU's readout with no windows stands %.3e from its prompt's run of the same rows", asPrompt)
	}
	if !slices.Equal(windowed, none) {
		t.Errorf("windows no block states move the CPU's rows by NMSE %.3e: a block that states none read them", unread)
	}
	for _, spec := range stepDevices() {
		t.Run(spec, func(t *testing.T) {
			g := openCentredCard(t, spec, 0)
			band, _ := nmseOf(windowedRows(t, m, g, m.Cfg.NLayer, short, nil)[w:], cpuPrompt[w:])
			g.Close()
			bound := centredBound * max(band, 1e-6)
			t.Logf("%s: a prompt's run of %d rows, one submission, against the CPU's: NMSE %.3e", spec, ns, band)
			if math.IsNaN(band) {
				t.Fatalf("a run one submission holds reads NMSE %v on %s", band, spec)
			}
			for _, rows := range []int{0, 64} {
				g := openCentredCard(t, spec, rows)
				for _, layers := range []int{m.Cfg.NLayer, m.Cfg.NLayer / 2} {
					for _, arm := range []struct {
						name string
						wins []nn.KeyRun
					}{{"no windows", nil}, {"windows no block reads", wins}} {
						phases := runPhases(g)
						got := runRows(t, m, g, layers, long, arm.wins, true)
						phases = runPhases(g) - phases
						nm, worst := nmseOf(got[w:], none[w:])
						t.Logf("%s, reserved %d, %d of %d blocks, %s: against the CPU's, NMSE %.3e (max|d| %.4f); bound %.3e; %d phase calls",
							spec, rows, layers, m.Cfg.NLayer, arm.name, nm, worst, bound, phases)
						if phases == 0 {
							t.Fatalf("the %d-row readout made no phase call: it did not go in pieces", n)
						}
						if math.IsNaN(nm) || nm > bound {
							t.Errorf("a readout of %d rows on %s, reserved %d, %d blocks, %s, reads NMSE %.3e against the CPU's, past %gx the band %.3e",
								n, spec, rows, layers, arm.name, nm, centredBound, band)
						}
					}
				}
				g.Close()
			}
		})
	}
}
