package model

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/internal/testmodels"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// swaSlice is how many logits per position a per-token run keeps: a slice is
// enough for an NMSE and keeps 712 positions of a 262,144-row head out of RAM.
const swaSlice = 4096

// swaRun prefills ids and returns logits on the device when g is non-nil: every
// position's first swaSlice logits for a per-token run, the last position's for
// a batched one (Prefill returns nothing else). It fails if the device let go
// of a block, since the State would silently finish on the host, and where the
// composer serves the blocks, unless it ran every block of every token.
func swaRun(t *testing.T, m *Model, g *tier.GPU, ids []int32, batched bool) [][]float32 {
	t.Helper()
	// One size for every arm, so no arm's result depends on the order the
	// arms ran in (TestATierServesALargerSecondSession covers that order).
	st := m.NewState(swaMaxSeq)
	defer st.Close()
	placed, before := 0, 0
	if g != nil {
		st.SetDevice(g)
		if placed = st.GPULayers(); placed == 0 {
			t.Fatalf("0 of %d blocks placed: %q", m.Cfg.NLayer, g.Err())
		}
		before = composedBlockSteps(g)
		if !batched {
			defer func() { composedBlocksRan(t, composedOf(m, g), before, len(ids)*placed, g) }()
		}
	}
	keep := func(lg []float32) []float32 { return append([]float32(nil), lg[:min(len(lg), swaSlice)]...) }
	var out [][]float32
	if batched {
		lg, err := st.Prefill(ids)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, keep(lg))
	} else {
		for _, id := range ids {
			lg, err := st.Forward(id)
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, keep(lg))
		}
	}
	if g != nil && st.GPULayers() != placed {
		t.Fatalf("after %d positions %d of the %d placed blocks are on the device (%q): the "+
			"arm measured the host", len(ids), st.GPULayers(), placed, g.Err())
	}
	return out
}

const swaMaxSeq = 1024

// swaWindow states m with a window of w keys: a config of its own, as every
// restatement of the model is, so the front end builds the blocks' graphs
// again from it (frontKey).
func swaWindow(m *Model, w int) { cfgMut(m, func(c *Config) { c.SWAWindow = w }) }

func swaNMSE(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		num, den = num+d*d, den+float64(b[i])*float64(b[i])
	}
	return num / den
}

// swaMedian is the median NMSE of two runs over the positions from `from` on
// -- or the one position a batched run has. A median because some positions
// are ill-conditioned on every engine, host and device alike.
func swaMedian(a, b [][]float32, from int) float64 {
	if len(a) == 1 {
		return swaNMSE(a[0], b[0])
	}
	var es []float64
	for i := from; i < len(a); i++ {
		es = append(es, swaNMSE(a[i], b[i]))
	}
	sort.Float64s(es)
	return es[len(es)/2]
}

// swaTier opens a device with the split tuner pinned; unpinned, it picks a
// reduction order per process and two tiers disagree with each other.
func swaTier(t *testing.T) *tier.GPU {
	t.Helper()
	o := append([]tier.Option{tier.WithDeviceTune(tier.TuneOff)}, testTierOpts(t)...)
	if testdevices.Selected() {
		o = append(o, tier.WithDevices(testdevices.One("")))
	}
	g, err := tier.OpenWith(o...)
	if err != nil || g == nil {
		noDevice(t, "device", err)
	}
	return g
}

// testTierOpts is the device options every device gate here opens its tier
// with on top of its own: JITLLM_VK_MAXGROUPS lowers the workgroups one Vulkan
// dispatch carries (tier.WithVulkanMaxGroups), so a whole model runs through
// split launches on a device whose own limit nothing reaches, and
// JITLLM_SCALAR_LINEAR_ROWS forces a hybrid's ragged linear steps onto the
// scan's one-thread form (scalarLinearRows).
func testTierOpts(t *testing.T) []tier.Option {
	t.Helper()
	var o []tier.Option
	if scalarLinearRows() {
		o = append(o, tier.WithConfig(func(c *tier.Config) { c.ScalarLinearRows = true }))
	}
	v := os.Getenv("JITLLM_VK_MAXGROUPS")
	if v == "" {
		return o
	}
	n, err := strconv.ParseUint(v, 10, 32)
	if err != nil || n == 0 {
		t.Fatalf("JITLLM_VK_MAXGROUPS=%q: want a positive count of workgroups", v)
	}
	return append(o, tier.WithVulkanMaxGroups(uint32(n)))
}

// scalarLinearRows reports JITLLM_SCALAR_LINEAR_ROWS, which puts a hybrid's
// ragged linear steps on the scan's one-thread form on a device that has the
// subgroup (tier.Config.ScalarLinearRows): how that form, which a device
// without one runs, is gated on real hardware.
func scalarLinearRows() bool { return os.Getenv("JITLLM_SCALAR_LINEAR_ROWS") != "" }

// TestSlidingWindowRunsOnTheDevice holds a device to the host past a sliding
// window, in both the per-token and the batched paths. The no-window control is
// a window too wide to bite, not a window of zero: zeroing SWAWindow also moves
// gemma3's local layers onto the global rotary base.
func TestSlidingWindowRunsOnTheDevice(t *testing.T) {
	for _, c := range []struct {
		file string
		// n is the prompt length, past the window; floor is the least the
		// device-vs-host bound may be, which otherwise follows the device's own
		// band.
		n     int
		floor float64
		// window, when set, narrows the model's own: see the second row.
		window int
	}{
		// gemma3's own window: five local layers in six, 512 keys.
		{"gemma-3-1b-it-Q4_K_M.jlm", 712, 1e-3, 0},
		// Every layer local, phi3's pattern (SWAPeriod NLayer+1), on a model
		// the device holds whole. phi3's window semantics are proved against
		// transformers on the host (synth-phi3); Phi-3.5 itself at a narrowed
		// window is chaotic on every engine, so it cannot serve here.
		{"gemma-3-1b-it-Q4_K_M.jlm", 400, 1e-3, 128},
	} {
		t.Run(fmt.Sprintf("%s/window%d", c.file, c.window), func(t *testing.T) {
			p := testmodels.Path(c.file)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
			}
			m, err := Open(p)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if c.window > 0 {
				cfgMut(m, func(cf *Config) { cf.SWAWindow, cf.SWAPeriod = c.window, cf.NLayer+1 })
			}
			w := m.Cfg.SWAWindow
			if w <= 0 || w >= c.n {
				t.Fatalf("window %d does not bite at %d positions: the gate would prove nothing", w, c.n)
			}
			ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200), true)[:c.n]

			host := swaRun(t, m, nil, ids, false)
			swaWindow(m, swaMaxSeq*4)
			open := swaRun(t, m, nil, ids, false)
			swaWindow(m, w)
			for _, batched := range []bool{false, true} {
				href, oref := host, open
				if batched {
					href, oref = host[len(host)-1:], open[len(open)-1:]
				}
				bites := swaMedian(href, oref, w)
				// The bound is this device's own band, measured with a window
				// that cannot bite, on its own tier.
				gb := swaTier(t)
				swaWindow(m, swaMaxSeq*4)
				base := swaMedian(swaRun(t, m, gb, ids, batched), oref, w)
				swaWindow(m, w)
				gb.Close()
				// At least a twentieth of what the window itself moves, because
				// base is one noisy sample and rounding amplifies over many
				// windowed layers. A device that drops the window still fails
				// by ~20x.
				bound := max(c.floor, 10*base, bites/20)
				if bites < 5*bound {
					t.Fatalf("batched=%v: the window moves the logits by only %.2e against a band "+
						"of %.2e -- too little to tell a device that applies it from one that "+
						"does not", batched, bites, bound)
				}
				g := swaTier(t)
				dev := swaRun(t, m, g, ids, batched)
				g.Close()
				d := swaMedian(dev, href, w)
				t.Logf("window %d, %d positions, batched=%v: device-vs-host %.2e (band %.2e), the window itself %.2e",
					w, c.n, batched, d, base, bites)
				if d > bound {
					t.Errorf("batched=%v: device-vs-host %.2e (bound %.2e); device-vs-no-window %.2e",
						batched, d, bound, swaMedian(dev, oref, w))
				}
			}
		})
	}
}

// TestATierServesALargerSecondSession runs a short session and then a long one
// on one tier, and holds the long one to the host. The tier must not keep the
// first session's KV ceiling or its scratch plan (a session shorter than the
// window gets no window, see windowFor). The short session stays open, so its
// cache is carried through the growth too.
func TestATierServesALargerSecondSession(t *testing.T) {
	p := testmodels.Path("gemma-3-1b-it-Q4_K_M.jlm")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("MODEL MISSING: %v (set JITLLM_MODELS to the model directory) -- this gate proved nothing", err)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	w := m.Cfg.SWAWindow
	const short, n = 400, 712
	if w <= short || w >= n {
		t.Fatalf("window %d must lie between the short session (%d) and the run (%d)", w, short, n)
	}
	ids := m.Vocab.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200), true)[:n]
	host := swaRun(t, m, nil, ids, false)
	swaWindow(m, swaMaxSeq*4)
	open := swaRun(t, m, nil, ids, false)
	swaWindow(m, w)

	g := swaTier(t)
	defer g.Close()
	a := m.NewState(short)
	defer a.Close()
	a.SetDevice(g)
	if a.GPULayers() != m.Cfg.NLayer {
		t.Fatalf("the short session placed %d of %d blocks: %q", a.GPULayers(), m.Cfg.NLayer, g.Err())
	}
	for _, id := range ids[:64] {
		if _, err := a.Forward(id); err != nil {
			t.Fatal(err)
		}
	}
	// swaRun opens the long session (swaMaxSeq positions) on the same tier and
	// fails if any placed block leaves the device during it.
	dev := swaRun(t, m, g, ids, false)
	d, bites := swaMedian(dev, host, w), swaMedian(host, open, w)
	t.Logf("second session past the window: device-vs-host %.2e, the window itself %.2e", d, bites)
	if d > 1e-3 {
		t.Errorf("device-vs-host %.2e past the window; device-vs-no-window %.2e -- the second "+
			"session ran without it", d, swaMedian(dev, open, w))
	}
	if a.GPULayers() != m.Cfg.NLayer {
		t.Errorf("the short session lost blocks to the long one: %d of %d", a.GPULayers(), m.Cfg.NLayer)
	}
}
