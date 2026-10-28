package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
	"github.com/jitllm/jitllm/jit/gpu/tier"
)

// A tower's composed arm on the CPU device: every tower block the front end
// states (Uncovered) composed for the CPU (WithComposedCPU) and run over the
// picture's rows as one run attending both ways, held to main's host tower
// after ONE block (int8 activations compound past it) at the device arm's
// bound, its whole encode at the device arm's whole-tower bound
// (towerOnDevices), and its encode counted: as many blocks served as the
// tower has, none declined. A gate whose blocks all declined is no pass.

// coveredTower is how many of m's tower blocks the front end states.
func coveredTower(m *Model) int {
	tw, n := m.Tower(), 0
	for i := range tw.Cfg.NLayer {
		if m.Uncovered(tw.base+i) == Covered {
			n++
		}
	}
	return n
}

// towerEncode is s's residual after tower block 0 and its encode of px;
// the residual is nil where block 0 ends no run of s's (a device's range
// past it).
func towerEncode(t *testing.T, s *State, px []float32) (after0, out []float32) {
	t.Helper()
	s.vis.afterBlock = func(li int, x []float32) {
		if li == 0 && after0 == nil {
			after0 = append([]float32(nil), x...)
		}
	}
	e, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	return after0, append([]float32(nil), e...)
}

// towerEncodeAt is towerEncode where block 0 ends a run.
func towerEncodeAt(t *testing.T, s *State, px []float32) (after0, out []float32) {
	t.Helper()
	after0, out = towerEncode(t, s, px)
	if after0 == nil {
		t.Fatal("no residual captured after block 0")
	}
	return after0, out
}

// composedTowerOnCPU is the gate over the tower open opens, its picture px
// (nil for the tower's own synthetic one).
func composedTowerOnCPU(t *testing.T, open func(testing.TB, ...Option) (*Model, *Tower)) {
	ref, rtw := open(t, noTune)
	defer ref.Close()
	c := rtw.Cfg
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	hs := rtw.testState()
	want0, want := towerEncodeAt(t, hs, px)
	hs.Close()

	m, tw := open(t, noTune, WithComposedCPU(true))
	defer m.Close()
	if n := coveredTower(m); n != c.NLayer {
		t.Fatalf("the front end states %d of %d tower blocks (block 0: %s, %v)", n, c.NLayer,
			m.Uncovered(tw.base), m.UncoveredErr(tw.base))
	}
	s := tw.testState()
	defer s.Close()
	got0, got := towerEncodeAt(t, s, px)
	served, declined := s.CPUBlockSteps()
	t.Logf("composed CPU tower: %d block runs served, declines %v (form error: %v)", served, nonZero(declined[:]), s.CPUFormErr())
	if served != c.NLayer {
		t.Fatalf("the CPU served %d of the tower's %d blocks (declines %v, form error %v)", served, c.NLayer,
			nonZero(declined[:]), s.CPUFormErr())
	}
	nmse0, worst0 := nmseOf(got0, want0)
	nmse, worst := nmseOf(got, want)
	t.Logf("after block 0: NMSE %.3e, max|d| %.4g; the encode: NMSE %.3e, max|d| %.4g over %d rows",
		nmse0, worst0, nmse, worst, len(want)/max(c.ProjDim, 1))
	if !(nmse0 < 5e-5) {
		t.Fatalf("NMSE %.3e after ONE block (bound 5e-5): the composed CPU is not running this block's graph", nmse0)
	}
	if !(nmse < 1e-2) {
		t.Fatalf("the composed CPU's encode is NMSE %.3e from main's (bound 1e-2)", nmse)
	}

	// A warm encode on the composed CPU allocates nothing
	// (TestWarmImageEncodeDoesNotAllocate's measure).
	s.vis.afterBlock = nil
	allocs := testing.AllocsPerRun(1, func() {
		if _, err := s.Encode(px); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("a warm composed encode makes %.1f allocations", allocs)
	if allocs != 0 {
		t.Fatalf("a warm composed encode made %.1f heap allocations", allocs)
	}

	// The violation: the rows attend causally, not both ways.
	s.vis.causalFault = true
	bad0, _ := towerEncodeAt(t, s, px)
	s.vis.causalFault = false
	nb, _ := nmseOf(bad0, want0)
	t.Logf("violation (causal rows): NMSE %.3e after block 0", nb)
	if !(nb > 100*nmse0) || !(nb > 5e-4) {
		t.Fatalf("the gate cannot see causal rows: NMSE %.3e against a clean %.3e", nb, nmse0)
	}
}

// nonZero is the declines a census counted, by name.
func nonZero(d []int) map[string]int {
	out := map[string]int{}
	for i, n := range d {
		if n != 0 {
			out[jit.DeclineName(i)] = n
		}
	}
	return out
}

// qvlOpener opens the Qwen-VL fixture name (openQVL).
func qvlOpener(name string) func(testing.TB, ...Option) (*Model, *Tower) {
	return func(tb testing.TB, opts ...Option) (*Model, *Tower) {
		m, _ := openQVL(tb.(*testing.T), name, opts...)
		return m, m.Tower()
	}
}

// composedTower is a tower the composed gates run: its name and how it
// opens.
type composedTower struct {
	name string
	open func(testing.TB, ...Option) (*Model, *Tower)
}

// composedTowers is the towers the composed gates run on every device.
func composedTowers() []composedTower {
	return []composedTower{
		{"SmolVLM", openTower},
		{"llava", openLlava},
		{"Qwen2-VL", qvlOpener("synth-qwen2vl")},
		{"Qwen2.5-VL", qvlOpener("synth-qwen25vl")},
		{"gemma3", openGemma3},
	}
}

func TestComposedTowerOnCPU(t *testing.T) {
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) { composedTowerOnCPU(t, f.open) })
	}
}

// TestTowerCensus states every projector family's tower blocks through the
// front end and names what each one it states none of lacks: the work list
// of towers the composer does not serve yet. A graph built wrong
// (UncoveredUnverified) fails it.
func TestTowerCensus(t *testing.T) {
	for _, f := range towerFamilies() {
		t.Run(f.name, func(t *testing.T) {
			m, tw := f.open(t)
			defer m.Close()
			why := map[string]int{}
			for i := range tw.Cfg.NLayer {
				u := m.Uncovered(tw.base + i)
				if u == UncoveredUnverified {
					t.Errorf("tower block %d's graph does not verify: %v", i, m.UncoveredErr(tw.base+i))
				}
				k := u.String()
				if err := m.UncoveredErr(tw.base + i); err != nil {
					k += ": " + err.Error()
				}
				why[k]++
			}
			t.Logf("%s (%s): %d tower blocks, %v", f.name, tw.Cfg.Projector, tw.Cfg.NLayer, why)
		})
	}
}

// towerParts is what a tower gate's composed arm asks of a tier: blocks,
// the head and the tower's blocks (tier.ComposedTower).
const towerParts = composedParts | tier.ComposedTower

// composedTowerOnDevices is the device arm of the composed tower: on each
// device, a tier asked for towers (towerParts), every tower block placed
// and served by the composer -- as many served block runs as the tower has
// blocks, no decline -- the whole encode held to main's host at the device
// arm's whole-tower bound (towerOnDevices), and one block placed, held to the
// host after it at the one-block bound.
func composedTowerOnDevices(t *testing.T, open func(testing.TB, ...Option) (*Model, *Tower)) {
	m, tw := open(t, noTune)
	defer m.Close()
	c := tw.Cfg
	if n := coveredTower(m); n != c.NLayer {
		t.Fatalf("the front end states %d of %d tower blocks (block 0: %s, %v)", n, c.NLayer,
			m.Uncovered(tw.base), m.UncoveredErr(tw.base))
	}
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255)/127.5 - 1
	}
	hs := tw.testState()
	want0, want := towerEncodeAt(t, hs, px)
	hs.Close()
	ran := 0
	for _, spec := range stepDevices() {
		g, err := tier.OpenWith(tier.WithDevices(spec), composedOpt(towerParts))
		if err != nil || g == nil {
			t.Logf("%s: not present (%v)", spec, err)
			continue
		}
		ran++
		t.Run(spec, func(t *testing.T) {
			defer g.Close()
			all := tw.testState()
			defer all.Close()
			all.SetDevice(g)
			if n := all.GPUBlocks(); n != c.NLayer {
				t.Fatalf("%s took %d of %d tower blocks (%s)", spec, n, c.NLayer, g.Err())
			}
			s0, _, _ := g.Stats().ServedPart(tier.ComposedTower)
			_, got := towerEncode(t, all, px)
			st := g.Stats()
			served, declined, unasked := st.ServedPart(tier.ComposedTower)
			t.Logf("%s: %d tower block runs served by the composer, %d declined %v, %d unasked (%s)", spec,
				served-s0, declined, nonZero(st.BlockDeclines[:]), unasked, g.Err())
			if served-s0 != c.NLayer || declined != 0 {
				t.Fatalf("%s: the composer served %d of the tower's %d blocks, declined %d %v (%s)", spec,
					served-s0, c.NLayer, declined, nonZero(st.BlockDeclines[:]), g.Err())
			}
			nm, worst := nmseOf(got, want)
			t.Logf("%s, all %d blocks composed: projector output NMSE %.3e, max|d| %.4g", spec, c.NLayer, nm, worst)
			if !(nm < 1e-2) {
				t.Errorf("%s: the composed tower's encode reads NMSE %.3e against the host (bound 1e-2)", spec, nm)
			}

			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			one := tw.testState()
			defer one.Close()
			one.SetDevice(g)
			if one.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", one.GPUBlocks())
			}
			got0, _ := towerEncodeAt(t, one, px)
			nm0, worst0 := nmseOf(got0, want0)
			bound := 5e-5
			if st := g.Stats(); st.VoltaGemm+st.GemmF16 > 0 || spec == "metal" {
				bound = 5e-4
			}
			t.Logf("%s, after block 0 composed: NMSE %.3e, max|d| %.4g (bound %.0e; tower chunk %d, %d phase calls)", spec, nm0, worst0, bound,
				g.TowerChunk(), g.Stats().TowerPhases)
			if !(nm0 < bound) {
				t.Fatalf("%s: NMSE %.3e after ONE composed block (bound %.0e)", spec, nm0, bound)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host: the gate proved nothing")
	}
}

// TestComposedTowerOnDevice runs the device arm over the towers the tier
// serves (tier.ComposedTower), Gemma 3's SigLIP among them: its 4096 rows
// of 1152 are the flattened space whose live elements once wrapped 32 bits
// (jit.TestComposedGPUAttendBidirRows).
func TestComposedTowerOnDevice(t *testing.T) {
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) { composedTowerOnDevices(t, f.open) })
	}
}
