package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
	"github.com/jitllm/jitllm/format/jlm"

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
	ps, pd := s.CPUProjectorSteps()
	pwhy, perr := s.CPUProjectorUncovered()
	t.Logf("composed CPU projector: %d pictures served, declines %v (%s %v)", ps, nonZero(pd[:]), pwhy, perr)
	if ps != 1 {
		t.Fatalf("the CPU's composed projector served %d of the encode's 1 picture (declines %v; %s %v)", ps, nonZero(pd[:]), pwhy, perr)
	}
	es, ed := s.CPUEntrySteps()
	ewhy, eerr := s.CPUEntryUncovered()
	t.Logf("composed CPU entry: %d pictures served, declines %v (%s %v)", es, nonZero(ed[:]), ewhy, eerr)
	if es != 1 {
		t.Fatalf("the CPU's composed entry served %d of the encode's 1 picture (declines %v; %s %v)", es, nonZero(ed[:]), ewhy, eerr)
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

	// A warm encode on the composed CPU allocates nothing: the engine's
	// allocations, told from the runtime's per-P caches as the decode gate
	// tells them (countAllocs). A process-wide count read 0 to 23 for the
	// same encode, from one run to the next.
	s.vis.afterBlock = nil
	encode := func() {
		if _, err := s.Encode(px); err != nil {
			t.Fatal(err)
		}
	}
	encode()
	r0 := m.PagerReads()
	allocVerdict(t, "a warm composed encode", countAllocs(encode), 1, m.PagerReads()-r0, 0)

	// The violation of the entry: each patch row gathered from the next patch.
	s.cpu.fault = cpuFaultEntryRows
	bade, _ := towerEncodeAt(t, s, px)
	s.cpu.fault = cpuFaultNone
	ne, _ := nmseOf(bade, want0)
	t.Logf("violation (each row the next patch): NMSE %.3e after block 0", ne)
	if !(ne > 100*nmse0) || !(ne > 5e-4) {
		t.Fatalf("the gate cannot see the entry's rows misplaced: NMSE %.3e against a clean %.3e", ne, nmse0)
	}

	// The violation of the run's first pass: each call's rows read only the
	// keys written before them, where the picture takes several calls.
	if rows := len(want0) / c.NEmbd; rows > cpuRowsCap {
		// Held to the clean composed encode, every block's: a windowed
		// block whose windows end at a call's rows has nothing to violate.
		s.cpu.fault = cpuFaultKeysFirst
		_, bad := towerEncodeAt(t, s, px)
		s.cpu.fault = cpuFaultNone
		nb, _ := nmseOf(bad, got)
		t.Logf("violation (no first pass over %d rows): NMSE %.3e from the clean composed encode", rows, nb)
		if !(nb > 5e-4) {
			t.Fatalf("the gate cannot see a run without its first pass: NMSE %.3e from the clean encode", nb)
		}
	} else {
		t.Logf("%d rows run in one call: no first pass to violate", rows)
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

// towerOf is open's model and its tower, for an opener that returns the
// model beside its golden.
func towerOf[G any](open func(testing.TB, ...Option) (*Model, G)) func(testing.TB, ...Option) (*Model, *Tower) {
	return func(tb testing.TB, opts ...Option) (*Model, *Tower) {
		m, _ := open(tb, opts...)
		return m, m.Tower()
	}
}

// towerOfModel is open's model and its tower.
func towerOfModel(open func(testing.TB, ...Option) *Model) func(testing.TB, ...Option) (*Model, *Tower) {
	return func(tb testing.TB, opts ...Option) (*Model, *Tower) {
		m := open(tb, opts...)
		return m, m.Tower()
	}
}

// composedTower is a tower the composed gates run: its name, how it opens,
// and the bound a whole encode of the gate's picture on a device is held to
// against the host's: its family's device gate's where that gate holds the
// tier's encode against the host's (main's tier is held to the same), 1e-2
// where it holds it against transformers' at towerBound on its own picture.
type composedTower struct {
	name  string
	open  func(testing.TB, ...Option) (*Model, *Tower)
	bound float64
}

// composedTowers is the towers the composed gates run on every device.
func composedTowers() []composedTower {
	return []composedTower{
		{"SmolVLM", openTower, 1e-2},
		{"llava", openLlava, 1e-2},
		{"Qwen2-VL", qvlOpener("synth-qwen2vl"), 1e-2},
		{"Qwen2.5-VL", qvlOpener("synth-qwen25vl"), 1e-2},
		{"gemma3", openGemma3, 1e-2},
		{"Qwen3-VL", qvlOpener("synth-qwen3vl"), 1e-2},
		{"Qwen3.5", qvlOpener("synth-qwen35vl"), 1e-2},
		{"GLM-4.1V", qvlOpener("synth-glm4v"), 1e-2},
		{"InternVL", openInternVL, 1e-2},
		{"Janus", towerOfModel(openJanus), 2e-2},
		{"MiniCPM-V", towerOfModel(openMiniCPMV), 1e-2},
		{"Pixtral", towerOf(openPixtral), 1e-3},
		{"Llama4", towerOf(openLlama4V), 1e-3},
		{"Phi-4", towerOf(openPhi4VFixture), 1e-3},
		{"Kimi-VL", qvlOpener("synth-kimivl"), 1e-2},
		{"HunyuanVL", qvlOpener("synth-hunyuanvl"), 1e-2},
		{"Gemma4", towerOf(openGemma4V), 1e-3},
		{"LFM2-VL", towerOfModel(openD1VL), 1e-2},
	}
}

func TestComposedTowerOnCPU(t *testing.T) {
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) { composedTowerOnCPU(t, f.open) })
	}
}

// TestTowerCensus states every projector family's tower blocks, its entry
// and its projector through the front end and names what each one it
// states none of lacks: the work list
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
			_, ew, ee := m.towerEntryOf()
			_, pw, pe := m.towerProjOf(false)
			for _, u := range []Uncovered{ew, pw} {
				if u == UncoveredUnverified {
					t.Errorf("the entry's or the projector's graphs do not verify: %v %v", ee, pe)
				}
			}
			t.Logf("%s: entry %s %v; projector %s %v", f.name, ew, ee, pw, pe)
		})
	}
}

// towerParts is what a tower gate's composed arm asks of a tier: blocks,
// the head, the tower's blocks (tier.ComposedTower) and its entry and
// projector (tier.ComposedRows).
const towerParts = composedParts | tier.ComposedTower | tier.ComposedRows

// composedTowerOnDevices is the device arm of the composed tower: on each
// device, a tier asked for towers (towerParts), every tower block placed
// and served by the composer -- as many served block runs as the tower has
// blocks, no decline -- the whole encode held to main's host at the device
// arm's whole-tower bound (towerOnDevices), and one block placed, held to the
// host after it at the one-block bound.
func composedTowerOnDevices(t *testing.T, open func(testing.TB, ...Option) (*Model, *Tower), bound float64) {
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
			t.Logf("%s, all %d blocks composed: projector output NMSE %.3e, max|d| %.4g; the tier's own kernels on the same picture %.3e",
				spec, c.NLayer, nm, worst, tierTowerNMSE(t, tw, spec, px, want))
			if !(nm < bound) {
				t.Errorf("%s: the composed tower's encode reads NMSE %.3e against the host (bound %.0e)", spec, nm, bound)
			}
			ce, cp, miss := all.CardRowsSteps()
			rs, rd, _ := g.Stats().ServedPart(tier.ComposedRows)
			t.Logf("%s: the card served the entry of %d pictures and the projector of %d, %d parts missed; %d picture-row parts served, %d declined (%s)",
				spec, ce, cp, miss, rs, rd, g.Err())
			if ce != 1 || cp != 1 || miss != 0 {
				t.Fatalf("%s: the card served the entry of %d and the projector of %d of 1 picture, %d parts missed (%s)", spec, ce, cp, miss, g.Err())
			}
			// The violations of the card's picture rows, each against the clean
			// composed encode: a patch row gathered from the next patch, then
			// the projector's own op.
			vio := func(name string, set func(s *State), unset func()) {
				bs := tw.testState()
				defer bs.Close()
				bs.SetDevice(g)
				set(bs)
				_, bad := towerEncode(t, bs, px)
				unset()
				be, bp, _ := bs.CardRowsSteps()
				nb, _ := nmseOf(bad, got)
				t.Logf("%s: violation (%s): NMSE %.3e from the clean encode (card entry %d, projector %d)", spec, name, nb, be, bp)
				if be != 1 || bp != 1 || !(nb > 1e-4) {
					t.Fatalf("%s: the gate cannot see %s on the card: NMSE %.3e (entry %d, projector %d)", spec, name, nb, be, bp)
				}
			}
			vio("each row the next patch", func(s *State) { s.cpu.fault = cpuFaultEntryRows }, func() {})
			switch {
			case c.WinPattern > 0:
				vio("the merged rows left in the window order", func(*State) { qwenNoUnwindow = true }, func() { qwenNoUnwindow = false })
			case c.Scale > 1 && c.projK() == c.NEmbd:
				tw.Cfg.fault = towerFaultPoolCorner
				vio("the pool reads a group's first patch alone", func(*State) {}, func() { tw.Cfg.fault = towerFaultNone })
			case c.Scale > 1:
				vio("the regroup's slots swapped", func(*State) { tw.Cfg.fault = towerFaultShuffleSwap }, func() { tw.Cfg.fault = towerFaultNone })
			}

			defer func(was int) { tw.opt.towerGPULayers = was }(tw.opt.towerGPULayers)
			tw.opt.towerGPULayers = 1
			one := tw.testState()
			defer one.Close()
			one.SetDevice(g)
			if one.GPUBlocks() != 1 {
				t.Fatalf("asked for 1 device block, got %d", one.GPUBlocks())
			}
			// The block's binary16 matrix forms -- the composer's, as main's
			// binary16 GEMMs on its own tier -- take the binary16 bound.
			mx := g.Stats().MatrixPart(tier.ComposedTower)
			got0, _ := towerEncodeAt(t, one, px)
			mx = g.Stats().MatrixPart(tier.ComposedTower) - mx
			nm0, worst0 := nmseOf(got0, want0)
			bound0 := 5e-5
			if st := g.Stats(); st.VoltaGemm+st.GemmF16 > 0 || mx > 0 || spec == "metal" {
				bound0 = 5e-4
			}
			t.Logf("%s, after block 0 composed: NMSE %.3e, max|d| %.4g (bound %.0e; %d binary16 tier GEMMs, %d composed matrix launches; tower chunk %d, %d phase calls)",
				spec, nm0, worst0, bound0, g.Stats().VoltaGemm+g.Stats().GemmF16, mx, g.TowerChunk(), g.Stats().TowerPhases)
			if !(nm0 < bound0) {
				t.Fatalf("%s: NMSE %.3e after ONE composed block (bound %.0e)", spec, nm0, bound0)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no device on this host: the gate proved nothing")
	}
}

// tierTowerNMSE is the whole encode of px with every block of tw on spec
// run by the tier's own kernels, against want: what the composed arm's
// reading is set beside.
func tierTowerNMSE(t *testing.T, tw *Tower, spec string, px, want []float32) float64 {
	t.Helper()
	g, err := tier.OpenWith(tier.WithDevices(spec), composedOpt(0))
	if err != nil || g == nil {
		t.Fatalf("%s: %v", spec, err)
	}
	defer g.Close()
	s := tw.testState()
	defer s.Close()
	s.SetDevice(g)
	if n := s.GPUBlocks(); n != tw.Cfg.NLayer {
		t.Fatalf("%s took %d of %d tower blocks on its own kernels (%s)", spec, n, tw.Cfg.NLayer, g.Err())
	}
	_, got := towerEncode(t, s, px)
	nm, _ := nmseOf(got, want)
	return nm
}

// TestComposedTowerOnDevice runs the device arm over the towers the tier
// serves (tier.ComposedTower), Gemma 3's SigLIP among them: its 4096 rows
// of 1152 are the flattened space whose live elements once wrapped 32 bits
// (jit.TestComposedGPUAttendBidirRows).
func TestComposedTowerOnDevice(t *testing.T) {
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) { composedTowerOnDevices(t, f.open, f.bound) })
	}
}

// towerArm is one arm of a tower family's device gate: a device, its blocks
// on the tier's own kernels or served by the composer (tier.ComposedStep,
// tower blocks among its parts).
type towerArm struct {
	spec     string
	composed bool
}

// towerArms is each device's two arms.
func towerArms() []towerArm {
	var out []towerArm
	for _, spec := range stepDevices() {
		out = append(out, towerArm{spec, false}, towerArm{spec, true})
	}
	return out
}

func (a towerArm) String() string {
	if a.composed {
		return a.spec + "/composed"
	}
	return a.spec
}

// option is the tier option that asks for the arm's parts.
func (a towerArm) option() tier.Option {
	if a.composed {
		return composedOpt(tier.ComposedStep)
	}
	return composedOpt(0)
}

// served runs encode and, in a composed arm, holds g's composer to having
// served runs tower block runs in it and declined none: a composed arm whose
// blocks ran on the tier's own kernels is no pass.
func (a towerArm) served(t *testing.T, g *tier.GPU, runs int, encode func()) {
	t.Helper()
	s0, d0, _ := g.Stats().ServedPart(tier.ComposedTower)
	encode()
	if !a.composed {
		return
	}
	st := g.Stats()
	s1, d1, u := st.ServedPart(tier.ComposedTower)
	t.Logf("%s: %d tower block runs served by the composer, %d declined %v, %d unasked", a, s1-s0, d1-d0,
		nonZero(st.BlockDeclines[:]), u)
	if s1-s0 != runs || d1 != d0 {
		t.Fatalf("%s: the composer served %d of %d tower block runs, declined %d %v (%s)", a, s1-s0, runs, d1-d0,
			nonZero(st.BlockDeclines[:]), g.Err())
	}
}

// TestTowerRopeAxesAreStated holds the front end's statement of a tower's
// rotary axes to its runs: contiguous runs as sections, MoonViT's
// alternating pairs as a cycle at each cycle's first pair's frequency, and
// runs it cannot state -- a cycle at another frequency, an axis coming back
// out of turn -- refused.
func TestTowerRopeAxesAreStated(t *testing.T) {
	kimi := TowerConfig{Kind: jlm.ProjKimiVL, HeadDim: 8}.kimiRope()
	a, ok := towerRopeAxes(kimi)
	if !ok || a.Cycle != 2 || a.CycleAxes != [4]uint8{1, 0} || a.Sections != [4]int{} {
		t.Fatalf("MoonViT's rotary stated as %+v (%v), want a cycle of axes 1, 0", a, ok)
	}
	pix := TowerConfig{Kind: jlm.ProjPixtral, HeadDim: 8}.pixtralRope()
	if a, ok := towerRopeAxes(pix); !ok || a.Cycle != 0 || a.Sections != [4]int{2, 2} {
		t.Fatalf("Pixtral's rotary stated as %+v (%v), want sections 2, 2", a, ok)
	}
	off := TowerConfig{Kind: jlm.ProjKimiVL, HeadDim: 8}.kimiRope()
	off.Runs[2].Freq++
	if a, ok := towerRopeAxes(off); ok {
		t.Fatalf("a cycle whose third pair turns at another frequency stated as %+v", a)
	}
	back := nn.Rope{NRot: 8, Runs: []nn.RopeRun{{Pairs: 1, Axis: 0}, {Pairs: 1, Axis: 1}, {Pairs: 1, Axis: 1, Freq: 2}, {Pairs: 1, Axis: 0, Freq: 2}}}
	if a, ok := towerRopeAxes(back); ok {
		t.Fatalf("axes 0, 1, 1, 0 stated as %+v", a)
	}
}

// restateTower has the front end state m's tower blocks again, from the
// tower's configuration as it now stands: a violation that changes a tower's
// configuration reaches the composer through the graphs it states.
func (m *Model) restateTower() {
	m.front.mu.Lock()
	m.front.restated++
	m.front.mu.Unlock()
}

// unclampTower stores tw's clipped linears' bounds as infinite until the
// returned undo: the bounds a clamp reads, on every path, no longer clip.
func unclampTower(tw *Tower) func() {
	var was [][]float32
	for i := range tw.Cfg.NLayer {
		l := &tw.model.layers[tw.base+i]
		was = append(was, append([]float32(nil), l.clamp...))
		for j := range l.clamp {
			l.clamp[j] = float32(math.Inf(2*(j%2) - 1))
		}
	}
	return func() {
		for i := range tw.Cfg.NLayer {
			copy(tw.model.layers[tw.base+i].clamp, was[i])
		}
	}
}

// TestComposedProjectorOnCPU holds the projector ALONE, composed for the
// CPU (towerProjOf), to main's on the same rows that are no picture: each
// tower's regroup, pool, norm, matrices and activation and the window
// order's put-back, counted served, and each family's violation of the op
// it adds seen -- the regroup's slots swapped, the pool reading a group's
// first patch alone, the merged rows left in the window order.
func TestComposedProjectorOnCPU(t *testing.T) {
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) {
			ref, rtw := f.open(t, noTune)
			defer ref.Close()
			c := rtw.Cfg
			side := c.ImageSz / c.PatchSz
			x := projectorRows(side*side, c.NEmbd)
			project := func(tw *Tower, composed bool) []float32 {
				s := tw.testState()
				defer s.Close()
				s.windowOrder(side, side)
				got, err := s.project(append([]float32(nil), x...))
				if err != nil {
					t.Fatal(err)
				}
				served, declined := s.CPUProjectorSteps()
				why, werr := s.CPUProjectorUncovered()
				if composed && served != 1 {
					t.Fatalf("the composed projector served %d of 1 (declines %v; %s %v)", served, nonZero(declined[:]), why, werr)
				}
				if !composed && served != 0 {
					t.Fatalf("main's arm ran the composed projector")
				}
				return append([]float32(nil), got...)
			}
			want := project(rtw, false)
			m, tw := f.open(t, noTune, WithComposedCPU(true))
			defer m.Close()
			got := project(tw, true)
			nm, worst := nmseOf(got, want)
			t.Logf("the composed projector alone: NMSE %.3e, max|d| %.4g over %d rows", nm, worst, len(want)/c.ProjDim)
			if !(nm < 1e-6) {
				t.Fatalf("the composed projector alone reads NMSE %.3e from main's (bound 1e-6)", nm)
			}
			var vio string
			switch {
			case c.WinPattern > 0:
				vio, qwenNoUnwindow = "the merged rows left in the window order", true
				defer func() { qwenNoUnwindow = false }()
			case c.Scale > 1 && c.projK() == c.NEmbd:
				vio, tw.Cfg.fault = "the pool reads a group's first patch alone", towerFaultPoolCorner
			case c.Scale > 1:
				vio, tw.Cfg.fault = "the regroup's slots swapped", towerFaultShuffleSwap
			default:
				t.Logf("a projector of a patch a row adds no op to violate")
				return
			}
			bad := project(tw, true)
			tw.Cfg.fault = towerFaultNone
			nb, _ := nmseOf(bad, want)
			t.Logf("violation (%s): NMSE %.3e", vio, nb)
			if !(nb > 1e-4) {
				t.Fatalf("the gate cannot see %s: NMSE %.3e", vio, nb)
			}
		})
	}
}

// TestComposedPictureRowsEveryFamily holds every tower family's entry and
// projector, where the front end states them, composed for the CPU to
// main's on the same picture and the same rows: the entry's rows before
// block 0, the projector's output alone, each counted served. A family
// whose entry or projector the front end states none of is named and
// counted declined, never served; a projector it states none of is named.
func TestComposedPictureRowsEveryFamily(t *testing.T) {
	for _, f := range towerFamilies() {
		t.Run(f.name, func(t *testing.T) {
			m, tw := f.open(t)
			defer m.Close()
			c := tw.Cfg
			_, ew, ee := m.towerEntryOf()
			_, pw, pe := m.towerProjOf(false)
			px := rainbow(c.ImageSz)
			entry := func() ([]float32, *State) {
				s := tw.testState()
				var out []float32
				s.vis.afterBlock = func(li int, x []float32) {
					if li == -1 && out == nil {
						out = append([]float32(nil), x...)
					}
				}
				if _, err := s.Encode(px); err != nil {
					t.Fatal(err)
				}
				return out, s
			}
			side := c.ImageSz / c.PatchSz
			rows := projectorRows(side*side, c.NEmbd)
			project := func() ([]float32, *State) {
				s := tw.testState()
				s.windowOrder(side, side)
				got, err := s.project(append([]float32(nil), rows...))
				if err != nil {
					t.Fatal(err)
				}
				return append([]float32(nil), got...), s
			}
			wantE, s := entry()
			s.Close()
			var wantP, gotP []float32
			if pw == Covered {
				wantP, s = project()
				s.Close()
			}
			defer func(was bool) { m.opt.composedCPU = was }(m.opt.composedCPU)
			m.opt.composedCPU = true
			gotE, s := entry()
			es, ed := s.CPUEntrySteps()
			s.Close()
			var ps int
			var pd [jit.NumDeclines]int
			if pw == Covered {
				gotP, s = project()
				ps, pd = s.CPUProjectorSteps()
				s.Close()
			}
			if ew == Covered {
				nm, worst := nmseOf(gotE, wantE)
				t.Logf("entry: served %d, NMSE %.3e, max|d| %.4g", es, nm, worst)
				if es != 1 || !(nm < 1e-5) {
					t.Errorf("the composed entry served %d of 1 at NMSE %.3e from main's (bound 1e-5; declines %v)", es, nm, nonZero(ed[:]))
				}
			} else {
				t.Logf("entry: %s (%v), declined %v", ew, ee, nonZero(ed[:]))
				if es != 0 {
					t.Errorf("an entry the front end states none of was served %d times", es)
				}
			}
			if pw == Covered {
				nm, worst := nmseOf(gotP, wantP)
				t.Logf("projector: served %d, NMSE %.3e, max|d| %.4g", ps, nm, worst)
				if ps != 1 || !(nm < 1e-6) {
					t.Errorf("the composed projector served %d of 1 at NMSE %.3e from main's (bound 1e-6; declines %v)", ps, nm, nonZero(pd[:]))
				}
			} else {
				t.Logf("projector: %s (%v)", pw, pe)
			}
		})
	}
}
