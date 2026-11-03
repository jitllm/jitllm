package model

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/engine/device/cpu"
	"github.com/jitllm/jitllm/format/jlm"
)

// TestTowerGridTableMatchesVisRope holds the tower's rotary table as the
// composer forms it on the CPU -- each row's coordinates a cell of the grid,
// the pairs' axes after the planes (towertable.go) -- to the table main's
// encode turned its rows by (visRopeOracle, the CPU JIT's table kernel at
// each row's coordinates), on every tower whose blocks rotate, at the
// picture each composed tower gate encodes: within an ulp's worth of the
// table's range. With every pair's axis swapped -- a row turned by its
// column where its row was meant -- it must part.
func TestTowerGridTableMatchesVisRope(t *testing.T) {
	ran := 0
	for _, f := range composedTowers() {
		t.Run(f.name, func(t *testing.T) {
			m, tw := f.open(t, noTune())
			defer m.Close()
			c := tw.Cfg
			if !c.Rope {
				t.Skipf("%s's tower turns by no rotary", f.name)
			}
			ran++
			px := make([]float32, c.ImageSz*c.ImageSz*3)
			for i := range px {
				px[i] = float32((i*37)%255)/127.5 - 1
			}
			s := tw.testState()
			defer s.Close()
			var want, got, bad []float32
			var err error
			s.vis.afterBlock = func(li int, _ []float32) {
				if li != 0 || want != nil {
					return
				}
				n := s.vis.prog.n
				want = visRopeOracle(s, n)
				got = make([]float32, n*c.HeadDim)
				if err = s.cpuTowerTableRows(n, got); err != nil {
					return
				}
				st := &s.cpu.towerTab
				clean := st.planes
				swapped := append([]float32(nil), clean...)
				for i := len(swapped) - c.HeadDim/2; i < len(swapped); i++ {
					swapped[i] = 1 - swapped[i]
				}
				st.planes = swapped
				bad = make([]float32, n*c.HeadDim)
				err = s.cpuTowerTableRows(n, bad)
				st.planes = clean
			}
			if _, e := s.Encode(px); e != nil {
				t.Fatal(e)
			}
			if err != nil {
				t.Fatal(err)
			}
			if want == nil {
				t.Fatal("no block ran on the CPU: the table was never compared")
			}
			worst, at := 0.0, -1
			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > worst || d != d {
					worst, at = d, i
				}
			}
			nb, _ := nmseOf(bad, want)
			t.Logf("%s: %d rows of %d: worst |d| %.3g (element %d); every pair's axis swapped: NMSE %.3e",
				f.name, len(want)/c.HeadDim, c.HeadDim, worst, at, nb)
			if !(worst <= 2e-6) {
				t.Fatalf("the composed grid table parts from visRope by %.3g at element %d (row %d)", worst, at, at/c.HeadDim)
			}
			if !(nb > 1e-3) {
				t.Fatalf("every pair's axis swapped reads NMSE %.3e: the gate cannot see the axes", nb)
			}
		})
	}
	if ran == 0 {
		t.Fatal("no tower turned by a rotary: the gate proved nothing")
	}
}

// visRopeOracle is every row's {cos, sin} row as main's encode built it: the
// row's coordinates on towerRope's two axes -- a patch's (grid row, grid
// column), Llama 4's from one with its class row at the origin, Gemma 4's
// column first -- through the CPU JIT's rotary table kernel.
func visRopeOracle(s *Instance, n int) []float32 {
	c := s.vis.t.Cfg
	gh, gw := s.patchGrid()
	out := make([]float32, n*c.HeadDim)
	r := c.towerRope()
	for i := range n {
		row := out[i*c.HeadDim : (i+1)*c.HeadDim]
		switch c.Kind {
		case jlm.ProjLlama4:
			var pos [2]int
			llama4Coords(i, gh*gw, gw, pos[:], c.viol)
			s.jit.RopeTableAt(r, row, pos[:])
		case jlm.ProjGemma4V:
			p := s.patchAt(i)
			pos := []int{p % gw, p / gw}
			if c.viol == violRowFirst {
				pos[0], pos[1] = pos[1], pos[0]
			}
			s.jit.RopeTableAt(r, row, pos)
		default:
			p := s.patchAt(i)
			s.jit.RopeTableAt(r, row, []int{p / gw, p % gw})
		}
	}
	return out
}

// noTowerRotary turns s's tower rows by nothing: the CPU's grid planes with
// every frequency zero (their scale and the pairs' axes kept), so each row's
// angles are zero -- the violation of a tower whose 2-D rotary was dropped.
func noTowerRotary(s *Instance) {
	p := s.cpu.towerTab.planes
	stride := cpu.RopeTabStride(s.vis.t.Cfg.HeadDim / 2)
	for i := range 8 * stride {
		p[i] = 0
	}
}
