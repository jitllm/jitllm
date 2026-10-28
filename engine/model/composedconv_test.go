package model

import (
	"testing"

	"github.com/jitllm/jitllm/jit"
)

// TestComposedConvTowerOnCPU is the composed CPU arm of a convolutional
// tower (Gemma 3n's MobileNet-V5): every block the front end states
// (frontconv.go) composed for the CPU at its own grid and run over the
// encode's own buffers, the rest named -- an attention over another grid's
// keys (UncoveredGridAttend) and nothing else -- and declined by name at
// the run. Its output is held to main's runner's after ONE block at the
// composed tower's one-block bound and over the whole encode at the device
// arm's whole-tower bound, every block's reading logged;
// a warm composed encode allocates nothing; and the depthwise filters bound
// transposed must part from the clean encode.
func TestComposedConvTowerOnCPU(t *testing.T) {
	ref, _ := openGemma3nV(t)
	defer ref.Close()
	rtw := ref.Tower()
	c := rtw.Cfg
	px := make([]float32, c.ImageSz*c.ImageSz*3)
	for i := range px {
		px[i] = float32((i*37)%255) / 255
	}
	hs := rtw.testState()
	var wantBlocks [][]float32
	hs.vis.afterBlock = func(li int, x []float32) {
		if li >= 0 {
			wantBlocks = append(wantBlocks, append([]float32(nil), x...))
		}
	}
	want, err := hs.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	want = append([]float32(nil), want...)
	hs.Close()

	m, _ := openGemma3nV(t, WithComposedCPU(true))
	defer m.Close()
	tw := m.Tower()
	stated, why := 0, map[string]int{}
	for i := range c.NLayer {
		u := m.Uncovered(tw.base + i)
		switch u {
		case Covered:
			stated++
		case UncoveredGridAttend:
		default:
			t.Fatalf("tower block %d: %s (%v)", i, u, m.UncoveredErr(tw.base+i))
		}
		why[u.String()]++
	}
	t.Logf("%d of %d tower blocks stated: %v", stated, c.NLayer, why)
	if stated == 0 {
		t.Fatal("the front end states none of the tower's blocks: the gate proves nothing")
	}
	s := tw.testState()
	defer s.Close()
	var got [][]float32
	s.vis.afterBlock = func(li int, x []float32) {
		if li >= 0 {
			got = append(got, append([]float32(nil), x...))
		}
	}
	out, err := s.Encode(px)
	if err != nil {
		t.Fatal(err)
	}
	out = append([]float32(nil), out...)
	served, declined := s.CPUBlockSteps()
	t.Logf("composed CPU tower: %d block runs served, declines %v (form error: %v)", served, nonZero(declined[:]), s.CPUFormErr())
	if served != stated || declined[jit.DeclineNoGraph] != c.NLayer-stated || served+declined[jit.DeclineNoGraph] != c.NLayer {
		t.Fatalf("the CPU served %d of the %d stated blocks, declines %v (form error %v)", served, stated, nonZero(declined[:]), s.CPUFormErr())
	}
	if len(got) != len(wantBlocks) {
		t.Fatalf("%d block outputs against main's %d", len(got), len(wantBlocks))
	}
	nm0, w0 := nmseOf(got[0], wantBlocks[0])
	worst, at := 0.0, -1
	for i := range got {
		if nm, _ := nmseOf(got[i], wantBlocks[i]); nm > worst || nm != nm {
			worst, at = nm, i
		}
	}
	nm, wd := nmseOf(out, want)
	t.Logf("after block 0: NMSE %.3e, max|d| %.4g; worst block %d at %.3e; the encode: NMSE %.3e, max|d| %.4g over %d rows",
		nm0, w0, at, worst, nm, wd, len(want)/c.ProjDim)
	if !(nm0 < 5e-5) {
		t.Fatalf("NMSE %.3e after ONE block (bound 5e-5)", nm0)
	}
	if !(nm < 1e-2) {
		t.Fatalf("the composed CPU's encode is NMSE %.3e from main's (bound 1e-2)", nm)
	}

	s.vis.afterBlock = nil
	encode := func() {
		if _, err := s.Encode(px); err != nil {
			t.Fatal(err)
		}
	}
	encode()
	r0 := m.PagerReads()
	allocVerdict(t, "a warm composed convolutional encode", countAllocs(encode), 1, m.PagerReads()-r0, 0)

	s.cpu.fault = cpuFaultFilterT
	bad, err := s.Encode(px)
	s.cpu.fault = cpuFaultNone
	if err != nil {
		t.Fatal(err)
	}
	nb, _ := nmseOf(bad, out)
	t.Logf("violation (depthwise filters transposed): NMSE %.3e from the clean composed encode", nb)
	if !(nb > 1e-4) {
		t.Fatalf("the gate cannot see the filters transposed: NMSE %.3e", nb)
	}
}
