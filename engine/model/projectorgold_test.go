package model

import (
	"math"
	"path/filepath"
	"testing"
)

// projectorRows is scripts/visiongold.py's patch_rows: a tower's last hidden
// state that is no picture, every row and channel distinct.
func projectorRows(n, width int) []float32 {
	out := make([]float32, n*width)
	for p := 0; p < n; p++ {
		for c := 0; c < width; c++ {
			fp, fc := float64(p), float64(c)
			out[p*width+c] = float32(math.Sin(0.37*fp+0.11*fc) + 0.5*math.Cos(0.013*fp*fc))
		}
	}
	return out
}

// TestProjectorMatchesTransformers holds the projector ALONE to transformers'
// on rows that are no picture: gemma3's pool, RMSNorm and matrix, InternVL's
// shuffle, LayerNorm and MLP. With the tower out of it the only error is the
// projector's own int8 activations, so a pool or shuffle in the wrong order --
// which the reference itself barely notices on a picture, since after the
// blocks neighbouring patches look alike (InternVL's 2x2 transpose moves its
// rainbow features by NMSE 1.9e-2, inside the tower's precision) -- cannot hide.
// Its composed arm runs the projector the composer forms for the CPU
// (towerProjOf), counted served once a run.
func TestProjectorMatchesTransformers(t *testing.T) {
	for _, fam := range visionFamilies() {
		for _, composed := range []bool{false, true} {
			name := fam.name
			if composed {
				name += "/composed"
			}
			t.Run(name, func(t *testing.T) { projectorMatchesTransformers(t, fam, composed) })
		}
	}
}

func projectorMatchesTransformers(t *testing.T, fam visionFamily, composed bool) {
	m, tw := fam.open(t)
	defer m.Close()
	c := tw.Cfg
	x := projectorRows(c.Patches(), c.NEmbd)
	want := readF32(t, filepath.Join(visionGoldDir(fam.name), "projector.features.f32"))
	run := func(f towerFault) float64 {
		defer func(was towerFault) { tw.Cfg.fault = was }(tw.Cfg.fault)
		tw.Cfg.fault = f
		s := tw.testState()
		defer s.Close()
		// project reads its rows from the residual it is handed and
		// writes its own scratch, so the input is a copy.
		got, err := s.project(append([]float32(nil), x...))
		if err != nil {
			t.Fatal(err)
		}
		if served, declined := s.CPUProjectorSteps(); composed && served != 1 {
			why, werr := s.CPUProjectorUncovered()
			t.Fatalf("the composed projector served %d of 1 (declines %v; %s %v)", served, nonZero(declined[:]), why, werr)
		}
		if len(got) != len(want) {
			t.Fatalf("%d values, transformers %d", len(got), len(want))
		}
		nm, _ := nmseOf(got, want)
		return nm
	}
	clean := run(towerFaultNone)
	t.Logf("projector alone: NMSE %.3e against transformers", clean)
	if math.IsNaN(clean) || clean > 1e-3 {
		t.Errorf("the projector alone reads NMSE %.3e (bound 1e-3)", clean)
	}
	vio := towerFaultShuffleSwap
	if fam.name == "gemma3" {
		vio = towerFaultPoolCorner
	}
	v := run(vio)
	t.Logf("violation (%d): NMSE %.3e", vio, v)
	if v < 100*clean {
		t.Errorf("the violation reads NMSE %.3e against a clean %.3e: the gate cannot see it", v, clean)
	}
}
