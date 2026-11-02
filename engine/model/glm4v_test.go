package model

import (
	"testing"
)

// glmTowerGate is TestQwenTowerMatchesTransformers' half for GLM-4.xV's tower:
// each of its own pieces taken out of the engine's copy must move the output
// far past the bound -- the 2-D rotary, the RMSNorm over the patch embedding,
// the table's bicubic resampling (bilinear in its place, and no resampling at
// all) and the merging convolution's column order.
func glmTowerGate(t *testing.T, tw *Tower, g *qvlGolden, clean float64) {
	t.Helper()
	for _, v := range []struct {
		name  string
		mut   func(*Instance)
		fault towerFault
	}{
		{"no 2-D rotary", func(s *Instance) {
			gh, gw := s.patchGrid()
			s.vis.ropeSC = make([]float32, gh*gw*tw.Cfg.HeadDim)
			for i := 0; i < len(s.vis.ropeSC); i += 2 {
				s.vis.ropeSC[i] = 1
			}
			s.vis.ropeAt = [2]int{gh, gw}
		}, towerFaultNone},
		{"no RMSNorm over the patch embedding", nil, faultNoEmbdNorm},
		{"the table resampled bilinear", nil, faultBilinearTable},
		{"the table read unresampled", nil, faultNoPosLerp},
		{"the merge group transposed", nil, towerFaultShuffleSwap},
	} {
		tw.Cfg.fault = v.fault
		bad, _ := qvlEncode(t, tw, g, v.mut)
		tw.Cfg.fault = towerFaultNone
		nb, _ := nmse32(g.Image.EmbdA8, bad)
		t.Logf("violation %q: NMSE %.3e", v.name, nb)
		if !(nb > 10*towerBound) {
			t.Errorf("%s reads %.3e against a clean %.3e: the gate cannot see it", v.name, nb, clean)
		}
	}
}
