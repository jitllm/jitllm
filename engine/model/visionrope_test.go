package model

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/engine/nn"
)

// TestVisionRopeMatchesTheFloat64Rotation holds the vision tower's 2-D rotary
// (ropeRows over ropeTable) to the float64 rotation it was before the table
// moved onto JIT.RopeTable: the row coordinate on the first quarter of each
// head's pairs, the column on the second, NEOX pairing (i, i+half), angle
// pos * 10000^(-2j/half). The caption gates cannot see a wrong rotation (they
// stayed green with cos and sin swapped), so it is checked against its
// reference directly.
func TestVisionRopeMatchesTheFloat64Rotation(t *testing.T) {
	// Qwen2-VL-2B's geometry at a smaller grid: 80-wide heads, so a quarter of
	// the pairs is 20 and the two halves are not symmetric around a power of two.
	c := TowerConfig{NHead: 4, HeadDim: 80, NEmbd: 320, ImageSz: 14 * 6, PatchSz: 14}
	tw := &Tower{Cfg: c, opt: &modelOpts{elemPool: true}}
	s := &Instance{vis: &visRun{t: tw}, jit: nn.NewJIT(c.NEmbd, c.NEmbd, nil, nn.WithTune(nn.TuneOff), nn.WithQuietTuner(true))}
	defer s.jit.Close()
	side := c.ImageSz / c.PatchSz
	n := side * side
	rng := rand.New(rand.NewSource(3))
	x := make([]float32, n*c.NEmbd)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	want := make([]float64, len(x))
	half, quart := c.HeadDim/2, c.HeadDim/4
	for p := 0; p < n; p++ {
		row, col := p/side, p%side
		for h := 0; h < c.NHead; h++ {
			base := p*c.NEmbd + h*c.HeadDim
			for i := 0; i < half; i++ {
				pos, j := row, i
				if i >= quart {
					pos, j = col, i-quart
				}
				ang := float64(pos) / math.Pow(10000, float64(2*j)/float64(half))
				sin, cos := math.Sincos(ang)
				a, b := float64(x[base+i]), float64(x[base+i+half])
				want[base+i], want[base+i+half] = a*cos-b*sin, a*sin+b*cos
			}
		}
	}
	// The rotation attnPrep applies to an image row: the text block's kernel over
	// the row's table.
	tab := s.visRope(n)
	for p := 0; p < n; p++ {
		nn.RoPE32JIT(x[p*c.NEmbd:(p+1)*c.NEmbd], c.HeadDim, tab[p*c.HeadDim:(p+1)*c.HeadDim], true)
	}
	var sse, sy2, worst float64
	for i, w := range want {
		d := float64(x[i]) - w
		sse, sy2 = sse+d*d, sy2+w*w
		worst = math.Max(worst, math.Abs(d))
	}
	nmse := sse / sy2
	t.Logf("%d patches x %d heads: NMSE %.3e, max|d| %.3e against the float64 rotation", n, c.NHead, nmse, worst)
	if math.IsNaN(nmse) || nmse > 1e-12 || worst > 1e-5 {
		t.Fatalf("the vision rotary does not rotate as the float64 reference: NMSE %.3e, max|d| %.3e", nmse, worst)
	}
}
