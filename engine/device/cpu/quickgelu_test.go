package cpu

import (
	"math"
	"testing"
	"unsafe"
)

// quickGELURef is the definition, in float64: x * sigma(1.702x).
func quickGELURef(x float64) float64 { return x / (1 + math.Exp(-1.702*x)) }

// TestQuickGELUMatchesItsDefinition gates the third ungated activation against
// the arithmetic it is named after, on the host architecture, and asserts it
// is far from SiLU and GELU-tanh (which a wrong kernel could plausibly emit).
// The separation bars sit just under the measured separations over [-12,12].
func TestQuickGELUMatchesItsDefinition(t *testing.T) {
	c := onHost(t)(hostTable().Act(ActQuickGELU))
	consts := ActConsts()

	n := 4096
	x := make([]float32, n)
	for i := range x {
		// -12..+12, which covers the saturating ends and the part of the range
		// a ViT's FFN actually spends its time in.
		x[i] = float32(-12 + 24*float64(i)/float64(n-1))
	}
	got := append([]float32(nil), x...)
	call(c, got, consts)

	var worst float64
	var wi int
	for i := range x {
		want := quickGELURef(float64(x[i]))
		d := math.Abs(float64(got[i]) - want)
		if d > worst {
			worst, wi = d, i
		}
	}
	if worst > 1e-5 {
		t.Errorf("quick-GELU worst |error| %.3e at x=%v (kernel %v, want %v)",
			worst, x[wi], got[wi], quickGELURef(float64(x[wi])))
	}

	// The separation check: the other two kernels over the same input must
	// differ by a wide margin.
	for _, other := range []struct {
		name string
		k    ActKind
		min  float64
	}{{"silu", ActSiLU, 0.15}, {"gelu-tanh", ActGELU, 0.02}} {
		oc := onHost(t)(hostTable().Act(other.k))
		buf := append([]float32(nil), x...)
		call(oc, buf, consts)
		sep := 0.0
		for i := range buf {
			if d := math.Abs(float64(buf[i]) - float64(got[i])); d > sep {
				sep = d
			}
		}
		if sep < other.min {
			t.Errorf("quick-GELU and %s differ by at most %.4f over [-12,12]; this gate "+
				"cannot tell them apart, so it would not catch the wrong one shipping",
				other.name, sep)
		} else {
			t.Logf("quick-GELU vs %s: max separation %.4f", other.name, sep)
		}
	}
	t.Logf("quick-GELU worst |error| %.3e over %d points in [-12,12]", worst, n)
}

// TestUngatedActivationsAreAllDistinct is the violation check for the enum:
// three kinds must map to three different kernels, so a mapping that collapsed
// two of them fails.
func TestUngatedActivationsAreAllDistinct(t *testing.T) {
	consts := ActConsts()
	n := 512
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(-6 + 12*float64(i)/float64(n-1))
	}
	outs := make([][]float32, len(Ungated))
	for i, k := range Ungated {
		c := onHost(t)(hostTable().Act(k))
		outs[i] = append([]float32(nil), x...)
		call(c, outs[i], consts)
	}
	for i := range outs {
		for j := i + 1; j < len(outs); j++ {
			same := true
			for e := range outs[i] {
				if outs[i][e] != outs[j][e] {
					same = false
					break
				}
			}
			if same {
				t.Errorf("%v and %v emit identical results: two names for one kernel",
					Ungated[i], Ungated[j])
			}
		}
	}
}

// call runs an ungated activation kernel over buf in place.
func call(c *Code, buf []float32, consts []float32) {
	args := Args{
		Out: &buf[0],
		Scr: (*byte)(unsafe.Pointer(&consts[0])),
		K:   int64(len(buf) / ElemLanes),
	}
	c.Call(&args)
}
