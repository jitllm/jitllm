//go:build amd64 || arm64

package nn

import (
	"math"
	"testing"

	"github.com/jitllm/jitllm/jit_old/cpu"
)

// The rotary table's violations: a gate that has never fired is not a gate.
//
// The violations go in through the constant block, not a mutated emitter, so
// they reach every tier: three of the kernel's four structural steps read a
// word of Scr that only that step reads (the mod-4 magic, the integer that
// turns n into n+1 for cosine's quadrant, and the negation words). Every tier
// must read these from the block rather than from immediates, or its
// violations read green while doing nothing. The sign step has two words
// because amd64 XORs a sign mask where NEON selects against an FNEG
// (ropetab_const.go).
func ropeWithConsts(t *testing.T, mutate func([]float32)) func() {
	t.Helper()
	old := ropeTabConsts
	next := append([]float32(nil), old...)
	mutate(next)
	ropeTabConsts = next
	return func() { ropeTabConsts = old }
}

// worstAgainstOracle is the same comparison the gate makes, as a number.
func worstAgainstOracle(r Rope, pos int) float64 {
	npairs := r.NRot / 2
	cs := make([]float32, r.NRot)
	r.Table(cs, pos)
	want := ropeOracle(r, pos, npairs)
	var worst float64
	for i := range want {
		if d := math.Abs(float64(cs[i]) - want[i]); d > worst {
			worst = d
		}
	}
	return worst
}

func TestRopeTableGateCatchesItsViolations(t *testing.T) {
	const bound = 3e-7
	r := Rope{NRot: 128, Base: 1000000}

	// The honest arm first, so a violation that reads large because the whole
	// kernel is broken cannot be mistaken for a violation that fired.
	if w := worstAgainstOracle(r, 12345); w > bound {
		t.Fatalf("the unviolated kernel is already %.3e out", w)
	}

	type word struct{ off, n int }
	for _, v := range []struct {
		name  string
		what  string
		words []word
	}{
		{"quadrant-selection", "cosine's quadrant is taken from n instead of n+1",
			[]word{{cpu.RopeTabOneVecWord, 8}}},
		// One step, two words: each ISA family leaves the other's word inert,
		// so both are zeroed in one subtest.
		{"sign-step", "bit 1 of the quadrant stops deciding the sign, so two of four quadrants are wrong",
			[]word{{cpu.RopeTabSignVecWord, 8}, {cpu.RopeTabTwoVecWord, 8}}},
		{"no-mod-4-fold", "the position's digit products are never folded, so the angle is formed large",
			[]word{{cpu.RopeTabMagicWord, 1}}},
	} {
		t.Run(v.name, func(t *testing.T) {
			restore := ropeWithConsts(t, func(c []float32) {
				for _, w := range v.words {
					for i := 0; i < w.n; i++ {
						c[w.off+i] = 0
					}
				}
			})
			defer restore()
			var worst float64
			for _, pos := range ropePositions() {
				if pos >= cpu.RopeTabMaxPos {
					continue
				}
				if w := worstAgainstOracle(r, pos); w > worst {
					worst = w
				}
			}
			if worst <= bound {
				t.Fatalf("%s: the gate did not fire -- worst |d| %.3e is still inside the "+
					"%.1e bound, so this gate certifies nothing about that step", v.what, worst, bound)
			}
			t.Logf("%s -> worst |d| %.3e", v.what, worst)
		})
	}

	// The interleave is the one step no constant can reach, so its violation
	// is made in the comparison instead: against a transposed oracle.
	t.Run("swapped-sin-cos", func(t *testing.T) {
		npairs := r.NRot / 2
		cs := make([]float32, r.NRot)
		r.Table(cs, 12345)
		want := ropeOracle(r, 12345, npairs)
		var worst float64
		for p := 0; p < npairs; p++ {
			for _, d := range []float64{
				math.Abs(float64(cs[2*p]) - want[2*p+1]),
				math.Abs(float64(cs[2*p+1]) - want[2*p]),
			} {
				if d > worst {
					worst = d
				}
			}
		}
		if worst <= bound {
			t.Fatalf("a transposed {cos, sin} is within %.1e of the table -- the gate "+
				"cannot tell the two apart at this position", bound)
		}
		t.Logf("{cos, sin} stored the other way round -> worst |d| %.3e", worst)
	})
}
