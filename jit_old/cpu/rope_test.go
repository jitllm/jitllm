//go:build amd64 || arm64

package cpu

import (
	"math"
	"testing"
)

// TestEmitRoPEMatchesReference rotates several heads by a real {cos, sin}
// table in both layouts, against a float64 rotation written here.
//
// Every pair count from 1 to 33 and a partial rotary: the dimensions past nrot
// and the heads past Rows must come back bit-identical. The angles differ per
// pair and the test asserts the two layouts' outputs differ, so a kernel that
// emitted NORM for both fails.
func TestEmitRoPEMatchesReference(t *testing.T) {
	ran := 0
	for _, neox := range []bool{false, true} {
		for pairs := 1; pairs <= 33; pairs++ {
			for _, extra := range []int{0, 6} { // 6: a partial rotary tail
				nrot := 2 * pairs
				hd := nrot + extra
				checkRoPE(t, hd, nrot, neox, 3)
				ran++
			}
		}
	}
	// The two layouts are different rotations.
	a, b := ropeRun(t, 16, 16, false, 1), ropeRun(t, 16, 16, true, 1)
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
		}
	}
	if same {
		t.Fatal("NORM and NEOX produced the same head: one layout was emitted for both")
	}
	t.Logf("%d (layout, pairs, partial) cases, three heads each, and one head beyond them untouched", ran)
}

func ropeTable(nrot int) []float32 {
	cs := make([]float32, nrot)
	for p := 0; p < nrot/2; p++ {
		th := 7.3 * math.Pow(10000, -2*float64(p)/float64(nrot))
		cs[2*p], cs[2*p+1] = float32(math.Cos(th)), float32(math.Sin(th))
	}
	return cs
}

func ropeInput(hd, heads int) []float32 {
	x := make([]float32, hd*(heads+1))
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.61) * float64(1+i%5))
	}
	return x
}

func ropeRun(t *testing.T, hd, nrot int, neox bool, heads int) []float32 {
	t.Helper()
	c := onHost(t)(hostTable().RoPE(hd, nrot, neox))
	defer c.Close()
	x := ropeInput(hd, heads)
	cs := ropeTable(nrot)
	c.Call(&Args{Out: &x[0], AScale: &cs[0], Rows: int64(heads)})
	return x
}

func checkRoPE(t *testing.T, hd, nrot int, neox bool, heads int) {
	t.Helper()
	got := ropeRun(t, hd, nrot, neox, heads)
	in := ropeInput(hd, heads)
	cs := ropeTable(nrot)
	half := nrot / 2
	for h := 0; h <= heads; h++ {
		x := in[h*hd : (h+1)*hd]
		want := make([]float64, hd)
		for i, v := range x {
			want[i] = float64(v)
		}
		if h < heads {
			for p := 0; p < half; p++ {
				i0, i1 := 2*p, 2*p+1
				if neox {
					i0, i1 = p, p+half
				}
				c, s := float64(cs[2*p]), float64(cs[2*p+1])
				x0, x1 := float64(x[i0]), float64(x[i1])
				want[i0], want[i1] = x0*c-x1*s, x0*s+x1*c
			}
		}
		for i := 0; i < hd; i++ {
			g := got[h*hd+i]
			rotated := h < heads && (i < nrot)
			if !rotated {
				if math.Float32bits(g) != math.Float32bits(x[i]) {
					t.Fatalf("hd=%d nrot=%d neox=%v head %d dim %d: %v should be untouched (%v)",
						hd, nrot, neox, h, i, g, x[i])
				}
				continue
			}
			if d := math.Abs(float64(g) - want[i]); d > 1e-5*(1+math.Abs(want[i])) {
				t.Fatalf("hd=%d nrot=%d neox=%v head %d dim %d: %v, want %v",
					hd, nrot, neox, h, i, g, want[i])
			}
		}
	}
}
