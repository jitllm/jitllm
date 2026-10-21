//go:build amd64 || arm64

package cpu

import (
	"math"
	"runtime"
	"testing"
)

// TestEmitRoPESplitMatchesReference rotates heads by XD-RoPE's two tables --
// A on each NEOX pair's first half, B on its second -- against a float64
// rotation written here, every pair count from 1 to 33 with and without a
// partial tail, on the host's own tier and, on amd64, the SSE tier too. With
// A equal to B it must be EmitRoPE's NEOX rotation to the bit; with them
// apart it must differ from it (a kernel that read A twice passes the first
// check and fails this one).
func TestEmitRoPESplitMatchesReference(t *testing.T) {
	// The host tier first, then the SSE tier wherever it is a second one this
	// host runs: on an SSE-only host the host tier IS the SSE tier.
	tiers := []*Emitters{hostTable()}
	if runtime.GOARCH == "amd64" && HostTier() != TierSSE {
		tiers = append(tiers, EmittersFor(TierSSE))
	}
	run := func(t *testing.T, code []byte, hd, heads int, cs []float32) []float32 {
		t.Helper()
		c := onHost(t)(code, nil)
		defer c.Close()
		x := ropeInput(hd, heads)
		c.Call(&Args{Out: &x[0], AScale: &cs[0], Rows: int64(heads)})
		return x
	}
	cases := 0
	for _, tr := range tiers {
		for pairs := 1; pairs <= 33; pairs++ {
			for _, extra := range []int{0, 6} {
				nrot := 2 * pairs
				hd, heads, half := nrot+extra, 3, pairs
				a := ropeTable(nrot)
				b := make([]float32, nrot)
				for p := 0; p < half; p++ {
					th := 2.9 * math.Pow(10000, -2*float64(p)/float64(nrot))
					b[2*p], b[2*p+1] = float32(math.Cos(th)), float32(math.Sin(th))
				}
				code, err := tr.RoPESplit(hd, nrot)
				if err != nil {
					t.Fatal(err)
				}
				got := run(t, code, hd, heads, append(append([]float32(nil), a...), b...))
				in := ropeInput(hd, heads)
				for h := 0; h <= heads; h++ {
					x := in[h*hd : (h+1)*hd]
					for i := 0; i < hd; i++ {
						want := float64(x[i])
						if h < heads && i < nrot {
							p := i % half
							x0, x1 := float64(x[p]), float64(x[p+half])
							if i < half {
								want = x0*float64(a[2*p]) - x1*float64(a[2*p+1])
							} else {
								want = x0*float64(b[2*p+1]) + x1*float64(b[2*p])
							}
						}
						g := got[h*hd+i]
						if d := math.Abs(float64(g) - want); d > 1e-5*(1+math.Abs(want)) {
							t.Fatalf("%s hd=%d nrot=%d head %d dim %d: %v, want %v", tr.Tier, hd, nrot, h, i, g, want)
						}
					}
				}
				// A equal to B: EmitRoPE's NEOX rotation to the bit.
				same := run(t, code, hd, heads, append(append([]float32(nil), a...), a...))
				plain, err := tr.RoPE(hd, nrot, true)
				if err != nil {
					t.Fatal(err)
				}
				ref := run(t, plain, hd, heads, a)
				for i := range same {
					if math.Float32bits(same[i]) != math.Float32bits(ref[i]) {
						t.Fatalf("%s hd=%d nrot=%d: with A == B, element %d is %v where the NEOX kernel gives %v",
							tr.Tier, hd, nrot, i, same[i], ref[i])
					}
				}
				differs := false
				for i := range got {
					if got[i] != ref[i] {
						differs = true
					}
				}
				if !differs {
					t.Fatalf("%s hd=%d nrot=%d: B's table changed nothing", tr.Tier, hd, nrot)
				}
				cases++
			}
		}
	}
	t.Logf("%d (tier, pairs, partial) cases, three heads each and one beyond them untouched", cases)
}
