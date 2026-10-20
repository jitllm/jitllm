package graph

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// gaussGraph is a gate row of n through OpGaussTopK at bound.
func gaussGraph(n int, bound float64) (*Graph, ValueID, ValueID) {
	g := &Graph{Version: Version}
	x := g.value("gate", n)
	y := g.value("gate_sparse", n)
	g.op(Op{Kind: OpGaussTopK, In: []ValueID{x}, Out: []ValueID{y}, Attr: Attrs{Bound: bound}})
	return g, x, y
}

func seqSum(x []float32) float32 {
	var s float32
	for _, v := range x {
		s += v
	}
	return s
}

// hostSum is the order of the host's generated row statistic: four chains of
// eight lanes over each 32 elements, then one chain of eight over each eight
// left, then the last elements one by one into lane 0, the chains folded
// (c0+c1)+(c2+c3), the two halves of the lanes added, and the pairs.
func hostSum(x []float32) float32 {
	var c [4][8]float32
	i := 0
	for ; i+32 <= len(x); i += 32 {
		for k := 0; k < 4; k++ {
			for l := 0; l < 8; l++ {
				c[k][l] += x[i+8*k+l]
			}
		}
	}
	for ; i+8 <= len(x); i += 8 {
		for l := 0; l < 8; l++ {
			c[0][l] += x[i+l]
		}
	}
	for ; i < len(x); i++ {
		c[0][0] += x[i]
	}
	var a [8]float32
	for l := 0; l < 8; l++ {
		a[l] = (c[0][l] + c[1][l]) + (c[2][l] + c[3][l])
	}
	var h [4]float32
	for l := 0; l < 4; l++ {
		h[l] = a[l] + a[l+4]
	}
	return (h[0] + h[1]) + (h[2] + h[3])
}

// gaussRef is the host routine transcribed: the mean times 1/n, the
// population variance times 1/n, the cut mean + bound*sd (multiply, then
// add), each value max(x - cut, 0). Its sums are sum's. Fault arms change one
// thing each.
func gaussRef(x []float32, bound float32, sum func([]float32) float32, fault string) []float32 {
	n := float32(len(x))
	inv := 1 / n
	mean := sum(x) * inv
	sq := make([]float32, len(x))
	for i, v := range x {
		d := v - mean
		sq[i] = d * d
	}
	div := inv
	if fault == "unbiased" {
		div = 1 / (n - 1)
	}
	vr := sum(sq) * div
	if fault == "epsilon" {
		vr += 1e-3
	}
	sd := float32(math.Sqrt(float64(vr)))
	if fault == "variance" {
		sd = vr
	}
	cut := sd*bound + mean
	switch fault {
	case "no bound":
		cut = sd + mean
	case "no mean":
		cut = sd * bound
	}
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = max(v-cut, 0)
		if fault == "no clamp" {
			out[i] = v - cut
		}
	}
	return out
}

func gaussRow(seed int64, n int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()*2 + 0.7)
	}
	return x
}

// TestGaussTopKDecomposesToItsReference holds the decomposition to the host's
// routine over rows of several widths (whole vectors, ragged tails) and
// bounds: bit for bit to the routine transcribed with the decomposition's
// sequential sums, and to a few ulps of the routine with the host's chain
// order of sums (hostSum, held to the generated kernel itself where the
// transcription was made). Each defect a wrong
// reading would introduce must part from it.
func TestGaussTopKDecomposesToItsReference(t *testing.T) {
	faults := []string{"unbiased", "epsilon", "variance", "no bound", "no mean", "no clamp"}
	for _, n := range []int{1, 7, 8, 33, 64, 1000, 1003, 4096} {
		for _, bound := range []float64{1.6448536269514722, 0.5, -0.25} {
			g, xv, yv := gaussGraph(n, bound)
			d := Decompose(g)
			for _, o := range d.Ops {
				if o.Kind != OpMap && o.Kind != OpReduce {
					t.Fatalf("n=%d: the decomposition holds %s", n, o.Kind)
				}
			}
			shown := map[string]bool{}
			for seed := int64(1); seed <= 6; seed++ {
				x := gaussRow(seed, n)
				got := evalPrims(t, d, map[ValueID][]float32{xv: x}, nil)[yv]
				b32 := float32(bound)
				exact := gaussRef(x, b32, seqSum, "")
				for i := range got {
					if got[i] != exact[i] {
						t.Fatalf("n=%d bound=%g seed=%d: element %d is %g, the routine's %g", n, bound, seed, i, got[i], exact[i])
					}
				}
				// The routine with the host's order of sums.
				chain := gaussRef(x, b32, hostSum, "")
				scale := float64(0)
				for _, v := range x {
					scale = math.Max(scale, math.Abs(float64(v)))
				}
				tol := 4e-6 * scale
				for i := range got {
					if math.Abs(float64(got[i]-chain[i])) > tol {
						t.Fatalf("n=%d bound=%g seed=%d: element %d is %g, the host order's %g", n, bound, seed, i, got[i], chain[i])
					}
				}
				for _, f := range faults {
					bad := gaussRef(x, b32, seqSum, f)
					for i := range got {
						if math.Abs(float64(got[i]-bad[i])) > tol {
							shown[f] = true
							break
						}
					}
				}
			}
			if n >= 33 {
				for _, f := range faults {
					if !shown[f] {
						t.Errorf("n=%d bound=%g: the comparison does not see %q", n, bound, f)
					}
				}
			}
		}
	}
}

// TestAGaussTopKOfAnIndexRowStaysWhole is the variant the primitives do not
// state: a row that is not f32 stays one op.
func TestAGaussTopKOfAnIndexRowStaysWhole(t *testing.T) {
	g, xv, _ := gaussGraph(16, 1.5)
	g.Values[xv].Elem = ElemU32
	if d := Decompose(g); d.Ops[0].Kind != OpGaussTopK || len(d.Ops) != 1 {
		t.Fatalf("a u32 row was decomposed into %d ops starting with %s", len(d.Ops), d.Ops[0].Kind)
	}
	g, _, _ = gaussGraph(16, math.NaN())
	if d := Decompose(g); d.Ops[0].Kind != OpGaussTopK {
		t.Fatalf("a bound that is not a number was decomposed")
	}
}

// laurelGraph is LAuReL's branch over a row of n at rank.
func laurelGraph(n, rank int, eps float64) (*Graph, ValueID, ValueID) {
	g := &Graph{Version: Version}
	h := g.value("attn_in", n)
	o := g.value("laurel", n)
	g.op(Op{Kind: OpLaurel, In: []ValueID{h}, Out: []ValueID{o}, Weight: []Binding{
		{Role: jlm.RoleLaurelL, K: n, Rows: rank}, {Role: jlm.RoleLaurelR, K: rank, Rows: n},
		{Role: jlm.RoleLaurelPostNorm, K: n}}, Attr: Attrs{Eps: eps}})
	return g, h, o
}

// laurelRef is the host's routine: h + rmsnorm(R(L(h))). Fault arms change one
// thing each.
func laurelRef(h, lw, rw, pw []float32, rank int, eps float64, fault string) []float32 {
	n := len(h)
	mv := func(m, x []float32, rows int) []float32 {
		out := make([]float32, rows)
		for r := range out {
			var s float32
			for c := range x {
				s += m[r*len(x)+c] * x[c]
			}
			out[r] = s
		}
		return out
	}
	low := mv(lw, h, rank)
	up := mv(rw, low, n)
	if fault == "swapped" {
		low = mv(rw, h, rank)
		up = mv(lw, low, n)
	}
	norm := make([]float32, n)
	oracle.RMSNorm32(norm, up, pw, eps)
	if fault == "eps" {
		oracle.RMSNorm32(norm, up, pw, eps*1e6+1e-2)
	}
	if fault == "unnormed" {
		copy(norm, up)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = norm[i] + h[i]
		if fault == "no residual" {
			out[i] = norm[i]
		}
	}
	return out
}

// TestLaurelDecomposesToItsReference holds the decomposition to the host's
// routine over several shapes and seeds, and each defect (the matrices
// swapped, the norm skipped or at another epsilon, the input not added back)
// must part from it.
func TestLaurelDecomposesToItsReference(t *testing.T) {
	faults := []string{"swapped", "eps", "unnormed", "no residual"}
	for _, c := range []struct{ n, rank int }{{16, 4}, {48, 8}, {64, 64}, {33, 5}} {
		g, hv, ov := laurelGraph(c.n, c.rank, 1e-6)
		d := Decompose(g)
		want := []OpKind{OpMatVec, OpMatVec, OpNorm, OpAdd}
		if len(d.Ops) != len(want) {
			t.Fatalf("%+v: the decomposition has %d ops", c, len(d.Ops))
		}
		for i := range want {
			if d.Ops[i].Kind != want[i] {
				t.Fatalf("%+v: op %d is %s, not %s", c, i, d.Ops[i].Kind, want[i])
			}
		}
		if d.Ops[3].Out[0] != ov {
			t.Fatalf("%+v: the add does not write the op's own output", c)
		}
		shown := map[string]bool{}
		for seed := int64(1); seed <= 6; seed++ {
			rng := rand.New(rand.NewSource(seed))
			fill := func(n int, s float32) []float32 {
				f := make([]float32, n)
				for i := range f {
					f[i] = (rng.Float32()*2 - 1) * s
				}
				return f
			}
			h := fill(c.n, 1)
			lw, rw, pw := fill(c.rank*c.n, 0.5), fill(c.n*c.rank, 0.5), fill(c.n, 1)
			got := evalPrims(t, d, map[ValueID][]float32{hv: h},
				map[jlm.Role][]float32{jlm.RoleLaurelL: lw, jlm.RoleLaurelR: rw, jlm.RoleLaurelPostNorm: pw})[ov]
			ref := laurelRef(h, lw, rw, pw, c.rank, 1e-6, "")
			for i := range got {
				if got[i] != ref[i] {
					t.Fatalf("%+v seed %d: element %d is %g, the routine's %g", c, seed, i, got[i], ref[i])
				}
			}
			for _, f := range faults {
				bad := laurelRef(h, lw, rw, pw, c.rank, 1e-6, f)
				for i := range got {
					if math.Abs(float64(got[i]-bad[i])) > 1e-4 {
						shown[f] = true
						break
					}
				}
			}
		}
		for _, f := range faults {
			if !shown[f] {
				t.Errorf("%+v: the comparison does not see %q", c, f)
			}
		}
	}
}

// TestALaurelWhoseWidthsDoNotChainStaysWhole is the variant the primitives
// here do not state: matrices that do not chain, or a binding it does not
// know, stay one op.
func TestALaurelWhoseWidthsDoNotChainStaysWhole(t *testing.T) {
	g, _, _ := laurelGraph(16, 4, 1e-6)
	g.Ops[0].Weight[1].K = 5
	if d := Decompose(g); d.Ops[0].Kind != OpLaurel {
		t.Fatalf("matrices that do not chain were decomposed")
	}
	g, _, _ = laurelGraph(16, 4, 1e-6)
	g.Ops[0].Weight[2].Role = jlm.RoleAttnNorm
	if d := Decompose(g); d.Ops[0].Kind != OpLaurel {
		t.Fatalf("an unknown binding was decomposed")
	}
}
