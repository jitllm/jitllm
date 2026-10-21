package graph

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// opNamed is the index of the op whose first output is the value called name.
func opNamed(t *testing.T, g *Graph, name string) int {
	t.Helper()
	for i, o := range g.Ops {
		if len(o.Out) > 0 && g.Values[o.Out[0]].Name == name {
			return i
		}
	}
	t.Fatalf("no op writes %q", name)
	return -1
}

// kernelSparseMixer is the device kernel's arithmetic
// (jit_old/gpu/kernels/sparsemixer.go) in float32 with the sums in expert order:
// the picks, then 1/SUM of each pick's kept terms.
func kernelSparseMixer(s []float32, eps float32) (sel [2]int, w [2]float32) {
	thr := float32(2 * float64(eps))
	first := func(skip int) int {
		best := -1
		for j, x := range s {
			if j != skip && (best < 0 || x > s[best]) {
				best = j
			}
		}
		return best
	}
	i1 := first(-1)
	i2 := first(i1)
	term := func(x, m float32) float32 {
		abs := max(x, x*-1)
		gap := (m - x) / max(abs, m)
		if thr < gap {
			return 0
		}
		return float32(math.Exp(float64(x - m)))
	}
	for p, i := range []int{i1, i2} {
		m := s[i]
		var sum float32
		for j, x := range s {
			if p == 1 && j == i1 {
				continue
			}
			sum += term(x, m)
		}
		w[p] = 1 / sum
	}
	return [2]int{i1, i2}, w
}

// bandedCase is one set of logits, and the margin it is routed under.
type bandedCase struct {
	name string
	eps  float32
	gen  func(rng *rand.Rand) []float32
}

func uniformLogits(n int, lo, hi float32) func(*rand.Rand) []float32 {
	return func(rng *rand.Rand) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = lo + (hi-lo)*rng.Float32()
		}
		return x
	}
}

// bandedRun routes logits x (a matvec against the identity, so the logits are
// x exactly) through the decomposition of a sparse-mixer route, mutated by
// mutate where it is not nil, and returns the selection and weights.
func bandedRun(t *testing.T, x []float32, eps float32, mutate func(*testing.T, *Graph)) (sel [2]int, w [2]float32) {
	t.Helper()
	n := len(x)
	g, xv, selv := routeGraph(routeCase{router: RouterSparseMixer, experts: n, k: 2, dim: n, margin: eps})
	d := Decompose(g)
	var wv ValueID = -1
	for _, o := range d.Ops {
		switch o.Kind {
		case OpRoute:
			t.Fatalf("the route was not decomposed")
		case OpCombine:
			if len(o.In) != 3 {
				t.Fatalf("the combine reads %d inputs, not the pairs, the selection and its weights", len(o.In))
			}
			wv = o.In[2]
		}
	}
	if mutate != nil {
		mutate(t, d)
	}
	id := make([]float32, n*n)
	for i := 0; i < n; i++ {
		id[i*n+i] = 1
	}
	got := evalPrims(t, d, map[ValueID][]float32{xv: x}, map[jlm.Role][]float32{jlm.RoleRouter: id})
	for j := 0; j < 2; j++ {
		sel[j], w[j] = int(got[selv][j]), got[wv][j]
	}
	return sel, w
}

var bandedCases = []bandedCase{
	{"wide logits", 0.01, uniformLogits(16, -3, 3)},
	{"64 experts, wide", 0.01, uniformLogits(64, -3, 3)},
	{"64 experts, clustered under one", 0.01, uniformLogits(64, 0.9, 1)},
	{"clustered, a wide band", 0.2, uniformLogits(32, 0.5, 1)},
	{"all negative", 0.01, uniformLogits(32, -3, -0.1)},
	{"the maximum tied", 0.01, func(rng *rand.Rand) []float32 {
		x := uniformLogits(16, -2, 0)(rng)
		x[3], x[9], x[12] = 0.5, 0.5, 0.5
		return x
	}},
	{"every logit zero (a 0/0 gap)", 0.01, func(*rand.Rand) []float32 { return make([]float32, 16) }},
	{"a logit just inside the clamped band", 0.01, func(rng *rand.Rand) []float32 {
		// 0.9802 is outside the band for a gap over |s| alone ((1-s)/s >
		// 0.02) and inside it for the gap over max(|s|, m).
		x := uniformLogits(8, -1, -0.5)(rng)
		x[0], x[5] = 1, 0.9802
		return x
	}},
}

// TestSparseMixerRouteDecomposesToItsReference holds the banded route's
// decomposition to the kernel's arithmetic (bit for bit: every sum in expert
// order, every constant the kernel's) and to oracle.SparseMixer (the
// selection exactly, the weights to the f64 sums' rounding), then runs the
// oracle's catalogued defects and deliberate breaks of the decomposition
// against the same inputs: each must part from the comparison on some case.
func TestSparseMixerRouteDecomposesToItsReference(t *testing.T) {
	const seeds = 8
	faultShown := map[oracle.SparseMixerFault]bool{}
	for _, c := range bandedCases {
		t.Run(c.name, func(t *testing.T) {
			for seed := int64(1); seed <= seeds; seed++ {
				x := c.gen(rand.New(rand.NewSource(seed)))
				sel, w := bandedRun(t, x, c.eps, nil)
				ksel, kw := kernelSparseMixer(x, c.eps)
				if sel != ksel || w != kw {
					t.Fatalf("seed %d: the decomposition picks %v at %v; the kernel's arithmetic %v at %v", seed, sel, w, ksel, kw)
				}
				ref := oracle.SparseMixer(x, c.eps, oracle.SparseMixerFaultNone)
				for j := 0; j < 2; j++ {
					if int32(sel[j]) != ref.Sel[j] || math.Abs(float64(w[j]-ref.Wt[j])) > 5e-6*math.Abs(float64(ref.Wt[j])) {
						t.Fatalf("seed %d: pick %d is %d at %g; the oracle's %d at %g", seed, j, sel[j], w[j], ref.Sel[j], ref.Wt[j])
					}
				}
				for _, f := range []oracle.SparseMixerFault{oracle.SparseMixerFaultNoMask, oracle.SparseMixerFaultKeepFirst,
					oracle.SparseMixerFaultRenormalised, oracle.SparseMixerFaultAbsFactor} {
					bad := oracle.SparseMixer(x, c.eps, f)
					for j := 0; j < 2; j++ {
						if math.Abs(float64(w[j]-bad.Wt[j])) > 5e-6*math.Abs(float64(bad.Wt[j])) {
							faultShown[f] = true
						}
					}
				}
			}
		})
	}
	for _, f := range []oracle.SparseMixerFault{oracle.SparseMixerFaultNoMask, oracle.SparseMixerFaultKeepFirst,
		oracle.SparseMixerFaultRenormalised, oracle.SparseMixerFaultAbsFactor} {
		if !faultShown[f] {
			t.Errorf("the decomposition matches the oracle under defect %d on every case: the comparison cannot see it", f)
		}
	}

	// Deliberate breaks of the decomposition: each must part from the
	// kernel's arithmetic on some case and seed.
	breaks := map[string]func(*testing.T, *Graph){
		"the threshold is eps, not 2*eps": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "route.w1.over")].Attr.Const /= 2
			d.Ops[opNamed(t, d, "route.w2.over")].Attr.Const /= 2
		},
		"the first pick's mask lets a masked term count 1": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "route.w2.kept")].Attr.Const = 1
		},
		"the first pick is not left out of the second sum": func(t *testing.T, d *Graph) {
			o := &d.Ops[opNamed(t, d, "route.w2.kept")]
			o.In[0] = d.Ops[opNamed(t, d, "route.w2.over")].Out[0]
		},
		"the weights are in the wrong order": func(t *testing.T, d *Graph) {
			o := &d.Ops[opNamed(t, d, "route.weights_b")]
			o.In[1], o.In[2] = o.In[2], o.In[1]
		},
		"the factor is |s| alone": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "route.w1.denominator")].In[1] = d.Ops[opNamed(t, d, "route.w1.denominator")].In[0]
			d.Ops[opNamed(t, d, "route.w2.denominator")].In[1] = d.Ops[opNamed(t, d, "route.w2.denominator")].In[0]
		},
		"the sums run pairwise": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "route.w1.sum")].Attr.Order = OrderTree
			d.Ops[opNamed(t, d, "route.w2.sum")].Attr.Order = OrderTree
		},
		"the second pick's logit is the first's": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "route.m2")].In[0] = d.Ops[opNamed(t, d, "route.m1")].In[0]
		},
	}
	for name, mutate := range breaks {
		shown := false
		for _, c := range bandedCases {
			for seed := int64(1); seed <= seeds && !shown; seed++ {
				x := c.gen(rand.New(rand.NewSource(seed)))
				sel, w := bandedRun(t, x, c.eps, mutate)
				ksel, kw := kernelSparseMixer(x, c.eps)
				shown = sel != ksel || w != kw
			}
		}
		if !shown {
			t.Errorf("%s: the comparison does not see it", name)
		}
	}
}

// TestSparseMixerVariantsStayWhole holds the routes the banded decomposition
// does not state to one op: a pick count other than two, a renormalisation, a
// scale, groups, a selection bias, and a route that states no margin.
func TestSparseMixerVariantsStayWhole(t *testing.T) {
	base := routeCase{router: RouterSparseMixer, experts: 16, k: 2, dim: 8, margin: 0.01}
	if d := Decompose(func() *Graph { g, _, _ := routeGraph(base); return g }()); d.Ops[0].Kind == OpRoute {
		t.Fatalf("the plain banded route stayed whole")
	}
	for name, c := range map[string]routeCase{
		"three picks":      {router: RouterSparseMixer, experts: 16, k: 3, dim: 8, margin: 0.01},
		"renormalised":     {router: RouterSparseMixer, experts: 16, k: 2, dim: 8, margin: 0.01, norm: true},
		"scaled":           {router: RouterSparseMixer, experts: 16, k: 2, dim: 8, margin: 0.01, scale: 2},
		"grouped":          {router: RouterSparseMixer, experts: 16, k: 2, dim: 8, margin: 0.01, groups: 4, used: 2},
		"a selection bias": {router: RouterSparseMixer, experts: 16, k: 2, dim: 8, margin: 0.01, selBias: true},
		"no stated margin": {router: RouterSparseMixer, experts: 16, k: 2, dim: 8},
		"a single expert":  {router: RouterSparseMixer, experts: 1, k: 2, dim: 8, margin: 0.01},
	} {
		g, _, _ := routeGraph(c)
		if d := Decompose(g); d.Ops[0].Kind != OpRoute {
			t.Errorf("%s: decomposed into %s", name, d.Ops[0].Kind)
		}
	}
}

// gateGraph is a shared expert's gate over a row of dim, scaling a row of out.
func gateGraph(dim, out int, w Binding) (g *Graph, x, sd, y ValueID) {
	g = &Graph{Version: Version}
	x = g.value("x", dim)
	sd = g.value("sd", out)
	y = g.value("y", out)
	g.op(Op{Kind: OpGateScalar, In: []ValueID{x, sd}, Out: []ValueID{y}, Weight: []Binding{w}})
	return g, x, sd, y
}

// TestGateScalarDecomposesToItsReference holds the gate's decomposition to
// sharedExpert's arithmetic (engine/model/moe.go): the gate's logit a dot of
// the row, the sigmoid of it, and the shared expert's output times that.
func TestGateScalarDecomposesToItsReference(t *testing.T) {
	const dim, out = 24, 24
	breaks := map[string]func(*testing.T, *Graph){
		"the gate is not a sigmoid": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "gate.score")].Attr.Fn = MapExp
		},
		"the output is added, not scaled": func(t *testing.T, d *Graph) {
			d.Ops[len(d.Ops)-1].Attr.Fn = MapAdd
		},
		"the gate reads the expert's output, not the row": func(t *testing.T, d *Graph) {
			d.Ops[opNamed(t, d, "gate.logit")].In[0] = d.Ops[len(d.Ops)-1].In[0]
		},
	}
	shown := map[string]bool{}
	for seed := int64(1); seed <= 8; seed++ {
		rng := rand.New(rand.NewSource(seed))
		fill := func(n int, s float32) []float32 {
			f := make([]float32, n)
			for i := range f {
				f[i] = (rng.Float32()*2 - 1) * s
			}
			return f
		}
		x, sdv, gate := fill(dim, 2), fill(out, 3), fill(dim, 0.5)
		// sharedExpert's arithmetic: the logit, its sigmoid, the output times it.
		var logit float32
		for i := range x {
			logit += gate[i] * x[i]
		}
		s := float32(1 / (1 + math.Exp(-float64(logit))))
		want := make([]float32, out)
		for i := range want {
			want[i] = sdv[i] * s
		}
		run := func(mutate func(*testing.T, *Graph)) []float32 {
			g, xv, sdn, y := gateGraph(dim, out, Binding{Role: jlm.RoleShRouter, K: dim})
			d := Decompose(g)
			for _, o := range d.Ops {
				if o.Kind == OpGateScalar {
					t.Fatalf("the gate was not decomposed")
				}
			}
			if mutate != nil {
				mutate(t, d)
			}
			return evalPrims(t, d, map[ValueID][]float32{xv: x, sdn: sdv}, map[jlm.Role][]float32{jlm.RoleShRouter: gate})[y]
		}
		if got := run(nil); fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("seed %d: the decomposition is %v; the reference %v", seed, got, want)
		}
		for name, mutate := range breaks {
			if fmt.Sprint(run(mutate)) != fmt.Sprint(want) {
				shown[name] = true
			}
		}
	}
	for name := range breaks {
		if !shown[name] {
			t.Errorf("%s: the comparison does not see it", name)
		}
	}
}

// TestGateScalarVariantsStayWhole holds a gate that is not a vector over the
// row to one op: a matrix of several rows, or a width the row does not have.
func TestGateScalarVariantsStayWhole(t *testing.T) {
	for name, w := range map[string]Binding{
		"a matrix of four rows": {Role: jlm.RoleShRouter, K: 24, Rows: 4},
		"a bank of sheets":      {Role: jlm.RoleShRouter, K: 24, Sheets: 2},
		"the wrong width":       {Role: jlm.RoleShRouter, K: 16},
	} {
		g, _, _, _ := gateGraph(24, 24, w)
		if d := Decompose(g); d.Ops[0].Kind != OpGateScalar {
			t.Errorf("%s: decomposed into %s", name, d.Ops[0].Kind)
		}
	}
	// One row stored as a matrix of one row is the same vector.
	g, _, _, _ := gateGraph(24, 24, Binding{Role: jlm.RoleShRouter, K: 24, Rows: 1})
	if d := Decompose(g); d.Ops[0].Kind == OpGateScalar {
		t.Errorf("a gate of one row stayed whole")
	}
}
