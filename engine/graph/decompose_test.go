package graph

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// evalPrims runs g's ops over one token by the primitives' definitions, in
// float32 with every reduction in its stated order: the reference a
// decomposition is held to. A matvec's weight is read as rows of K floats.
// Index values are held as their float32 values.
func evalPrims(t *testing.T, g *Graph, in map[ValueID][]float32, w map[jlm.Role][]float32) map[ValueID][]float32 {
	t.Helper()
	v := map[ValueID][]float32{}
	for id, x := range in {
		v[id] = x
	}
	for _, o := range g.Ops {
		width := g.Values[o.Out[0]].Width
		out := make([]float32, width)
		switch o.Kind {
		case OpMatVec:
			x, m := v[o.In[0]], w[o.Weight[0].Role]
			k := len(x)
			for r := range out {
				var s float32
				for c := 0; c < k; c++ {
					s += m[r*k+c] * x[c]
				}
				out[r] = s
			}
		case OpNorm:
			// An RMSNorm in float32 with float64 statistics: the oracle's.
			oracle.RMSNorm32(out, v[o.In[0]], w[o.Weight[0].Role], o.Attr.Eps)
		case OpAdd:
			a, b := v[o.In[0]], v[o.In[1]]
			for i := range out {
				out[i] = a[i] + b[i]
			}
		case OpMap:
			var ops [][]float32
			for _, id := range o.In {
				ops = append(ops, v[id])
			}
			if len(o.Weight) > 0 {
				ops = append(ops, w[o.Weight[0].Role])
			}
			if len(ops) < o.Attr.Fn.Arity() {
				ops = append(ops, []float32{float32(o.Attr.Const)})
			}
			at := func(op []float32, i int) float32 { return op[i/(width/len(op))] }
			for i := range out {
				a := at(ops[0], i)
				switch o.Attr.Fn {
				case MapAdd:
					out[i] = a + at(ops[1], i)
				case MapSub:
					out[i] = a - at(ops[1], i)
				case MapMul:
					out[i] = a * at(ops[1], i)
				case MapDiv:
					out[i] = a / at(ops[1], i)
				case MapMax:
					out[i] = max(a, at(ops[1], i))
				case MapMin:
					out[i] = min(a, at(ops[1], i))
				case MapExp:
					out[i] = float32(math.Exp(float64(a)))
				case MapSigmoid:
					out[i] = float32(1 / (1 + math.Exp(-float64(a))))
				case MapSoftplus:
					out[i] = float32(math.Log1p(math.Exp(float64(a))))
				case MapSqrt:
					out[i] = float32(math.Sqrt(float64(a)))
				case MapWhere:
					out[i] = at(ops[2], i)
					if a > 0 {
						out[i] = at(ops[1], i)
					}
				case MapLess:
					if a < at(ops[1], i) {
						out[i] = 1
					}
				case MapFma:
					out[i] = float32(math.FMA(float64(a), float64(at(ops[1], i)), float64(at(ops[2], i))))
				default:
					t.Fatalf("map %s", o.Attr.Fn)
				}
			}
		case OpReduce:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			for gi := range out {
				grp := x[gi*span : (gi+1)*span]
				var acc float32
				switch o.Attr.Fn {
				case MapAdd:
				case MapMax, MapMin:
					acc = grp[0]
				}
				for _, e := range grp {
					switch o.Attr.Fn {
					case MapAdd:
						acc += e
					case MapMax:
						acc = max(acc, e)
					case MapMin:
						acc = min(acc, e)
					}
				}
				out[gi] = acc
			}
		case OpTopK, OpRank:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			rank := func(grp []float32, e int) int {
				n := 0
				for f, y := range grp {
					if y > grp[e] || y == grp[e] && f < e {
						n++
					}
				}
				return n
			}
			if o.Kind == OpRank {
				for e := range out {
					gi := e / span
					out[e] = float32(rank(x[gi*span:(gi+1)*span], e%span))
				}
				break
			}
			k := o.Attr.TopK
			vals := make([]float32, width)
			for gi := 0; gi < len(x)/span; gi++ {
				grp := x[gi*span : (gi+1)*span]
				for e := range grp {
					if r := rank(grp, e); r < k {
						out[gi*k+r], vals[gi*k+r] = float32(e), grp[e]
					}
				}
			}
			v[o.Out[1]] = vals
		case OpTake:
			x, ix := v[o.In[0]], v[o.In[1]]
			for j := range out {
				out[j] = x[int(ix[j])]
			}
		case OpTile:
			x := v[o.In[0]]
			span := o.Attr.Span
			if span == 0 {
				span = len(x)
			}
			for i := range out {
				out[i] = x[i%span]
			}
		case OpSlice:
			// The whole-row form: Width elements from Offset.
			copy(out, v[o.In[0]][o.Attr.Offset:o.Attr.Offset+o.Attr.Width])
		case OpAdd:
			a, b := v[o.In[0]], v[o.In[1]]
			for i := range out {
				out[i] = a[i] + b[i]
			}
		case OpHeadNorm:
			// A weightless norm of each head: x * (1/sqrt(mean(x^2) + eps)),
			// the mean accumulated in f64 as the oracle's does.
			x, hd := v[o.In[0]], o.Attr.HeadDim
			for h := 0; h < len(x)/hd; h++ {
				var sum float64
				for _, e := range x[h*hd : (h+1)*hd] {
					sum += float64(e) * float64(e)
				}
				scale := 1 / math.Sqrt(sum/float64(hd)+o.Attr.Eps)
				for i := 0; i < hd; i++ {
					out[h*hd+i] = float32(float64(x[h*hd+i]) * scale)
				}
			}
		case OpAct, OpActMul:
			x := v[o.In[0]]
			for i := range out {
				e := float64(x[i])
				switch {
				case o.Kind == OpAct && o.Attr.Act == ActSiLU:
					out[i] = float32(e / (1 + math.Exp(-e)))
				case o.Kind == OpActMul && o.Attr.Act == ActSigmoid:
					out[i] = float32(1/(1+math.Exp(-e))) * v[o.In[1]][i]
				default:
					t.Fatalf("activation %d of %s", o.Attr.Act, o.Kind)
				}
			}
		case OpCombine:
			continue
		default:
			t.Fatalf("op %s is not a primitive", o.Kind)
		}
		v[o.Out[0]] = out
	}
	return v
}

// routeCase is one router shape the decomposition is held to the oracle on.
type routeCase struct {
	name            string
	router          Router
	norm            bool
	scale           float64
	groups, used    int
	bias, selBias   bool
	experts, k, dim int
	ties            bool // every logit equal: the tie rule decides the selection
}

// routeGraph is a route over a row of c.dim, its selection read by a combine.
func routeGraph(c routeCase) (*Graph, ValueID, ValueID) {
	g := &Graph{Version: Version}
	x := g.value("x", c.dim)
	sel := g.value("sel", c.k)
	ws := []Binding{{Role: jlm.RoleRouter, K: c.dim, Rows: c.experts}}
	if c.bias {
		ws = append(ws, Binding{Role: jlm.RoleRouterBias, K: c.experts})
	}
	if c.selBias {
		ws = append(ws, Binding{Role: jlm.RoleExpProbsB, K: c.experts})
	}
	g.op(Op{Kind: OpRoute, In: []ValueID{x}, Out: []ValueID{sel}, Weight: ws,
		Attr: Attrs{Experts: c.experts, TopK: c.k, Router: c.router, RouteNorm: c.norm, Scale: c.scale,
			Groups: c.groups, GroupsUsed: c.used}})
	pairs := g.value("pairs", c.k*4)
	y := g.value("y", 4)
	g.op(Op{Kind: OpCombine, In: []ValueID{pairs, sel}, Out: []ValueID{y}})
	return g, x, sel
}

// TestRouteDecomposesToTheOracle holds the route's decomposition into
// primitives to oracle.MoERouter -- the router's reference, its selection
// exactly and its weights to a few ulps -- over softmax routers renormalised
// or not and with a selection bias on the probability, sigmoid and
// sqrt-softplus gates with a logit bias, a selection bias, groups and a
// scale, and every logit tied. It runs the defects the oracle catalogues
// against the same inputs over eight seeds: each one a case can show must part
// from the decomposition on one of them, or the comparison cannot see it.
func TestRouteDecomposesToTheOracle(t *testing.T) {
	cases := []routeCase{
		{name: "softmax renormalised", router: RouterSoftmax, norm: true, experts: 64, k: 8, dim: 48},
		{name: "softmax over every expert", router: RouterSoftmax, experts: 64, k: 8, dim: 48},
		{name: "softmax, selection bias on the probability", router: RouterSoftmax, norm: true, selBias: true, experts: 64, k: 8, dim: 48},
		{name: "sigmoid, groups, bias, scale", router: RouterSigmoid, norm: true, scale: 2.5, groups: 8, used: 4, selBias: true, experts: 64, k: 8, dim: 48},
		{name: "sigmoid, logit bias, no groups", router: RouterSigmoid, norm: true, bias: true, experts: 32, k: 4, dim: 40},
		{name: "sqrt-softplus, selection bias", router: RouterSqrtSoftplus, norm: true, scale: 1.5, selBias: true, experts: 48, k: 6, dim: 40},
		{name: "sigmoid, groups, every logit tied", router: RouterSigmoid, norm: true, groups: 4, used: 2, selBias: true, experts: 32, k: 4, dim: 16, ties: true},
		{name: "softmax, every logit tied", router: RouterSoftmax, norm: true, experts: 16, k: 4, dim: 16, ties: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			seen := map[oracle.MoEFault]bool{}
			shown := map[oracle.MoEFault]bool{}
			for seed := int64(1); seed <= 8; seed++ {
				routeSeed(t, c, seed, seen, shown)
			}
			for f := range seen {
				if !shown[f] {
					t.Errorf("the decomposition matches the oracle under %q on every seed: the comparison cannot see it", f)
				}
			}
		})
	}
}

// routeSeed holds one seed's route to the oracle, and records each defect the
// case can show (seen) and whether it parted from the decomposition (shown).
func routeSeed(t *testing.T, c routeCase, seed int64, seen, shown map[oracle.MoEFault]bool) {
	t.Helper()
	{
		{
			rng := rand.New(rand.NewSource(seed))
			fill := func(n int, s float32) []float32 {
				f := make([]float32, n)
				for i := range f {
					f[i] = (rng.Float32()*2 - 1) * s
				}
				return f
			}
			x := fill(c.dim, 1)
			w := map[jlm.Role][]float32{jlm.RoleRouter: fill(c.experts*c.dim, 0.5)}
			if c.ties {
				w[jlm.RoleRouter] = make([]float32, c.experts*c.dim)
			}
			if c.bias {
				w[jlm.RoleRouterBias] = fill(c.experts, 0.3)
			}
			if c.selBias {
				w[jlm.RoleExpProbsB] = fill(c.experts, 0.05)
				if c.ties {
					w[jlm.RoleExpProbsB] = make([]float32, c.experts)
				}
			}
			g, xv, sel := routeGraph(c)
			d := Decompose(g)
			for _, o := range d.Ops {
				if o.Kind == OpRoute {
					t.Fatalf("the route was not decomposed")
				}
			}
			var wv ValueID = -1
			for _, o := range d.Ops {
				if o.Kind == OpCombine {
					if len(o.In) != 3 {
						t.Fatalf("the combine reads %d inputs, not the pairs, the selection and its weights", len(o.In))
					}
					wv = o.In[2]
				}
			}
			got := evalPrims(t, d, map[ValueID][]float32{xv: x}, w)

			// The oracle's inputs: the logits by the same matvec, gated or
			// softmaxed in float64 and rounded, as the reference takes them.
			logits := make([]float32, c.experts)
			for r := range logits {
				var s float32
				for i := 0; i < c.dim; i++ {
					s += w[jlm.RoleRouter][r*c.dim+i] * x[i]
				}
				if c.bias {
					s += w[jlm.RoleRouterBias][r]
				}
				logits[r] = s
			}
			scores := make([]float32, c.experts)
			switch c.router {
			case RouterSigmoid:
				for e, l := range logits {
					scores[e] = float32(1 / (1 + math.Exp(-float64(l))))
				}
			case RouterSqrtSoftplus:
				for e, l := range logits {
					scores[e] = float32(math.Sqrt(math.Log1p(math.Exp(float64(l)))))
				}
			default:
				m := logits[0]
				for _, l := range logits {
					m = max(m, l)
				}
				var z float32
				for _, l := range logits {
					z += float32(math.Exp(float64(l - m)))
				}
				for e, l := range logits {
					scores[e] = float32(math.Exp(float64(l-m))) / z
				}
			}
			gate := oracle.MoEGate{NGroup: c.groups, NGroupUsed: c.used, Norm: c.norm, Scale: float32(c.scale)}
			if c.selBias {
				gate.Bias = w[jlm.RoleExpProbsB]
			}
			ref := func(f oracle.MoEFault) oracle.MoERoute { return oracle.MoERouter(scores, c.k, gate, f) }
			same := func(r oracle.MoERoute) (bool, string) {
				for j := 0; j < c.k; j++ {
					if int(got[sel][j]) != int(r.Sel[j]) {
						return false, fmt.Sprintf("pick %d is expert %d, the oracle's %d", j, int(got[sel][j]), r.Sel[j])
					}
					if d := math.Abs(float64(got[wv][j] - r.Wt[j])); d > 4e-7*math.Max(1, math.Abs(float64(r.Wt[j]))) {
						return false, fmt.Sprintf("weight %d is %g, the oracle's %g", j, got[wv][j], r.Wt[j])
					}
				}
				return true, ""
			}
			if ok, why := same(ref(oracle.MoEFaultNone)); !ok {
				t.Fatalf("seed %d: the decomposition parts from the oracle: %s", seed, why)
			}
			// The defects these inputs can show must part from it.
			for _, f := range oracle.MoEFaults {
				shows := false
				switch f {
				case oracle.MoEFaultLastOfTie:
					shows = c.ties
				case oracle.MoEFaultBiasedWeights:
					shows = c.selBias && !c.ties
				case oracle.MoEFaultGroupTop1, oracle.MoEFaultMaskAfterTopK:
					shows = c.groups > 1 && !c.ties
				case oracle.MoEFaultScaleBeforeNorm:
					shows = c.norm && c.scale != 0 && c.scale != 1
				}
				if !shows {
					continue
				}
				seen[f] = true
				if ok, _ := same(ref(f)); !ok {
					shown[f] = true
				}
			}
		}
	}
}

// TestASparseMixerRouteStaysWhole is the route the primitives here do not yet
// state: it stays one op, counted as composite where it reaches the device.
func TestASparseMixerRouteStaysWhole(t *testing.T) {
	g, _, _ := routeGraph(routeCase{router: RouterSparseMixer, experts: 16, k: 2, dim: 8})
	d := Decompose(g)
	if d.Ops[0].Kind != OpRoute {
		t.Fatalf("a sparsemixer route was decomposed into %s", d.Ops[0].Kind)
	}
}
