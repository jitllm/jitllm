package graph

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// altCase is one AltUp shape the decompositions are held to the host on.
type altCase struct {
	name       string
	n, d, p    int
	act        Act
	eps        float64
	magnitudes float32 // the router's input scale: large drives tanh to saturation
}

// altRef transcribes the host's altPredict and altCorrect
// (engine/model/altup.go), over one row, in float32: the coefficient sums by
// fused accumulate from zero, every stream in order. fused false is the same
// routine with the multiply rounded before the add, the form the SSE tier
// runs.
type altRef struct {
	altCase
	t     *testing.T
	fused bool
}

func (r altRef) fma(a, b, c float32) float32 {
	if r.fused {
		return fma32(a, b, c)
	}
	return float32(a*b) + c
}

// expf is the engine's exp as the primitives' evaluator defines it.
func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

// modalities is tanh(R rmsnorm(x0) / d), the tanh from exp as the softcap
// kernel computes it.
func (r altRef) modalities(x0, norm, router []float32) []float32 {
	h := make([]float32, r.d)
	oracle.RMSNorm32(h, x0, norm, r.eps)
	for i := range h {
		h[i] *= float32(1 / float64(r.d))
	}
	m := make([]float32, r.n)
	for i := range m {
		var s float32
		for c := 0; c < r.d; c++ {
			s += router[i*r.d+c] * h[c]
		}
		z := expf(s*2) + 1
		m[i] = 1 - 2/z
	}
	return m
}

// coefs is sum_i m_i T_i, the table's rows of width w, accumulated from zero.
func (r altRef) coefs(m, t []float32, w int) []float32 {
	c := make([]float32, w)
	for i, mi := range m {
		for j := range c {
			c[j] = r.fma(mi, t[i*w+j], c[j])
		}
	}
	return c
}

func (r altRef) predict(x, norm, router, t []float32) []float32 {
	n, d := r.n, r.d
	c := r.coefs(r.modalities(x[:d], norm, router), t, n*n)
	pred := append([]float32(nil), x...)
	for k := 0; k < n; k++ {
		for j := 0; j < n; j++ {
			for e := 0; e < d; e++ {
				pred[k*d+e] = r.fma(c[k*n+j], x[j*d+e], pred[k*d+e])
			}
		}
	}
	return pred
}

func (r altRef) correct(pred, out, ple, norm, router, t, scale, gate, proj, post []float32) []float32 {
	n, d, p := r.n, r.d, r.p
	cc := r.coefs(r.modalities(out, norm, router), t, n)
	for i := range cc {
		cc[i] += 1
	}
	inn := make([]float32, d)
	for e := range inn {
		inn[e] = out[e] - pred[e]
	}
	x := append([]float32(nil), pred...)
	for k := 0; k < n; k++ {
		for e := 0; e < d; e++ {
			x[k*d+e] = r.fma(cc[k], inn[e], x[k*d+e])
		}
	}
	h := make([]float32, d)
	for e := range h {
		h[e] = x[e] * scale[e]
	}
	g := make([]float32, p)
	for i := range g {
		var s float32
		for c := 0; c < d; c++ {
			s += gate[i*d+c] * h[c]
		}
		g[i] = float32(refAct(r.t, r.act, float64(s))) * ple[i]
	}
	po := make([]float32, d)
	for i := range po {
		var s float32
		for c := 0; c < p; c++ {
			s += proj[i*p+c] * g[c]
		}
		po[i] = s
	}
	pn := make([]float32, d)
	oracle.RMSNorm32(pn, po, post, r.eps)
	for k := 1; k < n; k++ {
		for e := 0; e < d; e++ {
			x[k*d+e] += pn[e]
		}
	}
	return x
}

// altGraphs builds one block's prologue and its epilogue as the dense
// builder states them.
func altGraphs(c altCase) (pg *Graph, cg *Graph) {
	n, d, p := c.n, c.d, c.p
	pg = &Graph{Version: Version}
	x := pg.value("x", n*d)
	pg.In = []ValueID{x}
	pred, act := pg.value("alt_pred", n*d), pg.value("alt_active", d)
	pg.op(Op{Kind: OpAltPredict, In: []ValueID{x}, Out: []ValueID{pred, act},
		Weight: []Binding{{Role: jlm.RoleAltUpRouterNorm, K: d}, {Role: jlm.RoleAltUpRouter, K: d, Rows: n},
			{Role: jlm.RoleAltUpPredCoef, K: n * n, Rows: n}}, Attr: Attrs{Streams: n, Eps: c.eps}})
	pg.Out = []ValueID{pred, act}

	cg = &Graph{Version: Version}
	pr, out, ple := cg.value("alt_pred", n*d), cg.value("x_out", d), cg.value("ple_in", p)
	cg.In = []ValueID{pr, out, ple}
	xs := cg.value("x_streams", n*d)
	cg.op(Op{Kind: OpAltCorrect, In: []ValueID{pr, out, ple}, Out: []ValueID{xs}, Weight: []Binding{
		{Role: jlm.RoleAltUpRouterNorm, K: d}, {Role: jlm.RoleAltUpRouter, K: d, Rows: n},
		{Role: jlm.RoleAltUpCorrCoef, K: n, Rows: n}, {Role: jlm.RoleAltUpCorrScale, K: d},
		{Role: jlm.RolePLEGate, K: d, Rows: p}, {Role: jlm.RolePLEProj, K: p, Rows: d},
		{Role: jlm.RolePLEPostNorm, K: d}}, Attr: Attrs{Streams: n, Eps: c.eps, Act: c.act}})
	cg.Out = []ValueID{xs}
	return pg, cg
}

// altInputs draws every input and weight of c from seed.
func altInputs(c altCase, seed int64) (map[jlm.Role][]float32, []float32, []float32, []float32, []float32) {
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, s float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = (rng.Float32()*2 - 1) * s
		}
		return f
	}
	w := map[jlm.Role][]float32{
		jlm.RoleAltUpRouterNorm: fill(c.d, 1),
		jlm.RoleAltUpRouter:     fill(c.n*c.d, 0.5*c.magnitudes),
		jlm.RoleAltUpPredCoef:   fill(c.n*c.n*c.n, 0.6),
		jlm.RoleAltUpCorrCoef:   fill(c.n*c.n, 0.6),
		jlm.RoleAltUpCorrScale:  fill(c.d, 1),
		jlm.RolePLEGate:         fill(c.p*c.d, 0.4),
		jlm.RolePLEProj:         fill(c.d*c.p, 0.4),
		jlm.RolePLEPostNorm:     fill(c.d, 1),
	}
	return w, fill(c.n*c.d, 1), fill(c.n*c.d, 1), fill(c.d, 1), fill(c.p, 1)
}

var altCases = []altCase{
	{name: "four streams", n: 4, d: 24, p: 10, act: ActGELUTanh, eps: 1e-6, magnitudes: 1},
	{name: "three streams, tanh saturated", n: 3, d: 16, p: 7, act: ActGELUTanh, eps: 1e-6, magnitudes: 40},
	{name: "two streams, silu", n: 2, d: 9, p: 5, act: ActSiLU, eps: 1e-5, magnitudes: 1},
}

func sameBits(a, b []float32) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i, false
		}
	}
	return 0, true
}

// altMutations are deliberately wrong variants of the decomposition: each
// edits one op, found by the name of the value it writes.
var altMutations = []struct {
	name string
	in   string // the value written by the op to edit
	edit func(o *Op)
}{
	{"a wrong constant in the width scale", "alt.scaled", func(o *Op) { o.Attr.Const *= 1.01 }},
	{"the exponential's argument not doubled", "alt.modalities.twice", func(o *Op) { o.Attr.Const = 1 }},
	{"tanh's difference reversed", "alt.modalities", func(o *Op) { o.In[0], o.In[1] = o.In[1], o.In[0] }},
	{"every coefficient row the first", "alt.coef_row", func(o *Op) { o.Attr.Offset = 0 }},
	{"the accumulator dropped", "alt.coefs", func(o *Op) {
		if len(o.In) == 3 {
			o.Attr.Fn = MapMul
			o.In = o.In[:2]
		}
	}},
	{"every coefficient column the first", "alt.coef_col", func(o *Op) { o.Attr.Offset = 0 }},
	{"a stream tiled from the wrong offset", "alt.stream", func(o *Op) { o.Attr.Offset = 0 }},
	{"the change taken the wrong way", "alt.change", func(o *Op) { o.In[0], o.In[1] = o.In[1], o.In[0] }},
	{"the coefficients' one left out", "alt.coefs_one", func(o *Op) { o.Attr.Const = 0 }},
	{"the per-layer input added to the active stream too", "alt.others_in", func(o *Op) { o.Attr.Offset = 0 }},
}

// TestAltUpDecomposesToItsReference holds the prologue's and epilogue's
// decompositions to the host routines (engine/model/altup.go altPredict,
// altCorrect) bit for bit over several seeds, then to the same routine with
// the accumulate unfused to show the comparison sees a fused multiply-add,
// then to each deliberately wrong variant of the decomposition.
func TestAltUpDecomposesToItsReference(t *testing.T) {
	for _, c := range altCases {
		t.Run(c.name, func(t *testing.T) {
			pg, cg := altGraphs(c)
			pd, cd := Decompose(pg), Decompose(cg)
			for _, d := range []*Graph{pd, cd} {
				for _, o := range d.Ops {
					if o.Kind == OpAltPredict || o.Kind == OpAltCorrect {
						t.Fatalf("%s was not decomposed", o.Kind)
					}
				}
			}
			ref := altRef{altCase: c, t: t, fused: true}
			unfused := altRef{altCase: c, t: t}
			sawUnfused := 0
			for seed := int64(1); seed <= 6; seed++ {
				w, x, pred, out, ple := altInputs(c, seed)
				got := evalPrims(t, pd, map[ValueID][]float32{pg.In[0]: x}, w)
				want := ref.predict(x, w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter], w[jlm.RoleAltUpPredCoef])
				if i, ok := sameBits(got[pd.Out[0]], want); !ok {
					t.Fatalf("seed %d: prediction %d is %g, the host's %g", seed, i, got[pd.Out[0]][i], want[i])
				}
				if i, ok := sameBits(got[pd.Out[1]], want[:c.d]); !ok {
					t.Fatalf("seed %d: the active stream %d is %g, the host's %g", seed, i, got[pd.Out[1]][i], want[i])
				}
				if _, ok := sameBits(unfused.predict(x, w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter],
					w[jlm.RoleAltUpPredCoef]), want); !ok {
					sawUnfused++
				}

				cin := map[ValueID][]float32{cg.In[0]: pred, cg.In[1]: out, cg.In[2]: ple}
				gotC := evalPrims(t, cd, cin, w)
				args := func(r altRef) []float32 {
					return r.correct(pred, out, ple, w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter], w[jlm.RoleAltUpCorrCoef],
						w[jlm.RoleAltUpCorrScale], w[jlm.RolePLEGate], w[jlm.RolePLEProj], w[jlm.RolePLEPostNorm])
				}
				wantC := args(ref)
				if i, ok := sameBits(gotC[cd.Out[0]], wantC); !ok {
					t.Fatalf("seed %d: corrected stream element %d is %g, the host's %g", seed, i, gotC[cd.Out[0]][i], wantC[i])
				}
				if _, ok := sameBits(args(unfused), wantC); !ok {
					sawUnfused++
				}

				// The modalities are a tanh: held to the library's.
				m := ref.modalities(x[:c.d], w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter])
				h := make([]float32, c.d)
				oracle.RMSNorm32(h, x[:c.d], w[jlm.RoleAltUpRouterNorm], c.eps)
				for i := range m {
					var s float64
					for e := 0; e < c.d; e++ {
						s += float64(w[jlm.RoleAltUpRouter][i*c.d+e]) * float64(h[e]) / float64(c.d)
					}
					if d := math.Abs(float64(m[i]) - math.Tanh(s)); d > 1e-5 {
						t.Fatalf("seed %d: modality %d is %g, tanh gives %g", seed, i, m[i], math.Tanh(s))
					}
				}
			}
			if sawUnfused == 0 {
				t.Errorf("the unfused routine matches the fused on every seed: the comparison cannot see a fused multiply-add")
			}

			// Each wrong variant must part from the host on some seed.
			for _, mu := range altMutations {
				t.Run(mu.name, func(t *testing.T) {
					edited := 0
					parted := false
					for seed := int64(1); seed <= 6 && !parted; seed++ {
						w, x, pred, out, ple := altInputs(c, seed)
						for gi, d := range []*Graph{pd, cd} {
							bad := &Graph{Version: d.Version, Values: d.Values, In: d.In, Out: d.Out}
							for _, o := range d.Ops {
								o.In = append([]ValueID(nil), o.In...)
								if d.Values[o.Out[0]].Name == mu.in {
									mu.edit(&o)
									edited++
								}
								bad.Ops = append(bad.Ops, o)
							}
							in := map[ValueID][]float32{pg.In[0]: x}
							want := ref.predict(x, w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter], w[jlm.RoleAltUpPredCoef])
							if gi == 1 {
								in = map[ValueID][]float32{cg.In[0]: pred, cg.In[1]: out, cg.In[2]: ple}
								want = ref.correct(pred, out, ple, w[jlm.RoleAltUpRouterNorm], w[jlm.RoleAltUpRouter],
									w[jlm.RoleAltUpCorrCoef], w[jlm.RoleAltUpCorrScale], w[jlm.RolePLEGate], w[jlm.RolePLEProj],
									w[jlm.RolePLEPostNorm])
							}
							if _, ok := sameBits(evalPrims(t, bad, in, w)[bad.Out[0]], want); !ok {
								parted = true
							}
						}
					}
					if edited == 0 {
						t.Fatalf("no op writes %q: the variant edits nothing", mu.in)
					}
					if !parted {
						t.Errorf("the host and the decomposition agree under %q on every seed: the comparison cannot see it", mu.name)
					}
				})
			}
		})
	}
}

// TestAltUpTanhIsTheEnginesTanh holds the tanh the decompositions write to
// the library's across the range the softcap kernel saturates over, and to
// NaN in giving NaN.
func TestAltUpTanhIsTheEnginesTanh(t *testing.T) {
	g := &Graph{Version: Version}
	x := g.value("x", 9)
	y := prims{d: g}.tanh("t", x, 9)
	in := []float32{-30, -9, -1, -1e-3, 0, 1e-3, 1, 9, 30}
	got := evalPrims(t, g, map[ValueID][]float32{x: in}, nil)[y]
	for i, v := range in {
		if d := math.Abs(float64(got[i]) - math.Tanh(float64(v))); d > 2e-7 {
			t.Errorf("tanh(%g) = %g, want %g", v, got[i], math.Tanh(float64(v)))
		}
	}
}

// TestAltUpVariantsStayWhole is the shapes the decomposition does not state:
// one stream, a binding the epilogue is not given, a width the streams do not
// divide, a per-layer input of the wrong width.
func TestAltUpVariantsStayWhole(t *testing.T) {
	c := altCases[0]
	kinds := func(g *Graph) (alt, mapOps int) {
		for _, o := range g.Ops {
			switch o.Kind {
			case OpAltPredict, OpAltCorrect:
				alt++
			case OpMap:
				mapOps++
			}
		}
		return
	}
	for _, v := range []struct {
		name  string
		whole []OpKind // the ops the edit leaves undecomposable
		edit  func(pg, cg *Graph)
	}{
		{"one stream", []OpKind{OpAltPredict, OpAltCorrect}, func(pg, cg *Graph) { pg.Ops[0].Attr.Streams, cg.Ops[0].Attr.Streams = 1, 1 }},
		{"a width the streams do not divide", []OpKind{OpAltPredict, OpAltCorrect}, func(pg, cg *Graph) { pg.Ops[0].Attr.Streams, cg.Ops[0].Attr.Streams = 5, 5 }},
		{"an epilogue without its post norm", []OpKind{OpAltCorrect}, func(pg, cg *Graph) { cg.Ops[0].Weight = cg.Ops[0].Weight[:6] }},
		{"a prologue with a role it does not know", []OpKind{OpAltPredict}, func(pg, cg *Graph) { pg.Ops[0].Weight[2].Role = jlm.RoleAltUpCorrCoef }},
		{"a per-layer input that is not the gate's width", []OpKind{OpAltCorrect}, func(pg, cg *Graph) { cg.Values[cg.In[2]].Width++ }},
	} {
		t.Run(v.name, func(t *testing.T) {
			pg, cg := altGraphs(c)
			v.edit(pg, cg)
			for _, g := range []*Graph{pg, cg} {
				d := Decompose(g)
				alt, maps := kinds(d)
				if slices.Contains(v.whole, g.Ops[0].Kind) {
					if alt != 1 || maps != 0 || len(d.Ops) != 1 {
						t.Errorf("%s: %d AltUp ops and %d maps among %d ops, want the one op whole", g.Ops[0].Kind, alt, maps, len(d.Ops))
					}
				} else if alt != 0 {
					t.Errorf("%s stayed whole under an edit that does not touch it", g.Ops[0].Kind)
				}
			}
		})
	}
	// The unedited graphs decompose, to primitives only.
	pg, cg := altGraphs(c)
	for _, g := range []*Graph{pg, cg} {
		if alt, _ := kinds(Decompose(g)); alt != 0 {
			t.Errorf("%s stayed whole", fmt.Sprint(g.Ops[0].Kind))
		}
	}
}
