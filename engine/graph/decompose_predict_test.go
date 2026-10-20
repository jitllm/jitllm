package graph

import (
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// predictCase is a prediction prologue over a residual of s streams of d.
type predictCase struct {
	name string
	s, d int
	eps  float64
}

var predictCases = []predictCase{
	{name: "one stream", s: 1, d: 16, eps: 1e-6},
	{name: "one stream, ragged width", s: 1, d: 13, eps: 1e-5},
	{name: "four streams", s: 4, d: 16, eps: 1e-6},
	{name: "four streams, ragged width", s: 4, d: 13, eps: 1e-5},
	{name: "three streams", s: 3, d: 10, eps: 1e-6},
}

const (
	rEnorm, rHnorm, rEH = jlm.RoleNextnENorm, jlm.RoleNextnHNorm, jlm.RoleNextnEHProj
)

// predictGraph is the prologue as the block's graph states it: the three
// bindings the prologue reads and the four the draft reads after the block.
func predictGraph(c predictCase) (g *Graph, h, e, out ValueID) {
	g = &Graph{Version: Version}
	hd := c.s * c.d
	h, e, out = g.value("x", hd), g.value("next_embd", c.d), g.value("x_predict", hd)
	g.In = []ValueID{h, e}
	g.op(Op{Kind: OpPredict, In: []ValueID{h, e}, Out: []ValueID{out}, Attr: Attrs{Eps: c.eps}, Weight: []Binding{
		{Role: rEH, K: 2 * c.d, Rows: c.d}, {Role: rEnorm, K: c.d}, {Role: rHnorm, K: hd},
		{Role: jlm.RoleNextnHeadNorm, K: hd}, {Role: jlm.RoleNextnHead, K: c.d, Rows: 50},
		{Role: jlm.RoleNextnEmbd, K: c.d, Rows: 50}, {Role: jlm.RoleHCHeadNorm, K: hd}}})
	return g, h, e, out
}

func predictInputs(c predictCase, seed int64) (hid, emb []float32, w map[jlm.Role][]float32) {
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, s, off float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = off + (rng.Float32()*2-1)*s
		}
		return f
	}
	hd := c.s * c.d
	return fill(hd, 2, 0), fill(c.d, 2, 0), map[jlm.Role][]float32{
		rEnorm: fill(c.d, 0.5, 1), rHnorm: fill(hd, 0.5, 1), rEH: fill(2*c.d*c.d, 0.4, 0)}
}

// predictReference is the host's draftRows for one row, transcribed: per
// stream the row [enorm(e) ; hnorm_s(h_s)] (the embedding's half computed
// for stream 0 and copied to the rest), then eh_proj over each, a plain
// float32 sum in column order.
func predictReference(c predictCase, hid, emb []float32, w map[jlm.Role][]float32) []float32 {
	d0 := c.d
	eh := make([]float32, c.s*2*d0)
	for k := 0; k < c.s; k++ {
		row := eh[k*2*d0 : (k+1)*2*d0]
		if k == 0 {
			oracle.RMSNorm32(row[:d0], emb, w[rEnorm], c.eps)
		} else {
			copy(row[:d0], eh[:d0])
		}
		oracle.RMSNorm32(row[d0:], hid[k*d0:(k+1)*d0], w[rHnorm][k*d0:(k+1)*d0], c.eps)
	}
	out := make([]float32, c.s*d0)
	for k := 0; k < c.s; k++ {
		for r := 0; r < d0; r++ {
			var s float32
			for i := 0; i < 2*d0; i++ {
				s += w[rEH][r*2*d0+i] * eh[k*2*d0+i]
			}
			out[k*d0+r] = s
		}
	}
	return out
}

func decomposedPredict(t *testing.T, c predictCase, seed int64, edit func(*Graph)) (got, want []float32) {
	t.Helper()
	g, h, e, out := predictGraph(c)
	d := Decompose(g)
	for _, o := range d.Ops {
		if o.Kind == OpPredict {
			t.Fatalf("%s: the prologue was not decomposed", c.name)
		}
	}
	if edit != nil {
		edit(d)
	}
	hid, emb, w := predictInputs(c, seed)
	got = evalPrims(t, d, map[ValueID][]float32{h: hid, e: emb}, w)[out]
	return got, predictReference(c, hid, emb, w)
}

// predictMatches compares to the reference: exactly while one stream's norm
// is one weighted op, to a few ulps once the scales are stacked behind a
// weightless norm (two roundings where the host's has one).
func predictMatches(c predictCase, got, want []float32) bool {
	if c.s == 1 {
		if len(got) != len(want) {
			return false
		}
		for i := range want {
			if got[i] != want[i] {
				return false
			}
		}
		return true
	}
	_, ok := closeTo(got, want, 2e-5, 1e-2)
	return ok
}

// TestPredictDecomposesToItsReference holds the prologue, as primitives, to
// draftRows over several seeds, with the draft's own bindings beside it.
func TestPredictDecomposesToItsReference(t *testing.T) {
	for _, c := range predictCases {
		t.Run(c.name, func(t *testing.T) {
			for seed := int64(1); seed <= 6; seed++ {
				got, want := decomposedPredict(t, c, seed, nil)
				if !predictMatches(c, got, want) {
					t.Fatalf("seed %d: %v, the reference's %v", seed, got, want)
				}
			}
		})
	}
}

// TestPredictReadsOnlyWhatItsOutputStates: the head norm, head, embedding
// table and stream mix are the draft's, read after the block; the
// decomposition binds the prologue's three, and the source graph's op, which
// Decompose does not change, still carries all seven.
func TestPredictReadsOnlyWhatItsOutputStates(t *testing.T) {
	for _, c := range predictCases {
		g, _, _, _ := predictGraph(c)
		d := Decompose(g)
		got := map[jlm.Role]bool{}
		for _, o := range d.Ops {
			for _, w := range o.Weight {
				got[w.Role] = true
			}
		}
		if len(got) != 3 || !got[rEnorm] || !got[rHnorm] || !got[rEH] {
			t.Errorf("%s: the decomposition binds %v", c.name, got)
		}
		if len(g.Ops) != 1 || g.Ops[0].Kind != OpPredict || len(g.Ops[0].Weight) != 7 {
			t.Errorf("%s: the source graph was changed", c.name)
		}
	}
}

// TestPredictViolationArms: each defect, built as an edit of the
// decomposition, parts from the reference on some seed.
func TestPredictViolationArms(t *testing.T) {
	retarget := func(d *Graph, from, to jlm.Role) {
		for i := range d.Ops {
			for j := range d.Ops[i].Weight {
				if d.Ops[i].Weight[j].Role == from {
					d.Ops[i].Weight[j].Role = to
				}
			}
		}
	}
	arms := []struct {
		name string
		edit func(*Graph)
	}{
		{"the two halves swapped", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpConcat && len(o.In) == 2 {
					o.In[0], o.In[1] = o.In[1], o.In[0]
				}
			}
		}},
		{"the embedding normed by the hidden row's scale", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpNorm && o.Weight[0].Role == rEnorm {
					o.Weight[0].Role = rHnorm
				}
			}
		}},
		{"the hidden row normed by the embedding's scale", func(d *Graph) { retarget(d, rHnorm, rEnorm) }},
		{"the projection skipped", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpMatVec {
					w := d.Values[o.Out[0]].Width
					*o = Op{Kind: OpSlice, In: o.In, Out: o.Out, Attr: Attrs{Offset: w, Width: w}}
				}
			}
		}},
		{"the wrong epsilon", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpNorm || o.Kind == OpHeadNorm {
					o.Attr.Eps = 0.5
				}
			}
		}},
	}
	for _, arm := range arms {
		for _, c := range predictCases {
			t.Run(arm.name+"/"+c.name, func(t *testing.T) {
				shown := false
				for seed := int64(1); seed <= 6 && !shown; seed++ {
					got, want := decomposedPredict(t, c, seed, arm.edit)
					shown = !predictMatches(c, got, want)
				}
				if !shown {
					t.Fatalf("the comparison does not see: %s", arm.name)
				}
			})
		}
	}
	// Streams read their own scales: every stream normed against the first's.
	t.Run("every stream against stream 0's scale", func(t *testing.T) {
		c := predictCases[2]
		shown := false
		for seed := int64(1); seed <= 6 && !shown; seed++ {
			got, want := decomposedPredict(t, c, seed, func(d *Graph) {
				for i := range d.Ops {
					if o := &d.Ops[i]; o.Kind == OpSlice {
						o.Attr.Offset = 0
					}
				}
			})
			shown = !predictMatches(c, got, want)
		}
		if !shown {
			t.Fatalf("the comparison does not see one stream's row read for all")
		}
	})
}

// TestPredictVariantsStayWhole: a prologue whose bindings or widths the
// primitives do not chain is left as it is.
func TestPredictVariantsStayWhole(t *testing.T) {
	c := predictCases[2]
	without := func(role jlm.Role) func(*Graph) {
		return func(g *Graph) {
			o := &g.Ops[0]
			var ws []Binding
			for _, w := range o.Weight {
				if w.Role != role {
					ws = append(ws, w)
				}
			}
			o.Weight = ws
		}
	}
	for name, edit := range map[string]func(*Graph){
		"no embedding norm": without(rEnorm),
		"no hidden norm":    without(rHnorm),
		"no projection":     without(rEH),
		"a second projection": func(g *Graph) {
			g.Ops[0].Weight = append(g.Ops[0].Weight, Binding{Role: rEH, K: 2 * c.d, Rows: c.d})
		},
		"a projection of the wrong width": func(g *Graph) { g.Ops[0].Weight[0].K++ },
		"a hidden scale of one stream":    func(g *Graph) { g.Ops[0].Weight[2].K = c.d },
		"a ragged stream count": func(g *Graph) {
			g.Values[g.Ops[0].In[0]].Width++
			g.Values[g.Ops[0].Out[0]].Width++
		},
		"an output of another width": func(g *Graph) { g.Values[g.Ops[0].Out[0]].Width-- },
		"an index embedding":         func(g *Graph) { g.Values[g.Ops[0].In[1]].Elem = ElemU32 },
		"one input":                  func(g *Graph) { g.Ops[0].In = g.Ops[0].In[:1] },
	} {
		g, _, _, _ := predictGraph(c)
		edit(g)
		if d := Decompose(g); len(d.Ops) != 1 || d.Ops[0].Kind != OpPredict {
			t.Errorf("%s was decomposed into %d ops", name, len(d.Ops))
		}
	}
}
