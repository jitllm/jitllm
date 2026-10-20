package graph

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/internal/oracle"
)

// streamCase is a residual of h streams of d, a gate of rank r.
type streamCase struct {
	name       string
	h, d, rank int
	inject     bool
	eps        float64
}

var streamCases = []streamCase{
	{name: "four streams, placement", h: 4, d: 16, rank: 6, inject: true, eps: 1e-6},
	{name: "four streams, ragged width", h: 4, d: 13, rank: 5, inject: true, eps: 1e-5},
	{name: "three streams, no placement", h: 3, d: 10, rank: 4, eps: 1e-6},
	{name: "one stream", h: 1, d: 8, rank: 3, inject: true, eps: 1e-6},
}

const (
	rNorm, rDown, rUp, rInj = jlm.RoleHCAttnNorm, jlm.RoleHCAttnDown, jlm.RoleHCAttnUp, jlm.RoleHCAttnInject
)

func mixGraph(c streamCase) (g *Graph, in, row, place ValueID) {
	g = &Graph{Version: Version}
	hd := c.h * c.d
	in = g.value("x", hd)
	row = g.value("row", c.d)
	ws := []Binding{{Role: rNorm, K: hd}, {Role: rDown, K: hd, Rows: c.rank}, {Role: rUp, K: c.rank, Rows: hd}}
	out := []ValueID{row}
	place = -1
	if c.inject {
		ws = append(ws, Binding{Role: rInj, K: hd, Rows: c.h})
		place = g.value("place", c.h)
		out = append(out, place)
	}
	g.op(Op{Kind: OpStreamMix, In: []ValueID{in}, Out: out, Weight: ws, Attr: Attrs{Streams: c.h, Eps: c.eps}})
	return g, in, row, place
}

func streamInputs(c streamCase, seed int64) (x []float32, w map[jlm.Role][]float32) {
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, s, off float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = off + (rng.Float32()*2-1)*s
		}
		return f
	}
	hd := c.h * c.d
	x = fill(hd, 2, 0)
	w = map[jlm.Role][]float32{rNorm: fill(hd, 0.5, 1), rDown: fill(hd*c.rank, 0.4, 0), rUp: fill(c.rank*hd, 0.4, 0),
		rInj: fill(hd*c.h, 0.4, 0)}
	return x, w
}

// mixReference is the host's hcMix, transcribed for one row: each stream
// RMSNormed against its row of the scale, the gate's low-rank pair with the
// rank's input scaled by 1/streams, the gated streams added in order and
// scaled, and the placement weights 2*sigmoid(projection/streams).
func mixReference(c streamCase, x []float32, w map[jlm.Role][]float32) (row, place []float32) {
	hd := c.h * c.d
	xn := make([]float32, hd)
	for k := 0; k < c.h; k++ {
		oracle.RMSNorm32(xn[k*c.d:(k+1)*c.d], x[k*c.d:(k+1)*c.d], w[rNorm][k*c.d:(k+1)*c.d], c.eps)
	}
	mv := func(m, v []float32, rows int) []float32 {
		out := make([]float32, rows)
		for r := range out {
			var s float32
			for i := range v {
				s += m[r*len(v)+i] * v[i]
			}
			out[r] = s
		}
		return out
	}
	inv := 1 / float32(c.h)
	lo := mv(w[rDown], xn, c.rank)
	for i, e := range lo {
		e *= inv
		lo[i] = float32(float64(e) / (1 + math.Exp(-float64(e))))
	}
	gate := mv(w[rUp], lo, hd)
	for i, e := range gate {
		gate[i] = float32(1/(1+math.Exp(-float64(e)))) * xn[i]
	}
	row = append([]float32(nil), gate[:c.d]...)
	for k := 1; k < c.h; k++ {
		for i := range row {
			row[i] += gate[k*c.d+i]
		}
	}
	for i := range row {
		row[i] *= inv
	}
	if c.inject {
		place = mv(w[rInj], xn, c.h)
		for i, e := range place {
			e *= inv
			place[i] = float32(1/(1+math.Exp(-float64(e)))) * 2
		}
	}
	return row, place
}

// closeTo reports the first element of got that parts from want by more than
// tol relative to the larger of |want| and scale.
func closeTo(got, want []float32, tol, scale float64) (int, bool) {
	if len(got) != len(want) {
		return -1, false
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol*math.Max(math.Abs(float64(want[i])), scale) || math.IsNaN(d) {
			return i, false
		}
	}
	return 0, true
}

func decomposedMix(t *testing.T, c streamCase, seed int64, edit func(*Graph)) (row, place, wantRow, wantPlace []float32) {
	t.Helper()
	g, in, rv, pv := mixGraph(c)
	d := Decompose(g)
	for _, o := range d.Ops {
		if o.Kind == OpStreamMix {
			t.Fatalf("%s: the stream mix was not decomposed", c.name)
		}
	}
	if edit != nil {
		edit(d)
	}
	x, w := streamInputs(c, seed)
	got := evalPrims(t, d, map[ValueID][]float32{in: x}, w)
	wantRow, wantPlace = mixReference(c, x, w)
	if pv >= 0 {
		place = got[pv]
	}
	return got[rv], place, wantRow, wantPlace
}

// TestStreamMixDecomposesToItsReference holds the low-rank gated read of a
// stream-major residual, as primitives, to the host's hcMix: its row and its
// placement weights to a few ulps over several seeds.
func TestStreamMixDecomposesToItsReference(t *testing.T) {
	for _, c := range streamCases {
		t.Run(c.name, func(t *testing.T) {
			for seed := int64(1); seed <= 6; seed++ {
				row, place, wantRow, wantPlace := decomposedMix(t, c, seed, nil)
				if i, ok := closeTo(row, wantRow, 2e-5, 1e-3); !ok {
					t.Fatalf("seed %d: row[%d] = %g, the reference's %g", seed, i, row[i], wantRow[i])
				}
				if c.inject {
					if i, ok := closeTo(place, wantPlace, 2e-6, 1e-3); !ok {
						t.Fatalf("seed %d: placement[%d] = %g, the reference's %g", seed, i, place[i], wantPlace[i])
					}
				} else if wantPlace != nil {
					t.Fatalf("a mix with no placement wrote one")
				}
			}
		})
	}
}

// TestStreamMixViolationArms: each of the host's defects, built as an edit
// of the decomposition, parts from the reference on some seed.
func TestStreamMixViolationArms(t *testing.T) {
	c := streamCases[0]
	// constMap is the n-th OpMap of one operand and a constant, in order:
	// 0 the rank's input scale, 1 the mean's, 2 the placement projection's
	// scale, 3 the doubling.
	constMap := func(d *Graph, n int) *Op {
		for i := range d.Ops {
			if o := &d.Ops[i]; o.Kind == OpMap && o.Attr.Fn == MapMul && len(o.Weight) == 0 && len(o.In) == 1 {
				if n == 0 {
					return o
				}
				n--
			}
		}
		t.Fatalf("no constant multiply")
		return nil
	}
	arms := []struct {
		name  string
		edit  func(*Graph)
		place bool // shown by the placement weights rather than the row
	}{
		{"the gate's rank input unscaled", func(d *Graph) { constMap(d, 0).Attr.Const = 1 }, false},
		{"the mean not divided by the streams", func(d *Graph) { constMap(d, 1).Attr.Const = 1 }, false},
		{"the placement projection unscaled", func(d *Graph) { constMap(d, 2).Attr.Const = 1 }, true},
		{"the placement weights not doubled", func(d *Graph) { constMap(d, 3).Attr.Const = 1 }, true},
		{"the placement weights halved", func(d *Graph) { constMap(d, 3).Attr.Const = 0.5 }, true},
		{"a stream dropped from the mean", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpSlice && o.Attr.Offset == 3*c.d {
					o.Attr.Offset = 2 * c.d
				}
			}
		}, false},
		{"every stream normed together", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpHeadNorm {
					o.Attr.HeadDim = c.h * c.d
				}
			}
		}, false},
		{"the wrong norm epsilon", func(d *Graph) {
			for i := range d.Ops {
				if o := &d.Ops[i]; o.Kind == OpHeadNorm {
					o.Attr.Eps = 0.5
				}
			}
		}, false},
	}
	for _, arm := range arms {
		t.Run(arm.name, func(t *testing.T) {
			shown := false
			for seed := int64(1); seed <= 6 && !shown; seed++ {
				row, place, wantRow, wantPlace := decomposedMix(t, c, seed, arm.edit)
				if arm.place {
					_, ok := closeTo(place, wantPlace, 2e-6, 1e-3)
					shown = !ok
				} else {
					_, ok := closeTo(row, wantRow, 2e-5, 1e-3)
					shown = !ok
				}
			}
			if !shown {
				t.Fatalf("the comparison does not see: %s", arm.name)
			}
		})
	}
}

func placeGraph(c streamCase) (g *Graph, in, y, w, out ValueID) {
	g = &Graph{Version: Version}
	in, y, w = g.value("x", c.h*c.d), g.value("y", c.d), g.value("w", c.h)
	out = g.value("out", c.h*c.d)
	g.op(Op{Kind: OpStreamPlace, In: []ValueID{in, y, w}, Out: []ValueID{out}, Attr: Attrs{Streams: c.h}})
	return
}

// TestStreamPlaceDecomposesToItsReference: stream k becomes x_k + w_k*y with
// the product fused, exactly (the host's axpy is a fused multiply-add), and a
// multiply then an add, which rounds the product, is told apart from it.
func TestStreamPlaceDecomposesToItsReference(t *testing.T) {
	for _, c := range streamCases {
		t.Run(c.name, func(t *testing.T) {
			g, in, yv, wv, out := placeGraph(c)
			d := Decompose(g)
			for _, o := range d.Ops {
				if o.Kind == OpStreamPlace {
					t.Fatalf("the placement was not decomposed")
				}
			}
			unfused := false
			for seed := int64(1); seed <= 8; seed++ {
				rng := rand.New(rand.NewSource(seed))
				fill := func(n int) []float32 {
					f := make([]float32, n)
					for i := range f {
						f[i] = rng.Float32()*4 - 2
					}
					return f
				}
				x, y, w := fill(c.h*c.d), fill(c.d), fill(c.h)
				got := evalPrims(t, d, map[ValueID][]float32{in: x, yv: y, wv: w}, nil)[out]
				for k := 0; k < c.h; k++ {
					for i := 0; i < c.d; i++ {
						fused := float32(math.FMA(float64(w[k]), float64(y[i]), float64(x[k*c.d+i])))
						if got[k*c.d+i] != fused {
							t.Fatalf("seed %d: stream %d element %d is %g, the fused %g", seed, k, i, got[k*c.d+i], fused)
						}
						if plain := x[k*c.d+i] + float32(w[k]*y[i]); plain != fused {
							unfused = true
						}
					}
				}
			}
			if !unfused {
				t.Fatalf("no input separates a fused multiply-add from a multiply then an add: the comparison cannot see it")
			}
		})
	}
}

// TestStreamMixAndPlaceVariantsStayWhole: the Sinkhorn mix, the softmax
// bank's single-weight read, a placement with no weights and the Sinkhorn
// placement are not stated by these primitives and are left as they are.
func TestStreamMixAndPlaceVariantsStayWhole(t *testing.T) {
	c := streamCases[0]
	mix := func(edit func(*Op)) *Graph {
		g, _, _, _ := mixGraph(c)
		edit(&g.Ops[0])
		return g
	}
	place := func(edit func(*Op)) *Graph {
		g, _, _, _, _ := placeGraph(c)
		edit(&g.Ops[0])
		return g
	}
	for name, g := range map[string]*Graph{
		"sinkhorn mix":          mix(func(o *Op) { o.Attr.Sinkhorn = true }),
		"bank mix":              mix(func(o *Op) { o.Weight, o.Out, o.Attr.Groups = o.Weight[:1], o.Out[:1], 3 }),
		"mismatched down":       mix(func(o *Op) { o.Weight[1].K++ }),
		"mismatched stream":     mix(func(o *Op) { o.Attr.Streams = 3 }),
		"sinkhorn place":        place(func(o *Op) { o.Attr.Sinkhorn = true }),
		"bank place":            place(func(o *Op) { o.In = o.In[:2] }),
		"place of a checkpoint": place(func(o *Op) { o.Attr.Block = 1 }),
	} {
		d := Decompose(g)
		if len(d.Ops) != 1 || (d.Ops[0].Kind != OpStreamMix && d.Ops[0].Kind != OpStreamPlace) {
			t.Errorf("%s was decomposed into %d ops", name, len(d.Ops))
		}
	}
}
