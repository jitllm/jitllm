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

// TestStreamMixAndPlaceVariantsStayWhole: the Sinkhorn mix, a weighted
// placement of a checkpoint block and the Sinkhorn placement are not stated
// by these primitives and are left as they are.
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
		"mismatched down":       mix(func(o *Op) { o.Weight[1].K++ }),
		"mismatched stream":     mix(func(o *Op) { o.Attr.Streams = 3 }),
		"sinkhorn place":        place(func(o *Op) { o.Attr.Sinkhorn = true }),
		"place of a checkpoint": place(func(o *Op) { o.Attr.Block = 1 }),
	} {
		d := Decompose(g)
		if len(d.Ops) != 1 || (d.Ops[0].Kind != OpStreamMix && d.Ops[0].Kind != OpStreamPlace) {
			t.Errorf("%s was decomposed into %d ops", name, len(d.Ops))
		}
	}
}

// bankCase is a residual of h streams of d whose read mixes the first nb
// checkpoints with the running residual.
type bankCase struct {
	name     string
	h, d, nb int
}

var bankCases = []bankCase{
	{name: "four streams, the whole bank", h: 4, d: 16, nb: 3},
	{name: "five streams, one checkpoint", h: 5, d: 13, nb: 1},
	{name: "three streams, ragged width", h: 3, d: 7, nb: 2},
}

func bankGraph(c bankCase) (g *Graph, in, row ValueID) {
	g = &Graph{Version: Version}
	in, row = g.value("x", c.h*c.d), g.value("row", c.d)
	g.op(Op{Kind: OpStreamMix, In: []ValueID{in}, Out: []ValueID{row}, Weight: []Binding{{Role: jlm.RoleAttnResScore, K: c.d}},
		Attr: Attrs{Streams: c.h, Eps: 1e-6, Groups: c.nb}})
	return g, in, row
}

// bankReference is the host's k3Mix for one row: each term (checkpoints
// first, the running residual last) RMSNormed with a unit weight and dotted
// with the score row, the scores' softmax, and the terms added in order at
// their probabilities by a fused multiply-add.
func bankReference(c bankCase, x, w []float32) []float32 {
	ns := c.nb + 1
	scores := make([]float32, ns)
	ones := make([]float32, c.d)
	for i := range ones {
		ones[i] = 1
	}
	term := func(j int) []float32 {
		k := j + 1
		if j == c.nb {
			k = 0
		}
		return x[k*c.d : (k+1)*c.d]
	}
	for j := range scores {
		n := make([]float32, c.d)
		oracle.RMSNorm32(n, term(j), ones, 1e-6)
		var s float32
		for i := range n {
			s = fma32(w[i], n[i], s)
		}
		scores[j] = s
	}
	mx := scores[0]
	for _, s := range scores[1:] {
		mx = max(mx, s)
	}
	var sum float32
	for j := range scores {
		scores[j] = float32(math.Exp(float64(scores[j] - mx)))
		sum += scores[j]
	}
	out := make([]float32, c.d)
	for j := range scores {
		p := scores[j] / sum
		for i, v := range term(j) {
			out[i] = fma32(p, v, out[i])
		}
	}
	return out
}

func bankRow(t *testing.T, c bankCase, seed int64, edit func(*Graph)) (got, want []float32) {
	t.Helper()
	g, in, rv := bankGraph(c)
	d := Decompose(g)
	if edit != nil {
		edit(d)
	}
	rng := rand.New(rand.NewSource(seed))
	fill := func(n int, s float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = (rng.Float32()*2 - 1) * s
		}
		return f
	}
	x, w := fill(c.h*c.d, 2), fill(c.d, 1.5)
	got = evalPrims(t, d, map[ValueID][]float32{in: x}, map[jlm.Role][]float32{jlm.RoleAttnResScore: w})[rv]
	return got, bankReference(c, x, w)
}

// TestBankMixDecomposesToItsReference holds a residual attention's read, as
// primitives, to the host's k3Mix over several seeds.
func TestBankMixDecomposesToItsReference(t *testing.T) {
	for _, c := range bankCases {
		t.Run(c.name, func(t *testing.T) {
			g, _, _ := bankGraph(c)
			for _, o := range Decompose(g).Ops {
				if o.Kind == OpStreamMix {
					t.Fatalf("the bank's read was not decomposed")
				}
			}
			for seed := int64(1); seed <= 8; seed++ {
				got, want := bankRow(t, c, seed, nil)
				if i, ok := closeTo(got, want, 2e-5, 1e-3); !ok {
					t.Fatalf("seed %d: row[%d] = %g, the reference's %g", seed, i, got[i], want[i])
				}
			}
		})
	}
}

// TestBankMixViolationArms: a read that scores the terms in another order,
// drops the norm or mixes without the softmax's maximum parts from the
// reference on some seed.
func TestBankMixViolationArms(t *testing.T) {
	c := bankCases[0]
	for name, edit := range map[string]func(*Graph){
		"terms reordered": func(g *Graph) {
			for i, o := range g.Ops {
				if o.Kind == OpSlice && o.Attr.Offset == 0 {
					g.Ops[i].Attr.Offset = c.d
				}
			}
		},
		"no norm": func(g *Graph) {
			for i, o := range g.Ops {
				if o.Kind == OpMap && o.Attr.Fn == MapSqrt {
					g.Ops[i].Attr.Fn = MapMax
					g.Ops[i].In = []ValueID{o.In[0], o.In[0]}
				}
			}
		},
		"scores not centred": func(g *Graph) {
			for i, o := range g.Ops {
				if o.Kind == OpMap && o.Attr.Fn == MapSub {
					g.Ops[i].Attr.Fn = MapMul
				}
			}
		},
	} {
		parted := false
		for seed := int64(1); seed <= 8 && !parted; seed++ {
			got, want := bankRow(t, c, seed, edit)
			if _, ok := closeTo(got, want, 2e-5, 1e-3); !ok {
				parted = true
			}
		}
		if !parted {
			t.Errorf("%s: the decomposition still matches the reference", name)
		}
	}
}

// TestBankPlaceDecomposesToItsReference: the running residual plus the
// output, or on a checkpoint block the output alone with the block's input
// banked at its stream, every other stream as it was, exactly.
func TestBankPlaceDecomposesToItsReference(t *testing.T) {
	const h, w = 5, 9
	for k := 0; k < h; k++ {
		g := &Graph{Version: Version}
		in, y, out := g.value("x", h*w), g.value("y", w), g.value("out", h*w)
		g.op(Op{Kind: OpStreamPlace, In: []ValueID{in, y}, Out: []ValueID{out}, Attr: Attrs{Streams: h, Block: k}})
		d := Decompose(g)
		for _, o := range d.Ops {
			if o.Kind == OpStreamPlace {
				t.Fatalf("block %d: the placement was not decomposed", k)
			}
		}
		rng := rand.New(rand.NewSource(int64(k + 1)))
		x, yv := make([]float32, h*w), make([]float32, w)
		for i := range x {
			x[i] = rng.Float32()*4 - 2
		}
		for i := range yv {
			yv[i] = rng.Float32()*4 - 2
		}
		got := evalPrims(t, d, map[ValueID][]float32{in: x, y: yv}, nil)[out]
		want := append([]float32(nil), x...)
		if k == 0 {
			for i := range yv {
				want[i] = x[i] + yv[i]
			}
		} else {
			copy(want[k*w:(k+1)*w], x[:w])
			copy(want, yv)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("block %d: element %d is %g, the reference's %g", k, i, got[i], want[i])
			}
		}
	}
}
