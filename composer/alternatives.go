package composer

import (
	"reflect"
	"slices"

	"github.com/jitllm/jitllm/format/jlm"
	"github.com/jitllm/jitllm/graph"
)

// Band is the live rows of a call a code runs at: Upto, where set, at most
// Upto rows; Past, where set, more than Past rows, or a call stating none
// where Upto is not set.
type Band struct{ Upto, Past int }

// Runs reports whether a call of rows rows launches a code of b: a call of
// 0 rows runs the form's capacity.
func (b Band) Runs(rows int) bool {
	if b.Upto > 0 && (rows <= 0 || rows > b.Upto) {
		return false
	}
	return b.Past == 0 || rows <= 0 || rows > b.Past
}

// Cut is a matrix form's k cut at launch by the call's rows: at most Most
// segments over Units row tiles at a capacity of Live rows. Sums marks the
// combine, which a call cut in one segment does not run.
type Cut struct {
	Most, Units, Live int
	Sums              bool
}

// Of is the segments a call of rows rows is cut in: the most, as a power
// of two, whose tiles fit the form's, one at the capacity.
func (c Cut) Of(rows int) int {
	if c.Most < 2 || rows <= 0 || rows >= c.Live {
		return 1
	}
	q := c.Units / ((rows*c.Units + c.Live - 1) / c.Live)
	s := 1
	for s*2 <= c.Most && s*2 <= q {
		s *= 2
	}
	return s
}

// Runs reports whether a call of rows rows launches c: its band takes the
// rows, and a cut's combine runs only where the call is cut (0 rows is the
// form's capacity).
func (c *Code) Runs(rows int) bool {
	return c.Band.Runs(rows) && !(c.Cut.Sums && c.Cut.Of(rows) == 1)
}

// alternated reports whether req forms both ways (alternatives): a step of
// rows the device's matrix unit takes a projection over.
func alternated(req Request) bool {
	d := req.Device
	return d.Kind == DeviceGPU && d.Matrix[0] > 0 && d.Units > 0 && req.Step.Rows >= d.Matrix[0]
}

// matrixPrice is what the model prices a matrix kernel by: its form, the
// elements of k a trip stages, and the segments k is cut in at capacity.
type matrixPrice struct {
	p            Matrix
	depth, split int
}

// band is a form a call's projections run at when its live rows are more
// than the band below's and at most upto: the vector form at a tile of vec
// rows, or, where vec is 0, the matrix unit at tile.
type band struct {
	vec  int
	tile MatrixTile
	upto int
}

// rows is the rows a band's form holds, within the capacity's: its most,
// and at least a tile of them, so its form takes the tile the model priced.
func (b band) rows(capacity int) int {
	return min(capacity, max(b.upto, b.vec, b.tile.Rows))
}

// stepRows reports whether every value of g holds the step's rows, none
// stating rows of its own (graph.Value.Rows, a fold of rows): a band's form
// at fewer rows then holds fewer of each. One that states them is formed at
// the capacity's rows, which the rows it states are of.
func stepRows(g *graph.Graph) bool {
	for _, v := range g.Values {
		if v.Rows > 0 {
			return false
		}
	}
	return true
}

// bands is how req's projections run by the live rows of a call, where ks
// are its kernels formed with the matrix unit at its capacity: each band in
// order, and past the last the form ks are. At each call's rows the model
// (Matrix.cost; the vector form a pass over the weight each t rows, each
// weight element paying its read or its t rows' products, the greater) takes
// the least costly of the vector form at each tile of rows up to RowTiles'
// most, ks' form and the profile's matrix tiles. The bands run up from one
// row: the first call ks' own form is the least costly at ends them, and a
// form met again after another starts none.
func (r *Registry) bands(ks []*Kernel, req Request) []band {
	d, R := req.Device, req.Step.Rows
	var ms []*Kernel
	for _, k := range ks {
		if k.price.p.Rows > 0 {
			ms = append(ms, k)
		}
	}
	most := 0
	var tiles []MatrixTile
	for _, p := range r.Profiles[d.Kind].Passes {
		switch p := p.(type) {
		case RowTiles:
			most = p.most(d)
		case Matrix:
			tiles = append([]MatrixTile{p.Tile()}, p.Tiles...)
		}
	}
	if len(ms) == 0 {
		return nil
	}
	own := func(rows int) int {
		c := 0
		for _, k := range ms {
			s := k.price.split
			if k.Cut.Most > 1 {
				s = k.Cut.Of(rows)
			}
			c += k.price.p.cost(rows, k.Matrix[0], k.Matrix[1], s, k.price.depth, d)
		}
		return c
	}
	at := func(tile MatrixTile, rows int) int {
		c := 0
		for _, k := range ms {
			q := k.price.p.with(tile)
			n, kk := k.Matrix[0], k.Matrix[1]
			least := q.cost(rows, n, kk, 1, k.price.depth, d)
			for s := 2; s <= R/tile.Rows && kk%(s*k.price.depth) == 0 && kk/s >= 2*k.price.depth; s *= 2 {
				least = min(least, q.cost(rows, n, kk, s, k.price.depth, d))
			}
			c += least
		}
		return c
	}
	vec := func(rows, t int) int {
		c := 0
		for _, k := range ms {
			v := k.price.p.Vector
			c += (rows + t - 1) / t * max(v[0], v[1]*t) * (k.Matrix[0] * k.Matrix[1] / d.Units) * weightBits(k) / 256
		}
		return c
	}
	var opts []band
	for t := 2; t <= most; t *= 2 {
		opts = append(opts, band{vec: t})
	}
	for _, tile := range tiles {
		if tile.Rows < R {
			opts = append(opts, band{tile: tile})
		}
	}
	price := func(o band, rows int) int {
		if o.vec > 0 {
			return vec(rows, o.vec)
		}
		return at(o.tile, rows)
	}
	var out []band
	seen := map[int]bool{}
	cur := -1
	for rows := 1; rows <= R; rows++ {
		b, c := -1, own(rows)
		for i, o := range opts {
			if x := price(o, rows); x < c {
				b, c = i, x
			}
		}
		if b < 0 || seen[b] && b != cur {
			break
		}
		if b == cur {
			out[len(out)-1].upto = rows
			continue
		}
		cur, seen[b] = b, true
		o := opts[b]
		o.upto = rows
		out = append(out, o)
	}
	if len(out) > 0 && out[len(out)-1].upto >= R {
		out = out[:len(out)-1]
	}
	return out
}

// weightBits is sixteen times the bits k's weight stores an element in, its
// codes' planes and their scales' bytes (72 for 4.5 bits); a weight it does
// not state is priced as f32's.
func weightBits(k *Kernel) int {
	for _, b := range k.Bufs {
		o := b.Operand
		if o.Kind != OperandWeight || o.Region != 0 {
			continue
		}
		if l, ok := jlm.Layout(o.Format); ok {
			scales := 1
			if l.HasMin {
				scales = 2
			}
			return 16*(l.Planes[0]+l.Planes[1]) + 16*8*scales/l.Sub
		}
		switch o.Format {
		case jlm.TypeF16, jlm.TypeBF16:
			return 256
		case jlm.TypeF32:
			return 512
		}
	}
	return 512
}

// alternatives is one form of a request formed two ways: vec (the vector
// form, or a narrower band's tile, formed at its band's rows) and mat. The
// codes the two share, in order, run at every call; between them, vec's run
// at calls of at most r rows and mat's at the rest, within the band each
// already had. A kernel the two form alike but at different rows (kin) is
// one of each side by side, so each form's codes keep their places among
// the other's.
func alternatives(vec, mat []PartCode, r int) []PartCode {
	n, m := len(vec), len(mat)
	lcs := make([][]int, n+1)
	for i := range lcs {
		lcs[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			switch {
			case kin(vec[i], mat[j]):
				lcs[i][j] = lcs[i+1][j+1] + 1
			case lcs[i+1][j] >= lcs[i][j+1]:
				lcs[i][j] = lcs[i+1][j]
			default:
				lcs[i][j] = lcs[i][j+1]
			}
		}
	}
	var out []PartCode
	var gv, gm []PartCode
	flush := func() {
		for _, c := range gv {
			c.Band = Band{Upto: r}
			out = append(out, c)
		}
		for _, c := range gm {
			c.Band.Past = max(c.Band.Past, r)
			out = append(out, c)
		}
		gv, gm = gv[:0], gm[:0]
	}
	i, j := 0, 0
	for i < n || j < m {
		switch {
		case i < n && j < m && kin(vec[i], mat[j]):
			flush()
			v := vec[i]
			v.Rows = mat[j].Rows
			if reflect.DeepEqual(v, mat[j]) {
				out = append(out, mat[j])
			} else {
				gv, gm = append(gv, vec[i]), append(gm, mat[j])
				flush()
			}
			i, j = i+1, j+1
		case j == m || i < n && lcs[i+1][j] >= lcs[i][j+1]:
			gv = append(gv, vec[i])
			i++
		default:
			gm = append(gm, mat[j])
			j++
		}
	}
	flush()
	return out
}

// kin reports whether a and b are one kernel, formed alike but perhaps at
// different rows: its name, passes, part and matrix, and its bindings'
// operands and uses, alike.
func kin(a, b PartCode) bool {
	if a.Name != b.Name || a.Part != b.Part || a.Matrix != b.Matrix || a.Sweep != b.Sweep || !slices.Equal(a.Passes, b.Passes) || len(a.Binds) != len(b.Binds) {
		return false
	}
	for k, x := range a.Binds {
		y := b.Binds[k]
		if x.Operand != y.Operand || x.Writes != y.Writes || x.Read != y.Read || x.Own != y.Own {
			return false
		}
	}
	return true
}
