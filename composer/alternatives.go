package composer

import "reflect"

// Band is the live rows of a call a code runs at: Upto, where set, at most
// Upto rows; Past, where set, more than Past rows or a call stating none.
type Band struct{ Upto, Past int }

// Runs reports whether a call of rows rows launches a code of b: a call of
// 0 rows runs the form's capacity.
func (b Band) Runs(rows int) bool {
	switch {
	case b.Upto > 0:
		return rows > 0 && rows <= b.Upto
	case b.Past > 0:
		return rows <= 0 || rows > b.Past
	}
	return true
}

// Cut is a matrix form's k cut at launch by the call's rows, over Units row
// tiles at a capacity of Live rows: a call reaching u of the tiles is cut in
// 2<<i segments for the last i whose Upto[i] is u or more, one where none
// is, Most at the fewest tiles. Sums marks the combine, which a call cut in
// one segment does not run.
type Cut struct {
	Upto              []int
	Most, Units, Live int
	Sums              bool
}

// Of is the segments a call of rows rows is cut in (0 rows: the capacity).
func (c Cut) Of(rows int) int {
	u := c.Units
	if rows > 0 && rows < c.Live {
		u = (rows*c.Units + c.Live - 1) / c.Live
	}
	return c.at(u)
}

func (c Cut) at(units int) int {
	s := 1
	for i, t := range c.Upto {
		if units <= t {
			s = 2 << i
		}
	}
	return s
}

// Grid is the most row tiles of segments any call launches: its planes'
// rows, a row tile's each.
func (c Cut) Grid() int {
	g := c.Units
	for u := 1; u <= c.Units; u++ {
		g = max(g, u*c.at(u))
	}
	return g
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

// crossing is the rows a projection's tile holds in req's vector form, and
// the most rows a call runs it at, where ks are req's kernels formed with
// the matrix unit: the model's cost of the matrix kernels at a call's rows
// (Matrix.cost, cut as the call's rows cut them) against the vector form's
// at a tile of t rows -- a pass over the weight each t rows of the call,
// each weight element paying its read or its t rows' products, the greater.
// The vector form runs while it costs no more than the matrix form; of the
// tiles, the one whose calls cost least over the rows any tile runs at, the
// smaller at a tie. 0 rows where the vector form is never the cheaper.
func (r *Registry) crossing(ks []*Kernel, req Request) (int, int) {
	d, R := req.Device, req.Step.Rows
	var ms []*Kernel
	for _, k := range ks {
		if k.price.p.Rows > 0 {
			ms = append(ms, k)
		}
	}
	most := 0
	for _, p := range r.Profiles[d.Kind].Passes {
		if t, ok := p.(RowTiles); ok {
			most = t.most(d)
		}
	}
	if len(ms) == 0 || most < 2 {
		return 0, 0
	}
	mat := func(rows int) int {
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
	vec := func(rows, t int) int {
		c := 0
		for _, k := range ms {
			v := k.price.p.Vector
			c += (rows + t - 1) / t * max(v[0], v[1]*t) * (k.Matrix[0] * k.Matrix[1] / d.Units)
		}
		return c
	}
	var tiles, upto []int
	top := 0
	for t := 2; t <= most; t *= 2 {
		n := 1
		for n+1 < R && vec(n+1, t) <= mat(n+1) {
			n++
		}
		tiles, upto, top = append(tiles, t), append(upto, n), max(top, n)
	}
	best, least := -1, 0
	for i, t := range tiles {
		c := 0
		for rows := 2; rows <= top; rows++ {
			if rows <= upto[i] {
				c += vec(rows, t)
			} else {
				c += mat(rows)
			}
		}
		if upto[i] > 1 && (best < 0 || c < least) {
			best, least = i, c
		}
	}
	if best < 0 {
		return 0, 0
	}
	return tiles[best], upto[best]
}

// alternatives is one form of a request formed two ways at one capacity:
// vec without the matrix unit, its projections a tile of rows a thread over
// the weight read once, and mat with it. The codes the two share, in order,
// run at every call; between them, vec's run at calls of at most r rows and
// mat's at the rest.
func alternatives(vec, mat []PartCode, r int) []PartCode {
	n, m := len(vec), len(mat)
	lcs := make([][]int, n+1)
	for i := range lcs {
		lcs[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			switch {
			case reflect.DeepEqual(vec[i], mat[j]):
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
			c.Band = Band{Past: r}
			out = append(out, c)
		}
		gv, gm = gv[:0], gm[:0]
	}
	i, j := 0, 0
	for i < n || j < m {
		switch {
		case i < n && j < m && reflect.DeepEqual(vec[i], mat[j]):
			flush()
			out = append(out, vec[i])
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
