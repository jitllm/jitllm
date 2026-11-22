package composer

import "reflect"

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

// narrowBand is a matrix tile the model prices least at the calls of more
// than the band below's rows and at most upto rows.
type narrowBand struct {
	tile MatrixTile
	upto int
}

// bands is how req's projections run by the live rows of a call, where ks
// are its kernels formed with the matrix unit at its capacity: the vector
// form at a tile of t rows at calls of at most v rows, then each of narrow's
// tiles in its band, and past the last the form ks are. At each call's rows
// the model (Matrix.cost; the vector form a pass over the weight each t
// rows, each weight element paying its read or its t rows' products, the
// greater) takes the least costly of the vector form, ks' form and the
// profile's tiles; a tile is a band of its own only where it is the least
// costly over consecutive rows below ks' form's. v is 0 where the vector
// form is never the cheaper.
func (r *Registry) bands(ks []*Kernel, req Request) (t, v int, narrow []narrowBand) {
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
		return 0, 0, nil
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
	// best is the matrix option least costly at rows: -1 for ks' own form,
	// else the tile's index.
	best := func(rows int) (int, int) {
		b, c := -1, own(rows)
		for i, tile := range tiles {
			if tile.Rows >= R {
				continue
			}
			if x := at(tile, rows); x < c {
				b, c = i, x
			}
		}
		return b, c
	}
	mat := func(rows int) int {
		_, c := best(rows)
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
	if most >= 2 {
		var ts, upto []int
		top := 0
		for t := 2; t <= most; t *= 2 {
			n := 1
			for n+1 < R && vec(n+1, t) <= mat(n+1) {
				n++
			}
			ts, upto, top = append(ts, t), append(upto, n), max(top, n)
		}
		pick, least := -1, 0
		for i := range ts {
			c := 0
			for rows := 2; rows <= top; rows++ {
				if rows <= upto[i] {
					c += vec(rows, ts[i])
				} else {
					c += mat(rows)
				}
			}
			if upto[i] > 1 && (pick < 0 || c < least) {
				pick, least = i, c
			}
		}
		if pick >= 0 {
			t, v = ts[pick], upto[pick]
		}
	}
	// The bands run up from the vector form's; the first call ks' own form
	// is the least costly at ends them, and a tile met again starts none.
	seen := map[int]bool{}
	cur := -1
	for rows := max(v, 1) + 1; rows <= R; rows++ {
		b, _ := best(rows)
		if b < 0 || seen[b] && b != cur {
			break
		}
		if b == cur {
			narrow[len(narrow)-1].upto = rows
			continue
		}
		cur, seen[b] = b, true
		narrow = append(narrow, narrowBand{tiles[b], rows})
	}
	if len(narrow) > 0 && narrow[len(narrow)-1].upto >= R {
		narrow = narrow[:len(narrow)-1]
	}
	return t, v, narrow
}

// alternatives is one form of a request formed two ways at one capacity:
// vec (the vector form, or a narrower band's tile) and mat. The codes the
// two share, in order, run at every call; between them, vec's run at calls
// of at most r rows and mat's at the rest, within the band each already
// had.
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
			c.Band.Past = max(c.Band.Past, r)
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
