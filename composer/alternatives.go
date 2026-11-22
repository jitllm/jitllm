package composer

import "github.com/jitllm/jitllm/format/jlm"

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
// elements of k a trip stages, the segments k is cut in at capacity, and
// its weight's role, which its vector form's counts are found by.
type matrixPrice struct {
	p            Matrix
	depth, split int
	role         jlm.Role
}

// band is a form a call's projections run at when its live rows are more
// than the band below's and at most upto: the vector form at a tile of vec
// rows by out outputs, or, where vec is 0, the matrix unit at tile.
type band struct {
	vec, out int
	tile     MatrixTile
	upto     int
}

// bands is how req's projections run by the live rows of a call, where ks
// are its kernels formed with the matrix unit at its capacity: each band in
// order, and past the last the form ks are. At each call's rows the model
// takes the least costly of the vector form at each tile of rows by outputs
// whose sums and terms RowTiles' registers hold (vectorCost), ks' form and
// the profile's matrix tiles (Matrix.cost). The bands run up from one row:
// the first call ks' own form is the least costly at ends them, and a form
// met again after another starts none.
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
	var splits []Split
	for _, p := range r.Profiles[d.Kind].Passes {
		switch p := p.(type) {
		case RowTiles:
			most = p.most(d)
		case Matrix:
			tiles = append([]MatrixTile{p.Tile()}, p.Tiles...)
		case Split:
			splits = append(splits, p)
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
	var opts []band
	var counts map[jlm.Role]vectorOps
	if most >= 2 {
		counts = r.vectorCounts(req)
	}
	vectors := len(counts) > 0
	for _, k := range ms {
		_, ok := counts[k.price.role]
		vectors = vectors && ok
	}
	for t := 2; vectors && t <= most; t *= 2 {
		for u := 1; t*u <= most; u *= 2 {
			fit := true
			for _, k := range ms {
				fit = fit && (u == 1 || outputsFit(k.Matrix[0], u, tileLanes(d)))
			}
			if fit {
				opts = append(opts, band{vec: t, out: u})
			}
		}
	}
	for _, tile := range tiles {
		if tile.Rows < R {
			opts = append(opts, band{tile: tile})
		}
	}
	price := func(o band, rows int) int {
		if o.vec == 0 {
			return at(o.tile, rows)
		}
		c := 0
		for _, k := range ms {
			v := counts[k.price.role]
			x := vectorTile{rows: rows, t: o.vec, u: o.out, n: k.Matrix[0], k: k.Matrix[1], bits: weightBits(k), lanes: tileLanes(d), segments: 1}
			for _, s := range splits {
				if s.tiles(v.kind) {
					x.segments = s.tileSegments(x.elements(), v.trips, o.vec*o.out, d)
					break
				}
			}
			c += k.price.p.vectorCost(x, v, d)
		}
		return c
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

// vectorTile is a call of rows rows of a projection of n outputs over k,
// bits its weightBits, in its vector form at a tile of t rows by u outputs
// a thread, an output copy lanes outputs from the next (tileLanes), its sum
// cut in segments within a workgroup (Split.tileSegments).
type vectorTile struct {
	rows, t, u, n, k, bits, lanes, segments int
}

// elements is the threads one tile of rows takes before its sum is cut.
func (x vectorTile) elements() int {
	if x.u == 1 {
		return x.n
	}
	return (x.n + x.u*x.lanes - 1) / (x.u * x.lanes) * x.lanes
}

// vectorCost is the model's time for x: a pass over the weight each t rows,
// each pass paying the greater of its read (Vector.Read a sixteen stored
// bits) and its issue -- every thread's trips, the greater of their loads
// (Vector.Load each) and their other ops (Vector.Op each), which separate
// pipes issue -- and Vector.Exposed sixteenths of the lesser, spread over
// the device's units; a call launching fewer than Vector.Warps warps a unit
// pays as many times more as it falls short.
func (p Matrix) vectorCost(x vectorTile, v vectorOps, d Device) int {
	threads := x.elements()
	loads, ops := v.at(x.t, x.u)
	issue := threads * v.trips / d.Units * max(p.Vector.Load*loads, p.Vector.Op*ops)
	read := x.n * x.k / d.Units * x.bits / 256 * p.Vector.Read
	passes := (x.rows + x.t - 1) / x.t
	c := passes * (max(read, issue) + min(read, issue)*p.Vector.Exposed/16)
	launched, want := passes*threads*x.segments, p.Vector.Warps*d.Units*x.lanes
	if launched < want {
		c = c * want / launched
	}
	return c
}

// vectorOps is what one trip of a projection's vector form issues a
// thread, untiled, by what each value depends on: its loads and its other
// ops each [neither, the row, the output, both]; trips is its sum's trips.
// A tile of t rows by u outputs issues the row's t times, the output's u
// times, both's t*u times and the rest once (RowTiles). An op only an
// address reads is left out: it folds into the loads' offsets (Affine).
type vectorOps struct {
	loads, ops [4]int
	trips      int
	kind       PrimKind
}

// at is the loads and the other ops a trip of a tile of t rows by u outputs
// issues.
func (v vectorOps) at(t, u int) (loads, ops int) {
	w := [4]int{1, t, u, t * u}
	for c := range w {
		loads += w[c] * v.loads[c]
		ops += w[c] * v.ops[c]
	}
	return loads, ops
}

// countVector is k's vectorOps, false where RowTiles does not take k.
func countVector(k *Kernel) (vectorOps, bool) {
	if !tileable(k) || len(k.Space) != 2 {
		return vectorOps{}, false
	}
	row, out := k.coordDependent(0), k.coordDependent(1)
	live := k.Live()
	// addr marks the nodes whose every reader takes them as an address.
	addr := make([]bool, len(k.Nodes))
	read := make([]bool, len(k.Nodes))
	var buf [3]int
	for i := range k.Nodes {
		for _, a := range k.args(i, &buf) {
			read[a] = true
		}
	}
	for i := range addr {
		addr[i] = read[i]
	}
	for _, s := range k.Stores {
		if !s.Barrier && !s.Run {
			addr[s.Val] = false
		}
	}
	for changed := true; changed; {
		changed = false
		for i, nd := range k.Nodes {
			if !live[i] {
				continue
			}
			for _, a := range k.args(i, &buf) {
				if addr[a] && nd.Op != OpLoad && !addr[i] {
					addr[a], changed = false, true
				}
			}
		}
	}
	v := vectorOps{trips: k.Loops[0].Trip, kind: k.Kind}
	for i, nd := range k.Nodes {
		switch {
		case !live[i] || addr[i] || !k.Varies(i, 0):
			continue
		case nd.Op == OpConst || nd.Op == OpCoord || nd.Op == OpIter || nd.Op == OpThread || nd.Op == OpFold:
			continue
		}
		c := 0
		if row[i] {
			c++
		}
		if out[i] {
			c += 2
		}
		if nd.Op == OpLoad {
			v.loads[c]++
		} else {
			v.ops[c]++
		}
	}
	return v, true
}

// vectorCounts is the vectorOps of each projection req forms in its vector
// form, by its weight's role: req formed through the device's passes before
// RowTiles, with no matrix unit, its addresses simplified as the device's
// Affine does.
func (r *Registry) vectorCounts(req Request) map[jlm.Role]vectorOps {
	req.Device.Matrix, req.Device.Tile, req.Device.RowTile, req.Device.OutTile = [3]int{}, MatrixTile{}, 0, 0
	prof := r.Profiles[req.Device.Kind].on(req.Device)
	var affine []Pass
	for _, p := range prof.Passes {
		if a, ok := p.(Affine); ok {
			affine = append(affine, a)
		}
	}
	for i, p := range prof.Passes {
		if _, ok := p.(RowTiles); ok {
			prof.Passes = append(prof.Passes[:i:i], affine...)
			break
		}
	}
	ks, err := r.formWith(req, prof, nil)
	if err != nil {
		return nil
	}
	out := map[jlm.Role]vectorOps{}
	for _, k := range ks {
		if v, ok := countVector(k); ok {
			if w, ok := weightOf(k); ok {
				out[w] = v
			}
		}
	}
	return out
}

// weightOf is the role of the weight matrix k reads.
func weightOf(k *Kernel) (jlm.Role, bool) {
	for _, b := range k.Bufs {
		if b.Kind == BufIn && b.Operand.Kind == OperandWeight {
			return b.Operand.Role, true
		}
	}
	return 0, false
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
